from __future__ import division
import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from collections import defaultdict
import cPickle as pickle
from full_marginal import compute_means_covs
import gp


class MultiEGKSampler(object):
    def __init__(self, gamma, n_sample=100, normalize=False, random_seed=1):
        self.gamma = gamma
        self.n_sample = n_sample
        self.normalize = normalize
        self.random_seed = random_seed

    def fit(self, means, covs):
        """
        n: number of data cases
        k: number of kernels
        m: number features
        d: dimension of gaussians

        means: (n, k, d)
        covs: (n, k, d, d)
        """

        rnd = np.random.RandomState(self.random_seed)
        n_data, n_kernel, n_dim = means.shape

        # shape (m, k, d)
        self.random_weight = rnd.normal(0, 1 / self.gamma,
                                        size=(self.n_sample, n_kernel, n_dim))

        # shape (m, k)
        self.random_offset = rnd.uniform(0, 2 * np.pi,
                                         size=(self.n_sample, n_kernel))

        return self

    def transform(self, means, covs):
        w = self.random_weight
        b = self.random_offset
        n_data, n_kernel, n_dim = means.shape
        n_sample = self.n_sample * n_kernel

        w_ = w[..., np.newaxis]
        covs_ = covs[:, np.newaxis]

        fvec = (np.exp(-0.5 * ((w_ * covs_).sum(axis=-2) * w).sum(axis=-1)) *
                np.cos((w * means[:, np.newaxis]).sum(axis=-1) + b))
        fvec = fvec.reshape(n_data, n_sample)

        if self.normalize:
            return fvec / norm(fvec, 2, axis=1)[:, np.newaxis]
        return fvec * np.sqrt(2 / n_sample)

    def fit_transform(self, means, covs):
        return self.fit(means, covs).transform(means, covs)


def grid_search_cv(means, covs, label, gamma_grid, c_grid, n_sample=50,
                   normalize=True):
    best_score = -np.inf

    for gamma in gamma_grid:
        rp = MultiEGKSampler(gamma, n_sample=n_sample, normalize=normalize)
        scores = defaultdict(float)
        for idx_train, idx_test in StratifiedKFold(label):
            X_train = rp.fit_transform(means[idx_train], covs[idx_train])
            X_test = rp.transform(means[idx_test], covs[idx_test])
            l_train = label[idx_train]
            l_test = label[idx_test]
            for C in c_grid:
                clf = LinearSVC(C=C)
                clf.fit(X_train, l_train)
                l_predict = clf.predict(X_test)
                accuracy = np.mean(l_predict == l_test)
                scores[C] += accuracy

        best_C_score, best_C = max((score, C)
                                   for (C, score) in scores.iteritems())
        #print gamma, scores
        if best_C_score > best_score:
            best_score = best_C_score
            best_parms = {'gamma': gamma, 'C': best_C}

    return best_parms


def main():
    with open('data/ECG200-50.pkl', 'rb') as f:
        ts_train, ts_test, l_train, l_test = pickle.load(f)
    gp_parms = gp.learn_hyperparms(ts_train)
    t_ref = np.linspace(0, 1, 300)
    winsize = 20
    train_means, train_covs = compute_means_covs(ts_train, t_ref,
                                                 gp_parms, winsize)
    test_means, test_covs = compute_means_covs(ts_test, t_ref,
                                               gp_parms, winsize)

    total_sample = 5000
    n_kernel = train_means.shape[1]
    n_sample = total_sample // n_kernel
    normalize = True

    best_parms = grid_search_cv(train_means, train_covs, l_train,
                                gamma_grid=[0.1, 0.5, 1, 5, 10, 20, 50],
                                c_grid=[1, 10, 100, 1e3, 1e4, 1e5],
                                n_sample=n_sample,
                                normalize=normalize)
    print best_parms

    rp = MultiEGKSampler(best_parms['gamma'], n_sample=n_sample,
                         normalize=normalize)
    clf = LinearSVC(C=best_parms['C'])
    X_train = rp.fit_transform(train_means, train_covs)
    clf.fit(X_train, l_train)

    X_test = rp.transform(test_means, test_covs)
    l_predict = clf.predict(X_test)

    accuracy = np.mean(l_predict == l_test)
    print accuracy


if __name__ == '__main__':
    main()

