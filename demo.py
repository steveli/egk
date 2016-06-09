import numpy as np
import cPickle as pickle
from sklearn.svm import LinearSVC
import gp
from full_marginal import compute_means_covs
from fastfood import FastfoodEGK


def main():
    np.random.seed(111)
    with open('data/ECG200-50.pkl', 'rb') as f:
        ts_train, ts_test, l_train, l_test = pickle.load(f)

    # Estimate GP hyperparameters and the noise parameter by maximizing
    # the marginal likelihood.
    gp_parms = gp.learn_hyperparms(ts_train)

    # All time series are defined over a common time interval [0, 1].
    # We use 300 evenly-spaced reference time points between [0, 1]
    # to represent each time series.
    t_ref = np.linspace(0, 1, 300)

    # Compute the marginal posterior mean and covariance matrix for
    # both training and test time series
    train_means, train_covs = compute_means_covs(ts_train, t_ref, gp_parms)
    test_means, test_covs = compute_means_covs(ts_test, t_ref, gp_parms)

    # We use 500 random features with low-rank approximation, rank 10 in this
    # case, and normalize the random feature vector to have unit length.
    # By dropping the rank argument or set rank to 0 turns off the low rank
    # approximation.
    # The parameters gamma and C can be chosen using cross validation.
    rp = FastfoodEGK(gamma=20, n_sample=500, rank=10,
                     normalize=True)
    clf = LinearSVC(C=100)
    X_train = rp.fit_transform(train_means, train_covs)
    clf.fit(X_train, l_train)
    X_test = rp.transform(test_means, test_covs)
    l_predict = clf.predict(X_test)
    accuracy = np.mean(l_predict == l_test)
    print accuracy


if __name__ == '__main__':
    main()
