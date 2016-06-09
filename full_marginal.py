from __future__ import division
import numpy as np
from numpy.linalg import solve


def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])


def compute_means_covs(ts, t_ref, gp_parms, winsize=0, mean_shift=True):
    """
    Compute the posterior GP means and covariance matrices.

    ts: time series
    t_ref: reference time points the posterior GP is marginalized over
    gp_parms: GP hyperparameters and the noise term
    winsize: window size, 0 for using the full Gaussian (over t_ref)
    """
    a, b, c = gp_parms
    K_test = cov_mat(t_ref, t_ref, a, b)

    n_ts = len(ts)
    n_sample = len(t_ref)
    if winsize == 0:
        post_means = np.empty((n_ts, n_sample))
        post_covs = np.empty((n_ts, n_sample, n_sample))
    else:
        n_kernel = n_sample - winsize + 1
        post_means = np.empty((n_ts, n_kernel, winsize))
        post_covs = np.empty((n_ts, n_kernel, winsize, winsize))

    for idx, (t, y) in enumerate(ts):
        mean_list, cov_list = [], []
        K_train = reg_cov_mat(t, a, b, c)
        K_train_test = cov_mat(t, t_ref, a, b)
        Ktr_inv_Ktt = solve(K_train, K_train_test)
        if mean_shift:
            mu = np.mean(y)
            mean_test = mu + Ktr_inv_Ktt.T.dot(y - mu)
        else:
            mean_test = Ktr_inv_Ktt.T.dot(y)

        full_cov = K_test - K_train_test.T.dot(Ktr_inv_Ktt)
        if winsize == 0:
            post_means[idx] = mean_test
            post_covs[idx] = full_cov
        else:
            for i in xrange(n_sample - winsize + 1):
                post_means[idx, i] = mean_test[i:(i + winsize)]
                post_covs[idx, i] = full_cov[i:(i + winsize), i:(i + winsize)]

    return post_means, post_covs

