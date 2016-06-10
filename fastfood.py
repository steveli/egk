from __future__ import division
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from sklearn.decomposition import randomized_svd
import fht


def low_rank_cov_root(covs, rank, implementation='randomized_svd'):
    """
    return X: (n_data, n_dim, rank) matrix so that X[i].dot(X[i].T) ~ covs[i]
    """
    n_data, n_dim = covs.shape[:2]
    if implementation == 'randomized_svd':
        X = np.empty((n_data, n_dim, rank))
        for i in xrange(n_data):
            U, s, V = randomized_svd(covs[i], rank)
            X[i] = U * np.sqrt(s)
    elif implementation == 'scipy':
        X = np.empty((n_data, n_dim, rank))
        for i in xrange(n_data):
            eigval, eigvec = eigh(covs[i],
                                  eigvals=(n_dim - rank, n_dim - 1))
            X[i] = eigvec * np.sqrt(eigval)
    elif implementation == 'numpy':
        eigval, eigvec = np.linalg.eigh(covs)
        idx = np.argsort(eigval, axis=-1)[:, -rank:]
        val_idx = np.ogrid[0:n_data, 0:n_dim]
        vec_idx = np.ogrid[0:n_data, 0:n_dim, 0:n_dim]
        X = (eigvec[vec_idx[0], vec_idx[1], idx[:, np.newaxis]] *
                np.sqrt(eigval[val_idx[0], idx][:, np.newaxis]))
    return X


class FastfoodEGK(object):
    def __init__(self, gamma, n_sample=None, normalize=False, rank=0,
                 random_seed=1):
        """
        Apply low-rank approximation with rank > 0
        """
        self.gamma = gamma
        self.n_sample = n_sample
        self.normalize = normalize
        self.rank = rank
        self.random_seed = random_seed

    def fit(self, means, covs):
        """
        n: number of data cases
        m: number of features
        d: dimension of Gaussians
        means: (n, d)
        covs: (n, d, d)
        """
        rnd = np.random.RandomState(self.random_seed)
        n_dim = means.shape[1]
        n_dim_pow2 = 2**int(np.ceil(np.log2(n_dim)))
        if self.n_sample is None:
            self.n_sample = n_dim_pow2
        n_sample = self.n_sample
        n_block = int(np.ceil(n_sample / n_dim_pow2))

        # Generate fastfood components
        # B: diagonal binary scaling matrix
        # Pi: permutation matrix
        # G: diagonal Gaussian matrix, G_{ii} ~ N(0, 1)
        # S: diagonal scaling matrix
        B = rnd.choice([-1, 1], size=(n_block, n_dim_pow2))
        G = rnd.normal(0, 1, size=(n_block, n_dim_pow2))
        Pi = np.empty((n_block, n_dim_pow2), dtype=int)
        S = np.sqrt(rnd.chisquare(n_dim_pow2, size=(n_block, n_dim_pow2)))
        for i in xrange(n_block):
            S[i] /= np.linalg.norm(G[i], 2)
            Pi[i] = rnd.permutation(n_dim_pow2)

        self.B = B
        self.G = G
        self.Pi = Pi
        self.S = S

        self.random_offset = rnd.uniform(0, 2 * np.pi, size=self.n_sample)

        self.n_dim = n_dim
        self.n_dim_pow2 = n_dim_pow2
        self.n_block = n_block
        return self

    def fastfood_2d(self, X):
        n_data, n_dim = X.shape
        B = self.B
        G = self.G
        Pi = self.Pi
        S = self.S

        n_block = self.n_block

        # Fastfood
        V = np.empty((n_data, n_dim * n_block))
        idx_lo = 0
        for i in xrange(n_block):
            BX = B[i] * X
            HBX = fht.fht2(BX, 1)
            PiHBX = HBX[:, Pi[i]]
            GPiHBX = PiHBX * G[i]
            HGPiHBX = fht.fht2(GPiHBX, 1)
            SHGPiHBX = HGPiHBX * S[i]
            idx_hi = idx_lo + n_dim
            V[:, idx_lo:idx_hi] = SHGPiHBX
            idx_lo = idx_hi
        V *= np.sqrt(n_dim) / self.gamma
        if self.n_sample != V.shape[1]:
            V = V[:, :self.n_sample]
        features = np.sqrt(2 / self.n_sample) * np.cos(V + self.random_offset)
        return features

    def exp_quadratic(self, X):
        n_data, n_dim = X.shape[:2]

        B = self.B
        G = self.G
        Pi = self.Pi
        S = self.S

        n_block = self.n_block

        # Fastfood
        V = np.empty((n_data, n_dim * n_block))
        idx_lo = 0
        for i in xrange(n_block):
            BX = B[i] * X
            HBX = fht.fht3(BX, 2)
            PiHBX = HBX[:, :, Pi[i]]
            GPiHBX = PiHBX * G[i]
            HGPiHBX = fht.fht3(GPiHBX, 2)
            SHGPiHBX = HGPiHBX * S[i]

            BX = B[i, :, np.newaxis] * SHGPiHBX
            HBX = fht.fht3(BX, 1)
            PiHBX = HBX[:, Pi[i]]
            GPiHBX = PiHBX * G[i, :, np.newaxis]
            HGPiHBX = fht.fht3(GPiHBX, 1)
            diag = HGPiHBX.diagonal(axis1=1, axis2=2)

            idx_hi = idx_lo + n_dim
            V[:, idx_lo:idx_hi] = diag * S[i]
            idx_lo = idx_hi

        if self.n_sample != V.shape[1]:
            V = V[:, :self.n_sample]

        return np.exp(-0.5 * V * n_dim / self.gamma**2)

    def exp_low_rank(self, X):
        n_data, n_dim = X.shape[:2]

        B = self.B[..., np.newaxis]
        G = self.G[..., np.newaxis]
        Pi = self.Pi
        S = self.S[..., np.newaxis]
        n_block = self.n_block

        # Fastfood
        V = np.empty((n_data, n_dim * n_block))
        idx_lo = 0
        for i in xrange(n_block):
            BX = B[i] * X
            HBX = fht.fht3(BX, 1)
            PiHBX = HBX[:, Pi[i]]
            GPiHBX = PiHBX * G[i]
            HGPiHBX = fht.fht3(GPiHBX, 1)
            SHGPiHBX = HGPiHBX * S[i]

            idx_hi = idx_lo + n_dim
            V[:, idx_lo:idx_hi] = np.power(SHGPiHBX, 2).sum(axis=2)
            idx_lo = idx_hi

        if self.n_sample != V.shape[1]:
            V = V[:, :self.n_sample]

        return np.exp(-0.5 * V * n_dim / self.gamma**2)

    def transform(self, means, covs):
        n_data, n_dim = means.shape
        n_dim_pow2 = self.n_dim_pow2

        if self.rank > 0:
            covs = low_rank_cov_root(covs, self.rank)
            root_cov = True
        else:
            root_cov = False

        if n_dim == n_dim_pow2:
            means_padded = means
            covs_padded = covs
        else:
            means_padded = np.zeros((n_data, n_dim_pow2))
            means_padded[:, :n_dim] = means
            if root_cov:
                covs_padded = np.zeros((n_data, n_dim_pow2, self.rank))
                covs_padded[:, :n_dim] = covs
            else:
                covs_padded = np.zeros((n_data, n_dim_pow2, n_dim_pow2))
                covs_padded[:, :n_dim, :n_dim] = covs

        cos = self.fastfood_2d(means_padded)

        if root_cov:
            exp_quad = self.exp_low_rank(covs_padded)
        else:
            exp_quad = self.exp_quadratic(covs_padded)

        features = exp_quad * cos

        if self.normalize:
            return features / norm(features, 2, axis=1)[:, np.newaxis]
        return features

    def fit_transform(self, means, covs):
        return self.fit(means, covs).transform(means, covs)

