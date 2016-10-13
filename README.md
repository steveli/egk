# Random Features for Expected Gaussian Kernels

Implementation of the random Fourier features for expected Gaussian kernels
with Fastfood and low rank approximation as described in the paper
[Classification of Sparse and Irregularly Sampled Time Series with
Mixtures of Expected Gaussian Kernels and Random Features]
(http://auai.org/uai2015/proceedings/papers/41.pdf).

## Requirements

* [Python 2.7](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/)
* [fht](https://pypi.python.org/pypi/fht)
  (Python module for fast Hadamard transform)

## Example

The following example (`demo.py`) demonstrates how to classify
irregularly sampled time series data using random features for
the expected Gaussian kernels.
The data file `data/ECG200-50.pkl` used in the example is
created based on the `ECG200` data set from the [UCR time series
classification archive](http://www.cs.ucr.edu/~eamonn/time_series_data/)
with 50% sampling density.
The data file is in Python's pickle format that stores the data set
in the form of `ts_train, ts_test, l_train, l_test`, where

* `ts_train` is the training time series as a list of tuples.
  Each tuple contains two numpy ndarrays `(t, y)`
  where `t` stores the time stamps and `y` stores the corresponding
  observed values.
* `l_train` stores the labels associated with the training time series
  `ts_train` as a numpy vector (integer).
* `ts_test` and `l_test` are the time series and corresponding labels
  of the test set.

```python
import numpy as np
import cPickle as pickle
from sklearn.svm import LinearSVC
import gp
from full_marginal import compute_means_covs
from fastfood import FastfoodEGK

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
```

## Support

The development of this code was supported by the National Science Foundation through award # IIS-1350522.

## License

MIT
