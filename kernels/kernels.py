"""
    Kernels in sklearn format
"""
from sklearn.metrics.pairwise import *

def anova_kernel(X, Y=None, max_degree=3, gamma=None, degree=1, dense_output=True):
    """
    Compute the anova kernel between X and Y.
    Parameters
    ----------
    X : array of shape (n_samples_1, n_features)
    Y : array of shape (n_samples_2, n_features)
    max_degree : int, default 3
    gamma : float, default None
        if None, defaults to 1.0 / n_features
    degree : int, default 1
    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
        .. versionadded:: 0.20
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(1, max_degree):
        dk = -gamma * euclidean_distances(np.power(X, k), np.power(Y, k), squared=True)
        np.exp(dk, dk) # In place
        np.power(dk, degree)
        K += dk
    return K


def rational_kernel(X, Y=None, c=1):
    """
    Compute the rational kernel between X and Y
    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)
    Y : ndarray of shape (n_samples_2, n_features)
    c: float, default 1
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    K = euclidean_distances(X, Y, squared=True) 
    np.divide(K, K + c, K)

    return 1 - K

def spherical_kernel(X, Y=None, gamma=None):
    """
    Compute the spherical kernel between X and Y
    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)
    Y : ndarray of shape (n_samples_2, n_features)
    gamma: float, default None
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y)
    K = np.where(K <= gamma, K, 0)
    K = 1. - 3./2. * K / gamma + 1./2. * np.power(K / gamma, 3)

    return K

def cauchy_kernel(X, Y=None, gamma=None):
    """
    Compute the cauchy kernel between X and Y
    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)
    Y : ndarray of shape (n_samples_2, n_features)
    gamma: float, default None
    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True) / (gamma ** 2)
    
    return 1. / (1. + K)

ADD_PAIRWISE_KERNEL_FUNCTIONS = {
    'anova': anova_kernel,
    'rational': rational_kernel,
    'spherical': spherical_kernel,
    'cauchy': cauchy_kernel}

PAIRWISE_KERNEL_FUNCTIONS.update(ADD_PAIRWISE_KERNEL_FUNCTIONS)

ADD_KERNEL_PARAMS = {
    'anova': frozenset(["max_degree", "gamma", "degree"]),
    'rational': frozenset(["c"]),
    'spherical': frozenset(["gamma"]),
    'cauchy': frozenset(["gamma"])}
KERNEL_PARAMS.update(ADD_KERNEL_PARAMS)