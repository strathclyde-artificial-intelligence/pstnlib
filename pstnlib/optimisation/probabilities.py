from scipy import stats
import numpy as np
from math import log, sqrt

inf = 1e9

def rectangular_probability(mean: np.ndarray, cov: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Given a multivariate normal distribution X with mean vector and covariance matrix, calculates the rectangular probability P(l <= X <= u).
    ------------------------
    Params:
        mean: np.ndarray
            Mean vector of random variable
        cov: np.ndarray
            Covariance matrix
        lower: np.ndarray
            Vector of lower bounds
        upper: np.ndarray
            Vector of upper bounds
    ------------------------
    returns:
        float
    """
    # Checks dimensions
    assert len(mean) == len(upper)
    assert len(mean) == len(lower)
    assert np.shape(cov)[0] == len(mean)
    assert np.shape(cov)[1] == len(mean)
    # Converts into form F(xi <= z), where z = [u, -l]^T.
    omega = np.zeros((2 * len(mean), len(mean)))
    for i in range(len(mean)):
        omega[i, i] = 1
        omega[i + len(mean), i] = -1
    mean = omega @ mean
    cov = omega @ cov @ omega.transpose()

    lower = -1 * lower
    z = np.concatenate([upper, lower])
    return stats.multivariate_normal(mean, cov, allow_singular=True).cdf(z)

def rectangular_gradient(mean: np.ndarray, cov: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    '''
    Calculates the gradient of the function F(l, u) = P(l <= X <= u). This uses the formula given in:
    Van Ackooij, W., Henrion, R., Möller, A. and Zorgati, R., 2010.
    On probabilistic constraints induced by rectangular sets and multivariate normal distributions. 
    Mathematical Methods of Operations Research, 71(3), pp.535-549.
    ------------------------
    Params:
        mean: np.ndarray
            Mean vector of random variable
        cov: np.ndarray
            Covariance matrix
        lower: np.ndarray
            Vector of lower bounds
        upper: np.ndarray
            Vector of upper bounds
    ------------------------
    returns:
        tuple(np.array, np.array)
    '''
    dl, du = [], []
    I = np.eye(len(mean))
    for i in range(len(mean)):
        D = np.delete(I, i, 0)
        cov_i = np.c_[cov[:, i]]
        bar_cov = D @ (cov - 1/cov[i, i] * cov_i @ cov_i.transpose()) @ D.transpose()
        bar_u = np.delete(upper, i)
        bar_l = np.delete(lower, i)
        # For gradient with respect to upper bound of index i.
        bar_mean_u = D @ (np.c_[mean] + 1/cov[i, i] * (upper[i] - mean[i]) * cov_i)
        bar_mean_u = np.transpose(bar_mean_u).flatten()
        bar_F_u = rectangular_probability(bar_mean_u, bar_cov, bar_l, bar_u)
        fu = stats.norm(mean[i], sqrt(cov[i, i])).pdf(upper[i])
        du.append(fu * bar_F_u)
        # For gradient with respect to lower bound of index i.
        bar_mean_l = D @ (np.c_[mean] + 1/cov[i, i] * (lower[i] - mean[i]) * cov_i)
        bar_mean_l = np.transpose(bar_mean_l).flatten()
        bar_F_l = rectangular_probability(bar_mean_l, bar_cov, bar_l, bar_u)
        
        fl = stats.norm(mean[i], sqrt(cov[i, i])).pdf(lower[i])
        dl.append(-fl * bar_F_l)
    return (np.array(dl), np.array(du))
