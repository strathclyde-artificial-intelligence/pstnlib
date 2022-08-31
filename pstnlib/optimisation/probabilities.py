from scipy import stats
import numpy as np

def rectangular_probability(mean: np.ndarray, cov: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """
    Given a multivariate normal distribution X with mean vector and covariance matrix, calculates the rectangular probability P(l <= X <= u).
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