import numpy as np
import numpy as np
from scipy.stats import multivariate_normal as norm
from scipy import stats
import sys

def flatten(column_vector):
    '''
    Description:    Changes numpy column vector to array    
    Input:          A numpy P x 1 column vector
    Output:         A numpy p dimensional array 
    '''
    return np.transpose(column_vector).flatten()

def prob(z, mean, cov):
    '''
    Description:    Scipy multivariate normal cdf to evaluate the probablity F(z)
    Input:          z:      a vector at which to evaluate the probability
                    mean:   a vector of mean values
                    cov:    a covariance matrix
    Output          float:  Value of F(z) for N(mean, cov)
    '''
    prob = norm(mean, cov, allow_singular=True).cdf(z)
    if prob == 0:
        return sys.float_info.min
    else:
        return prob

def generate_random_correlation(n, eta, size=1):
    """
    Description:    Code for generating random positive semidefinite correlation matrices. Taken from https://gist.github.com/junpenglao/b2467bb3ad08ea9936ddd58079412c1a
                    based on code from "Generating random correlation matrices based on vines and extended onion method", Daniel Lewandowski, Dorots Kurowicka and Harry Joe, 2009.
    Input:          n:      Size of correlation matrix
                    eta:    Parameter - the larger eta is, the closer to the identity matrix will be the correlation matrix (more details see https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices)
                    size:   Number of samples
    Output:         Correlation matrix
    """
    beta0 = eta - 1 + n/2
    shape = n * (n-1) // 2
    triu_ind = np.triu_indices(n, 1)
    beta_ = np.array([beta0 - k/2 for k in triu_ind[0]])
    # partial correlations sampled from beta dist.
    P = np.ones((n, n) + (size,))
    P[triu_ind] = stats.beta.rvs(a=beta_, b=beta_, size=(size,) + (shape,)).T
    # scale partial correlation matrix to [-1, 1]
    P = (P-.5)*2
    
    for k, i in zip(triu_ind[0], triu_ind[1]):
        p = P[k, i]
        for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
            p = p * np.sqrt((1 - P[l, i]**2) *
                            (1 - P[l, k]**2)) + P[l, i] * P[l, k]
        P[k, i] = p
        P[i, k] = p
    return np.transpose(P, (2, 0 ,1))[0]