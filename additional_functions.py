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