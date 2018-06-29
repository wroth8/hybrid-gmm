'''
Train GMMs with the EM algorithm

@author Wolfgang Roth
'''

import numpy as np

from time import time
from sklearn.mixture.gaussian_mixture import GaussianMixture

def trainGmmEM(x, t, K, cov_type='full', n_max_iter=500, n_restarts_random=20, n_restarts_kmeans=20, regularize=1e-2, uniform_class_prior=True):
    '''
    Trains a GMM for each class of the given data. If EM is run for several
    times, the models with the largest log-likelihood are kept.
    
    x: The input features
    t: The target values
    K: List containing the number of components per class
    cov_type: Which covariance type should be used ('full' or 'diag')
    n_max_iter: Maximum number of iterations used for EM training
    n_restarts_random: Number of random restarts with random initial values
    n_restarts_kmeans: Number of random restarts where initial values are
      computed with the k-means algorithm.
    regularize: Regularizer for the diagonal of the covariance matrices
    uniform_class_prior: If True, the class prior is set to 1/C for each class,
      otherwise it is computed as the fraction of samples per class.
      
    Returns a dictionary containing all parameters. 
    '''
    assert n_restarts_random + n_restarts_kmeans > 0

    C = int(np.max(t) + 1)
    params = {}
    for c in range(C):
        print 'EM training for class %d/%d with %d components (N=%d, D=%d)' % (c, C, K[c], np.sum(t == c), x.shape[1])
        t_start = time()
        if n_restarts_random > 0:
            gmm1 = GaussianMixture(n_components=K[c],
                                   covariance_type=cov_type,
                                   reg_covar=regularize,
                                   max_iter=n_max_iter,
                                   n_init=n_restarts_random,
                                   init_params='random',
                                   verbose=2)
            gmm1.fit(x[t == c, :])
        if n_restarts_kmeans > 0:
            gmm2 = GaussianMixture(n_components=K[c],
                                   covariance_type=cov_type,
                                   reg_covar=regularize,
                                   max_iter=n_max_iter,
                                   n_init=n_restarts_kmeans,
                                   init_params='kmeans',
                                   verbose=2)
            gmm2.fit(x[t == c, :])
        t_elapsed = time() - t_start
        print 'EM training for class %d/%d finished in %f seconds' % (c, C, t_elapsed)

        # Select the better model of gmm1 and gmm2
        # Don't use gmm.lower_bound_, it returns the last logl and not the best
        score1 = gmm1.score(x[t == c, :]) if n_restarts_random > 0 else -np.Inf
        score2 = gmm2.score(x[t == c, :]) if n_restarts_kmeans > 0 else -np.Inf
        gmm = gmm1 if score1 > score2 else gmm2

        params['alpha_%d' % (c)] = gmm.weights_
        params['mu_%d' % (c)] = gmm.means_
        params['Sigma_%d' % (c)] = gmm.covariances_
        params['Lambda_%d' % (c)] = gmm.precisions_
    
    if uniform_class_prior == True:
        params['prior'] = np.full((C,), 1. / C, 'float32')
    else:
        _, counts = np.unique(t, return_counts=True)
        counts = np.asarray(counts, 'float32')
        params['prior'] = counts / np.sum(counts)

    return params
    