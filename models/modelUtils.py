'''
Some utility functions regarding GMMs.

@author Wolfgang Roth
'''

import numpy as np

from numpy.linalg import  slogdet
from scipy.linalg import eigh
from scipy.misc import logsumexp

def classifyGmmFullCov(gmm_params, K, x, t):
    '''
    Returns the classification error for GMMs with full covariance matrices.
    '''
    C = len(K)
    logl = np.zeros((x.shape[0], C))
     
    for c in range(C):
        logl_class = np.zeros((x.shape[0], K[c]))
         
        alpha = gmm_params['alpha_%d' % (c)]
        mu = gmm_params['mu_%d' % (c)]
        Lambda = gmm_params['Lambda_%d' % (c)]
         
        for k in range(K[c]):
            x_sub_mu = x - mu[k, :]
            logdet = slogdet(Lambda[k,:,:])[1]
            logl_class[:, k] = -x_sub_mu.shape[1] * 0.5 * np.log(2 * np.pi) + 0.5 * logdet - 0.5 * np.sum(np.dot(x_sub_mu, Lambda[k,:,:]) * x_sub_mu, axis=1)
             
        logl_class = logl_class + np.log(alpha)
        logl[:, c] = logsumexp(logl_class, axis=1)
    logl += np.log(gmm_params['prior'])
 
    ce = np.mean(t != np.argmax(logl, axis=1))
    return ce

def classifyGmmDiagCov(gmm_params, K, x, t):
    '''
    Returns the classification error for GMMs with diagonal covariance matrices.
    '''
    C = len(K)
    logl = np.zeros((x.shape[0], C))
     
    for c in range(C):
        logl_class = np.zeros((x.shape[0], K[c]))
         
        alpha = gmm_params['alpha_%d' % (c)]
        mu = gmm_params['mu_%d' % (c)]
        Lambda = gmm_params['Lambda_%d' % (c)]
         
        for k in range(K[c]):
            x_sub_mu = x - mu[k, :]
            logdet = np.sum(np.log(Lambda[k,:]))
            logl_class[:, k] = -x_sub_mu.shape[1] * 0.5 * np.log(2 * np.pi) + 0.5 * logdet - 0.5 * np.sum((x_sub_mu ** 2.) * Lambda[k,:], axis=1)
             
        logl_class = logl_class + np.log(alpha)
        logl[:, c] = logsumexp(logl_class, axis=1)
    logl += np.log(gmm_params['prior'])
 
    ce = np.mean(t != np.argmax(logl, axis=1))
    return ce

def getDplrParameters(gmm_params, S, epsilon, use_precision=False):
    '''
    Converts GMM parameters with full covariance matrices stored in a dictionary
    (as obtained from trainGmmEM) into the format that is used by the
    GMMClassifier class. The full covariance matrices are converted into the
    DPLR matrix structure described in [1]. The model parameters returned by
    this function can be readily used as initial parameters for GMMClassifier.
    
    [1] Roth W., Peharz R., Tschiatschek S., Pernkopf F., Hybrid generative-
    discriminative training of Gaussian mixture models, Pattern Recognition
    Letters 2018, (accepted)
    '''
    C = len(gmm_params['prior'])
    D = gmm_params['mu_0'].shape[1]
    K = []
    for c in range(C):
        K.append(len(gmm_params['alpha_%d' % (c)]))
    K_all = np.sum(K)
    init_mu = np.zeros((K_all, D))
    init_s = np.zeros((K_all, D, S))
    init_d_rho = np.zeros((D, K_all))
    init_prior_k_rho = np.zeros((K_all,))
    init_prior_c_rho = np.log(gmm_params['prior'])
    
    for c in range(C):
        k1 = int(np.sum(K[:c]))
        k2 = k1 + int(K[c])
        init_prior_k_rho[k1:k2] = np.log(gmm_params['alpha_%d' % (c)])
        if use_precision:
            for k in range(K[c]):
                k_idx = k1 + k
                w, v = eigh(gmm_params['Lambda_%d' % (c)][k, :, :], eigvals=(D-S,D-1))
                s = v * np.sqrt(w)
                Lambda = np.dot(s, s.T)
                d = np.diag(gmm_params['Lambda_%d' % (c)][k, :, :] - Lambda)
                d = np.maximum(d, 1e-8)
                init_mu[k_idx, :] = gmm_params['mu_%d' % (c)][k, :]
                init_s[k_idx, :, :] = s
                max_d_0 = np.maximum(d, 0) # log-sum-exp trick for numerical stability
                init_d_rho[:, k_idx] = np.log(np.exp(d - max_d_0) - np.exp(0 - max_d_0)) + max_d_0 # inverse softplus
        else:
            for k in range(K[c]):
                k_idx = k1 + k
                w, v = eigh(gmm_params['Sigma_%d' % (c)][k, :, :], eigvals=(D-S,D-1))
                s = v * np.sqrt(w)
                Sigma = np.dot(s, s.T)
                d = np.diag(gmm_params['Sigma_%d' % (c)][k, :, :] - Sigma)
                d = np.maximum(d - epsilon, 1e-8) # epsilon is added again in GMMClassifier
                init_mu[k_idx, :] = gmm_params['mu_%d' % (c)][k, :]
                init_s[k_idx, :, :] = s
                max_d_0 = np.maximum(d, 0) # log-sum-exp trick for numerical stability
                init_d_rho[:, k_idx] = np.log(np.exp(d - max_d_0) - np.exp(0 - max_d_0)) + max_d_0 # inverse softplus

    return init_mu, init_s, init_d_rho, init_prior_k_rho, init_prior_c_rho

def getDiagParameters(gmm_params, epsilon, use_precision=False):
    '''
    Converts GMM parameters with diagonal covariance matrices stored in a
    dictionary (as obtained from trainGmmEM) into the format that is used by the
    GMMClassifierDiag class. The model parameters returned by this function can
    be readily used as initial parameters for GMMClassifierDiag.
    '''
    C = len(gmm_params['prior'])
    D = gmm_params['mu_0'].shape[1]
    K = []
    for c in range(C):
        K.append(len(gmm_params['alpha_%d' % (c)]))
    K_all = np.sum(K)
    init_mu = np.zeros((K_all, D))
    init_d_rho = np.zeros((D, K_all))
    init_prior_k_rho = np.zeros((K_all,))
    init_prior_c_rho = np.log(gmm_params['prior'])
    
    for c in range(C):
        k1 = int(np.sum(K[:c]))
        k2 = k1 + int(K[c])
        init_prior_k_rho[k1:k2] = np.log(gmm_params['alpha_%d' % (c)])
        if use_precision:
            for k in range(K[c]):
                k_idx = k1 + k
                init_mu[k_idx, :] = gmm_params['mu_%d' % (c)][k, :]
                init_d_rho[:, k_idx] = np.log(1 - np.exp(-gmm_params['Lambda_%d' % (c)][k, :])) + gmm_params['Lambda_%d' % (c)][k, :] # inverse softplus
        else:
            for k in range(K[c]):
                k_idx = k1 + k
                init_mu[k_idx, :] = gmm_params['mu_%d' % (c)][k, :]
                d = np.maximum(gmm_params['Sigma_%d' % (c)][k, :] - float(epsilon), 1e-15)  # epsilon is added again in GMMClassifierDiag
                init_d_rho[:, k_idx] = np.log(1 - np.exp(-d)) + d # inverse softplus
    
    return init_mu, init_d_rho, init_prior_k_rho, init_prior_c_rho