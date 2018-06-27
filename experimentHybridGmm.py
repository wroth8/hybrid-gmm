#!/usr/bin/python

'''
In this experiment a GMM with full covariance matrices is fit to each class of a
data set using the EM algorithm. These GMMs then serve as initial parameters for
hybrid training of the GMMs.

The script automatically downloads MNIST and MNIST Basic to the file system if
they are not present already.

This experiment replicates some of the experiments resulting in the values of
Table 1 from [1]. Due to randomness/multithreading/etc, the exact results are
probably not reproducible.

[1] Roth W., Peharz R., Tschiatschek S., Pernkopf F., Hybrid generative-
    discriminative training of Gaussian mixture models, Pattern Recognition
    Letters 2018, (accepted)

@author Wolfgang Roth
'''

import numpy as np

from os.path import isfile

from datasets.mnist import downloadMnist
from datasets.mnist_basic import downloadMnistBasic

from sklearn.decomposition import PCA

from models.GMMClassifier import GMMClassifier
from optimization.trainGmmEM import trainGmmEM
from optimization.trainGmmAdam import trainGmmAdam

from models.modelUtils import classifyGmmFullCov, getDplrParameters

def pcaTransform(x_tr_np, x_va_np, x_te_np, n_components=50):
    '''
    Computes the principal components for the given training data, reduces the
    number of dimensions by projecting the data sets to these principal
    components, and performs whitening.
    
    x_tr_np: The training data
    x_va_np: The validation data
    x_te_np: The test data
    n_components: The number of principal components to use
    '''
    # Compute PCA
    pca = PCA(n_components)
    pca.fit(x_tr_np)
    data_mean = np.mean(x_tr_np, axis=0)
    # PCA transformation
    x_tr_np = np.dot(x_tr_np - data_mean, pca.components_.T)
    x_va_np = np.dot(x_va_np - data_mean, pca.components_.T)
    x_te_np = np.dot(x_te_np - data_mean, pca.components_.T)
    # Whitening
    data_std = np.std(x_tr_np, axis=0)
    x_tr_np = (x_tr_np * (1 / data_std))
    x_va_np = (x_va_np * (1 / data_std))
    x_te_np = (x_te_np * (1 / data_std))
    
    return x_tr_np, x_va_np, x_te_np

def loadDataset(dataset, pca=None):
    '''
    Loads the given data set and returns its training set, validation set, and
    test set. If the given file is not found on the file system, it is
    downloaded from the internet first. Optionally, PCA is performed to reduce
    the input features to fewer dimensions and to perform whitening.
    
    dataset: The data set to be loaded. 'mnist' and 'mnist_basic' will be
      downloaded if they are not found on the file system. Other data sets must
      be located in a file named '<dataset>.npz'
    pca: The number of dimensions to which the input features should be reduced
      using PCA or None if no PCA should be performed
    '''
    if dataset == 'mnist':
        if not isfile('mnist.npz'):
            downloadMnist('mnist.npz')
    elif dataset == 'mnist_basic':
        if not isfile('mnist_basic.npz'):
            downloadMnistBasic('mnist_basic.npz')
    
    with np.load(dataset + '.npz') as data:
        x_tr_np = data['x_tr_np']
        t_tr_np = data['t_tr_np']
        x_va_np = data['x_va_np']
        t_va_np = data['t_va_np']
        x_te_np = data['x_te_np']
        t_te_np = data['t_te_np']

    if pca is not None:
        x_tr_np, x_va_np, x_te_np = pcaTransform(x_tr_np, x_va_np, x_te_np, pca)
    
    return x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np

def getParameters(dataset, pca=None, objective='hybrid_mm'):
    '''
    Returns parameters found with Bayesian optimization [1] for the following
    combinations of arguments:
      <mnist,mnist_basic>,<None/50>,<hybrid_mm/hybrid_cll>
    For all other combinations a set of default parameters is returned.
    
    dataset: The data set
    pca: The number of dimensions to which the input features should be reduced
      using PCA or None if no PCA should be performed
    objective: The objective that is used to train
    
    [1] Snoek J., Larochelle H., Adams R.P., Practical Bayesian optimization of
        machine learning algorithms, in: NIPS 2012, pp. 2960-2968
    '''
    default_params = {}
    default_params['K'] = [10,10,10,10,10,10,10,10,10,10]
    default_params['epsilon'] = 1e-2
    default_params['S'] = 25
    default_params['step_size'] = 1e-3
    default_params['gamma'] = 100.0
    default_params['lambda'] = 0.1
    default_params['parameterization'] = 'covariance'
    default_params['n_epochs'] = 50
    
    params = {}
    if dataset == 'mnist':
        if pca is None and objective == 'hybrid_mm':
            params['K'] = [19,20,20,17,18,20,11,20,16,9]
            params['epsilon'] = 1e-1
            params['S'] = 25
            params['step_size'] = 0.00056718476264925525
            params['gamma'] = 100.0
            params['lambda'] = 0.082224613316728826
            params['parameterization'] = 'covariance'
            params['n_epochs'] = 100
        elif pca is None and objective == 'hybrid_cll':
            params['K'] = [19,20,20,17,18,20,11,20,16,9]
            params['epsilon'] = 1e-1
            params['S'] = 25
            params['step_size'] = 0.0004282175778818493
            params['gamma'] = np.NaN
            params['lambda'] = 0.057993548043951952
            params['parameterization'] = 'covariance'
            params['n_epochs'] = 100
        elif pca is 50 and objective == 'hybrid_mm':
            params['K'] = [20,4,15,12,10,8,15,13,20,15]
            params['epsilon'] = 1e-3
            params['S'] = 25
            params['step_size'] = 0.0020745026134397354
            params['gamma'] = 59.725819772414454
            params['lambda'] = 0.24579488636781685
            params['parameterization'] = 'precision'
            params['n_epochs'] = 100
        elif pca is 50 and objective == 'hybrid_cll':
            params['K'] = [17,5,19,7,20,11,19,8,20,10]
            params['epsilon'] = 1e-1 
            params['S'] = 22
            params['step_size'] = 0.00012883582913784423
            params['gamma'] = np.NaN
            params['lambda'] = 0.00091105119440064539
            params['parameterization'] = 'precision'
            params['n_epochs'] = 100
        else:
            params = default_params
    elif dataset == 'mnist_basic':
        if pca is None and objective == 'hybrid_mm':
            params['K'] = [17,20,20,7,11,9,20,18,20,20]
            params['epsilon'] = 1e-1
            params['S'] = 25
            params['step_size'] = 0.00019201285614918643
            params['gamma'] = 100.0
            params['lambda'] = 0.7974772494299518
            params['parameterization'] = 'covariance'
            params['n_epochs'] = 500
        elif pca is None and objective == 'hybrid_cll':
            params['K'] = [17,20,20,7,11,9,20,18,20,20]
            params['epsilon'] = 1e-1
            params['S'] = 25
            params['step_size'] = 0.00080984777623103815
            params['gamma'] = np.NaN
            params['lambda'] = 0.99570251907029028
            params['parameterization'] = 'covariance'
            params['n_epochs'] = 500
        elif pca == 50 and objective == 'hybrid_mm':
            params['K'] = [6,19,20,4,13,4,9,6,7,7]
            params['epsilon'] = 1e-1
            params['S'] = 24
            params['step_size'] = 0.0001
            params['gamma'] = 30.472564842678782
            params['lambda'] = 0.0088875223665784917
            params['parameterization'] = 'precision'
            params['n_epochs'] = 500
        elif pca == 50 and objective == 'hybrid_cll':
            params['K'] = [4,2,11,4,1,4,1,1,8,1]
            params['epsilon'] = 1e-3 
            params['S'] = 25
            params['step_size'] = 0.0033225802170367556
            params['gamma'] = np.NaN
            params['lambda'] = 0.26341790385068836
            params['parameterization'] = 'covariance'
            params['n_epochs'] = 500
        else:
            params = default_params
    else:
        params = default_params
    return params

def getGmmEM(x, t, K, epsilon, filename):
    '''
    Trains a GMM using the EM algorithm on the given data and stores it to the
    given file. If the file is already present, the GMM is loaded from that
    file.
    
    x: The input features
    t: The target values
    K: List with C entries corresponding to the number of components per class
    epsilon: The regularizer that is added to the diagonal during EM training
    filename: The location where the GMM should be stored to/loaded from
    '''
    C = len(K)
    if isfile(filename):
        print 'Initial GMM already computed with EM algorithm. Loading parameters from file...'
        params = {}
        with np.load(filename) as data:
            params['prior'] = data['prior']
            for c in range(C):
                params['alpha_%d' % (c)] = data['alpha_%d' % (c)]
                params['mu_%d' % (c)] = data['mu_%d' % (c)]
                params['Sigma_%d' % (c)] = data['Sigma_%d' % (c)]
                params['Lambda_%d' % (c)] = data['Lambda_%d' % (c)]
    else:
        print 'Training initial GMM with the EM algorithm'
        params = trainGmmEM(x, t, K, cov_type='full', n_restarts_random=5,
                            n_restarts_kmeans=5, regularize=epsilon,
                            uniform_class_prior=True)
        np.savez(filename, **params)
    
    return params

if __name__ == '__main__':
    '''
    Specify the following three variables to determine the experiment:
    dataset: What data set to use. The data set is expected to be stored in a
      file called '<dataset>.npz'. For the format take a look at the datasets-
      package. 'mnist' and 'mnist_basic' will automatically trigger a download,
      process the data, and create the npz-file.
    pca: Set to None to leave the input data unchanged. If set to a number, PCA
      will be used to perform whitening and reduce the number of dimensions to
      the given value.
    objective: Supported objectives for training are
      'hybrid_mm': Hybrid large-margin objective
      'hybrid_cll' Hybrid conditional-log-likelihood objective
      'disc_mm': Discriminative large-margin objective
      'disc_cll': Discriminative conditional-log-likelihood objective
    ''' 
    dataset = 'mnist'
    pca = 50
    objective = 'hybrid_mm' 
    
    np.random.seed(1234)
    
    x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np = loadDataset(dataset, pca)
    
    C = int(np.max([np.max(t_tr_np), np.max(t_va_np), np.max(t_te_np)]) + 1)
    D = x_tr_np.shape[1]
    
    params = getParameters(dataset, pca, objective)
    
    gmm_em_filename = 'gmm_em_%s%s_eps%1.0e_K_%s.npz' % (dataset, '' if pca is None else ('_pca%d' % (pca)), params['epsilon'], '_'.join([str(e) for e in params['K']]))
    print 'GMM EM Filename', gmm_em_filename
    gmm_params = getGmmEM(x_tr_np, t_tr_np, params['K'], params['epsilon'], gmm_em_filename)
    
    print 'Classification Results for ML GMMs'
    print 'CE Tr:', classifyGmmFullCov(gmm_params, params['K'], x_tr_np, t_tr_np)
    print 'CE Va:', classifyGmmFullCov(gmm_params, params['K'], x_va_np, t_va_np)
    print 'CE Te:', classifyGmmFullCov(gmm_params, params['K'], x_te_np, t_te_np)

    init_params = getDplrParameters(gmm_params, params['S'], params['epsilon'], params['parameterization'] == 'precision')
    if params['parameterization'] == 'precision':
        model = GMMClassifier(C, D, params['K'], params['S'], epsilon=1e-6, use_precision=True,
                              tradeoff=params['lambda'], gamma=params['gamma'], eta=10,
                              init_params=init_params)
    else:
        model = GMMClassifier(C, D, params['K'], params['S'], epsilon=params['epsilon'],
                              use_precision=False, tradeoff=params['lambda'], gamma=params['gamma'],
                              eta=10, init_params=init_params)
    
    optim_cfg = {'objective' : objective,
                 'n_batch' : 100,
                 'n_epochs' : params['n_epochs'],
                 'step_size' : params['step_size']}
    model = trainGmmAdam(model, optim_cfg, 5678, x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np)