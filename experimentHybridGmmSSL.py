#!/usr/bin/python

'''
In this experiment, we train GMMs using both labeled and unlabeled data. First,
GMMs are trained with the EM algorithm using the labeled data only. Next, hybrid
GMMs are initialized using these GMMs and trained according to the hybrid
objective for semi-supervised learning on the whole data set which also contains
unlabeled data. 

The script automatically downloads MNIST and MNIST Basic to the file system if
they are not present already.

This experiment replicates some of the experiments resulting in the values of
Table 2 from [1]. Due to randomness/multithreading/etc, the exact results are
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

from models.GMMClassifierSSL import GMMClassifierSSL
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

def loadDataset(dataset, pca=None, n_labeled=50000, n_unlabeled=0):
    '''
    Loads the given data set and returns its training set, validation set, and
    test set. If the given file is not found on the file system, it is
    downloaded from the internet first. Optionally, PCA is performed to reduce
    the input features to fewer dimensions and to perform whitening. The number
    of labeled and unlabeled training samples can be varied. Larger values of
    n_labeled result in a strict superset of labeled samples. For fixed
    n_labeled samples, larger values of n_unlabeled result in a strict superset
    of unlabeled samples. A fixed seed for the random number generator is used
    to obtain the same data sets used for the experiments in [1].
    
    dataset: The data set to be loaded. 'mnist' and 'mnist_basic' will be
      downloaded if they are not found on the file system. Other data sets must
      be located in a file named '<dataset>.npz'
    pca: The number of dimensions to which the input features should be reduced
      using PCA or None if no PCA should be performed
    n_labeled: The number of labeled training samples in the data set.
    n_unlabeled: The number of unlabeled training samples in the data set. Note
      that if n_labeled and n_unlabeled do not sum up to the number of total
      samples, some samples are removed from the data set.
      
    [1] Roth W., Peharz R., Tschiatschek S., Pernkopf F., Hybrid generative-
        discriminative training of Gaussian mixture models, Pattern Recognition
        Letters 2018, (accepted)
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

    # Generate random labeled/unlabeled data. The labels are removed as follows:
    # (1) Generate a random permutation of size N_tr
    # (2) The first n_labeled samples of the random permutation are indices of
    #     labeled samples.
    # (3) The next n_unlabeled samples of the random permutation are indices of
    #     unlabeled samples. Note that some samples are excluded if n_labeled+
    #     n_unlabeled is not equal to the total number of samples.
    # (4) The first n_labeled+n_unlabeled samples of the random permutation are
    #     the used samples. These indices are sorted to preserve the ordering of
    #     the original data set.
    if n_labeled != x_tr_np.shape[0]:
        rng_ssl = np.random.RandomState(1)
        randperm = rng_ssl.permutation(np.arange(x_tr_np.shape[0]))
        idx_used = np.sort(randperm[:n_labeled+n_unlabeled])
        idx_unlabeled = randperm[n_labeled:n_labeled+n_unlabeled]
        t_tr_np[idx_unlabeled] = -1 # set to unlabeled
        t_tr_np = t_tr_np[idx_used]
        x_tr_np = x_tr_np[idx_used[:,None], np.arange(x_tr_np.shape[1])]

    return x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np

def getParameters(dataset, pca=None, objective='hybrid_mm', n_labeled=50000, n_unlabeled=0):
    '''
    Returns parameters found with Bayesian optimization [1] for the following
    combinations of arguments:
      <mnist>,<50>,<hybrid_mm/hybrid_cll>,<100,250,500,1000,2500,5000,10000,25000,50000>,<50000-n_labeled>
    For all other combinations a set of default parameters is returned.
    
    dataset: The data set
    pca: The number of dimensions to which the input features should be reduced
      using PCA or None if no PCA should be performed
    objective: The objective that is used to train
    n_labeled: The number of labeled samples
    n_unlabeled: The number of unlabeled samples
    
    [1] Snoek J., Larochelle H., Adams R.P., Practical Bayesian optimization of
        machine learning algorithms, in: NIPS 2012, pp. 2960-2968
    '''
    default_params = {}
    default_params['K'] = [3,3,3,3,3,3,3,3,3,3]
    default_params['epsilon'] = 1e-2
    default_params['S'] = 25
    default_params['step_size'] = 1e-3
    default_params['gamma'] = 100.0
    default_params['lambda'] = 0.1
    default_params['kappa'] = 0.9
    default_params['parameterization'] = 'covariance'
    default_params['n_epochs'] = 50
    
    params = {}
    if dataset == 'mnist':
        if pca is 50 and objective == 'hybrid_mm_ssl':
            if n_labeled == 100 and n_unlabeled == 49900:
                params['K'] = [3 for _ in range(10)]
                params['epsilon'] = 1e-1
                params['S'] = 25
                params['step_size'] = 0.00015022193396674232
                params['gamma'] = 0.28556901004722451
                params['lambda'] = 0.00091105119440064539
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 250 and n_unlabeled == 49750:
                params['K'] = [3 for _ in range(10)]
                params['epsilon'] = 1e-1
                params['S'] = 25
                params['step_size'] = 0.0013854144382826103
                params['gamma'] = 0.01
                params['lambda'] = 0.9990889488055994
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 500 and n_unlabeled == 49500:
                params['K'] = [4 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 0.00071464003944782663
                params['gamma'] = 0.14415026918651322
                params['lambda'] = 0.00092408352746785501
                params['kappa'] = 0.00091202883409830806
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 1000 and n_unlabeled == 49000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1  
                params['S'] = 25
                params['step_size'] = 0.00013282383873645714
                params['gamma'] = 0.01
                params['lambda'] = 0.99908811716149704
                params['kappa'] = 0.91822332231709558
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 2500 and n_unlabeled == 47500:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1  
                params['S'] = 25
                params['step_size'] = 0.00071935314140847862
                params['gamma'] = 0.01
                params['lambda'] = 0.039814728883153454
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 5000 and n_unlabeled == 45000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1  
                params['S'] = 25
                params['step_size'] = 1.0000000000000001e-05
                params['gamma'] = 0.01
                params['lambda'] = 0.9990889488055994
                params['kappa'] = 0.001080381219209626
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 10000 and n_unlabeled == 40000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-2  
                params['S'] = 25
                params['step_size'] = 0.0053244946358643307
                params['gamma'] = 100.0
                params['lambda'] = 0.12268095457584757
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 25000 and n_unlabeled == 25000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-2  
                params['S'] = 25
                params['step_size'] = 0.0010307776842991062
                params['gamma'] = 100.0
                params['lambda'] = 0.73755283094689794
                params['kappa'] = 0.9990889488055994
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 50000 and n_unlabeled == 0:
                params['K'] = [20,4,15,12,10,8,15,13,20,15]
                params['epsilon'] = 1e-3
                params['S'] = 25
                params['step_size'] = 0.0020745026134397354
                params['gamma'] = 59.725819772414454
                params['lambda'] = 0.24579488636781685
                params['kappa'] = 1.
                params['parameterization'] = 'precision'
                params['n_epochs'] = 100
            else:
                params = default_params
        elif pca is 50 and objective == 'hybrid_cll_ssl':
            if n_labeled == 100 and n_unlabeled == 49900:
                params['K'] = [3 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 0.00059702580875359854
                params['gamma'] = np.NaN
                params['lambda'] = 0.00091105119440064539
                params['kappa'] = 0.98364898292515046
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 250 and n_unlabeled == 49750:
                params['K'] = [3 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 23
                params['step_size'] = 0.0011663186526704411
                params['gamma'] = np.NaN
                params['lambda'] = 0.00091105119440064539
                params['kappa'] = 0.081709111626870515
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 500 and n_unlabeled == 49500:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 4.3321340485604352e-05
                params['gamma'] = np.NaN
                params['lambda'] = 0.9990889488055994
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 1000 and n_unlabeled == 49000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 1.0000000000000001e-05
                params['gamma'] = np.NaN
                params['lambda'] = 0.99885485542162145
                params['kappa'] = 0.56250847702953022
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 2500 and n_unlabeled == 47500:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 0.00043645995441589071
                params['gamma'] = np.NaN
                params['lambda'] = 0.00091105119440064539
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 5000 and n_unlabeled == 45000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 9.9955252166410638e-05
                params['gamma'] = np.NaN
                params['lambda'] = 0.033042557396901172
                params['kappa'] = 0.00091105119440064539
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 10000 and n_unlabeled == 40000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 1.0000000000000001e-05
                params['gamma'] = np.NaN
                params['lambda'] = 0.00092330375738138037
                params['kappa'] = 0.0056619335270511866
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 25000 and n_unlabeled == 25000:
                params['K'] = [5 for _ in range(10)]
                params['epsilon'] = 1e-1 
                params['S'] = 25
                params['step_size'] = 0.00040242764647642623
                params['gamma'] = np.NaN
                params['lambda'] = 0.00092988652712629888
                params['kappa'] = 0.9990889488055994
                params['parameterization'] = 'covariance'
                params['n_epochs'] = 100
            elif n_labeled == 50000 and n_unlabeled == 0:
                params['K'] = [17,5,19,7,20,11,19,8,20,10]
                params['epsilon'] = 1e-1 
                params['S'] = 22
                params['step_size'] = 0.00012883582913784423
                params['gamma'] = np.NaN
                params['lambda'] = 0.00091105119440064539
                params['kappa'] = 1.
                params['parameterization'] = 'precision'
                params['n_epochs'] = 100
            else:
                params = default_params
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
        params = trainGmmEM(x, t, K, cov_type='full', n_restarts_random=20,
                            n_restarts_kmeans=20, regularize=epsilon,
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
      'hybrid_mm_ssl': Hybrid large-margin objective
      'hybrid_cll_ssl' Hybrid conditional-log-likelihood objective
    n_labeled: The number of labeled samples to use. The remaining samples from
      the training set are used as additional unlabeled data.
    ''' 
    dataset = 'mnist'
    pca = 50
    objective = 'hybrid_mm_ssl'
    n_labeled = 50000
    n_unlabeled = 50000 - n_labeled
    
    np.random.seed(1234)
    
    x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np = loadDataset(dataset, pca, n_labeled=n_labeled, n_unlabeled=n_unlabeled)
    
    C = int(np.max([np.max(t_tr_np), np.max(t_va_np), np.max(t_te_np)]) + 1)
    D = x_tr_np.shape[1]
    
    params = getParameters(dataset, pca, objective, n_labeled=n_labeled, n_unlabeled=n_unlabeled)
    
    gmm_em_filename = 'gmmSSL_em_%s%sL%dUl%d_eps%1.0e_K_%s.npz' % (dataset, '' if pca is None else ('_pca%d' % (pca)), n_labeled, n_unlabeled, params['epsilon'], '_'.join([str(e) for e in params['K']]))
    print 'GMM EM Filename', gmm_em_filename
    gmm_params = getGmmEM(x_tr_np[t_tr_np != -1, :], t_tr_np[t_tr_np != -1], params['K'], params['epsilon'], gmm_em_filename)
    
    print 'Classification Results for ML GMMs'
    print 'CE Tr [labeled only]:', classifyGmmFullCov(gmm_params, params['K'], x_tr_np[t_tr_np != -1, :], t_tr_np[t_tr_np != -1])
    print 'CE Va:', classifyGmmFullCov(gmm_params, params['K'], x_va_np, t_va_np)
    print 'CE Te:', classifyGmmFullCov(gmm_params, params['K'], x_te_np, t_te_np)

    init_params = getDplrParameters(gmm_params, params['S'], params['epsilon'], params['parameterization'] == 'precision')
    if params['parameterization'] == 'precision':
        model = GMMClassifierSSL(C, D, params['K'], params['S'], epsilon=1e-6, use_precision=True,
                              tradeoff_hybrid=params['lambda'], tradeoff_ssl=params['kappa'],
                              gamma=params['gamma'], eta=10, init_params=init_params)
    else:
        model = GMMClassifierSSL(C, D, params['K'], params['S'], epsilon=params['epsilon'], use_precision=False,
                              tradeoff_hybrid=params['lambda'], tradeoff_ssl=params['kappa'],
                              gamma=params['gamma'], eta=10, init_params=init_params)
    
    optim_cfg = {'objective' : objective,
                 'n_batch' : 100,
                 'n_epochs' : params['n_epochs'],
                 'step_size' : params['step_size']}
    model = trainGmmAdam(model, optim_cfg, 5678, x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np)
