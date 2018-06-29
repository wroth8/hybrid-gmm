'''
Train GMMs with the EM algorithm

@author Wolfgang Roth
'''

import numpy as np
from theano import *
import theano.tensor as T

from adam import Adam
from time import time

def trainGmmAdam(gmmc, optim_cfg, rng_seed, x_tr_np, t_tr_np, x_va_np, t_va_np, x_te_np, t_te_np):
    '''
    Trains hybrid/discriminativce GMMs on the given training data.
    
    gmmc: GMMClassifier object containing the Theano computation graph
    optim_cfg: Dictionary containing optimization settings. It contains the
      following fields:
      - 'objective': Which objective to use. Must be one of 'hybrid_mm',
          'hybrid_cll', 'disc_mm', 'disc_cll', 'hybrid_mm_ssl', or
          'hybrid_cll_ssl'. The SSL objectives should be used in conjunction
          with GMMClassifierSSL, the other objectives should be used with
          GMMClassifier(Diag)
      - 'n_batch': The batch size
      - 'n_epochs': Number of epochs to train
      - 'step_size': The step size of the ADAM algorithm
    rng_seed: Seed for the random number generator used to shuffle the training
      data after each epoch
    x_tr_np: The input features of the training set that are used for training.
    t_tr_np: The targets of the training set that are used for training
    x_va_np: The input features of the validation set. The validation set is
      evaluated after every epoch and the model for the 
    t_va_np: The targets of the validation set
    x_te_np: The input features of the test set
    t_te_np: The targets of the test set
    
    Returns a dictionary with the following fields:
      - 'costs': List containing the cost that is actually being minimized for
           each epoch
      - 'costs_gen': List containing the log-likelihood for each epoch
      - 'costs_gen_usv': List containing the log-likelihood for the unlabeled
           data in case of semi-supervised learning, and None otherwise.
      - 'costs_disc': List containing either the MM or the CLL for each epoch
      - 'errs_tr': List containing the error on the training set for each epoch
      - 'errs_va': List containing the error on the validation set for each epoch
      - 'errs_te': List containing the error on the test set for each epoch
      - 'best_model': The parameters of the model resulting in the best
           validation performance.
    '''
    x_tr_np = np.asarray(x_tr_np, theano.config.floatX)
    x_va_np = np.asarray(x_va_np, theano.config.floatX)
    x_te_np = np.asarray(x_te_np, theano.config.floatX)
    t_tr_np = np.asarray(t_tr_np, 'int32')
    t_va_np = np.asarray(t_va_np, 'int32')
    t_te_np = np.asarray(t_te_np, 'int32')
    
    x_tr = theano.shared(x_tr_np, borrow=True)
    x_va = theano.shared(x_va_np, borrow=True)
    x_te = theano.shared(x_te_np, borrow=True)
    t_tr = theano.shared(t_tr_np, borrow=True)
    t_va = theano.shared(t_va_np, borrow=True)
    t_te = theano.shared(t_te_np, borrow=True)
    
    N_tr = x_tr_np.shape[0]
    N_va = x_va_np.shape[0]
    N_te = x_te_np.shape[0]
    batch_size_eval = 10000
    n_batches = int(float(N_tr) / optim_cfg['n_batch'])
    rng = np.random.RandomState(rng_seed)
    
    print 'Compiling CE_TR function'
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    if optim_cfg['objective'] == 'hybrid_mm':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_hybrid_mm, gmmc.cost_nll, gmmc.cost_mm],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'hybrid_cll':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_hybrid_cll, gmmc.cost_nll, gmmc.cost_cll],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'disc_mm':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_mm, gmmc.cost_nll, gmmc.cost_mm],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'disc_cll':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_cll, gmmc.cost_nll, gmmc.cost_cll],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'hybrid_mm_ssl':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_hybrid_mm, gmmc.cost_nll_sv_normalized, gmmc.cost_nll_usv_normalized, gmmc.cost_mm_normalized],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'hybrid_cll_ssl':
        ce_tr = theano.function(inputs=[start_idx, end_idx],
                                outputs=[gmmc.ce, gmmc.cost_hybrid_cll, gmmc.cost_nll_sv_normalized, gmmc.cost_nll_usv_normalized, gmmc.cost_cll_normalized],
                                givens={gmmc.x:x_tr[start_idx : end_idx],
                                        gmmc.t:t_tr[start_idx : end_idx]})
    else:
        raise Exception('Unknown objective ''%s''' % optim_cfg['objective'])
        
    print 'Compiling CE_VA function'
    ce_va = theano.function(inputs=[start_idx, end_idx],
                            outputs=gmmc.ce,
                            givens={gmmc.x:x_va[start_idx : end_idx],
                                    gmmc.t:t_va[start_idx : end_idx]})
    print 'Compiling CE_TE function'
    ce_te = theano.function(inputs=[start_idx, end_idx],
                            outputs=gmmc.ce,
                            givens={gmmc.x:x_te[start_idx : end_idx],
                                    gmmc.t:t_te[start_idx : end_idx]})

    # precompute weights for weighted averages
    i0_tr, i0_va, i0_te, i1_tr, i1_va, i1_te = [], [], [], [], [], []
    w_tr, w_va, w_te = [], [], []
    n_batches_tr = int(np.ceil(float(N_tr)/batch_size_eval))
    n_batches_va = int(np.ceil(float(N_va)/batch_size_eval))
    n_batches_te = int(np.ceil(float(N_te)/batch_size_eval))
    for batch_idx in range(n_batches_tr):
        i0_tr.append(batch_idx * batch_size_eval)
        i1_tr.append(min([(batch_idx + 1) * batch_size_eval, N_tr]))
        w_tr.append(i1_tr[-1] - i0_tr[-1])
    for batch_idx in range(n_batches_va):
        i0_va.append(batch_idx * batch_size_eval)
        i1_va.append(min([(batch_idx + 1) * batch_size_eval, N_va]))
        w_va.append(i1_va[-1] - i0_va[-1])
    for batch_idx in range(n_batches_te):
        i0_te.append(batch_idx * batch_size_eval)
        i1_te.append(min([(batch_idx + 1) * batch_size_eval, N_te]))
        w_te.append(i1_te[-1] - i0_te[-1])
    
    cost = np.zeros((n_batches_tr,), dtype=np.float32)
    cost_gen = np.zeros((n_batches_tr,), dtype=np.float32)
    cost_gen_usv = np.zeros((n_batches_tr,), dtype=np.float32) # for SSL only
    cost_disc = np.zeros((n_batches_tr,), dtype=np.float32)
    err_tr = np.zeros((n_batches_tr,), dtype=np.float32)
    err_va = np.zeros((n_batches_va,), dtype=np.float32)
    err_te = np.zeros((n_batches_te,), dtype=np.float32)

    # Evaluate model on data
    if optim_cfg['objective'] == 'hybrid_mm_ssl' or optim_cfg['objective'] == 'hybrid_cll_ssl':
        for batch_idx in range(n_batches_tr):
            err_tr[batch_idx], cost[batch_idx], cost_gen[batch_idx], cost_gen_usv[batch_idx], cost_disc[batch_idx] = ce_tr(i0_tr[batch_idx], i1_tr[batch_idx])
        costs_gen_usv = [np.average(cost_gen_usv, weights=w_tr)]
    else:
        for batch_idx in range(n_batches_tr):
            err_tr[batch_idx], cost[batch_idx], cost_gen[batch_idx], cost_disc[batch_idx] = ce_tr(i0_tr[batch_idx], i1_tr[batch_idx])
        costs_gen_usv = None
    for batch_idx in range(n_batches_va):
        err_va[batch_idx] = ce_va(i0_va[batch_idx], i1_va[batch_idx])
    for batch_idx in range(n_batches_te):
        err_te[batch_idx] = ce_te(i0_te[batch_idx], i1_te[batch_idx])
    
    costs = [np.average(cost, weights=w_tr)]
    costs_gen = [np.average(cost_gen, weights=w_tr)]
    costs_disc = [np.average(cost_disc, weights=w_tr)]
    errs_tr = [np.average(err_tr, weights=w_tr)]
    errs_va = [np.average(err_va, weights=w_va)]
    errs_te = [np.average(err_te, weights=w_te)]
    best_err_va = np.Inf
    best_model = [p.get_value(borrow=True) for p in gmmc.params]
    if optim_cfg['objective'] == 'hybrid_mm' or optim_cfg['objective'] == 'disc_mm':
        print 'Epoch %5d/%5d: cost=%11.5e, cost_nll=%11.5e, cost_mm=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f' % (0, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1])
    elif optim_cfg['objective'] == 'hybrid_cll' or optim_cfg['objective'] == 'disc_cll':
        print 'Epoch %5d/%5d: cost=%11.5e, cost_nll=%11.5e, cost_cll=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f' % (0, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1])
    elif optim_cfg['objective'] == 'hybrid_mm_ssl':
        print 'Epoch %5d/%5d: cost=%11.5e, cost_nll_sv=%11.5e, cost_nll_usv=%11.5e, cost_mm=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f' % (0, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_gen_usv[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1])
    elif optim_cfg['objective'] == 'hybrid_cll_ssl':
        print 'Epoch %5d/%5d: cost=%11.5e, cost_nll_sv=%11.5e, cost_nll_usv=%11.5e, cost_cll=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f' % (0, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_gen_usv[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1])
    
    print 'Compiling training function'
    if optim_cfg['objective'] == 'hybrid_mm' or optim_cfg['objective'] == 'hybrid_mm_ssl':
        print 'Training with hybrid-max-margin objective'
        gmm_train = theano.function(inputs=[start_idx, end_idx],
                                    outputs=gmmc.cost_hybrid_mm,
                                    updates=Adam(gmmc.cost_hybrid_mm, gmmc.params, lr=optim_cfg['step_size']),
                                    givens={gmmc.x:x_tr[start_idx : end_idx],
                                            gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'hybrid_cll' or optim_cfg['objective'] == 'hybrid_cll_ssl':
        print 'Training with hybrid-conditional-log-likelihood objective'
        gmm_train = theano.function(inputs=[start_idx, end_idx],
                                    outputs=gmmc.cost_hybrid_cll,
                                    updates=Adam(gmmc.cost_hybrid_cll, gmmc.params, lr=optim_cfg['step_size']),
                                    givens={gmmc.x:x_tr[start_idx : end_idx],
                                            gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'disc_mm':
        print 'Training with discriminative-max-margin objective'
        gmm_train = theano.function(inputs=[start_idx, end_idx],
                                    outputs=gmmc.cost_mm,
                                    updates=Adam(gmmc.cost_mm, gmmc.params, lr=optim_cfg['step_size']),
                                    givens={gmmc.x:x_tr[start_idx : end_idx],
                                            gmmc.t:t_tr[start_idx : end_idx]})
    elif optim_cfg['objective'] == 'disc_cll':
        print 'Training with discriminative-conditional-log-likelihood objective'
        gmm_train = theano.function(inputs=[start_idx, end_idx],
                                    outputs=gmmc.cost_cll,
                                    updates=Adam(gmmc.cost_cll, gmmc.params, lr=optim_cfg['step_size']),
                                    givens={gmmc.x:x_tr[start_idx : end_idx],
                                            gmmc.t:t_tr[start_idx : end_idx]})
    else:
        raise Exception('Unknown objective ''%s''' % optim_cfg['objective'])
    
    # Create start/end idx for every batch
    batches = np.array_split(np.arange(N_tr), n_batches)
    batches = [(idx[0], idx[-1] + 1) for idx in batches]

    print 'Start training (#batches=%d)...' % (n_batches)
    for epoch_idx in range(optim_cfg['n_epochs']):
        t_start = time()
        for batch_idx in range(n_batches):
            batch = batches[batch_idx]
            gmm_train(batch[0], batch[1])

        if optim_cfg['objective'] == 'hybrid_mm_ssl' or optim_cfg['objective'] == 'hybrid_cll_ssl':
            for batch_idx in range(n_batches_tr):
                err_tr[batch_idx], cost[batch_idx], cost_gen[batch_idx], cost_gen_usv[batch_idx], cost_disc[batch_idx] = ce_tr(i0_tr[batch_idx], i1_tr[batch_idx])
            costs_gen_usv = [np.average(cost_gen_usv, weights=w_tr)]
        else:
            for batch_idx in range(n_batches_tr):
                err_tr[batch_idx], cost[batch_idx], cost_gen[batch_idx], cost_disc[batch_idx] = ce_tr(i0_tr[batch_idx], i1_tr[batch_idx])
        for batch_idx in range(n_batches_va):
            err_va[batch_idx] = ce_va(i0_va[batch_idx], i1_va[batch_idx])
        for batch_idx in range(n_batches_te):
            err_te[batch_idx] = ce_te(i0_te[batch_idx], i1_te[batch_idx])
        costs.append( np.average(cost, weights=w_tr) )
        costs_gen.append( np.average(cost_gen, weights=w_tr) )
        costs_disc.append( np.average(cost_disc, weights=w_tr) )
        errs_tr.append( np.average(err_tr, weights=w_tr) )
        errs_va.append( np.average(err_va, weights=w_va) )
        errs_te.append( np.average(err_te, weights=w_te) )

        t_elapsed = time() - t_start
        if optim_cfg['objective'] == 'hybrid_mm' or optim_cfg['objective'] == 'disc_mm':
            print 'Epoch %5d/%5d: cost=%11.5e, cost_nll=%11.5e, cost_mm=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f (t_elapsed=%f seconds)' % (epoch_idx, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1], t_elapsed)
        elif optim_cfg['objective'] == 'hybrid_cll' or optim_cfg['objective'] == 'disc_cll':
            print 'Epoch %5d/%5d: cost=%11.5e, cost_nll=%11.5e, cost_cll=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f (t_elapsed=%f seconds)' % (epoch_idx, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1], t_elapsed)
        elif optim_cfg['objective'] == 'hybrid_mm_ssl':
            print 'Epoch %5d/%5d: cost=%11.5e, cost_nll_sv=%11.5e, cost_nll_usv=%11.5e, cost_mm=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f (t_elapsed=%f seconds)' % (epoch_idx, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_gen_usv[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1], t_elapsed)
        elif optim_cfg['objective'] == 'hybrid_cll_ssl':
            print 'Epoch %5d/%5d: cost=%11.5e, cost_nll_sv=%11.5e, cost_nll_usv=%11.5e, cost_cll=%11.5e, ce_tr=%7.5f, ce_va=%7.5f, ce_te=%7.5f (t_elapsed=%f seconds)' % (epoch_idx, optim_cfg['n_epochs'], costs[-1], costs_gen[-1], costs_gen_usv[-1], costs_disc[-1], errs_tr[-1], errs_va[-1], errs_te[-1], t_elapsed)
    
        if errs_va[-1] < best_err_va:
            print 'err_va improved --> storing model'
            best_err_va = errs_va[-1]
            best_model = gmmc.getParameters()
    
        if len(costs) > 5 and np.all(np.asarray(costs)[-5:-1] == costs[-1]):
            print 'cost was constant over five epochs: stopping optimization'
            break
        
        # shuffle
        randperm = rng.permutation(N_tr)
        x_tr_np = x_tr_np[randperm[:, None], np.arange(x_tr_np.shape[1])]
        t_tr_np = t_tr_np[randperm]
        x_tr.set_value(x_tr_np, borrow=True)
        t_tr.set_value(t_tr_np, borrow=True)

    return {'costs':costs,
            'costs_gen':costs_gen,
            'costs_gen_usv':costs_gen_usv,
            'costs_disc':costs_disc,
            'errs_tr':errs_tr,
            'errs_va':errs_va,
            'errs_te':errs_te,
            'best_model':best_model}