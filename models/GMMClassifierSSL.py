'''
Theano computation graph that defines the hybrid GMM objectives for semi-
supervised learning for DPLR covariance matrices from [1].

[1] Roth W., Peharz R., Tschiatschek S., Pernkopf F., Hybrid generative-
    discriminative training of Gaussian mixture models, Pattern Recognition
    Letters 2018, (accepted)

@author Wolfgang Roth
'''

import numpy as np
import theano
import theano.tensor as T

from theanoOpLogDetPSD import logdet_psd

class GMMClassifierSSL:
    def __init__(self, C, D, K, S, rng_state=None, epsilon=1e-2, use_precision=True, tradeoff_hybrid=0.5, tradeoff_ssl=0.9, gamma=1, eta=10, init_params=None):
        '''
        Constructs the Theano computation graph for the given parameters
        
        C: Number of classes
        D: Number of input features
        K: List of length C containing the number of components per class
        S: Number of dimensions of the low-rank approximation of the DPLR matrix
          structure. Note that this parameter is actually called 'R' in the
          paper.
        rng_state: Random number generator seed to use if the parameters should
          be initialized randomly. This parameter is ignored if 'init_params' is
          given.
        epsilon: Regularizer for the diagonal of the covariance matrices.
        use_precision: Determines if precisions or covariances should be used.
          The precision is the inverse of the covariance matrix.
        tradeoff_hybrid: The lambda parameter of the hybrid objective. Close to
          1 means very generative, close to 0 means very discriminative.
        tradeoff_ssl: The kappa parameter of the hybrid objective for semi-
          supervised learning. Close to 1 puts more weight onto the labeled
          samples, close to 0 puts more weights onto the unlabeled samples.
        gamma: The gamma parameter of the MM/LM objective.
        eta: The parameter for the softmax approximation.
        init_params: Use this to provide initial parameters. Usually parameters
          obtained with the EM algorithm are provided here. init_params must be
          a five-tuple containing
            - mu_vals (K_all x D): Mean values for each component
            - s_vals (K_all x D x S): Low-rank matrices for each component
            - d_rho_vals (D x K_all): Diagonal variances for each component (inverse softplus values)
            - prior_k_rho_vals (K_all): Logits of the component priors
            - prior_c_rho_vals (C): Logits of the class priors
          with K_all = sum(K). The component parameters are stored linearly for
          all classes. E.g. the first K[0] entries correspond to components of
          class 0. If precisions are used instead of covariances, the use of
          s_vals and d_rho_vals changes accordingly.
        '''
        self.x = T.matrix('x')
        self.t = T.ivector('t')
        self.tradeoff_hybrid = tradeoff_hybrid
        self.tradeoff_ssl = tradeoff_ssl
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon

        K_all = np.sum(K)
        
        if init_params is None:
            rng = np.random.RandomState(rng_state)
            mu_vals = rng.normal(0., 1., size=(K_all, D))
            s_vals = rng.normal(0., 1., size=(K_all, D, S))
            d_rho_vals = rng.normal(0, 0.1, size=(D, K_all))
            prior_k_rho_vals = np.zeros((np.sum(K),))
            prior_c_rho_vals = np.zeros((C,))
        else:
            assert len(init_params) == 5
            mu_vals, s_vals, d_rho_vals, prior_k_rho_vals, prior_c_rho_vals = init_params
            assert mu_vals.shape == (K_all, D)
            assert s_vals.shape == (K_all, D, S)
            assert d_rho_vals.shape == (D, K_all)
            assert prior_k_rho_vals.shape == (K_all,)
            assert prior_c_rho_vals.shape == (C,)

        mu_vals = np.asarray(mu_vals, dtype=theano.config.floatX)
        s_vals = np.asarray(s_vals, dtype=theano.config.floatX)
        d_rho_vals = np.asarray(d_rho_vals, dtype=theano.config.floatX)
        prior_k_rho_vals = np.asarray(prior_k_rho_vals, dtype=theano.config.floatX)
        prior_c_rho_vals = np.asarray(prior_c_rho_vals, dtype=theano.config.floatX)
        
        # Shared variables
        self.means = theano.shared(mu_vals, name='means', borrow=True) 
        self.s = theano.shared(s_vals, name='s', borrow=True)
        self.d_rho = theano.shared(d_rho_vals, name='d_rho', borrow=True)
        self.prior_k_rho = theano.shared(prior_k_rho_vals, name='prior_k_rho', borrow=True)
        self.prior_c_rho = theano.shared(prior_c_rho_vals, name='prior_c_rho', borrow=True)
        self.params = [self.means, self.s, self.d_rho, self.prior_k_rho, self.prior_c_rho]

        self.d = T.nnet.softplus(self.d_rho) + self.epsilon
        if use_precision == True:
            # s and d are used to represent precision matrices
            self.exponent = T.dot(self.x ** 2, self.d) #xDx
            self.exponent -= 2 * T.dot(self.x, self.d * self.means.T) #-2xDm
            self.exponent += T.sum(self.means ** 2 * self.d.T, axis=1) # mDm
            self.exponent += T.sum(T.dot(self.x, self.s) ** 2, axis=2) # xSSx
            self.exponent -= 2 * T.sum(T.dot(self.x, self.s) * T.sum(self.s * self.means[:,:,None], axis=1)[None,:,:], axis=2) # -2xSSm
            self.exponent += T.sum(T.sum(self.s * self.means[:,:,None], axis=1) ** 2, axis=1) # mSSm
            self.exponent *= -0.5
    
            eye_S = T.eye(S, dtype=theano.config.floatX)
            self.aux_matrix = T.batched_tensordot(self.s / self.d.T[:,:,None], self.s, axes=(1, 1)) + eye_S
            self.aux_logdet, _ = theano.scan(fn=lambda aux: logdet_psd(aux),
                                         outputs_info=None,
                                         sequences=self.aux_matrix,
                                         non_sequences=None)
            self.logdet = T.sum(T.log(self.d), axis=0) + self.aux_logdet
            
            # logpK contains all log probabilities of all components in an (N x sum(K)) array
            # Note that the log component priors are not added yet
            self.logpK = -0.5 * D * T.log(2. * np.pi) + 0.5 * self.logdet  + self.exponent
        else:
            # s and d are used to represent covariance matrices
            if S == 1:
                self.aux_matrix = T.sum(self.s[:,:,0] / self.d.T * self.s[:,:,0], axis=1).reshape((K_all, 1, 1)) + 1.
            else:
                # Since the latest Cuda/Theano update, the following two lines
                # cause an error in the case of S=1.
                eye_S = T.eye(S, dtype=theano.config.floatX)
                self.aux_matrix = T.batched_tensordot(self.s / self.d.T[:,:,None], self.s, axes=(1, 1)) + eye_S
            (self.aux_inv, self.aux_logdet), _ = theano.scan(fn=lambda aux: [T.nlinalg.matrix_inverse(aux), logdet_psd(aux)],
                                                         outputs_info=None,
                                                         sequences=[self.aux_matrix],
                                                         non_sequences=None)
            self.logdet = T.sum(T.log(self.d), axis=0) + self.aux_logdet

            # Product inv(d) * s for all K --> K x D x S
            self.rs = self.s / self.d.T[:,:,None]
            # Product inv(d) * s * aux_inv for all K --> K x D x S
            self.ls = T.batched_dot(self.rs, self.aux_inv)

            # s and d are used to represent covariance matrices
            self.exponent = T.dot(self.x ** 2, 1. / self.d) #xDx
            self.exponent -= 2 * T.dot(self.x, (1. / self.d) * self.means.T) #-2xDm
            self.exponent += T.sum(self.means ** 2 * (1. / self.d.T), axis=1) # mDm
            self.exponent -= T.sum(T.dot(self.x, self.ls) * T.dot(self.x, self.rs), axis=2) # -x ls rs x
            self.exponent += 2 * T.sum(T.dot(self.x, self.ls) * T.sum(self.rs * self.means[:,:,None], axis=1)[None,:,:], axis=2) # 2x ls rs m
            self.exponent -= T.sum(T.sum(self.ls * self.means[:,:,None], axis=1) * T.sum(self.rs * self.means[:,:,None], axis=1), axis=1) # -m ls rs m
            self.exponent *= -0.5

            # logpK contains all log probabilities of all components in an (N x sum(K)) array
            # Note that the log component priors are not added yet
            self.logpK = -0.5 * D * T.log(2. * np.pi) - 0.5 * self.logdet + self.exponent

        # logpC contains the log joint probabilities p(x,c) in an (N x C) array
        self.logpC = self.logpK
        self.logpC_list = []
        for c in range(C):
            k1 = int(np.sum(K[:c]))
            k2 = int(k1 + K[c])
            self.logpC_list.append(self.logpC[:, k1:k2])
            aux_max = T.max(self.prior_k_rho[k1:k2]) # Compute log-probabilities without division
            log_prior_k = self.prior_k_rho[k1:k2] - T.log(T.sum(T.exp(self.prior_k_rho[k1:k2] - aux_max))) - aux_max
            self.logpC_list[c] += log_prior_k
            aux_max = T.max(self.logpC_list[c], axis=1, keepdims=True)
            self.logpC_list[c] = T.log(T.sum(T.exp(self.logpC_list[c] - aux_max), axis=1)) + aux_max.flatten()
        self.logpC = T.stack(self.logpC_list, axis=1)
        aux_max = T.max(self.prior_c_rho) # Compute log-probabilities without division
        log_prior_c = self.prior_c_rho - T.log(T.sum(T.exp(self.prior_c_rho - aux_max))) - aux_max
        self.logpC += log_prior_c 

        # mm and cll objective are only for labeled data
        # logl objective is slightly different for labeled and unlabeled data
        idx_sv = T.ge(self.t, 0).nonzero()
        idx_usv = T.lt(self.t, 0).nonzero()
        self.logpC_sv = self.logpC[idx_sv]
        self.logpC_usv = self.logpC[idx_usv]
        is_sv_empty = T.eq(self.logpC_sv.shape[0], 0)
        is_usv_empty = T.eq(self.logpC_usv.shape[0], 0)
        
        # If there are no supervised/unsupervised samples create a dummy entry
        # to avoid problems. The corresponding costs are set to 0 later. We set
        # the number of rows to 2 because 1 results in an error.
        # The problems appear to be CUDNN related if for instance a sum over an
        # empty tensor is computed.
        self.logpC_sv  = theano.ifelse.ifelse(is_sv_empty,  T.zeros((2,C), theano.config.floatX), self.logpC_sv)
        self.t_sv      = theano.ifelse.ifelse(is_sv_empty,  T.zeros((2,), 'int32')              , self.t[idx_sv])
        self.logpC_usv = theano.ifelse.ifelse(is_usv_empty, T.zeros((2,C), theano.config.floatX), self.logpC_usv)
        
        # Compute mean divisor since T.mean causes divisions by zero if there
        # are no labeled or unlabeled data in the minibatch. Therefore, we
        # compute T.mean with T.sum()/N
        self.aux_mean_divisor_sv = T.switch(is_sv_empty, 1., self.logpC_sv.shape[0])
        self.aux_mean_divisor_usv = T.switch(is_usv_empty, 1., self.logpC_usv.shape[0])

        # Create cost functions

        # Compute the log of the softmax of logpc which gives the log of the conditional likelihood
        self.cll_max_tmp = T.max(self.logpC_sv, axis=1, keepdims=True)
        self.cll_logsumexp = T.log(T.sum(T.exp(self.logpC_sv - self.cll_max_tmp), axis=1)) + T.reshape(self.cll_max_tmp, (self.cll_max_tmp.shape[0],))
        self.cost_cll = theano.ifelse.ifelse(is_sv_empty, 0., -T.sum(self.logpC_sv[T.arange(self.logpC_sv.shape[0]), self.t_sv] - self.cll_logsumexp))
        self.cost_cll_normalized = self.cost_cll / self.aux_mean_divisor_sv
        
        # Negative log-likelihood of labeled data
        self.cost_nll_sv = theano.ifelse.ifelse(is_sv_empty, 0., -T.sum(self.logpC_sv[T.arange(self.t_sv.shape[0]), self.t_sv]))
        self.cost_nll_sv_normalized = self.cost_nll_sv / self.aux_mean_divisor_sv
        
        # Negative log-likelihood of unlabeled data
        self.logpC_usv_max = T.max(self.logpC_usv, axis=1, keepdims=True)
        self.logpC_usv_logsumexp = T.log(T.sum(T.exp(self.logpC_usv - self.logpC_usv_max), axis=1)) + T.reshape(self.logpC_usv_max, (self.logpC_usv.shape[0],))
        self.cost_nll_usv = theano.ifelse.ifelse(is_usv_empty, 0., -T.sum(self.logpC_usv_logsumexp))
        self.cost_nll_usv_normalized = self.cost_nll_usv / self.aux_mean_divisor_usv
        
        # Total negative log-likelihood
        self.cost_nll = self.cost_nll_sv + self.cost_nll_usv
        self.cost_nll_normalized = self.cost_nll / self.x.shape[0]
        
        self.margin_start = self.gamma + self.logpC_sv - T.reshape(self.logpC_sv[T.arange(self.t_sv.shape[0]), self.t_sv], (self.t_sv.shape[0], 1))
        self.margin = self.gamma + self.logpC_sv - T.reshape(self.logpC_sv[T.arange(self.t_sv.shape[0]), self.t_sv], (self.t_sv.shape[0], 1))
        self.margin *= self.eta
        self.margin = T.set_subtensor(self.margin[T.arange(self.t_sv.shape[0]), self.t_sv], -np.inf)
        
        # Log-sum-exp trick
        self.margin_max_tmp = T.max(self.margin, axis=1, keepdims=True)
        self.max_margin = T.log(T.sum(T.exp(self.margin - self.margin_max_tmp), axis=1)) + T.reshape(self.margin_max_tmp, (self.margin.shape[0],))
        self.max_margin /= self.eta

        # The cast in the following statement resolves an error that says that
        # both paths of ifelse must be of equal type. Setting the dtype argument
        # of T.sum did not solve the problem.
        self.cost_mm = theano.ifelse.ifelse(is_sv_empty, 0., T.cast(T.sum(T.nnet.relu(self.max_margin)), theano.config.floatX))
        self.cost_mm_normalized = self.cost_mm / self.aux_mean_divisor_sv
        
        # Note: The division by self.x.shape[0] in the following two expressions
        # ensures that gradients of minibatches are unbiased.

        # Cost with CLL criterion
        self.cost_hybrid_cll = (self.tradeoff_hybrid * (self.tradeoff_ssl * self.cost_nll_sv + (1. - self.tradeoff_ssl) * self.cost_nll_usv) + (1. - self.tradeoff_hybrid) * self.cost_cll) / (self.x.shape[0])

        # Cost with MM criterion        
        self.cost_hybrid_mm = (self.tradeoff_hybrid * (self.tradeoff_ssl * self.cost_nll_sv + (1. - self.tradeoff_ssl) * self.cost_nll_usv) + (1. - self.tradeoff_hybrid) * self.cost_mm) / (self.x.shape[0])

        # Predictions and classification errors
        self.y = T.argmax(self.logpC, axis=1)
        self.y_sv = self.y[idx_sv]
        self.y_usv = self.y[idx_usv]
        self.ce = theano.ifelse.ifelse(is_sv_empty, 0., T.mean(T.neq(self.y_sv, self.t_sv), dtype=theano.config.floatX))

    def getParameters(self):
        params = {'mu' : self.means.get_value(borrow=True),
                  'd_rho' : self.d_rho.get_value(borrow=True),
                  'prior_k_rho' : self.prior_k_rho.get_value(borrow=True),
                  'prior_c_rho' : self.prior_c_rho.get_value(borrow=True)}
        return params

