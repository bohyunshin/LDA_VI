import numpy as np
from numpy import exp, log
from scipy.special import digamma, gamma, loggamma
from collections import Counter, OrderedDict
import pickle
import time
from _online_lda_fast import _dirichlet_expectation_2d, _dirichlet_expectation_1d_
from sklearn.feature_extraction.text import CountVectorizer

EPS = np.finfo(np.float).eps

class LDA_VI:
    def __init__(self, path_data, alpha, m, delta,  K, sampling):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))
        if sampling:
            np.random.seed(0)
            idx = np.random.choice(len(self.data), 1000, replace=False)
            self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.m = m  # background distribution; dimension: V * 1
        self.delta = delta # hyperparameter for exponential distribution
        self.K = K
        self.perplexity = []

    def _make_vocab(self):
        self.vocab = []
        for lst in self.data:
            self.vocab += lst
        self.vocab = sorted(list(set(self.vocab)))

        # make DTM
        self.data_join = [' '.join(doc) for doc in self.data]
        self.cv = CountVectorizer()
        self.X = self.cv.fit_transform(self.data_join).toarray()
        self.w2idx = self.cv.vocabulary_
        self.idx2w = {val: key for key, val in self.w2idx.items()}


    def _init_params(self):
        '''
        Initialize parameters for SAGE
        <variational free parameters>
        q(z_dn): Multi(phi)
        q(theta_d): Dir(gamma_d)
        q(tau_ki): Gamma(a,b)

        <latent variables>
        z_dn
        theta_d
        eta
        tau
        '''
        self.V = len(self.w2idx)
        self.D = len(self.data)
        self.Nd = [len(doc) for doc in self.data]

        # # set initial value for free variational parameters
        # self.phi = np.ones((self.V, self.K)) # dimension: for topic d, Nd * K

        '''
        Set initial values for variational parameters
        '''
        # initialize phi: different for each document (variational parameters for z_dn)
        self.phi = {}
        for d in range(self.D):
            self.phi[d] = np.ones((self.V, self.K))
        # initialize gamma (variational parameters for theta)
        np.random.seed(1)
        self.gam = np.random.gamma(100, 1/100, (self.D, self.K)) # dimension: D * K
        # initialize a,b (variational parameters for tau)
        np.random.seed(2)
        self.a = np.random.gamma(100, 1 / 100, (self.V, self.K)) # dimension: V * K
        np.random.seed(3)
        self.b = np.random.gamma(100, 1 / 100, (self.V, self.K))  # dimension: V * K

        # initialize latent eta parameters
        np.random.seed(4)
        self.eta = np.random.gamma(100, 1/100, (self.V, self.K))

        # initialize dirichlet expectation to reduce computation time
        self._update_gam_E_dir()

    def _ELBO(self):
        term1 = 0 # E[ log p( w | phi, z) ]
        term2 = 0 # E[ log p( z | theta) ]
        term3 = 0 # E[ log q(z) ]
        term4 = 0 # E[ log p( theta | alpha) ]
        term5 = 0 # E[ log q( theta ) ]
        term6 = 0 # E[ log p(beta | eta) ]
        term7 = 0 # E[ log q(beta) ]

        '''
        ELBO is calculated w.r.t. each document
        Update term1, term2, term5 together
        Update term3, term6 together
        Update term4, term7 together
        ELBO = term1 + term2 + term3 + term4 - term5 - term6 - term7
        '''

        # update term 1, 2, 3, 4, 5
        for d in range(self.D):

            ndw = self.X[d,:]
            for k in range(self.K):
                # update term 1
                tmp = self.lam_E[:,k] * self.phi[d][:,k]
                # ndw_vec = np.zeros(self.V) # V * 1
                # ndw_vec[ndw] += self.X[d,ndw]

                term1 += (tmp * ndw).sum()

                # update term 2
                tmp = (ndw * self.phi[d][:,k]).sum() # sum of V * 1 numpy arrays: scalar
                E_theta_dk = self.gam_E[d,k] # scalar
                term2 += E_theta_dk * tmp # scalar * scalar = scalar

                # update term 3
                tmp = self.phi[d][:,k] * log(self.phi[d][:,k] + 0.000000001)
                term3 += (tmp * ndw).sum()




            # update term 4
            term4 += loggamma(self.K * self.alpha) - log(self.K * gamma(self.alpha))
            term4 += (self.alpha - 1) * self.gam_E[d,:].sum()

            # update term 5
            term5 += loggamma(sum(self.gam[d,:])) - sum(loggamma(self.gam[d,:]))
            term5 += ( (self.gam[d,:]-1) * self.gam_E[d,:] ).sum()

        print('Done term 1 ~ 5')


        for k in range(self.K):
            # update term 6
            term6 += loggamma(self.V * self.eta) - log(self.V * gamma(self.eta))
            term6 +=  (self.eta-1) * self.lam_E[:,k].sum()

            # update term 7
            term7 += loggamma(sum( self.lam[:,k] )) - sum( loggamma(self.lam[:,k]) )
            term7 += ( ( self.lam[:,k]-1 ) * ( self.lam_E[:,k] ) ).sum()
        print('Done term 6, 7')

        return term1 + term2 - term3 + term4 - term5 + term6 - term7

    def _E_dir(self, params_mat):
        '''
        input: vector parameters of dirichlet
        output: Expecation of dirichlet - also vector
        '''
        return _dirichlet_expectation_2d(params_mat)

    def _E_dir_1d(self, params):
        return _dirichlet_expectation_1d_(params)

    def _update_gam_E_dir(self):
        self.gam_E = self._E_dir(self.gam.transpose()).transpose()

    def _update_lam_E_dir(self):
        self.lam_E = self._E_dir(self.lam.transpose()).transpose()

    def _cal_exp_prob(self, k, Nd_index):
        eta_k = np.exp(self.eta[Nd_index,k] + self.m)
        normalizer = eta_k.sum()
        return eta_k / normalizer

    def _cal_small_c_k(self):
        self.c_k = np.zeros((self.V, self.K)) # dimension: V * K
        for d in range(self.D):
            self.c_k = self.c_k + self.X[d,:][:,None] * self.phi[d]

    def _cal_large_C_k(self):
        self.C_k = np.sum(self.c_k, axis=0) # dimension: K * 1

    def _cal_beta(self):
        numerator = self.eta + self.m[:,None] # dimension: V * K
        normalizer = np.sum(numerator, axis=0) # dimension: 1 * K
        self.beta = numerator / normalizer[None,:]

    def _simple_newtons(self, x0, delta, tol, multivariate=False):
        x1 = x0 - delta
        if multivariate:
            while sum(abs(x1 - x0)) > tol:
                x0 = x1
                x1 = x0 - delta
        else:
            while abs(x1-x0) > tol:
                x0 = x1
                x1 = x0 - delta
        return x1



    def _update_phi(self, d):
        # duplicated words are ignored
        Nd_index = np.nonzero(self.X[d,:])[0]
        # get the proportional value in each topic t,
        # and then normalize to make as probabilities
        # the indicator of Z_dn remains only one term


        for k in range(self.K):
            prob_eta = self._cal_exp_prob(k, Nd_index) # Nd * 1: indexing for words in dth document
            E_theta = np.exp(self.gam_E[d,k]) # scalar: indexing for kth topic
            self.phi[d][Nd_index,k] = prob_eta * E_theta # Nd * 1
        ## vectorize to reduce time
        # to prevent overfloat
        #self.phi -= self.phi.min(axis=1)[:,None]
        self.phi[d][Nd_index,:] = exp(self.phi[d][Nd_index,:])
        # normalize prob
        self.phi[d][Nd_index,:] /= np.sum(self.phi[d][Nd_index,:], axis=1)[:,None]
        # print(None)

    def _update_gam(self,d):
        gam_d = np.repeat(self.alpha, self.K)
        ids = np.nonzero(self.X[d,:])[0]
        n_dw = self.X[d,:][ids] # ids*1
        phi_dwk = self.phi[d][ids,:] # ids*K
        gam_d = gam_d + np.dot(n_dw, phi_dwk) # K*1 + K*1

        self.gam[d,:] = gam_d
        self.gam_E[d, :] = self._E_dir_1d(self.gam[d, :])


    def _update_eta(self):
        for k in range(self.K):
            beta_k = self.beta[:,k]
            u = self.C_k[k] * np.outer(beta_k, beta_k) # K*K
            v = self.beta[:,k]                         # K*1
            E_tau_inv = 1/((self.a[:,k]-1) * self.b[:,k]) # K*1
            A = -np.diag( self.C_k[k] * beta_k + E_tau_inv) # K*K
            A_inv = np.linalg.inv(A)

            grad_k = self.c_k[:,k] - self.C_k[k] * self.beta[:,k] - E_tau_inv * self.eta[:,k]
            # Hessian_k = np.outer(u,v) + A
            # Hessian_k_inverse = A_inv - self.C_k[k] * np.dot(A_inv, np.outer(beta_k, beta_k), A_inv ) \
            #                     / 1 + self.C_k[k] * np.dot(beta_k, A_inv, beta_k)

            delta_eta_k = -np.dot(A_inv, grad_k) + self.C_k[k] * np.dot( np.dot(A_inv, beta_k),
                                                np.dot(beta_k, np.dot(A_inv, grad_k))) \
                                                (1 + self.C_k[k] * np.dot(beta_k, A_inv, beta_k))
            self._simple_newtons(np.random.gamma(100,1/100,(self.K,1)), delta_eta_k, tol=0.001)

    def train(self, threshold, max_iter):

        print('Making Vocabs...')
        self._make_vocab()

        print('Initializing Parms...')
        self._init_params()
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.V}')
        print(f'{self.K} topics chosen')

        print('Start optimizing!')
        # initialize ELBO
        ELBO_before = 0
        ELBO_after = 99999
        self._ELBO_history = []

        print('##################### start training #####################')
        for iter in range(max_iter):
            start = time.time()
            ELBO_before = ELBO_after
            print('\n')

            print('E step: start optimizing phi, gamma...')
            self.gam = np.ones((self.D, self.K))
            self._update_gam_E_dir()

            for d in range(self.D):

                gam_before = self.gam[d,:]
                gam_after = np.repeat(999,self.K)
                while sum(abs(gam_before - gam_after)) / self.K > threshold:
                    gam_before = gam_after
                    self._update_phi(d)
                    self._update_gam(d)
                    gam_after = self.gam[d,:]

            # update beta_star
            print('M step: Updating lambda..')
            self._update_lam()
            print('Finished Iteration!')
            print('\n')

            if iter % 50 == 0:
                print('Now calculating ELBO...')
                ELBO_after = self._ELBO()
                self._ELBO_history.append(ELBO_after)
                self._perplexity(ELBO_after)

                print(f'Before ELBO: {ELBO_before}')
                print(f'After ELBO: {ELBO_after}')
                print('\n')

                if abs(ELBO_before - ELBO_after) < threshold:
                    break

            print(f'Computation time: {(time.time()-start)/60} mins')
        print('Done Optimizing!')


    def _perplexity(self, ELBO):
        '''
        Calculates Approximated Perplexity
        '''
        denominator = sum(self.Nd)

        self.perplexity.append( exp(-ELBO / denominator) )


