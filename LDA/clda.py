import numpy as np
from numpy import exp, log
from scipy.special import digamma, gamma, loggamma
from collections import Counter, OrderedDict
import pickle
import time
from _online_lda_fast import _dirichlet_expectation_2d, _dirichlet_expectation_1d_
from sklearn.feature_extraction.text import CountVectorizer

EPS = np.finfo(np.float).eps

class CLDA_VI:
    def __init__(self, path_data, alpha, eta,  K, seed_words, sampling):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))
        if sampling:
            np.random.seed(0)
            idx = np.random.choice(len(self.data), 1000, replace=False)
            self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.eta = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.seed_words = seed_words
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

        for key in self.seed_words.keys():
            # filter seed words existing in corpus vocabulary
            self.seed_words[key] = [ i for i in self.seed_words[key] if i in list(self.w2idx.keys()) ]
            self.seed_words[key] = [ self.w2idx[i] for i in self.seed_words[key] ]


        # excluded words from count vectorizer
        stop_words = list(set(self.vocab) - set(list(self.w2idx.keys())))


    def _init_params(self):
        '''
        Initialize parameters for LDA
        This is variational free parameters each endowed to
        Z_dn: psi_star
        theta_d: alpha_star
        phi_t : beta_star
        '''
        self.V = len(self.w2idx)
        self.D = len(self.data)
        self.Nd = [len(doc) for doc in self.data]

        # # set initial value for free variational parameters
        # self.phi = np.ones((self.V, self.K)) # dimension: for topic d, Nd * K

        # set initial phi, kappa, delta: different for each document
        self.phi = {}
        self.kappa = {}
        self.kappa['b=1'] = {}
        self.kappa['b=0'] = {}
        self.delta = {}
        self.delta[1] = {}
        self.delta[2] = {}


        # dictionary to put in hyper parameters
        self.pi = {}
        self.pi[1] = {}
        self.pi[2] = {}


        # imposing skewed prior
        # noninformative prior (uniform) on 'not' seed words
        # informative prior for seed words
        temp_alpha = np.repeat(1, self.V * self.K).reshape(self.V, self.K)
        temp_beta = np.repeat(1, self.V * self.K).reshape(self.V, self.K)

        for k, key in enumerate(list(self.seed_words.keys())):
            seed_words_index = np.array(self.seed_words[key])
            # imposing Beta(50,2) prior for seed words (spike at 1)
            temp_alpha[seed_words_index,k] = 20
            temp_beta[seed_words_index, k] = 3

            other_key = list(set(list(self.seed_words.keys())) - set([key]))
            for kk in other_key:
                not_seed_words_index = np.array(self.seed_words[kk])
                # imposing Beta(2,50) prior for 'not' seed words (spike at 0)
                temp_alpha[not_seed_words_index,k] = 3
                temp_beta[not_seed_words_index, k] = 20
        for d in range(self.D):
                self.pi[1][d] = temp_alpha
                self.pi[2][d] = temp_beta

        # initialize phi
        for d in range(self.D):
            self.phi[d] = np.random.uniform(0,1,self.V*self.K).reshape(self.V, self.K)

        # initialize lambda, gamma of dirichlet through gamma distribution
        np.random.seed(1)
        self.lam = np.random.gamma(100, 1/100, (self.V, self.K)) # dimension: V * K
        np.random.seed(2)
        self.gam = np.random.gamma(100, 1/100, (self.D, self.K)) # dimension: D * K

        # initialize delta
        for d in range(self.D):
            self.delta[1][d] = np.random.gamma(3,1,(self.V, self.K))
            self.delta[2][d] = np.random.gamma(3,1,(self.V, self.K))

        # initialize dirichlet expectation
        self._update_lam_E_dir()
        self._update_gam_E_dir()
        self._update_nu_E()

        # initialize kappa using phi, expectation of nu, beta
        for d in range(self.D):
            self.kappa['b=1'][d] = np.exp(self.phi[d] * self.lam_E + self.nu_E[1][d]) # W*K Dimension
            self.kappa['b=0'][d] = np.exp(self.phi[d] * self.lam_E + self.nu_E[2][d])

            self.kappa['b=1'][d] = self.kappa['b=1'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])
            self.kappa['b=0'][d] = self.kappa['b=0'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])

        print('hello world!')


    def _ELBO(self):
        term1 = 0 # E[ log p( w | phi, z) ]
        term2 = 0 # E[ log p( z | theta) ]
        term3 = 0 # E[ log q(z) ]
        term4 = 0 # E[ log p( b | nu ) ]
        term5 = 0 # E[ log q(b) ]
        term6 = 0 # E[ log p( nu | pi ) ]
        term7 = 0 # E[ log q( nu ) ]


        term8 = 0 # E[ log p( theta | alpha) ]
        term9 = 0 # E[ log q( theta ) ]
        term10 = 0 # E[ log p(beta | eta) ]
        term11 = 0 # E[ log q(beta) ]

        '''
        ELBO is calculated w.r.t. each document
        Update term1, term2, term5 together
        Update term3, term6 together
        Update term4, term7 together
        ELBO = term1 + term2 + term3 + term4 - term5 - term6 - term7
        '''

        # update term 1, 2, 3, 4, 5, 6, 7
        for d in range(self.D):

            ndw = self.X[d,:]
            for k in range(self.K):
                # update term 1
                tmp = self.lam_E[:,k] * self.phi[d][:,k] * self.kappa['b=1'][d][:,k]
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
                kap = self.kappa['b=1'][d][:,k]
                E_log_nu = self.nu_E[1][d][:,k]
                E_log_one_minus_nu = self.nu_E[2][d][:,k]
                tmp = kap*E_log_nu + (1-kap)*E_log_one_minus_nu
                term4 += (tmp * ndw).sum()

                # update term 5
                tmp = kap*np.log(kap + EPS) + (1-kap)*np.log(1-kap + EPS)
                term5 += (tmp * ndw).sum()

                # update term 6
                first = (self.pi[1][d][:,k] - 1) * (self.nu_E[1][d][:,k])
                second = (self.pi[2][d][:,k] - 1) * (self.nu_E[2][d][:,k])
                tmp = first + second
                term6 += (tmp*ndw).sum()

                # update term 7
                first = (self.delta[1][d][:, k] - 1) * (self.nu_E[1][d][:,k])
                second = (self.delta[2][d][:, k] - 1) * (self.nu_E[2][d][:,k])

                tmp = log(
                    gamma(self.delta[1][d][:,k] + self.delta[2][d][:,k]) / \
                    (gamma(self.delta[1][d][:,k] * self.delta[2][d][:,k]))
                            ) * (first+second)
                term7 += (tmp*ndw).sum()



            # update term 8
            term8 += loggamma(self.K * self.alpha) - log(self.K * gamma(self.alpha))
            term8 += (self.alpha - 1) * self.gam_E[d,:].sum()

            # update term 9
            term9 += loggamma(sum(self.gam[d,:])) - sum(loggamma(self.gam[d,:]))
            term9 += ( (self.gam[d,:]-1) * self.gam_E[d,:] ).sum()

        print('Done term 1 ~ 9')


        for k in range(self.K):
            # update term 10
            term10 += loggamma(self.V * self.eta) - log(self.V * gamma(self.eta))
            term10 +=  (self.eta-1) * self.lam_E[:,k].sum()

            # update term 11
            term11 += loggamma(sum( self.lam[:,k] )) - sum( loggamma(self.lam[:,k]) )
            term11 += ( ( self.lam[:,k]-1 ) * ( self.lam_E[:,k] ) ).sum()
        print('Done term 10, 11')

        return term1 + term2 - term3 + term4 - term5 + term6 - term7 + term8 - term9 + term10 - term11

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

    def _update_nu_E(self):
        self.nu_E = {}
        self.nu_E[1] = {}
        self.nu_E[2] = {}

        for d in range(self.D):
            sum_delta = self.delta[1][d] + self.delta[2][d]
            self.nu_E[1][d] = digamma(self.delta[1][d]) - digamma(sum_delta)
            self.nu_E[2][d] = digamma(self.delta[2][d]) - digamma(sum_delta)


    def _update_phi(self, d):
        # duplicated words are ignored
        Nd_index = np.nonzero(self.X[d,:])[0]
        # get the proportional value in each topic t,
        # and then normalize to make as probabilities
        # the indicator of Z_dn remains only one term


        for k in range(self.K):
            E_beta = self.kappa['b=1'][d][Nd_index,k] * self.lam_E[Nd_index,k] # Nd * 1: indexing for words in dth document
            E_theta = self.gam_E[d,k] # scalar: indexing for kth topic
            self.phi[d][Nd_index,k] = E_beta + E_theta # Nd * 1
        ## vectorize to reduce time
        # to prevent overfloat
        #self.phi -= self.phi.min(axis=1)[:,None]
        self.phi[d][Nd_index,:] = exp(self.phi[d][Nd_index,:])
        # normalize prob
        self.phi[d][Nd_index,:] /= np.sum(self.phi[d][Nd_index,:], axis=1)[:,None]
        # print(None)

    def _update_kappa(self, d):
        self.kappa['b=1'][d] = np.exp(self.phi[d] * self.lam_E + self.nu_E[1][d])  # W*K Dimension
        self.kappa['b=0'][d] = np.exp(self.phi[d] * self.lam_E + self.nu_E[2][d])

        self.kappa['b=1'][d] = self.kappa['b=1'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])
        self.kappa['b=0'][d] = self.kappa['b=0'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])

    def _update_delta(self, d):
        self.delta[1][d] = self.kappa['b=1'][d] + self.pi[1][d]
        self.delta[2][d] = self.kappa['b=0'][d] + self.pi[2][d]

        sum_delta = self.delta[1][d] + self.delta[2][d]
        self.nu_E[1][d] = digamma(self.delta[1][d]) - digamma(sum_delta)
        self.nu_E[2][d] = digamma(self.delta[2][d]) - digamma(sum_delta)

    def _update_gam(self,d):
        gam_d = np.repeat(self.alpha, self.K)
        ids = np.nonzero(self.X[d,:])[0]
        n_dw = self.X[d,:][ids] # ids*1
        phi_dwk = self.phi[d][ids,:] # ids*K
        gam_d = gam_d + np.dot(n_dw, phi_dwk) # K*1 + K*1

        self.gam[d,:] = gam_d
        self.gam_E[d, :] = self._E_dir_1d(self.gam[d, :])


    def _update_lam(self):
        # for k in range(self.K):
        #     lam_k = np.sum(self.X, axis=0) * self.phi[d][:,k] + self.eta
        #     self.lam[:,k] = lam_k
        self.lam = np.zeros((self.V, self.K))
        for d in range(self.D):
            self.lam += self.X[d,:][:,None] * self.phi[d] * self.kappa['b=1'][d]
        self.lam += self.eta

        # update lambda dirichlet expectation
        self._update_lam_E_dir()

        # update kappa? do we update kappa before E-step?
        # for d in range(self.D):
        #     self.kappa['b=1'][d] = self.phi[d] * self.lam_E * self.nu['b=1']
        #     self.kappa['b=0'][d] = self.phi[d] * self.lam_E * self.nu['b=0']
        #
        #     self.kappa['b=1'][d] = self.kappa['b=1'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])
        #     self.kappa['b=0'][d] = self.kappa['b=0'][d] / (self.kappa['b=1'][d] + self.kappa['b=0'][d])

    def train(self, threshold, max_iter, max_iter_doc):

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

                a = 0
                while sum(abs(gam_before - gam_after)) / self.K > threshold:
                    gam_before = gam_after
                    self._update_phi(d)
                    self._update_kappa(d)
                    self._update_delta(d)
                    self._update_gam(d)
                    gam_after = self.gam[d,:]
                    a += 1
                    if a > max_iter_doc:
                        break

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


