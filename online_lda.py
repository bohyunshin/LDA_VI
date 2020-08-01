import numpy as np
from scipy.special import digamma, gamma, polygamma
import pickle
from numpy import exp, log
import time
from collections import Counter, OrderedDict

dir = 'preprocessed_review.pickle'
class LDA_VI:
    def __init__(self, path_data, alpha, eta,  K):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))[:1000]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.eta = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.K = K
        self.perplexity = []

    def _make_vocab(self):
        self.vocab = []
        for lst in self.data:
            self.vocab += lst
        self.vocab = sorted(list(set(self.vocab)))
        self.w2idx = {j:i for i,j in enumerate(self.vocab)}
        self.idx2w = {val:key for key, val in self.w2idx.items()}
        self.doc2idx = [ [self.w2idx[word] for word in doc] for doc in self.data]

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
        self.Ndw = [ OrderedDict(sorted(Counter(doc).items()))  for doc in self.doc2idx ]

        # set initial value for free variational parameters
        self.phi = np.ones((self.V, self.K)) # dimension: for topic d, Nd * K

        # initialize variational parameters of dirichlet through gamma distribution
        np.random.seed(1)
        self.lam = np.random.gamma(100, 1/100, (self.V, self.K)) # dimension: M * K
        np.random.seed(2)
        self.gam = np.random.gamma(100, 1/100, (self.D, self.K)) # dimension: D * K

        # initialize dirichlet expectation to reduce computation time
        self.lam_E = np.zeros((self.V, self.K))
        self.gam_E = np.zeros((self.D, self.K))

        print('calculating initial dirichlet expectation...')
        for k in range(self.K):
            self.lam_E[:,k] = self._E_dir(self.lam[:,k])
        for d in range(self.D):
            self.gam_E[d,:] = self._E_dir(self.gam[d,:])

        # self.alpha_star = np.zeros(self.T)
        # for d in range(self.D):
        #     tmp = np.repeat(self.Nd[d], self.T)/self.T + self.alpha
        #     self.alpha_star = np.vstack((self.alpha_star, tmp))
        # self.alpha_star = self.alpha_star[1:,:]

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

        # update term1, 2, 5
        for d in range(self.D):
            # update dirichlet expectation of beta_star
            for k in range(self.K):
                self.lam_E[:,k] = self._E_dir(self.lam[:,k])


            ndw = np.array(list(self.Ndw[d].keys()))
            for k in range(self.K):
                # update term 1
                tmp = self.lam_E[:,k] * self.phi[:,k]
                ndw_vec = np.zeros(self.V) # V * 1
                ndw_vec[ndw] += np.array(list(self.Ndw[d].values()))
                term1 += (tmp * ndw_vec).sum()

                # update term 2
                ndw_vec * self.phi[:,k] # V * 1
                theta_dk = self.gam_E[d,k] # scalar
                term2 += (theta_dk * ndw_vec).sum() # V * 1

                # update term 3
                tmp = self.phi[:,k] * log(self.phi[:,k] + 0.000000001)
                term3 += (tmp * ndw_vec).sum()


            # update term 4
            term4 += digamma(self.K * self.alpha) - log(self.K * gamma(self.alpha))
            term4 += (self.alpha - 1) * self.gam_E[d,:].sum()

            # update term 5
            term5 += digamma(sum(self.gam[d,:])) - sum(digamma(self.gam[d,:]))
            term5 += ( (self.gam[d,:]-1) * self.gam_E[d,:] ).sum()


        for k in range(self.K):
            # update term 6
            term6 += digamma(self.V * self.eta) - log(self.V * gamma(self.eta))
            term6 += ( (self.eta-1) * self.lam[:,k] ).sum()

            # update term 7
            term7 += digamma(sum( self.lam[:,k] )) - sum( digamma(self.lam[:,k]) )
            term7 += ( ( self.lam[:,k]-1 ) * ( self.lam_E[:,k] ) ).sum()

        return term1 + term2 - term3 + term4 - term5 + term6 - term7

    def _E_dir(self, params):
        '''
        input: vector parameters of dirichlet
        output: Expecation of dirichlet - also vector
        '''
        return polygamma(1,params) - polygamma(1,sum(params))

    def _logB(self, params):
        # log of dirichlet multiplicative factor
        return digamma(params) - digamma(sum(params))

    def _update_phi(self, d):
        # for topic t, the sum of probability should be one
        start = time.time()

        # duplicated words are ignored
        Nd_index = np.array(list(set(self.doc2idx[d])))
        # get the proportional value in each topic t,
        # and then normalize to make as probabilities
        # the indicator of Z_dn remains only one term


        for k in range(self.K):
            E_beta = self.lam_E[:,k][Nd_index] # Nd * 1: indexing for words in dth document
            E_theta = self.gam_E[d,:][k] # scalar: indexing for kth topic
            self.phi[Nd_index,k] = E_beta + E_theta # Nd * 1
        ## vectorize to reduce time
        # to prevent overfloat
        self.phi -= self.phi.min(axis=1)[:,None]
        self.phi = exp(self.phi)
        # normalize prob
        self.phi /= np.sum(self.phi, axis=1)[:,None]
        # print(None)

    def _update_gam(self,d):
        tmp = self.Ndw[d]
        gam_dk = np.repeat(self.alpha, self.K)
        for w in self.doc2idx[d]:
            gam_dk = gam_dk +  tmp[w] * self.phi[w,:] # K * 1 dimension
        self.gam[d,:] = gam_dk


    def _update_lam(self):
        for k in range(self.K):
            lam_k = np.repeat(self.eta, self.V)
            for d, doc in enumerate(self.doc2idx):
                for w in doc:
                    lam_kw = self.Ndw[d][w] * self.phi[w,k]
                    lam_k[w] += lam_kw
            print(f'{k} topic finished')


    def train(self, threshold):

        '''
        Variational inference using coordinate descent approach
        <Pseudo Code>
        while relative increase in ELBO_d > threshold:
            for each document in corpus:
                repeat relative increase in alpha_star > threshold
                    for each word in document:
                        for each topic:
                            update variational parameters of Z_dn: psi_star
                            update variational parameters of theta_d: alpha_star
            update variational parameters of phi_t
            calculate ELBO
        '''

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
        self._ELBO_history.append(ELBO_before)


        print('##################### start training #####################')
        while abs(ELBO_after - ELBO_before) > threshold:
            start = time.time()
            ELBO_before = ELBO_after
            print('\n')

            print('E step: start optimizing alpha_star...')
            # update psi_star, alpha_star
            for d in range(self.D):
                gam_before = self.gam[d,:]
                gam_after = np.repeat(999,self.K)
                while sum(abs(gam_before - gam_after)) / self.K > threshold:
                    gam_before = gam_after
                    self._update_phi(d)
                    self._update_gam(d)
                    gam_after = self.gam[d,:]

            # update beta_star
            print('M step: Updating Beta..')
            self._update_lam()
            print('Finished Inference!')
            print('\n')

            print('Now calculating ELBO...')
            ELBO_after = self._ELBO()

            self._ELBO_history.append(ELBO_after)

            self._perplexity(ELBO_after)

            print(f'Before ELBO: {ELBO_before}')
            print(f'After ELBO: {ELBO_after}')
            print('\n')
            print(f'Computation time: {(time.time()-start)/60} mins')

        print('Done Optimizing!')


    def _perplexity(self, ELBO):
        '''
        Calculates Approximated Perplexity
        '''
        denominator = sum(self.Nd)

        self.perplexity.append( exp(-ELBO / denominator) )


