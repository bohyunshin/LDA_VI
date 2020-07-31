import numpy as np
from scipy.special import digamma, gamma, polygamma
import pickle
from numpy import exp, log
import time
from collections import Counter

# define free variational parameters
K = 50
psi_star = np.repeat(1/K, 3) # dimension: M * K
# beta_star = np.repeat
# alpha_star = np.repeat
dir = 'preprocessed_review.pickle'
class LDA_VI:
    def __init__(self, path_data, alpha, beta,  T):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))[:1000]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.beta = beta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.T = T
        self.perplexity = []

    def _make_vocab(self):
        self.vocab = []
        for lst in self.data:
            self.vocab += lst
        self.vocab = sorted(list(set(self.vocab)))
        self.w2idx = {j:i for i,j in enumerate(self.vocab)}
        self.idx2w = {val:key for key, val in self.w2idx.items()}
        self.doc2idx = [ [self.w2idx[word] for word in doc] for doc in self.data]

        self.Nd = [len(doc) for doc in self.data]

    def _init_params(self):
        '''
        Initialize parameters for LDA
        This is variational free parameters each endowed to
        Z_dn: psi_star
        theta_d: alpha_star
        phi_t : beta_star
        '''
        self.M = len(self.w2idx)
        self.D = len(self.data)
        self.Nd = [len(doc) for doc in self.data]


        # set initial value for free variational parameters
        self.psi_star = [ np.ones((Nd, self.T))/self.T for Nd in self.Nd ] # dimension: for topic d, Nd * T

        # initialize variational parameters of dirichlet through gamma distribution
        np.random.seed(1)
        self.beta_star = np.random.gamma(100, 1/100, (self.M, self.T)) # dimension: M * T
        np.random.seed(2)
        self.alpha_star = np.random.gamma(100, 1/100, (self.D, self.T)) # dimension: D * T

        # initialize dirichlet expectation to reduce computation time
        self.beta_star_E = np.zeros((self.M, self.T))
        self.alpha_star_E = np.zeros((self.D, self.T))

        print('calculating initial dirichlet expectation...')
        for t in range(self.T):
            self.beta_star_E[:,t] = self._E_dir(self.beta_star[:,t])
        for d in range(self.D):
            self.alpha_star_E[d,:] = self._E_dir(self.alpha_star[d,:])

        # self.alpha_star = np.zeros(self.T)
        # for d in range(self.D):
        #     tmp = np.repeat(self.Nd[d], self.T)/self.T + self.alpha
        #     self.alpha_star = np.vstack((self.alpha_star, tmp))
        # self.alpha_star = self.alpha_star[1:,:]

    def _ELBO(self):
        term1 = 0 # E[ log p( w | phi, z) ]
        term2 = 0 # E[ log p( z | theta) ]
        term3 = 0 # E[ log p( phi | beta) ]
        term4 = 0 # E[ log p( theta | alpha) ]
        term5 = 0 # E[ log q( z | psi* ) ]
        term6 = 0 # E[ log q( phi | beta* ) ]
        term7 = 0 # E[ log q( theta | alpha* ) ]

        '''
        ELBO is calculated w.r.t. each document
        Update term1, term2, term5 together
        Update term3, term6 together
        Update term4, term7 together
        ELBO = term1 + term2 + term3 + term4 - term5 - term6 - term7
        '''

        # update term1, 2, 5
        for d in range(self.D):
            # update term1
            idx = np.array(self.doc2idx[d])


            # update dirichlet expectation of beta_star
            for t in range(self.T):
                self.beta_star_E[:,t] = self._E_dir(self.beta_star[:,t])


            tmp = self.psi_star[d] * self.beta_star_E[idx,:] # Nd * T product Nd * T
            term1 += tmp.sum()

            # update term2
            # update dirichlet expectation of alpha_star
            self.alpha_star_E[d, :] = self._E_dir(self.alpha_star[d, :])
            tmp2 = self.alpha_star_E[d, :] # T dimensional
            term2 += (self.psi_star[d] * tmp2[None,:]).sum()

            # update term5
            # for numerical stability,add delta.
            term5 += (self.psi_star[d] * log(self.psi_star[d] + 0.00000000001)).sum()

        print('Done term 1,2,5')
        print(f'term1: {term1}, term2: {term2}, term5: {term5}')

        # update term3, 6
        for t in range(self.T):
            # update term3
            term3 += self.beta_star_E[:, t].sum()

            # update term6
            logB = self._logB(self.beta_star[:, t]).sum()  # log ( dirichlet multiplicative factor)

            # to prevent float error
            if np.isnan(logB):
                logB = 0
            val = (self.beta_star[:,t] - 1) * self.beta_star_E[:, t]
            term6 += val.sum() + logB
        term3 *= self.beta - 1
        print('Done term 3,6')
        print(f'term3: {term3}, term6: {term6}')

        # update term 4, 7
        for d in range(self.D):
            # update term4
            term4 += self.alpha_star_E[d,:].sum()

            # update term7
            logB = self._logB(self.alpha_star[d,:]).sum()  # log ( dirichlet multiplicative factor)
            if np.isnan(logB):
                logB = 0
            val = (self.alpha_star[d,:]-1) * self.alpha_star_E[d,:]
            term7 += val.sum() + logB
        term4 *= self.alpha - 1

        print('Done term 4, 7')
        print(f'term4: {term4}, term7: {term7}')

        return term1 + term2 + term3 + term4 - term5 - term6 - term7

    def _E_dir(self, params):
        '''
        input: vector parameters of dirichlet
        output: Expecation of dirichlet - also vector
        '''
        return polygamma(1,params) - polygamma(1,sum(params))

    def _logB(self, params):
        # log of dirichlet multiplicative factor
        return digamma(params) - digamma(sum(params))

    def _update_Z_dn(self, d):
        # for topic t, the sum of probability should be one
        start = time.time()

        Nd_index = np.array(self.doc2idx[d])
        # get the proportional value in each topic t,
        # and then normalize to make as probabilities
        # the indicator of Z_dn remains only one term


        for t in range(self.T):
            phi_sum = self.beta_star_E[:,t][Nd_index] # Nd * 1: indexing for words in dth document
            theta_sum = self.alpha_star_E[d,:][t] # scalar: indexing for t th topic
            self.psi_star[d][:, t] = phi_sum + theta_sum # Nd * 1
        ## vectorize to reduce time
        # to prevent overfloat
        self.psi_star[d] -= self.psi_star[d].min(axis=1)[:,None]
        self.psi_star[d] = exp(self.psi_star[d])
        # normalize prob
        self.psi_star[d] /= np.sum(self.psi_star[d], axis=1)[:,None]
        # print(None)

        # store update psi
        # dimension of psi: Nd * T

    def _get_psi_tdn(self,t,d,n):
        '''
        <input>
        t: topic index
        d: document index
        n: word index. Note that this is in the whole corpus.
        <output>
        prob for nth word in dth doc be the topic t, i.e., P(psi_tdn = t)
        '''
        word_in_doc_index = self.doc2idx[d].index(n)
        return self.psi_star[d][word_in_doc_index, t]




    def _update_phi_t(self):
        for t in range(self.T):
            beta = np.repeat(self.beta, self.M)
            a = 0
            for d, doc in enumerate(self.doc2idx):
                idx = np.array(list(doc))
                beta[idx] += self.psi_star[d][:,t]
            print(f'{t} topic finished')

            self.beta_star[:,t] = beta



    def _update_theta_d(self,d):
        beta_star_dt = np.sum(self.psi_star[d], axis=0) + self.alpha
        self.alpha_star[d,:] = beta_star_dt
        # print(None)


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
        print(f'# of unique vocabs: {self.M}')
        print(f'{self.T} topics chosen')

        print('Start optimizing!')
        # initialize ELBO
        ELBO_before = 0
        ELBO_after = 99999
        self._ELBO_history = []
        self._ELBO_history.append(ELBO_before)


        max_iter = 0
        print('##################### start training #####################')
        while abs(ELBO_after - ELBO_before) > threshold:
            start = time.time()
            ELBO_before = ELBO_after
            print('\n')

            print('E step: start optimizing alpha_star...')
            # update psi_star, alpha_star
            for d in range(self.D):
                alpha_star_before = self.alpha_star[d,:]
                alpha_star_after = np.repeat(999,self.T)
                while sum(abs(alpha_star_before - alpha_star_after)) / self.T > threshold:
                    alpha_star_before = alpha_star_after
                    self._update_Z_dn(d)
                    self._update_theta_d(d)
                    alpha_star_after = self.alpha_star[d,:]

            # update beta_star
            print('M step: find phi_t that maximizes ELBO')
            self._update_phi_t()
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


