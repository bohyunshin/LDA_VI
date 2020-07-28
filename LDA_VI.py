import numpy as np
from scipy.special import polygamma, gamma
import pickle
from numpy import exp, log
import time

# define free variational parameters
K = 50
psi_star = np.repeat(1/K, 3) # dimension: M * K
# beta_star = np.repeat
# alpha_star = np.repeat
dir = 'preprocessed_review.pickle'
class LDA_VI:
    def __init__(self, path_data, alpha, beta,  T):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.beta = beta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.T = T

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
        self.beta_star = np.ones((self.M, self.T))/self.T # dimension: M * T

        self.alpha_star = np.zeros(self.T)
        for d in range(self.D):
            tmp = np.repeat(self.Nd[d], self.T)/self.T + self.alpha
            self.alpha_star = np.vstack((self.alpha_star, tmp))
        self.alpha_star = self.alpha_star[1:,:]

    def _ELBO(self):
        term1 = 0 # E[ log p( w | phi, z) ]
        term2 = 0 # E[ log p( z | theta) ]
        term3 = 0 # E[ log p( phi | beta) ]
        term4 = 0 # E[ log p( theta | alpha) ]
        term5 = 0 # E[ log q( z | psi* ) ]
        term6 = 0 # E[ log q( phi | beta* ) ]
        term7 = 0 # E[ log q( theta | alpha* ) ]

        '''
        Update term1, term2, term5 together
        Update term3, term6 together
        Update term4, term7 together
        ELBO = term1 + term2 + term3 + term4 - term5 - term6 - term7
        '''

        # update term1, 2, 5
        for d in range(self.D):
            # update term1
            idx = np.array(self.doc2idx[d])
            tmp = self.beta_star[idx,:]
            for t in range(self.T):
                tmp[:,t] = self._E_dir(tmp[:,t])
            tmp = self.psi_star[d] * tmp
            term1 += tmp.sum()

            # update term2
            tmp2 = self._E_dir(self.alpha_star[d,:]) # T dimensional
            term2 += (self.psi_star[d] * tmp2[None,:]).sum()

            # update term5
            # for numerical stability,add delta.
            term5 += (self.psi_star[d] * log(self.psi_star[d] + 0.00000000001)).sum()

        print('Done term 1,2,5')
        print(f'term1: {term1}, term2: {term2}, term5: {term5}')

        # update term3, 6
        for t in range(self.T):
            # update term3
            term3 += self._E_dir(self.beta_star[:, t]).sum()

            # update term6
            logB = log(self._B(self.beta_star[:, t]))  # log ( dirichlet multiplicative factor)

            if np.isnan(logB):
                logB = 0
            val = (self.beta_star[:,t] - 1) * self._E_dir(self.beta_star[:, t])
            term6 += val.sum() + logB
        term3 *= self.beta - 1
        print('Done term 3,6')
        print(f'term3: {term3}, term6: {term6}')

        # update term 4, 7
        for d in range(self.D):
            # update term4
            term4 += self._E_dir(self.alpha_star[d,:]).sum()

            # update term7
            logB = log(self._B(self.alpha_star[d,:]))  # log ( dirichlet multiplicative factor)
            if np.isnan(logB):
                logB = 0
            val = (self.alpha_star[d:,]-1) * self._E_dir(self.alpha_star[d,:])
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
        return polygamma(1, params) - polygamma(1, sum(params))

    def _B(self, params):
        # dirichlet multiplicative factor
        return  gamma(params.sum()) / np.prod(gamma(params))

    def _update_Z_dn(self):
        # for topic t, the sum of probability should be one
        start = time.time()
        a = 0
        for d in range(self.D):
            # for i in range(self.Nd[d]): # for word in dth document
            prob_t = []
            Nd_index = np.array(self.doc2idx[d])
            # get the proportional value in each topic t,
            # and then normalize to make as probabilities
            # the indicator of Z_dn remains only one term


            for t in range(self.T):
                phi_sum = sum( self._E_dir(self.beta_star[Nd_index,t]) )
                theta_sum =  self._E_dir(self.alpha_star[d,:])[t]
                self.psi_star[d][:, t] = phi_sum + theta_sum
            ## vectorize to reduce time
            # to prevent overfloat
            self.psi_star[d] -= self.psi_star[d].min(axis=1)[:,None]
            self.psi_star[d] = exp(self.psi_star[d])
            # normalize prob
            self.psi_star[d] /= np.sum(self.psi_star[d], axis=1)[:,None]

            # store update psi
            # dimension of psi: Nd * T
        print(f'q(Z) update하는데 걸린 시간은 {(time.time() - start)/60}')
        print('done')

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
                for n in idx:
                    beta[n] += self._get_psi_tdn(t, d, n)

            self.beta_star[:,t] = beta
            print(f'{t}번째 주제 완료!')


    def _update_theta_d(self):
        for d in range(self.D):
            beta_star_dt = np.sum(self.psi_star[d], axis=0) + self.alpha
            self.alpha_star[d,:] = beta_star_dt


    def train(self, threshold):

        print('Making Vocabs...')
        self._make_vocab()
        print('Done!')

        print('Initializing Parms...')
        self._init_params()
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.M}')
        print(f'{self.T} topics chosen')

        print('Start optimizing!')
        # initialize ELBO
        ELBO_before = self._ELBO()
        ELBO_after = 99999
        self._ELBO_history = []
        self._ELBO_history.append(ELBO_before)


        max_iter = 0
        print('##################### start training #####################')
        while abs(ELBO_after - ELBO_before) > threshold:
            start = time.time()
            ELBO_before = ELBO_after
            print('\n')
            self._update_Z_dn()
            print('\n')
            self._update_phi_t()
            print('done')
            print('\n')
            self._update_theta_d()
            print('done')
            print('\n')
            ELBO_after = self._ELBO()

            self._ELBO_history.append(ELBO_after)

            print(f'Before ELBO: {ELBO_before}')
            print(f'After ELBO: {ELBO_after}')
            print('\n')
            max_iter += 1

            if max_iter == 100: break
            print(f'걸린 시간은 {(time.time() - start)/60}분')

        return None

    def perplexity(self):
        return None


