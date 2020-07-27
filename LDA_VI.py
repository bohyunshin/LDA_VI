import numpy as np
from scipy.special import polygamma
import pickle
from numpy import exp
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
        self.alpha_star = np.ones((self.D , self.T))/self.T # dimension: D * T

    def _ELBO(self):
        # term 1 update
        term1 = 0
        for d in range(self.D):
            for n in self.Nd[d]:
                for t in range(self.T):
                    for m in range(self.M):
                        term1 += self._indicator(n, m) * self.psi_star[t,n,d] * self._E_dir(self.beta_star)

        return None

    def _E_dir(self, params):
        '''
        input: vector parameters of dirichlet
        output: Expecation of dirichlet - also vector
        '''
        return polygamma(1, params) - polygamma(1, sum(params))

    def _indicator(self, x,y):
        if x == y:
            return 1
        else:
            return 0

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


    def _coordinate_VI_LDA(self, max_iter, threshold):

        print('Making Vocabs...')
        self._make_vocab()
        print('Done!')
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.M}')
        print(f'{T} topics chosen')

        print('Initializing Parms...')
        self._init_params()

        print('Start optimizing!')
        # initialize ELBO
        ELBO_before = self._ELBO()
        ELBO_after = 99999
        self._ELBO_history = []
        self._ELBO_history.append(ELBO_before)

        while abs(ELBO_after - ELBO_before) > threshold:
            ELBO_before = ELBO_after
            self._update_Z_dn()
            self._update_phi_t()
            self._update_theta_d()
            ELBO_after = self.ELBO()

            self._ELBO_history.append(ELBO_after)

        return None

    def perplexity(self):
        return None


