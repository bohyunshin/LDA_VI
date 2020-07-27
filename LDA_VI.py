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
            # to prevent overfloat
            self.psi_star[d] -= self.psi_star[d].min(axis=1)[:,None]
            self.psi_star[d] = exp(self.psi_star[d])
            self.psi_star[d] /= np.sum(self.psi_star[d], axis=1)[:,None]
            # normalized prob

            # store update psi
            # dimension of psi: Nd * T
            # Note that row index does not correspond to N, but i.
            # self.psi_star[d][i,:] = prob
            a += 1

            if a % 100 == 0:
                print(a)
                print(self.psi_star[d][0,:])
        print(f'q(Z) update하는데 걸린 시간은 {(time.time() - start)/60}')


    def _update_phi_t(self):
        for t in range(self.T):
            _gamma = []
            for m in range(self.M):
                _sum = 0
                for d in range(self.D):
                    for n in self.Nd[d]:
                        _sum += self._indicator(n,m) * self.psi_star[t,n,d]
                _sum += self.beta
                _gamma.append(_sum)
            self.beta_star[:,t] = _gamma


    def _update_theta_d(self):
        for d in range(self.D):
            _eta = []
            for t in tange(self.T):
                _sum = 0
                for n in self.Nd[d]:
                    _sum += self.psi_star[t,n,d]
                _sum += self.alpha
                _eta.append(_sum)
            self.alpha_star[d,:] = _eta


    def _coordinate_VI_LDA(self, max_iter, threshold):

        print('Making Vocabs...')
        self._make_vocab()
        print('Done!')
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.M}')

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


