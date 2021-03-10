import numpy as np
from numpy import exp, log
from scipy.special import digamma, gamma, loggamma, gammaln, logsumexp
from collections import Counter, OrderedDict
import pickle
import time
from _online_lda_fast import _dirichlet_expectation_2d, _dirichlet_expectation_1d_
from sklearn.feature_extraction.text import CountVectorizer

EPS = np.finfo(np.float).eps

class LDA_VI:
    def __init__(self, alpha, eta, K, eta_seed=None, eta_not_seed=None, seed_words=None,
                 confirmatory=None, evaluate_every=10):
        # loading data
        # self.data = pickle.load(open(path_data, 'rb'))
        # np.random.seed(0)
        # idx = np.random.choice(len(self.data), 1000, replace=False)
        # self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        if confirmatory:
            self.eta_ordinary = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        else:
            self.eta = eta
        self.eta_seed = eta_seed
        self.eta_not_seed = eta_not_seed
        self.seed_words = seed_words
        self.K = K
        self.evaluate_every = evaluate_every
        self.confirmatory = confirmatory
        self.perplexity = []

        if self.confirmatory:
            if self.K != len(self.seed_words.keys()):
                raise ValueError('Input number of topics does not match number of topics in seed words')

    def _make_vocab(self):
        self.vocab = []
        # for lst in self.data:
        #     self.vocab += lst
        # self.vocab = sorted(list(set(self.vocab)))
        #
        # # make DTM
        # self.data_join = [' '.join(doc) for doc in self.data]
        # self.cv = CountVectorizer()
        # self.X = self.cv.fit_transform(self.data_join).toarray()
        # self.w2idx = self.cv.vocabulary_
        # self.idx2w = {val: key for key, val in self.w2idx.items()}

        # # excluded words from count vectorizer
        # stop_words = list(set(self.vocab) - set(list(self.w2idx.keys())))


    def _init_params(self, X, cv):
        '''
        Initialize parameters for LDA
        This is variational free parameters each endowed to variational distribution
        q(Z_{di} = k) ~ Multi(phi_{dwk})
        q(theta_d) ~ Dir(gamma_d)
        q(beta_k) ~ Dir(lambda_k)
        '''
        self.w2idx = cv.vocabulary_
        self.idx2w = {val: key for key, val in self.w2idx.items()}

        self.D, self.V = X.shape
        self.Nd = [len(np.nonzero(X[doc, :])[0]) for doc in range(self.D)]


        if self.confirmatory:

            self.seed_word_index = []

            # change words in seed_words dictionary to index
            for key in self.seed_words.keys():
                # filter seed words existing in corpus vocabulary
                self.seed_words[key] = [i for i in self.seed_words[key] if i in list(self.w2idx.keys())]
                self.seed_words[key] = [self.w2idx[i] for i in self.seed_words[key]]
                self.seed_word_index += self.seed_words[key]

            self.seed_word_index = list(set(self.seed_word_index))

            # make asseymmetric prior for word-topic distribution
            # different by each topic
            self.eta = self.eta_ordinary * np.ones((self.K, self.V))
            for k in range(self.K):
                setdiff_index = np.array(list(set(range(self.K)) - set([k])))
                key = list(self.seed_words.keys())[k]
                not_key = [key for i, key in enumerate(list(self.seed_words.keys())) if i in setdiff_index]
                self.eta[k, np.array(self.seed_words[key])] = self.eta_seed

                for kk in not_key:
                    self.eta[k, np.array(self.seed_words[kk])] = self.eta_not_seed

        # initialize variational parameters for q(beta) ~ Dir(lambda)
        np.random.seed(1)
        self.components_ = np.random.gamma(100, 1/100, (self.K, self.V)) # dimension: K * V
        self._update_lam_E_dir()

        np.random.seed(1)
        self.gamma = np.random.gamma(100., 1. / 100., self.K)  # 1*K



    # def _E_dir(self, params_mat):
    #     '''
    #     input: vector parameters of dirichlet
    #     output: Expecation of dirichlet - also vector
    #     '''
    #     return _dirichlet_expectation_2d(params_mat)
    #
    # def _E_dir_1d(self, params):
    #     return _dirichlet_expectation_1d_(params)

    # def _update_gam_E_dir(self):
    #     self.gam_E = self._E_dir(self.gam)

    def _update_lam_E_dir(self):
        self.Elogbeta = _dirichlet_expectation_2d(self.components_)
        self.expElogbeta = np.exp(self.Elogbeta)



    def _e_step(self, X, maxIter, threshold, random_state):

        """E-step in EM update.

        Parameters
        ----------
        X : Document-Term matrix

        maxIter : Maximum number of iterations for individual document loop.

        threshold : Threshold for individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        (gamma, sstats) :
            `gamma` is topic distribution for each
            document. In the literature, this is called `gamma`.
            `sstats` is expected sufficient statistics for the M-step.
            Computation of M-step is done in advance to reduce computation

        """

        np.random.seed(random_state)
        phi = np.random.gamma(100., 1./100., (self.D, self.K)) # D*K
        gamma = self.gamma # 1*K
        Elogtheta = _dirichlet_expectation_1d_(gamma) # 1*K
        expElogtheta = np.exp(Elogtheta)

        # e-step for each document
        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]

            cnts = X[d, Nd_index] # 1*Nd
            #lastphid = phi[d, :] # 1*K


            Elogbetad = self.Elogbeta[:,Nd_index] # K*Nd

            # normalizer for phi_{dk}
            # matrix product between 1*Nd, Nd*K array and then multiplied by
            # expElogtheta -> 1*K array
            phinorm =  np.dot(cnts, Elogbetad.T) + Elogtheta
            phinorm -= phinorm.max()
            phinorm = np.exp(phinorm)
            # phinorm = np.exp(  ) * expElogtheta
            phid = phinorm / phinorm.sum()

            phi[d,:] = phid

        gamma = phi.sum(axis=0) + self.alpha # 1*K
        return gamma, phi

    def do_e_step(self,X,  maxIter, threshold, random_state, parallel = None):
        """Parallel update for e-step

        Parameters
        ----------
        X : Document-Term matrix

        parallel : Pre-initialized joblib

        maxIter : Maximum number of iterations for individual document loop.

        threshold : Threshold for individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        (gamma, sstats) :
            `gamma` is topic distribution for each
            document. In the literature, this is called `gamma`.
            `sstats` is expected sufficient statistics for the M-step.
            Computation of M-step is done in advance to reduce computation
        """

        # Parallel job is not finished yet!!
        gamma, phi = self._e_step(X, maxIter, threshold, random_state)
        return gamma, phi

    def _em_step(self, X, maxIter, threshold, random_state):

        """EM-step for 1 iteration

        Parameters
        ----------
        X : Document-Term matrix

        maxIter : Maximum number of iterations for for individual document loop.

        threshold : Threshold for individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        (gamma, components_) :
            `gamma` is topic distribution for each
            document. In the literature, this is called `gamma`.
            `components_` is word distribution for each
            topic. In the literature, this is called 'lambda'.
            It has the same meaning as self.components_ in scikit-learn implementation

        """

        gamma, phi = self.do_e_step(X, maxIter, threshold, random_state)

        self.phi = phi
        self.gamma = gamma
        self.components_ = np.dot( phi.T, X ) + self.eta

        # update lambda related variables, expectation and exponential of expectation
        self._update_lam_E_dir()

        return


    def train(self , X, cv, maxIter, maxIterDoc, threshold, random_state):

        """Learn variational parameters using batch-approach
        Note: online-approach will be update shortly

        Parameters
        ----------
        X: Document-Term matrix

        maxIter : Maximum number of iterations for EM loop.

        maxIterDoc: Maximum number of iterations for individual loop

        threshold : Threshold for EM & individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        self

        """

        print('Initializing Parms...')
        self._init_params(X,cv)
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.V}')
        print(f'{self.K} topics chosen')

        print('Start optimizing!')
        # initialize ELBO
        ELBO_after = 99999
        self._ELBO_history = []

        print('##################### start training #####################')
        for iter in range(maxIter):

            ELBO_before = ELBO_after

            # do EM-step
            self._em_step(X, maxIterDoc, threshold, random_state)
            # print(iter)
            if iter % self.evaluate_every == 0:
                print('Now calculating ELBO...')
                ELBO_after = self._approx_bound(X)
                self._ELBO_history.append(ELBO_after)
                self._perplexity(ELBO_after)

                print(f'Current Iteration: {iter}')
                print(f'Before ELBO: {ELBO_before}')
                print(f'After ELBO: {ELBO_after}')
                print('\n')

                if abs(ELBO_before - ELBO_after) < threshold:
                    break

        print('Done Optimizing!')


    def _loglikelihood(self, prior, distr, Edirichlet, size, beta=None, theta=None):
        """Calculate loglikelihood for
        E[log p(theta | alpha) - log q(theta | gamma)]
        E[log p(beta | eta) - log q (beta | lambda)]

        Parameters
        ----------
        prior : Prior for each distribution. In literature,
        this is alpha and eta

        distr : Variational parameters for q(theta), q(beta)
        For q(theta), this is gamma, D*K dimensional array and
        for q(beta), this is beta, K*V dimensional array

        Edirichlet: Expectation for log dirichlet specified in distr.
        For q(theta), this is self.Elogtheta and
        for q(beta), this is self.Elogbeta

        """

        score = np.sum((prior - distr) * Edirichlet)
        score += np.sum(gammaln(distr) - gammaln(prior))
        if self.confirmatory and beta:
            score += np.sum(gammaln(np.sum(prior,1)) - gammaln(np.sum(distr, 1)))
        elif theta:
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr)))
        else:
            score += np.sum(gammaln(prior*size) - gammaln(np.sum(distr, 1)))
        return score


    def _approx_bound(self, X):
        """Estimate the variational bound, ELBO.

        Estimate the variational bound over "all documents". Since we
        cannot compute the exact loglikelihood for corpus, we estimate
        the lower bound of loglikelihood, ELBO in the literature.
        In mathematical formula, it is
        E[log p(w, z, theta, lambda)] - E[log q(z, theta, lambda)]

        Parameters
        ----------
        X : Document-Term matrix

        Returns
        -------
        score : float
        """

        Elogtheta = _dirichlet_expectation_1d_(self.gamma) # 1*K
        gamma = self.gamma  # 1*K
        Elogbeta = self.Elogbeta # K*V
        _lambda = self.components_  # K*V
        phi = self.phi # D*K
        alpha = self.alpha
        eta = self.eta

        ELBO = 0

        # E[log p(docs | theta, beta)]

        temp = np.dot( phi, Elogbeta ) * X
        ELBO += temp.sum().sum()

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        ELBO += self._loglikelihood(alpha, gamma,
                                Elogtheta, self.K, theta=True)

        # E[log p(beta | eta) - log q (beta | lambda)]
        if self.confirmatory:
            ELBO += self._loglikelihood(eta, self.components_,
                                    Elogbeta, self.V, beta=True)
        else:
            ELBO += self._loglikelihood(eta, self.components_,
                                        Elogbeta, self.V, beta=True)

        return ELBO


    def _perplexity(self, ELBO):
        '''
        Calculates Approximated Perplexity
        '''
        denominator = sum(self.Nd)

        self.perplexity.append( exp(-ELBO / denominator) )


