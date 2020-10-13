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
    def __init__(self, alpha, eta, K, evaluate_every=10):
        # loading data
        # self.data = pickle.load(open(path_data, 'rb'))
        # np.random.seed(0)
        # idx = np.random.choice(len(self.data), 1000, replace=False)
        # self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.eta = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.K = K
        self.evaluate_every = evaluate_every
        self.perplexity = []

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
        This is variational free parameters each endowed to
        Z_dn: psi_star
        theta_d: alpha_star
        phi_t : beta_star
        '''
        self.w2idx = cv.vocabulary_
        self.idx2w = {val: key for key, val in self.w2idx.items()}

        self.D, self.V = X.shape
        self.Nd = [len(np.nonzero(X[doc, :])[0]) for doc in range(self.D)]

        # initialize variational parameters for q(beta) ~ Dir(lambda)
        np.random.seed(1)
        self.components_ = np.random.gamma(100, 1/100, (self.K, self.V)) # dimension: K * V
        self._update_lam_E_dir()



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
        gamma = np.random.gamma(100., 1./100., (self.D, self.K))
        Elogtheta = _dirichlet_expectation_2d(gamma)
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self.components_.shape)

        # e-step for each document
        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]

            cnts = X[d, Nd_index] # 1*Nd
            gammad = gamma[d, :] # 1*K
            Elogthetad = Elogtheta[d, :] # 1*K
            expElogthetad = np.exp(Elogthetad) # 1*K
            expElogbetad = self.expElogbeta[:,Nd_index] # K*Nd

            # normalizer for phi_{dwk}
            # inner product between 1*K, 1*K array -> scalar normalizer
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate gamma, phi until gamma converges
            for it in range(maxIter):
                lastgamma = gammad

                # update for gamma_{dk} = alpha + \sum_w n_{dw} phi_{dwk}
                # here, phi_{dwk} is defined implicitly to save memory
                # elementwise product between 1*K and
                # innerproduct(1*Nd, Nd*K) = 1*K
                # -> 1*K dimension
                gammad = self.alpha + expElogthetad * \
                        np.dot(cnts / phinorm, expElogbetad.T)

                # for next iteration, update values
                Elogthetad = _dirichlet_expectation_1d_(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < threshold):

                    break

            # update dth gamma parameter after convergence
            gamma[d, :] = gammad

            # contribution of document d to the expected sufficient
            # statistics for the M step.
            # Note phi_{dwk} is not defined explicitly, but implicitly
            # by calculating expElogthetad, expElogbetad because of the memory issue
            # Here, we have not finished calculating lambda_{kw}
            # because we have not multiplied expElogbetad which will be done at the end of e-step
            sstats[:, Nd_index] += np.outer(expElogthetad.T, cnts/phinorm)


        # This step finished computing the sufficient statistics for the
        # M step, so that
        # sstats[k,w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{ Elogtheta_{dk} + Elogbeta_{kw} } / phinorm_{dw}
        sstats = sstats * self.expElogbeta

        return gamma, sstats

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
        gamma, sstats = self._e_step(X, maxIter, threshold, random_state)
        return gamma, sstats

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

        gamma, sstats = self.do_e_step(X, maxIter, threshold, random_state)

        self.gamma = gamma
        self.components_ = sstats + self.eta

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


    def _loglikelihood(self, prior, distr, Edirichlet, size):
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
        score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
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

        Elogtheta = _dirichlet_expectation_2d(self.gamma) # D*K
        gamma = self.gamma  # D*K
        Elogbeta = self.Elogbeta # K*V
        _lambda = self.components_  # K*V
        alpha = self.alpha
        eta = self.eta

        ELBO = 0

        # E[log p(docs | theta, beta)]
        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]
            cnts = X[d, Nd_index]  # 1*Nd

            temp = (Elogtheta[d, :, np.newaxis]
                    + Elogbeta[:, Nd_index])
            norm_phi = logsumexp(temp, axis=0)
            ELBO += np.dot(cnts, norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        ELBO += self._loglikelihood(alpha, gamma,
                                Elogtheta, self.K)

        # E[log p(beta | eta) - log q (beta | lambda)]
        ELBO += self._loglikelihood(eta, self.components_,
                                Elogbeta, self.V)

        return ELBO


    def _perplexity(self, ELBO):
        '''
        Calculates Approximated Perplexity
        '''
        denominator = sum(self.Nd)

        self.perplexity.append( exp(-ELBO / denominator) )


