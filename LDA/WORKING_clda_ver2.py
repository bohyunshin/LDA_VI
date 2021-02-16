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
    def __init__(self, alpha, eta, K,
                 seed_words=None, evaluate_every=10, confirmatory=None):
        # loading data
        # self.data = pickle.load(open(path_data, 'rb'))
        # np.random.seed(0)
        # idx = np.random.choice(len(self.data), 1000, replace=False)
        # self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.eta = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.seed_words = seed_words
        self.K = K
        self.evaluate_every = evaluate_every
        self.confirmatory = confirmatory
        self.perplexity = []

        self.iter = 0

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
        q(b_{dik}) ~ Ber(kap_{dik})
        q(nu_{dik}) ~ Beta(delta_{dik1}, delta_{dik2})
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
                # mapping filtered seed words to index
                self.seed_words[key] = [self.w2idx[i] for i in self.seed_words[key]]
                self.seed_word_index += self.seed_words[key]

            # This dictionary maps key -> seed words in other key
            # That is, suppose there are 4 topics, A,B,C,D.
            # Then, seedwords2other_keys[A] = seed words in topic B,C,D
            self.seedwords2other_keys = {}
            for k in range(self.K):
                setdiff_index = np.array(list(set(range(self.K)) - set([k])))
                key = list(self.seed_words.keys())[k]
                not_key = [key for i, key in enumerate(list(self.seed_words.keys())) if i in setdiff_index]

                other_seed_words = []
                for kk in not_key:
                    other_seed_words += self.seed_words[kk]
                self.seedwords2other_keys[key] = other_seed_words


            self.seed_word_index = list(set(self.seed_word_index))

        # initialize variational parameters for q(beta) ~ Dir(lambda)
        # initialize variational parameters for q(b) ~ Ber(kappa)
        # initialize variational parameters for q(v) ~ Beta(delta)
        np.random.seed(1)
        self.components_ = np.random.gamma(100, 1/100, (self.K, self.V)) # dimension: K * V
        self.kappa_1 = np.random.uniform(0, 1, (self.K, self.V))  # dimension: K * V
        self.kappa_2 = 1 - self.kappa_1
        self.delta_1 = np.random.gamma(100, 1 / 100, (self.K, self.V))  # dimension: K * V
        self.delta_2 = np.random.gamma(100, 1 / 100, (self.K, self.V))  # dimension: K * V
        # hyperparameters: Beta(1,1)
        self.pi1 = np.ones((self.K,self.V)) * 0.5
        self.pi2 = np.ones((self.K, self.V)) * 0.5

        # initialize logExpectation
        self._update_lam_E_dir()
        self._update_nu_E_beta()



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
        self.expKapElogbeta = np.exp(self.Elogbeta * self.kappa_1)

    def _update_nu_E_beta(self):
        # Elognu_1: E[log nu]
        self.Elognu_1 = digamma(self.delta_1) - digamma(self.delta_1 + self.delta_2)
        self.expElognu_1 = np.exp(self.Elognu_1)
        # Elognu_2: E[log (1-nu)]
        self.Elognu_2 = digamma(self.delta_2) - digamma(self.delta_1 + self.delta_2)
        self.expElognu_2 = np.exp(self.Elognu_2)

    def _make_pi_prior(self, Nd_index, alpha, beta):
        """Prior for informative prior, pi
        Basically, we assume uninformative prior, i.e., Beta(1,1)
        However, to incorpoarte information of seed words, we impose
        informative prior for beta distribution corresponding to seed words.

        If X ~ Beta(8,2), this distribution is skewed to 1 (mode near 1).
        If X ~ Beta(2,8), this distribution is skewed to 0 (mode near 0).

        Parameters
        ----------
        Nd_index: Word index for dth document
        alpha: First hyperparameter for beta distribution
        beta: Second hyperparameter for beta distirbution

        Returns
        -------
        (pi1, pi2): K*Nd dimensional matrix
        """
        pi1 = np.ones((self.K, len(Nd_index)))
        pi2 = np.ones((self.K, len(Nd_index)))

        if self.confirmatory is None:
            return pi1, pi2

        keys = list(self.seed_words.keys())

        for k in range(self.K):
            temp1 = pi1[k,:]
            temp2 = pi2[k, :]

            seed_index = [i for i,j in enumerate(Nd_index) if j in self.seed_words[keys[k]] ]
            other_seed_index = [i for i,j in enumerate(Nd_index) if j in self.seedwords2other_keys[keys[k]] ]

            if len(seed_index) >= 1:
                temp1[np.array(seed_index)] = alpha
                temp2[np.array(seed_index)] = beta

            if len(other_seed_index) >= 1:
                temp1[np.array(other_seed_index)] = beta
                temp2[np.array(other_seed_index)] = alpha

            # try:
            #     temp1[np.array(seed_index)] = alpha
            # except:
            #     pass
            # try:
            #     temp1[np.array(other_seed_index)] = beta
            # except:
            #     pass
            # try:
            #     temp2[np.array(seed_index)] = beta
            # except:
            #     pass
            # try:
            #     temp2[np.array(other_seed_index)] = alpha
            # except:
            #     pass

            pi1[k,:]=temp1
            pi2[k,:]=temp2

        return (pi1, pi2)

    def _make_Elognu(self, delta_j, delta_j_prime):
        """Expectation of log nu for jth

        Parameter
        ---------
        delta_j:

        delta_j_prime:


        Returns
        -------
        Elognu : K*Nd dimensional matrix
        """
        return digamma(delta_j) - digamma(delta_j + delta_j_prime)



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

        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]

            cnts = X[d, Nd_index]  # 1*Nd
            gammad = gamma[d, :]  # 1*K
            Elogthetad = Elogtheta[d, :]  # 1*K
            expElogthetad = np.exp(Elogthetad)  # 1*K
            expKapElogbetad = self.expKapElogbeta[:, Nd_index]  # K*Nd

            # normalizer for phi_{dwk}
            # inner product between 1*K, K*Nd array -> Nd array normalizer
            phinorm = np.dot(expElogthetad, expKapElogbetad) + 1e-100

            # Iterate gamma, phi until gamma converges
            for it in range(maxIter):
                lastgamma = gammad

                # update for gamma_{dk} = alpha + \sum_w n_{dw} phi_{dwk}
                # here, phi_{dwk} is defined implicitly to save memory
                # elementwise product between 1*K and
                # innerproduct(1*Nd, Nd*K) = 1*K
                # -> 1*K dimension
                gammad = self.alpha + expElogthetad * \
                         np.dot(cnts / phinorm, expKapElogbetad.T)

                # for next iteration, update values
                Elogthetad = _dirichlet_expectation_1d_(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expKapElogbetad) + 1e-100

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
            sstats[:, Nd_index] += np.outer(expElogthetad.T, cnts/phinorm) # K*Nd

        # This step finished computing the sufficient statistics for the
        # M step, so that
        # sstats[k,w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{ Elogtheta_{dk} + Elogbeta_{kw} } / phinorm_{dw}
        sstats_lambda = sstats * self.expKapElogbeta * self.kappa_1
        sstats_kappa = sstats * self.expKapElogbeta * self.Elogbeta

        return gamma, sstats_lambda, sstats_kappa

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
        gamma, sstats_lambda, sstats_kappa = self._e_step(X, maxIter, threshold, random_state)
        return gamma, sstats_lambda, sstats_kappa

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

        gamma, sstats_lambda, sstats_kappa = self.do_e_step(X, maxIter, threshold, random_state)

        self.gamma = gamma

        # finish calculating delta
        self.delta_1 = self.kappa_1 + self.pi1
        self.delta_2 = self.kappa_2 + self.pi2

        def normalize(arr1, arr2):
            K,V = arr1.shape
            for k in range(K):
                for v in range(V):
                    min_val = min(arr1[k,v], arr2[k,v])
                    arr1[k,v] -= min_val
                    arr2[k,v] -= min_val
            return arr1, arr2
        self.kappa_1, self.kappa_2 = normalize(sstats_kappa + self.Elognu_1, sstats_kappa + self.Elognu_2)
        self.kappa_1, self.kappa_2 = np.exp(self.kappa_1), np.exp(self.kappa_2)

        # finish calculating kappa
        # self.kappa_1 = np.exp(sstats_kappa + self.Elognu_1)
        # self.kappa_2 = np.exp(sstats_kappa + self.Elognu_2)
        kapnorm = self.kappa_1 + self.kappa_2 + EPS
        self.kappa_1 = self.kappa_1 / kapnorm
        self.kappa_2 = self.kappa_2 / kapnorm

        # finish calculating lambda
        self.components_ = sstats_lambda + self.eta

        # update lambda related variables, expectation and exponential of expectation
        self._update_lam_E_dir()
        self._update_nu_E_beta()

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

            self.iter += 1

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
        score += np.sum(gammaln(prior*size) - gammaln(np.sum(distr, 1)))
        return score

    def _loglikelihood_bernoulli(self, kappa1, kappa2, Elognu1, Elognu2):
        """Contribution of loglikelihood for
        E[log p(b | nu) - log q(b | kappa)]

        Parameter
        ---------
        kappa1, kappa2:

        Elognu1, Elognu2:

        Returns
        -------
        score : float
        """

        first = kappa1*Elognu1 + kappa2*Elognu2
        second = kappa1 * np.log(kappa1+EPS) + kappa2 * np.log(kappa2+EPS)
        score = first - second
        score = score.sum().sum()

        return score

    def _loglikelihood_beta(self, delta1, delta2, Elognu1, Elognu2):
        """Contribution of loglikelihood for
        E[log p(nu | pi) - log q(nu | delta)]

        Parameter
        ---------
        delta1, delta2:

        Elognu1, Elognu2:

        Returns
        -------
        score : float
        """
        pi1 = self.pi1 # K*V
        pi2 = self.pi2 # K*V
        def _coefficient_beta(delta1, delta2):
            denom = gamma(delta1) * gamma(delta2)
            numer = gamma(delta1 + delta2)
            return (numer / denom)

        coef = np.log(_coefficient_beta(delta1, delta2)) # K*V
        first = (pi1 - 1) * Elognu1 + (pi2 - 1) * Elognu2 # K*V
        second = (delta1 - 1) * Elognu1 + (delta2 - 1) * Elognu2 # K*V
        score = first - coef - second

        return score.sum().sum()

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
        ELBO = 0

        Elogtheta = _dirichlet_expectation_2d(self.gamma) # D*K
        gamma = self.gamma  # D*K
        Elogbeta = self.Elogbeta # K*V
        Ekaplogbeta = self.kappa_1 * self.Elogbeta # K*V
        _lambda = self.components_  # K*V
        alpha = self.alpha
        eta = self.eta

        # E[log p(docs | theta, beta)]
        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]
            cnts = X[d, Nd_index]  # 1*Nd

            temp = (Elogtheta[d, :, np.newaxis]
                    + Ekaplogbeta[:, Nd_index])
            norm_phi = logsumexp(temp, axis=0)
            ELBO += np.dot(cnts, norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        ELBO += self._loglikelihood(alpha, gamma,
                                Elogtheta, self.K)

        # compute E[log p(beta | eta) - log q (beta | lambda)]
        ELBO += self._loglikelihood(eta, self.components_,
                                Elogbeta, self.V)

        # compute E[log p(nu | pi) - log q(nu | delta)]
        ELBO += self._loglikelihood_bernoulli(self.kappa_1, self.kappa_2,
                                              self.Elognu_1, self.Elognu_2)

        # compute E[log p(b | nu) - log q(b | kappa)]
        ELBO += self._loglikelihood_beta(self.delta_1, self.delta_2,
                                         self.Elognu_1, self.Elognu_2)

        return ELBO


    def _perplexity(self, ELBO):
        '''
        Calculates Approximated Perplexity
        '''
        denominator = sum(self.Nd)

        self.perplexity.append( exp(-ELBO / denominator) )


