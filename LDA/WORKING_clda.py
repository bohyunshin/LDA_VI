import numpy as np
from numpy import exp, log
from scipy.special import digamma, gamma, loggamma, gammaln, logsumexp
from collections import Counter, OrderedDict
import pickle
import time
from _online_lda_fast import _dirichlet_expectation_2d, _dirichlet_expectation_1d_
from sklearn.feature_extraction.text import CountVectorizer

EPS = np.finfo(np.float).eps

class CLDA_VI:
    def __init__(self, alpha, eta, K, seed_words=None, evaluate_every=10):
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
        self.perplexity = []

        self.iter = 0

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

        # e-step for each document
        for d in range(self.D):
            Nd_index = np.nonzero(X[d, :])[0]

            # prior
            pi1, pi2 = self._make_pi_prior(Nd_index, alpha=8, beta=2) # K*Nd

            # initial kappa
            kappa1 = np.random.uniform(0,1,(self.K, len(Nd_index))) # K*Nd
            expkappa1 = np.exp(kappa1)
            kappa2 = 1-kappa1 # K*Nd

            # initial delta
            delta1 = pi1 + kappa1 # K*Nd
            delta2 = pi2 + kappa2 # K*Nd
            Elognu1 = self._make_Elognu(delta1, delta2)
            Elognu2 = self._make_Elognu(delta2, delta1)
            expElognu1 = np.exp(Elognu1)
            expElognu2 = np.exp(Elognu2)

            cnts = X[d, Nd_index] # 1*Nd
            gammad = gamma[d, :] # 1*K
            Elogthetad = Elogtheta[d, :] # 1*K
            Elogbetad = self.Elogbeta[:,Nd_index] # K*Nd
            expElogthetad = np.exp(Elogthetad) # 1*K
            expElogbetad = self.expElogbeta[:,Nd_index] # K*Nd

            # normalizer for phi_{dwk}
            # inner product between 1*K, K*Nd array -> Nd array normalizer
            phinorm = np.dot(expElogthetad, np.exp(Elogbetad*kappa1)) + 1e-100

            # Iterate gamma, phi, kappa, delta until gamma converges
            for it in range(maxIter):
                lastgamma = gammad

                phi = expElogthetad[:,None] * np.exp(Elogbetad*kappa1) / phinorm[None,:] # K*Nd

                kappa1 = np.exp(phi*Elogbetad) * expElognu1
                kappa2 = np.exp(phi*Elogbetad) * expElognu2
                kapnorm = kappa1+kappa2

                kappa1 = kappa1 / kapnorm
                kappa2 = kappa2 / kapnorm

                delta1 = kappa1 + pi1
                delta2 = kappa2 + pi2

                # update for gamma_{dk} = alpha + \sum_w n_{dw} phi_{dwk}
                gammad = self.alpha + np.dot(cnts, phi.T)




                # for next iteration, update values
                Elogthetad = _dirichlet_expectation_1d_(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, np.exp(Elogbetad*kappa1)) + 1e-100
                Elognu1 = self._make_Elognu(delta1, delta2)
                Elognu2 = self._make_Elognu(delta2, delta1)
                expElognu1 = np.exp(Elognu1)
                expElognu2 = np.exp(Elognu2)

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
            sstats[:, Nd_index] += phi*kappa1*cnts[None,:]


            # calculate contribution of document d to the ELBO in advance
            if self.iter % self.evaluate_every == 0:
                score = 0
                # contribution of phi
                score += np.sum(cnts[None,:]*phi)
                # contribution of b
                score += self._loglikelihood_bernoulli(cnts, kappa1, kappa2, Elognu1, Elognu2)
                # contribution of nu
                score += self._loglikelihood_beta(cnts, delta1, delta2, Elognu1, Elognu2)
            else:
                score = None


        # This step finished computing the sufficient statistics for the
        # M step, so that
        # sstats[k,w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{ Elogtheta_{dk} + Elogbeta_{kw} } / phinorm_{dw}
        # sstats = sstats * self.expElogbeta

        return gamma, sstats, score

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
        gamma, sstats, score = self._e_step(X, maxIter, threshold, random_state)
        return gamma, sstats, score

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

        gamma, sstats, score = self.do_e_step(X, maxIter, threshold, random_state)

        self.gamma = gamma
        self.components_ = sstats + self.eta

        if self.iter % self.evaluate_every == 0:
            self.score = score
        else:
            pass

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

    def _loglikelihood_bernoulli(self, cnts, kappa1, kappa2, Elognu1, Elognu2):
        """Contribution of loglikelihood for dth document:
        E[log p(b | nu) - log q(b | kappa)]

        Parameter
        ---------
        cnts : Nonzero word counts for dth document

        kappa1, kappa2:

        Elognu1, Elognu2:

        Returns
        -------
        score : float
        """

        mat = kappa1 * (Elognu1 - np.log(kappa1)) + kappa2 * (Elognu2 - np.log(kappa2)) # K*Nd
        mat = cnts[None,:]*mat

        return np.sum(mat)

    def _loglikelihood_beta(self, cnts, delta1, delta2, Elognu1, Elognu2):
        """Contribution of loglikelihood for dth document:
        E[log p(nu | pi) - log q(nu | delta)]

        Parameter
        ---------
        cnts : Nonzero word counts for dth document

        delta1, delta2:

        Elognu1, Elognu2:

        Returns
        -------
        score : float
        """

        def _coefficient_beta(delta1, delta2):
            denom = gamma(delta1) * gamma(delta2)
            numer = gamma(delta1 + delta2)
            return np.log(numer / denom)

        coef = _coefficient_beta(delta1, delta2) # K*Nd

        mat = coef * ( (delta1 - 1) * Elognu1 + (delta2 - 1) * Elognu2 )
        mat = cnts[None,:]*mat

        return np.sum(mat)

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

        ELBO = self.score

        # # E[log p(docs | theta, beta)]
        # for d in range(self.D):
        #     Nd_index = np.nonzero(X[d, :])[0]
        #     cnts = X[d, Nd_index]  # 1*Nd
        #
        #     temp = (Elogtheta[d, :, np.newaxis]
        #             + Elogbeta[:, Nd_index])
        #     norm_phi = logsumexp(temp, axis=0)
        #     ELBO += np.dot(cnts, norm_phi)

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


