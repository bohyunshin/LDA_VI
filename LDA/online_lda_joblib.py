import numpy as np
from numpy import exp, log
import pickle
from _online_lda_fast import _dirichlet_expectation_2d, _dirichlet_expectation_1d_
from sklearn.feature_extraction.text import CountVectorizer
from utils import gen_even_slices

from scipy.special import digamma, gamma, loggamma, gammaln, logsumexp
from joblib import Parallel, delayed, effective_n_jobs, cpu_count

EPS = np.finfo(np.float).eps


def _update_doc_distribution(X, components_, expElogbeta, cal_sstats, alpha, maxIter, threshold, random_state):
    """E-step in EM update.

    Parameters
    ----------
    X : Document-Term matrix whose dimension is D*V

    components_ : Word-topic distribution for corpus denoted
        by lambda in the literature

    expElogbeta : Exponential of expectation of log beta.

    cal_sstats : Whether to calculate expected sufficient statistics or not.
        In the literature, this corresponds to E[ log beta_{kw} ].
        By computing E[ log beta_{kw} ] in advance, we do note need to store
        phi_{dwk}, which accounts for huge memory if stored as a variable.

    alpha : Prior for document topic distribution
        denoted by alpha in the literature

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

    D = X.shape[0]
    K = expElogbeta.shape[0]


    np.random.seed(random_state)
    gamma = np.random.gamma(100., 1. / 100., (D, K))
    Elogtheta = _dirichlet_expectation_2d(gamma)
    # expElogtheta = np.exp(Elogtheta)

    sstats = np.zeros(components_.shape)

    # e-step for each document
    for d in range(D):
        Nd_index = np.nonzero(X[d, :])[0]

        cnts = X[d, Nd_index]  # 1*Nd
        gammad = gamma[d, :]  # 1*K
        Elogthetad = Elogtheta[d, :]  # 1*K
        expElogthetad = np.exp(Elogthetad)  # 1*K
        expElogbetad = expElogbeta[:, Nd_index]  # K*Nd

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
            gammad = alpha + expElogthetad * \
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
        if cal_sstats:
            sstats[:, Nd_index] += np.outer(expElogthetad.T, cnts / phinorm)

    return (gamma, sstats)




class LDA_VI:
    def __init__(self, path_data, alpha, eta,  K, n_jobs = None, verbose=0, evaluate_every = 30):
        # loading data
        # self.data = pickle.load(open(path_data, 'rb'))
        # np.random.seed(0)
        # idx = np.random.choice(len(self.data), 1000, replace=False)
        # self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.alpha = alpha # hyperparameter; dimension: T * 1 but assume symmetric prior
        self.eta = eta  # hyperparameter; dimension: M * 1 but assume symmetric prior
        self.K = K
        self.perplexity = []
        self.n_jobs = cpu_count() if n_jobs is None else n_jobs
        self.verbose = verbose
        self.evaluate_every = evaluate_every

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
        self.Nd = [len(np.nonzero(X[doc,:])[0]) for doc in range(self.D)]

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


    def _e_step(self, X,  maxIter, cal_sstats, threshold, random_state,  parallel = None):
        """Parallel update for e-step

        Parameters
        ----------
        X : Document-Term matrix

        parallel : Pre-initialized joblib

        maxIter : Maximum number of iterations for individual document loop.

        cal_sstats: Whether to compute sufficient statistics

        threshold : Threshold for individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        (gamma, sstats) :
            `gamma` is topic distribution for each
            document. In the literature, this is called `gamma`.
            `sstats` is expected sufficient statistics for the M-step.
            Computation of M-step is almost done in advance on the e-step
            to reduce computation
        """

        # Run e-step in parallel
        n_jobs = effective_n_jobs(self.n_jobs)
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0,
                                                           self.verbose - 1))
        results = parallel(
            delayed(_update_doc_distribution)(X[idx_slice, :],
                                              self.components_,
                                              self.expElogbeta,
                                              cal_sstats,
                                              self.alpha,
                                              maxIter, threshold, random_state)
            for idx_slice in gen_even_slices(X.shape[0], n_jobs))

        # merge result
        doc_topics, sstats_list = zip(*results)
        gamma = np.vstack(doc_topics)

        if cal_sstats:
            # This step finished computing the sufficient statistics for the
            # M step, so that
            # sstats[k,w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{ Elogtheta_{dk} + Elogbeta_{kw} } / phinorm_{dw}
            suff_stats = np.zeros(self.components_.shape)
            for sstats in sstats_list:
                suff_stats += sstats
            suff_stats *= self.expElogbeta
        else:
            suff_stats = None

        return (gamma, suff_stats)

    def _em_step(self, X, maxIter, threshold, random_state, parallel=None):

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

        # E-step
        _, suff_sstats = self._e_step(X,
                                     maxIter=maxIter,
                                     threshold=threshold,
                                     random_state=random_state,
                                     cal_sstats=True,
                                     parallel=parallel)

        # self.gamma = gamma

        # Finished M-step
        self.components_ = suff_sstats + self.eta

        # update lambda related variables, expectation and exponential of expectation
        self._update_lam_E_dir()

        return


    def train(self , X, cv, maxIter, maxIterDoc, threshold, random_state):

        """Learn variational parameters using batch-approach
        Note: online-approach will be update shortly

        Parameters
        ----------
        X : Document-Term matrix

        cv: CountVectorizer object made from X

        maxIter : Maximum number of iterations for EM loop.

        maxIterDoc: Maximum number of iterations for individual loop

        threshold : Threshold for EM & individual document loop

        random_state : Integer
            Random number of initialization of gamma parameters

        Returns
        -------
        self

        """

        # print('Making Vocabs...')
        # self._make_vocab()

        print('Initializing Parms...')
        self._init_params(X, cv)
        print(f'# of Documents: {self.D}')
        print(f'# of unique vocabs: {self.V}')
        print(f'{self.K} topics chosen')

        print('Start optimizing!')
        # initialize ELBO

        ELBO_after = 99999
        self._ELBO_history = []
        n_jobs = effective_n_jobs(self.n_jobs)
        print('##################### start training #####################')
        with Parallel(n_jobs=n_jobs,
                      verbose=max(0,self.verbose-1)) as parallel:
            for iter in range(maxIter):

                ELBO_before = ELBO_after

                # do batch EM
                self._em_step(X, maxIterDoc, threshold, random_state, parallel)

                print(iter)
                # calculate ELBO
                if iter % self.evaluate_every == 0:
                    print('Now calculating ELBO...')
                    gamma, _ = self._e_step(X,
                                            maxIter=maxIterDoc,
                                            cal_sstats=False,
                                            threshold=threshold,
                                            random_state=random_state,
                                            parallel=parallel
                                            )
                    ELBO_after = self._approx_bound(X,gamma)
                    self._ELBO_history.append(ELBO_after)
                    self._perplexity(ELBO_after)

                    print(f'Current Iteration: {iter}')
                    print(f'Before ELBO: {ELBO_before}')
                    print(f'After ELBO: {ELBO_after}')
                    print('\n')

                    if abs(ELBO_before - ELBO_after) < threshold:
                        break


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


    def _approx_bound(self, X, gamma):
        """Estimate the variational bound, ELBO.

        Estimate the variational bound over "all documents". Since we
        cannot compute the exact loglikelihood for corpus, we estimate
        the lower bound of loglikelihood, ELBO in the literature.
        In mathematical formula, it is
        E[log p(w, z, theta, lambda)] - E[log q(z, theta, lambda)]

        Parameters
        ----------
        X : Document-Term matrix

        gamma : doc_topic distribution, dimension is D*K
        Returns
        -------
        score : float
        """

        Elogtheta = _dirichlet_expectation_2d(gamma) # D*K
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


