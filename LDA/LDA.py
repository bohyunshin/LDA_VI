import random
import numpy as np
from scipy.special import gammaln, psi
from collections import Counter
import pickle

class LDA():
    def __init__(self, K, DataPath, iter, alpha=None, beta=None):
        # data format should be
        # [ [word1, word2, word3], [word1, word2, word3, word4], ... ]
        self.documents = pickle.load(open(DataPath, 'rb'))
        self.D = len(self.documents) # total number of documents
        # to make counter object
        self.documents_c = [Counter(doc) for doc in self.documents]
        Vocab = Counter()
        for counter in self.documents_c:
            Vocab += counter
        self.V = len(Vocab.keys())
        self.unique_words = sorted(Vocab.keys())
        self.word2index = {word:index for index, word in enumerate(self.unique_words)}
        self.index2word = {v:k for k, v in self.word2index.items()}
        self.K = K
        self.iter = iter

        self.ph = np.zeros((self.K, self.V))
        self.th = np.zeros((self.D, self.K))


        if alpha == None:
            self.alpha = [0.1 for _ in range(self.K)]
        elif len(alpha) == 1:
            alpha_scalar = alpha
            self.alpha = [alpha_scalar for _ in range(self.K)]
        else: self.alpha = alpha

        if beta == None:
            self.beta = [50/self.K for _ in range(self.V)]
        elif len(beta) == 1:
            beta_scalar = beta
            self.beta = [beta_scalar for _ in range(self.V)]
        else: self.beta = beta

    def assign_topics(self, doc, word):
        # doc: index of document
        # word: just word, not index
        # self.n_dk = np.zeros((self.D, self.K))
        # self.n_kw = np.zeros((self.K, self.V))
        d = doc
        w = self.word2index[word]

        topic_prob_word = []
        for k in range(self.K):
            self.n_dk[d, k] -= 1
            self.n_kw[k, w] -= 1
            self.n_k[k] -= 1

            left = (self.n_dk[d, k] + self.alpha[k]) / ( sum(self.n_dk[d,:]) + sum(self.alpha) )
            right = (self.n_kw[k, w] + self.beta[w]) / ( sum(self.n_kw[k,:]) + sum(self.beta) )
            prob_gibbs = left * right
            topic_prob_word.append(prob_gibbs)
        topic_prob_word /= sum(topic_prob_word)
        new_z = np.random.multinomial(1, topic_prob_word).argmax()

        self.n_dk[d, new_z] += 1
        self.n_kw[new_z, w] += 1
        self.n_k[new_z] += 1

    def estimate_phi_theta(self):

        for k in range(self.K):
            for w in range(self.V):
                self.ph[k,w] = (self.n_kw[k,w] + self.beta[w]) / (sum(self.n_kw[k,:] + self.beta))

        for d in range(self.D):
            for k in range(self.k):
                self.th[d,k] = (self.n_dk[d,k] + self.alpha[k]) / (sum(self.n_dk[d,:] + self.alpha))

    def loglikelihood(self):
        return None

    def train(self):
        print(f'Number of documents: {self.D}')
        print(f'Number of vocabs: {self.V}')
        print(f'Number of Topics: {self.K}')

        # Create variables
        self.n_dk = np.zeros((self.D, self.K))
        self.n_kw = np.zeros((self.K, self.V))
        self.n_k = np.zeros(self.K)
        self.len_doc = [len(d) for d in self.documents]

        self.doc_topic_assignment = {}       # {index of doc: [topic assignment for each word]}
        for i, d in enumerate(self.len_doc):
            self.doc_topic_assignment[i] = [0 for _ in range(d)]

        # Initialize topic assignments
        for d in range(self.D):
            for i, word in enumerate(self.documents[d]):
                w = self.word2index[word]
                t = random.randint(0, self.K-1)
                self.doc_topic_assignment[d][i] = t
                self.n_dk[d,t] += 1
                self.n_kw[t,w] += 1
                self.n_k[t] += 1

        # collapsed sampling
        print('starting collapsed gibbs sampling...')
        for it in range(self.iter):
            for d in range(self.D):
                for i, word in enumerate(self.documents[d]):
                    self.assign_topics(d, word)

                    # calculate loglikelihood here

        self.estimate_phi_theta()

        return None

    def estimate_theta_phi(self):
        return None

    def Perplexity(self):
        return None


DataPath = '/Users/shinbo/PycharmProjects/LDA/preprocessed_review.pickle'
K = 3
LDA = LDA(K=3, DataPath = DataPath, iter = 10)

LDA.train()
print(sum(sum(LDA.ph)))
