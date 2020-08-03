from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import numpy as np

cv = CountVectorizer()
dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed.pickle"

class LDA_sklearn:
    def __init__(self, path_data, alpha, eta,  K):
        # loading data
        self.data = pickle.load(open(path_data, 'rb'))
        np.random.seed(0)
        idx = np.random.choice(len(self.data), 1000, replace=False)
        self.data = [j for i, j in enumerate(self.data) if i in idx]
        self.data = [' '.join(doc) for doc in self.data]
        self.K = K
        self.alpha = alpha
        self.eta = eta

    def _make_vocab(self):
        self.vocab = []
        for lst in self.data:
            self.vocab += lst
        self.vocab = sorted(list(set(self.vocab)))
        self.w2idx = {j:i for i,j in enumerate(self.vocab)}
        self.idx2w = {val:key for key, val in self.w2idx.items()}
        self.doc2idx = [ [self.w2idx[word] for word in doc] for doc in self.data]

    def _train(self):
        cv = CountVectorizer(vocabulary=self.w2idx)
        df = cv.fit_transform(self.data)
        lda = LatentDirichletAllocation(n_components=self.K, random_state=42,
                                        doc_topic_prior = self.alpha, topic_word_prior=self.eta,
                                        learning_method='batch')
        lda.fit(df)
