import Cython
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma, psi, gammaln
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer

from clda import CLDA_VI
from online_lda_joblib import LDA_VI


# #run my LDA model
# dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed_corpus.pickle"
dir_ = 'preprocessed_review.pickle'

alpha = 5
eta = 0.1
K = 5
maxIter = 1000
maxIterDoc = 100
threshold = 0.01
random_state = 42

data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(dir_,alpha,eta,K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

pickle.dump(lda, open('/Users/shinbo/PycharmProjects/model/lda_modified_joblib.pickle','wb'))


# see word_topic distribution
model = pickle.load(open('/Users/shinbo/PycharmProjects/model/lda_modified_joblib.pickle','rb'))

lda_lam = [model.components_[k,:] for k in range(K)]

def print_top_words(lam, feature_names, n_top_words):
    for topic_id, topic in enumerate(lam):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
                       + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
print_top_words(lda_lam, list(model.cv.get_feature_names()), 200)
print(model.perplexity)
print(model._ELBO_history)