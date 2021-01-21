import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma, psi, gammaln
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from online_lda import LDA_VI

#run my LDA model

dir_ = '/Users/shinbo/Desktop/metting/FEKIM/finished_preprocessed_paper_ver2.pickle'

# for ordinary lda
alpha = 5
eta = 0.1
maxIter = 1000
maxIterDoc =100
threshold = 10
random_state = 42

# for naive_clda
eta_seed = 1000
eta_not_seed = 0.0000001
# set seed words
seed_words = pickle.load(open('/Users/shinbo/Desktop/metting/FEKIM/seed_words.pkl','rb'))
K = len(seed_words.keys())

data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

lda = LDA_VI(alpha=alpha,eta_seed=eta_seed, eta=0.1,
             eta_not_seed=eta_not_seed, K=K,
             seed_words=seed_words, confirmatory=True)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

save_dir = '../../model_lda/FEKIM_CLDA_result.pkl'
pickle.dump(lda, open(save_dir, 'wb'))