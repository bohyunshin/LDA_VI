import Cython
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma, psi, gammaln
import pickle
import time
from clda import CLDA_VI
from online_lda import LDA_VI


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
lda = LDA_VI(dir_,alpha,eta,K)
lda.train(maxIter, maxIterDoc, threshold, random_state)

pickle.dump(lda, open('/Users/shinbo/PycharmProjects/model/lda_modified.pickle','wb'))

