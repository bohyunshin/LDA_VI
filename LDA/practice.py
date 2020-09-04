import Cython
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma
import pickle
import time
from naive_clda import CLDA_VI

#run my LDA model
dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed.pickle"
seed_words = {'god':['believe','god','think','people'], 'image':['image','jpeg','pjg']}
lda = CLDA_VI(dir, 5, 0.1, 10, 2, seed_words)
lda.train(threshold=.01, max_iter=1000)
pickle.dump(lda, open('clda_model.pickle', 'wb'))

# a = np.array(range(6)).reshape(2,3)
# b = np.repeat(2,2)
# print(a)
# print(b[:,None])
# print(a + b[:,None])



#
#
# #
# # load lda model
# model = pickle.load(open('lda_model.pickle','rb'))
#
# pd.DataFrame(model.gam).to_csv('gamma.csv', index=False)
# pd.DataFrame(model.lam).to_csv('lambda.csv', index=False)
# # pd.DataFrame(model.phi).to_csv('phi.csv', index=False)
#
# # see word_topic distribution
# lda_lam = [model.lam[:,k] for k in range(10)]
# def print_top_words(lam, feature_names, n_top_words):
#     for topic_id, topic in enumerate(lam):
#         print('\nTopic Nr.%d:' % int(topic_id + 1))
#         print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
#                        + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
# print_top_words(lda_lam, list(model.cv.get_feature_names()), 200)
# print(model.perplexity)
# print(model._ELBO_history)

