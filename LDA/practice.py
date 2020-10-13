import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma, psi, gammaln
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from online_lda import LDA_VI


# #run my LDA model
# dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed_corpus.pickle"
dir_ = 'preprocessed_review.pickle'

# for ordinary lda
alpha = 5
eta = 0.1
K = 4
maxIter = 1000
maxIterDoc = 100
threshold = 0.01
random_state = 42

# for naive_clda
eta_seed = 1000
eta_not_seed = 0.0000001
# set seed words
seed_words = dict()
seed_words['price'] = ["price", "prices", "priced", "fee", "fees", "cost", "value", "money", "pay",
                             "expensive", "charge", "pricey", "paid", "cheaper", "rate", "reasonably", "cheap"]
seed_words['service'] = ["staff", "helpful", "service", "friendly", "pool", "desk",
                              "valet", "welcoming", "parking", "offered", "offer", "manager",
                              "weclome", "serviced", "service", "help", "courteous", "check", "towels",
                              "solved", "solve", "offers", 'serve']

seed_words['food'] = ['food', 'coffee','water','beverage','beverages','breakfast','luanch','dinner','tea',
                           'fruit','fruits','starbucks','cafe','cafes','coconut','restaurant', 'restaurants', 'starbucks','drink']
seed_words['accomodation'] = ['rooms','room','bathrooms','bathroom','lobby','hotel','stay','elevator','towel','balcony'
                                    'location','far','close','front','stay']

data = pickle.load(open(dir_, 'rb'))
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

# lda = LDA_VI(alpha=alpha,eta=eta,K=K, eta_seed=eta_seed,
#              eta_not_seed=eta_not_seed, seed_words=seed_words, confirmatory=True)
# lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)
#
# pickle.dump(lda, open('/Users/shinbo/PycharmProjects/model/clda_all_review.pickle','wb'))


# see word_topic distribution
model = pickle.load(open('/Users/shinbo/PycharmProjects/model/clda_all_review.pickle','rb'))

lda_lam = [model.components_[k,:] for k in range(K)]

def print_top_words(lam, feature_names, n_top_words):
    for topic_id, topic in enumerate(lam):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
                       + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
print_top_words(lda_lam, list(cv.get_feature_names()), 200)
print(model.perplexity)
print(model._ELBO_history)