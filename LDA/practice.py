import Cython
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma
import pickle
import time
from naive_clda import CLDA_VI
from online_lda import LDA_VI


# #run my LDA model
dir = "preprocessed_review.pickle"

# set seed words
seed_words_dict = dict()
seed_words_dict['price'] = ["price",  "fee",  "cost", "value", "money", "pay",
                             "expensive", "charge", "pricey",  "cheaper", "rate", "reasonably", "cheap"]
seed_words_dict['service'] = ["staff", "helpful", "service", "friendly", "pool", "desk",
                              "valet", "welcoming", "parking",  "offer", "manager",
                               "service", "help", "courteous", "check", "towel",
                              'serve']

seed_words_dict['food'] = ['food', 'coffee','water','beverage','breakfast','lunch','dinner','tea',
                           'fruit','starbucks','cafe', 'coconut','restaurant', 'drink']

seed_words_dict['accomodation'] = ['room','bathroom','lobby','hotel','stay','elevator','towel','balcony',
                                    'location','far','close','front','stay']

lda = CLDA_VI(dir, 5, 0.1, 0.000001, 100, 4, seed_words_dict, sampling=True)
lda.train(threshold=.01, max_iter=1000)
pickle.dump(lda, open('clda_model.pickle', 'wb'))
pickle.dump(lda, open('lda_model.pickle', 'wb'))

# load lda model
model = pickle.load(open('lda_model.pickle','rb'))
# model2 = pickle.load(open('lda_model.pickle','rb'))
# model = lda

# pd.DataFrame(model.gam).to_csv('/Users/shinbo/Desktop/metting/FEKIM/gamma.csv', index=False)
# pd.DataFrame(model.lam).to_csv('/Users/shinbo/Desktop/metting/FEKIM/lambda.csv', index=False)
# # pd.DataFrame(model.phi).to_csv('phi.csv', index=False)


# see word_topic distribution
# lda_lam = [model.kappa['b=1'][0][:,k] for k in range(4)]
lda_lam = [model.lam[:,k] for k in range(4)]

def print_top_words(lam, feature_names, n_top_words):
    for topic_id, topic in enumerate(lam):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
                       + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
print_top_words(lda_lam, list(model.cv.get_feature_names()), 200)
print(model.perplexity)
print(model._ELBO_history)


