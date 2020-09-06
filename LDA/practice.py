import Cython
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma
import pickle
import time
from naive_clda import CLDA_VI

# #run my LDA model
dir = "/Users/shinbo/Desktop/metting/FEKIM/finished_preprocessed_paper.pickle"
seed_words = {'microplastic':['plastic', 'microplastic', 'marine', 'water',  'particle', 'concentration', 'size', 'sediment', 'soil', 'debris', 'metal'],
              'human toxicity':['environment',  'pollution', 'exposure', 'chemical', 'food', 'health', 'potential', 'organic', 'risk']}
lda = CLDA_VI(dir, 5, 0.1, 0.00000001, 100, 2, seed_words, False)
lda.train(threshold=.01, max_iter=1000)
pickle.dump(lda, open('/Users/shinbo/Desktop/metting/FEKIM/clda_model.pickle', 'wb'))

# load lda model
model = pickle.load(open('/Users/shinbo/Desktop/metting/FEKIM/clda_model.pickle','rb'))

# pd.DataFrame(model.gam).to_csv('/Users/shinbo/Desktop/metting/FEKIM/gamma.csv', index=False)
# pd.DataFrame(model.lam).to_csv('/Users/shinbo/Desktop/metting/FEKIM/lambda.csv', index=False)
# # pd.DataFrame(model.phi).to_csv('phi.csv', index=False)

micro = pd.DataFrame({'lambda':model.lam[:,0], 'word':list(model.cv.get_feature_names())})
micro.sort_values(by='lambda', ascending=False).iloc[:100,:].to_csv('/Users/shinbo/Desktop/metting/FEKIM/micro.csv', index=False, encoding='utf-8')

toxi = pd.DataFrame({'lambda':model.lam[:,1], 'word':list(model.cv.get_feature_names())})
toxi.sort_values(by='lambda', ascending=False).iloc[:100,:].to_csv('/Users/shinbo/Desktop/metting/FEKIM/toxi.csv', index=False, encoding='utf-8')



# see word_topic distribution
lda_lam = [model.lam[:,k] for k in range(2)]
def print_top_words(lam, feature_names, n_top_words):
    for topic_id, topic in enumerate(lam):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
                       + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
print_top_words(lda_lam, list(model.cv.get_feature_names()), 100)
print(model.perplexity)
print(model._ELBO_history)


