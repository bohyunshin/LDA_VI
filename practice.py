import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma
import pickle
import time
from online_lda import LDA_VI

dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed.pickle"
lda = LDA_VI(dir, 5, 0.1, 10)
lda.train(0.00001)
print(lda._ELBO_history)
pd.DataFrame(lda.phi).to_csv('phi.csv', index=False)
pd.DataFrame(lda.lam).to_csv('lam.csv', index=False)
pd.DataFrame(lda.gam).to_csv('gam.csv', index=False)

# from collections import Counter, OrderedDict
# a = [1,2,3,3,3,1,1,2,2]
# print(Counter(a).items())
# a = OrderedDict(sorted(dict(Counter(a)).items()))
# print(a.values())



