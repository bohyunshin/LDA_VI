import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import gamma, digamma, polygamma
import pickle
import time
from LDA_VI import LDA_VI

dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed.pickle"
lda = LDA_VI(dir, 5, 0.1, 10)


lda.train(0.00001)
print(lda._ELBO_history)
pd.DataFrame(lda.psi_star[0]).to_csv('psi_star.csv', index=False)
pd.DataFrame(lda.alpha_star).to_csv('alpha_star.csv', index=False)
pd.DataFrame(lda.beta_star).to_csv('beta_star.csv', index=False)

