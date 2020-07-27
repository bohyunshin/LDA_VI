import numpy as np
import pickle
import time
from LDA_VI import LDA_VI

dir = "preprocessed_review.pickle"
lda = LDA_VI(dir, 0.1, 0.1, 10)
lda._make_vocab()
print('끝')
lda._init_params()
print('초기화끝')
a = lda._update_theta_d()
print(lda.alpha_star)

# T = 10
# M = len(lda.w2idx)
# D = len(lda.data)
#
# beta_star = np.ones((M,T)) / T
# alpha_star = np.ones((D,T)) / T
# psi_star = [ [np.ones((d,n)) for d in range(D) for n in lda.doc2idx[d]] for _ in range(T) ]

# self.psi_star = np.ones((self.T, self.M, self.D)) / self.T  # dimension: T*= * M * D
# self.beta_star = np.ones((self.M, self.T)) / self.T  # dimension: M * T
# self.alpha_star = np.ones((self.D, self.T)) / self.T  # dimension: D * T


# lda._init_params()
# print('done')
#
# start = time.time()
# lda._update_Z_dn()
# print((time.time() - start) / 60)

