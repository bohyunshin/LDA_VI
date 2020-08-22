# from autograd import grad, hessian
#
# gw = lambda x1, x2: x1**4 + x1**2 + 10*x2**3 + x2
# gradient = grad(gw)
# hessian = hessian(gw)
#
# print(gradient(10.0, 10.0))
# print(hessian)

import numpy as np
import pickle
from SAGE_VI import SAGE_VI
dir = "/Users/shinbo/Desktop/metting/LDA/0. data/20news-bydate/newsgroup_preprocessed.pickle"
sage = SAGE_VI(dir,5,0.1,10,True)
sage.train(0.01, 100)
pickle.dump(sage,open('sage_model.pickle', 'wb'))


a = np.array([1,2,3], dtype='float64')
print(np.power(a,-1))