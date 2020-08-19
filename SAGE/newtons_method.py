# from autograd import grad, hessian
#
# gw = lambda x1, x2: x1**4 + x1**2 + 10*x2**3 + x2
# gradient = grad(gw)
# hessian = hessian(gw)
#
# print(gradient(10.0, 10.0))
# print(hessian)

import numpy as np

a = np.array([1,2])
b = np.array(range(4)).reshape((2,2))
print(np.outer(a,a))

print(np.dot(b,b,b))