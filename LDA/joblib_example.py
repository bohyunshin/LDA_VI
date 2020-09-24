import time
import math
from joblib import Parallel, delayed
def my_fun_2p(i, j):
    """ We define a simple function with two parameters.
    """
    time.sleep(1)
    return math.sqrt(i**j)
j_num = 3
num = 10
start = time.time()
for i in range(num):
    for j in range(j_num):
        my_fun_2p(i, j)
end = time.time()
print('{:.4f} s'.format(end-start))

start = time.time()
# n_jobs is the number of parallel jobs
Parallel(n_jobs=2)(delayed(my_fun_2p)(i, j) for i in range(num) for j in range(j_num))
end = time.time()
print('{:.4f} s'.format(end-start))