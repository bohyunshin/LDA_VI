# Import the extension module hello.
import hello
import _online_lda_fast as lda_fast
import numpy as np

a = np.random.gamma(1,1,(5,5))
print(lda_fast._dirichlet_expectation_2d(a))


# Call the print_result method 
#hello.print_result(23.0)
