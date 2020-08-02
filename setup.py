from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

# define an extension that will be cythonized and compiled
ext = Extension(name="_online_lda_fast", sources=["_online_lda_fast.pyx"],
			include_dirs=[np.get_include()])
setup(ext_modules=cythonize(ext))

