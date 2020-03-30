import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange
cdef double PI = 3.1415926535
from scipy.sparse import spdiags


def addDelay(C, t, tC, tD, tW):
   f1=0
   for i in range(np.size(tD)):
       dt = (t-tD[i])/tW[i]
       f1 += 0.5*(1+np.tanh(dt))*C*tC[i]
   return f1
