import  numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_t

cdef class CommonMethods:
    cdef:
        readonly int N, M, nClass
        readonly np.ndarray population, Ni, CM, dxdt
        readonly dict paramList, readData

    cpdef set_contactMatrix(self, double t, contactMatrix)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(CommonMethods):
    """
    Susceptible, Infected, Removed (SIR)

    * Ia: asymptomatic
    * Is: symptomatic
    """

    cdef:
        readonly int kI, kE
        readonly np.ndarray gI, beta
        readonly double tsi_max, dt

    cpdef rhs(self, xt, tt)

    cpdef RK2_timestep(self, xt, t, contactMatrix)
