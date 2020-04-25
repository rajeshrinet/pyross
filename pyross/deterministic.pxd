import  numpy as np
cimport numpy as np
cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS:
    """
    Susceptible, Infected, Recovered, Susceptible (SIRS)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa, ep
        readonly np.ndarray rp0, Ni, drpdt, CM, FM, sa, iaa

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, gE, fsa
        readonly np.ndarray rp0, Ni, drpdt, CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI5R:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality

    S  ---> E
    E  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, R
    Ih ---> Ic, R
    Ic ---> Im, R
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gE, gIa, gIs, gIh, gIc, fsa, fh
        readonly np.ndarray rp0, Ni, drpdt, CM, FM, sa, iaa, hh, cc, mm

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR:
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    """
    cdef:
        readonly int N, M, ki
        readonly double alpha, beta, gI, fsa
        readonly np.ndarray rp0, Ni, drpdt,  CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR:
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    See: Lloyd, Theoretical Population Biology 60, 59􏰈71 (2001), doi:10.1006􏰅tpbi.2001.1525.
    """
    cdef:
        readonly int N, M, ki, ke
        readonly double alpha, beta, gI, fsa, gE
        readonly np.ndarray rp0, Ni, drpdt, CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIR:
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered (SEAIR)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, gE, gA, fsa
        readonly np.ndarray rp0, Ni, drpdt,  CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ:
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, gE, gA, fsa
        readonly double tS, tE, tA, tIa, tIs
        readonly np.ndarray rp0, Ni, drpdt,  CM, FM

    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R:
    """
    Susceptible, Exposed, Activates, Infected, Recovered (SEAIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality

    S  ---> E
    E  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, R
    Ih ---> Ic, R
    Ic ---> Im, R
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gE, gA, gIa, gIs, gIh, gIc, fsa, fh
        readonly np.ndarray rp0, Ni, drpdt, CM, FM, sa, iaa, hh, cc, mm

    cdef rhs(self, rp, tt)
