import  numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_t

cdef class IntegratorsClass:
    cdef:
        readonly int N, M, kI, kE, nClass
        readonly double beta, gE, gA, gIa, gIs, gIh, gIc, fsa, fh, ep, gI
        readonly double tS, tE, tA, tIa, tIs, gIsp, gIcp, gIhp
        readonly np.ndarray rp0, Ni, dxdt, CM, FM, sa, iaa, hh, cc, mm, alpha




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(IntegratorsClass):
    """
    Susceptible, Infected, Recovered, Susceptible (SIRS)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(IntegratorsClass):
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI5R(IntegratorsClass):
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
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI8R(IntegratorsClass):
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
    Is ---> Is',Ih, R
    Ih ---> Ih',Ic, R
    Ic ---> Ic',Im, R
    """
    cdef rhs(self, rp, tt)





@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    See: Lloyd, Theoretical Population Biology 60, 59􏰈71 (2001), doi:10.1006􏰅tpbi.2001.1525.
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIR(IntegratorsClass):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered (SEAIR)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R(IntegratorsClass):
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
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI8R(IntegratorsClass):
    """
    Susceptible, Exposed, Activates, Infected, Recovered (SEAIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality

    S  ---> E
    E  ---> A
    A  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, Is', R
    Ih ---> Ic, Ih', R
    Ic ---> Im, Ic', R
    """
    cdef rhs(self, rp, tt)




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(IntegratorsClass):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef rhs(self, rp, tt)


#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.cdivision(True)
#@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(True)
@cython.cdivision(False)
@cython.nonecheck(True)
cdef class Spp(IntegratorsClass):
    cdef model_term* linear_terms
    cdef model_term* infection_terms
    cdef int linear_terms_len
    cdef int infection_terms_len
    cdef np.ndarray infection_classes_indices
    cdef dict model_class_name_to_class_index
    cdef dict parameters
    cdef dict linear_param_to_model_term
    cdef dict infection_param_to_model_term
    cdef list model_classes
    cdef np.ndarray _lambdas

    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef rhs(self, rp, tt)

cdef struct model_term:
    # Represents a term in the model, either linear or non-linear
    int oi_pos # Which model class to add to
    int oi_neg # Which model class to subtract from
    int oi_coupling # Which model class that couples
    int infection_index # Class infection index (only used if infection term)
    DTYPE_t* param

    # Implement at some point in the future:
    #int* add_to # Which model classes to add to
    #int add_to_len
    #int* subtract_from # Which model classes to subtract from
    #int subtract_from_len
    #int coupling # Which model class that couples
    #int infection_index # Class infection index (only used if infection term)
    #DTYPE_t param