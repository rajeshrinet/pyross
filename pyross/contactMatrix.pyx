import  numpy as np
cimport numpy as np
import scipy.linalg as spl
cimport cython
import pandas as pd
import warnings
from types import ModuleType

import os 

try:
    curDir, theFilename = os.path.split(__file__)  ## this does not Work on Binder 
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='Austria')
except ImportError:
    curDir ='../'



def Austria():
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='Austria')
    CH = np.array(my_data)

    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='Austria')
    CW = np.array(my_data)

    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='Austria')
    CS = np.array(my_data)

    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='Austria')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def China():
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='China')
    CH = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='China')
    CW = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='China')
    CS = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='China')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def Denmark():
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='Denmark')
    CH = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='Denmark')
    CW = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='Denmark')
    CS = np.array(my_data)                         
                                                   
    my_data = pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='Denmark')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def Germany():
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='Germany')
    CH = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='Germany')
    CW = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='Germany')
    CS = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='Germany')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def India():
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='India')
    CH = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='India')
    CW = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='India')
    CS = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='India')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def Italy():
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_home_1.xlsx'), sheet_name='Italy')
    CH = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_work_1.xlsx'), sheet_name='Italy')
    CW = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='Italy')
    CS = np.array(my_data)                        
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'), sheet_name='Italy')
    CO = np.array(my_data)
    return CH, CW, CS, CO


def UK():
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_home_2.xlsx'), sheet_name='United Kingdom of Great Britain')
    CH0 = np.array(my_data)                       
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_work_2.xlsx'), sheet_name='United Kingdom of Great Britain')
    CW0 = np.array(my_data)                       
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_school_2.xlsx'), sheet_name='United Kingdom of Great Britain')
    CS0 = np.array(my_data)                       
                                                  
    my_data = pd.read_excel(os.path.join(curDir, 'data/contact_matrices_152_countries/MUestimates_other_locations_2.xlsx'), sheet_name='United Kingdom of Great Britain')
    CO0 = np.array(my_data)

    #hard coding the first row from the file as panda wont read them
    CH = np.zeros((16, 16))
    CH[0,:]= np.array((0.478812799633172, 0.55185413960287,0.334323605154544,0.132361228266194,0.138531587861408,0.281604887066586,0.406440258772792,0.493947983343078,0.113301080935514,0.0746826413664804,0.0419640342896305,0.0179831987029717,0.00553694264516568,0.00142187285266089,0,0.000505582193632659))
    for i in range(15):
        CH[i+1, :] = CH0[i, :]
    CW = np.zeros((16, 16))
    CW[0,:]= np.array((0,0,0,0,0,0,0,0,0,0,0,0,0,0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000820604524144799,0.0000120585150153575,0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000316436833811157))
    for i in range(15):
        CW[i+1, :] = CW0[i, :]
    CS = np.zeros((16, 16))
    CS[0,:]= np.array((0.974577996106766,0.151369805263473,0.00874880925953218,0.0262790907947637,0.0111281607429249,0.0891043051294382,0.125477587043249,0.0883182775274553,0.0371824197201174,0.0294092695284747,0.0000000000000000000000000000000000000510911446027435,0.0000000000000000000000000000000113982464440009,0.00758428705895781,0.00151636767747242,0.0000000000000000000000000000000000000000000000000123262013953524,0.000000000000000000000000000000000000000000000000000000000000000597486362181075))
    for i in range(15):
        CS[i+1, :] = CS0[i, :]
    CO = np.zeros((16, 16))
    CO[0,:]= np.array((0.257847576361162,0.100135168376607,0.0458036773638843,0.127084549151753,0.187303683093508,0.257979214509792,0.193228849121415,0.336594916946786,0.309223290169635,0.070538522966953,0.152218422246435,0.113554851510519,0.0615771477785246,0.040429874099682,0.0373564987094767,0.00669781557624776))
    for i in range(15):
        CO[i+1, :] = CO0[i, :]
    return CH, CW, CS, CO






DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class ContactMatrixFunction:
    cdef:
        np.ndarray CH, CW, CS, CO

    def __init__(self, CH, CW, CS, CO):
        self.CH, self.CW, self.CS, self.CO = CH, CW, CS, CO

    cpdef get_individual_contactMatrices(self):
        return self.CH, self.CW, self.CS, self.CO

    def constant_contactMatrix(self):
        cdef:
            np.ndarray C
        C = self.CH + self.CW + self.CS + self.CO
        def C_func(t):
            return C
        return C_func

    def constant_CM(self, t):
        return self.CH + self.CW + self.CS + self.CO

    def interventions_temporal(self,times,interventions):
        cdef:
            np.ndarray t_arr = np.array(times)
            np.ndarray prefac_arr = np.array(interventions)
            int index
            np.ndarray C
            np.ndarray CH = self.CH, CW = self.CW
            np.ndarray CS = self.CS, CO = self.CO
        # times: ordered array with temporal boundaries between the
        #        different interventions
        # interventions: ordered matrix with prefactors of CW, CS, CO matrices
        #                during the different time intervals
        # note that len(interventions) = len(times) + 1
        def C_func(t):
            index = np.argmin( t_arr < t)
            if index == 0:
                if t >= t_arr[len(t_arr)-1]:
                    index = -1
            #print("t = {0},\tprefac_arr = {1}".format(t,prefac_arr[index]))
            return CH + prefac_arr[index,0]*CW \
                      + prefac_arr[index,1]*CS \
                      + prefac_arr[index,2]*CO
        return C_func


    def interventions_threshold(self,thresholds,interventions):
        cdef:
            np.ndarray thresholds_ = np.array(thresholds)
            np.ndarray prefac_arr = np.array(interventions)
            int index
            np.ndarray C
            np.ndarray CH = self.CH, CW = self.CW
            np.ndarray CS = self.CS, CO = self.CO
        # thresholds: array of shape [K*M,3] with K*M population numbers (S,Ia,Is)
        # interventions: array of shape [K+1,3] with prefactors during different
        #                phases of intervention
        # The current state of the intervention is defined by the
        # largest integer "index" such that state[j] >= thresholds[index,j] for all j.
        #
        def C_func(t,S,Ia,Is):
            state = np.concatenate((S, Ia, Is))
            index = np.argmin((thresholds_ <= state ).all(axis=1))
            if index == 0:
                N = len(thresholds_)
                if (thresholds_[N-1] <= state ).all():
                    index = N
            return CH + prefac_arr[index,0]*CW \
                      + prefac_arr[index,1]*CS \
                      + prefac_arr[index,2]*CO
        return C_func
    



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(ContactMatrixFunction):
    cdef:
        np.ndarray CHS, CWS, CSS, COS








"""
KreissPy
https://gitlab.com/AustenBolitho/kreisspy
sublibrary dedicated to calculating the transient effects of non-normal 
contact matricies
"""  
def _epsilon_eval(z, A, ord=2):
    """
    Finds the value of \epsilon for a given complex number and matrix.
    Uses the first definition of the pseudospectrum in Trfethen & Embree
    ord="svd" uses fourth definition (may be faster)
    

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    order: order of the matrix norm given from associated vector norm
    default is regular L2 norm -> returns maximum singular value.
    accepted inputs are any in spl.norm or "svd"  
    """
    z=np.array(z)
    A=np.array(A)
    zc = complex(z[0], z[1])
    try :
        ep = 1/spl.norm(spl.inv(zc*np.eye(*A.shape)-A),ord=ord)
        # ep = spl.norm(zc*np.eye(*A.shape)-A,ord=ord)
    except TypeError:
        if ord=="svd":
            ep = np.min(spl.svdvals(zc*np.eye(*A.shape)-A))
        else: raise Exception("invalid method")
    return ep


def _inv_epsilon_eval(z, A, ord=2):
    """
    Finds the value of 1/\epsilon for a given complex number and matrix.
    Uses the first definition of the pseudospectrum in Trfethen & Embree
    ord="svd" uses fourth definition (may be faster)
    

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    order: order of the matrix norm given from associated vector norm
    default is regular L2 norm -> returns maximum singular value.
    accepted inputs are any in spl.norm or "svd"  
    """
    z=np.array(z)
    A=np.array(A)
    zc = complex(z[0], z[1])
    try :
        iep = spl.norm(spl.inv(zc*np.eye(*A.shape)-A),ord=ord)
    except TypeError:
        if ord=="svd":
            iep = 1/np.min(spl.svdvals(zc*np.eye(*A.shape)-A))
        else: raise Exception("invalid method")
    return iep


def _kreiss_eval(z, A, theta=0, ord=2):
    """
    Kreiss constant guess for a matrix and pseudo-eigenvalue.

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm
    """
    z=np.array(z)
    A=np.array(A)
    kg = (z[0]-theta)*_inv_epsilon_eval(z, A, ord=ord)
    return kg


def _inv_kreiss_eval(z, A, theta=0, ord=2):
    """
    1/Kreiss constant guess for a matrix and pseudo-eigenvalue.
    for minimizer

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm
    """
    z=np.array(z)
    A=np.array(A)
    ikg = _epsilon_eval(z, A, ord=ord)/np.real(z[0]-theta) if z[0]-theta > 0 else np.inf
    # print(z[0]-theta)
    return ikg
    

def _transient_properties(guess, A, theta=0, ord=2):
    """
    returns the maximal eigenvalue (spectral abcissa),
    initial groth rate (numerical abcissa),
    the Kreiss constant (minimum bound of transient)
    and time of transient growth

    inputs:
    A: an MxM matrix
    guess: initial guess for the minimizer
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm

    returns: [spectral abcissa, numerical abcissa, Kreiss constant ,
              duration of transient, henrici's departure from normalcy']
    """
    from scipy.optimize import minimize
    A = np.array(A)
    if np.array_equal(A@A.T, A.T@A):
        warnings.warn("The input matrix is normal")
        # print("The input matrix is normal")
    evals = spl.eigvals(A)
    sa = evals[
        np.where(np.real(evals) == np.amax(np.real(evals)))[0]
    ]
    na = np.real(np.max(spl.eigvals((A+A.T)/2)))
    m = minimize(_inv_kreiss_eval, guess, args=(A, theta, ord),
                 bounds=((0, None), (None, None)))
    K = 1/m.fun
    tau = np.log(1/m.fun)/m.x[0]
    evals2 = np.dot(evals,np.conj(evals))
    frobNorm = spl.norm(A,ord='fro')
    henrici = np.sqrt(frobNorm**2-evals2)#/frobNorm
    return np.array([sa, na, K, tau, henrici],dtype=np.complex64)


def _first_estimate( A, tol=0.001):
    """
    Takes the eigenvalue with the largest real part
    
    returns a first guess of the
    maximal pseudoeigenvalue in the complex plane
    """
    evals = spl.eigvals(A)
    revals = np.real(evals)
    idxs = np.where(revals == np.amax(revals))[0]
    mevals = evals[idxs]
    iguesses = []
    for evl in mevals:
        guess = []
        a, b = np.real(evl), np.imag(evl)
        if a > 0:
            guess = [a+tol, b]
        else:
            guess = [tol, b]
        iguesses.append(guess)
    return iguesses


def characterise_transient(A, tol=0.001, theta=0, ord=2):
    """
    returns the maximal eigenvalue (spectral abcissa),
    initial groth rate (numerical abcissa),
    the Kreiss constant (minimum bound of transient)
    and time of transient growth

    inputs:
    A: an MxM matrix
    tol: Used to find a first estimate of the pseudospectrum
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm

    returns: [spectral abcissa, numerical abcissa, Kreiss constant ,
              duration of transient, henrici's departure from normalcy']
    
    """
    guesses = _first_estimate(A, tol)
    transient_properties = [1, 0, 0]
    for guess in guesses:
        tp = _transient_properties(guess, A, theta, ord)
        if tp[2] > transient_properties[2]:
            transient_properties = tp
        else:
            pass
    return transient_properties

