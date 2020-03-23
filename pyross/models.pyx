import  numpy as np
cimport numpy as np
cimport cython
from scipy.io import savemat
import odespy
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)



cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M
        readonly double alpha, beta, gamma, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM
    
    def __init__(self, S0, Ia0, Is0, alpha, beta, gamma, fsa, M, Ni):

        self.alpha = alpha 
        self.beta  = beta
        self.gamma = gamma 
        self.fsa   = fsa

        self.N = np.sum(Ni)
        self.M = M

        self.rp0   = np.zeros( 3*self.M, dtype=DTYPE)        # initial distribution
        self.Ni    = np.zeros( self.M, dtype=DTYPE)          # # people in each age-group
        self.Ni           = Ni
        self.rp0[0:M]     = S0
        self.rp0[M:2*M]   = Ia0
        self.rp0[2*M:3*M] = Is0
        
        self.lld   = np.zeros( self.M, dtype=DTYPE)           # lambda matrix
        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C
        self.drpdt = np.zeros( 3*self.M, dtype=DTYPE)         # right hand side
    
       
    cdef rhs(self, rp):
        cdef: 
            int N=self.N, M=self.M, i, j
            double alpha=self.alpha, beta=self.beta, gamma=self.gamma, aa, bb
            double fsa=self.fsa
            double [:] S  = rp[0:M]        
            double [:] Ia = rp[M:2*M]       
            double [:] Is = rp[2*M:3*M]       
            double [:] Ni  = self.Ni       
            double [:] ld  = self.lld       
            double [:,:] CM = self.CM
            double [:] X = self.drpdt        

        
        for i in prange(M, nogil=True):
            bb=0
            for j in prange(M):
                 bb+= beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            ld[i] = bb
        for i in prange(M, nogil=True):
            aa = ld[i]*S[i]
            X[i]     = -aa
            X[i+M]   = alpha*aa     - gamma*Ia[i]
            X[i+2*M] = (1-alpha)*aa - gamma*Is[i]
        return

         
    def simulate(self, Tf, CM, filename='this.mat'):
        self.CM = CM
        time_points=np.linspace(0, Tf, Tf+1);  ## intervals at which output is returned by integrator. 
        dt = time_points[1] - time_points[0]
        def rhs0(rp, t):
            self.rhs(rp)
            #print (np.max(self.drpdt))
            return self.drpdt
            
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #solver = odespy.RKF45(rhs0)
        #solver = odespy.RK4(rhs0)
        solver.set_initial_condition(self.rp0)
        
        u, t = solver.solve(time_points)
        ss = 1# int(Npts/10000)

        savemat(filename, {'X':u[::ss], 't':t[::ss], 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gamma':self.gamma })
        return
        
