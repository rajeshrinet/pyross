import  numpy as np
cimport numpy as np
cimport cython
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
        readonly int N, M,
        readonly double alpha, beta, gamma, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gamma = parameters.get('gamma')                    # recovery rate
        self.zeta  = parameters.get('zeta')                     # inverse of the average incubation period
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter 

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 3*self.M, dtype=DTYPE)           # right hand side
    
       
    cdef rhs(self, rp, tt):
        cdef: 
            int N=self.N, M=self.M, i, j
            double alpha=self.alpha, beta=self.beta, gamma=self.gamma, aa, bb
            double fsa=self.fsa
            double [:] S    = rp[0:M]        
            double [:] Ia   = rp[M:2*M]       
            double [:] Is   = rp[2*M:3*M]       
            double [:] Ni   = self.Ni       
            double [:] ld   = self.lld       
            double [:,:] CM = self.CM
            double [:] X    = self.drpdt        

        for i in prange(M, nogil=True):
            bb=0
            for j in prange(M):
                 bb += beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa
            X[i+M]   = alpha*aa     - gamma*Ia[i]
            X[i+2*M] = (1-alpha)*aa - gamma*Is[i]
        return

         
    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='this.mat'):
        from scipy.integrate import odeint
        from scipy.io import savemat
        
        def rhs0(rp, t):
            self.rhs(rp, t)
            self.CM = contactMatrix(t)
            return self.drpdt
            
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator. 
        u = odeint(rhs0, np.concatenate((S0, Ia0, Is0)), time_points, mxstep=5000000)
        #elif integrator=='odespy-vode':
        #    import odespy
        #    solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #    #solver = odespy.RKF45(rhs0)
        #    #solver = odespy.RK4(rhs0)
        #    solver.set_initial_condition(self.rp0)
        #    u, time_points = solver.solve(time_points)
        savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gamma':self.gamma })
        return
        



cdef class SEIR:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gamma, zeta, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gamma = parameters.get('gamma')                    # recovery rate
        self.zeta  = parameters.get('zeta')                     # inverse of the average incubation period
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter 

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side
    
       
    cdef rhs(self, rp, tt):
        cdef: 
            int N=self.N, M=self.M, i, j
            double alpha=self.alpha, beta=self.beta, gamma=self.gamma, aa, bb
            double fsa=self.fsa, zeta=self.zeta, ce1=self.zeta*self.alpha, ce2=self.zeta*(1-self.alpha)
            double [:] S    = rp[0:M]        
            double [:] E    = rp[M:2*M]       
            double [:] Ia   = rp[2*M:3*M]       
            double [:] Is   = rp[3*M:4*M]       
            double [:] Ni   = self.Ni       
            double [:] ld   = self.lld       
            double [:,:] CM = self.CM
            double [:] X    = self.drpdt        

        for i in prange(M, nogil=True):
            bb=0
            for j in prange(M):
                 bb+ = beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa
            X[i+M]   = aa       - zeta*E[i]
            X[i+2*M] = ce1*E[i] - gamma*Ia[i]
            X[i+3*M] = ce2*E[i] - gamma*Is[i]
        return

         
    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='this.mat'):
        from scipy.integrate import odeint
        from scipy.io import savemat
        
        def rhs0(rp, t):
            self.rhs(rp, t)
            self.CM = contactMatrix(t)
            return self.drpdt
            
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator. 
        u = odeint(rhs0, np.concatenate((S0, E0, Ia0, Is0)), time_points, mxstep=5000000)
        #elif integrator=='odespy-vode':
        #    import odespy
        #    solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #    #solver = odespy.RKF45(rhs0)
        #    #solver = odespy.RK4(rhs0)
        #    solver.set_initial_condition(self.rp0)
        #    u, time_points = solver.solve(time_points)
        savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gamma':self.gamma })
        return
        
