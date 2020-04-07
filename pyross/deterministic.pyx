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
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIa')                      # recovery rate of Is
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
            double alpha=self.alpha, beta=self.beta, gIa=self.gIa, aa, bb
            double fsa=self.fsa, alphab=1-self.alpha,gIs=self.gIs
            double [:] S    = rp[0  :M]        
            double [:] Ia   = rp[M  :2*M]       
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
            X[i+M]   = alpha *aa - gIa*Ia[i]
            X[i+2*M] = alphab*aa - gIs*Is[i]
        return

         
    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='None'):
        from scipy.integrate import odeint
        
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
        
        if filename=='None':
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        else:
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
            from scipy.io import savemat
            savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs })
        return data
        



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
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gE    = parameters.get('gE')                       # recovery rate of E
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
            double alpha=self.alpha, beta=self.beta, gIa=self.gIa, gIs=self.gIs, aa, bb
            double fsa=self.fsa, gE=self.gE, ce1=self.gE*self.alpha, ce2=self.gE*(1-self.alpha)
            double [:] S    = rp[0  :  M]        
            double [:] E    = rp[  M:2*M]       
            double [:] Ia   = rp[2*M:3*M]       
            double [:] Is   = rp[3*M:4*M]       
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
            X[i+M]   = aa       - gE*  E[i]
            X[i+2*M] = ce1*E[i] - gIa*Ia[i]
            X[i+3*M] = ce2*E[i] - gIs*Is[i]
        return

         
    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='None'):
        from scipy.integrate import odeint
        
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
        
        if filename=='None':
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        else:    
            from scipy.io import savemat
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
            savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE})
        return data
        



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
        readonly double alpha, beta, gIa, gIs, gE, gAA, gAS, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gE    = parameters.get('gE')                       # recovery rate of E
        self.gAA   = parameters.get('gE')                       # rate to go from A to Ia
        self.gAS   = parameters.get('gE')                       # rate to go from A to Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter 

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 5*self.M, dtype=DTYPE)           # right hand side
    
       
    cdef rhs(self, rp, tt):
        cdef: 
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs
            double gAA=self.gAA*self.alpha, gAS=self.gAS*(1-self.alpha)

            double [:] S    = rp[0*M:M]        
            double [:] E    = rp[1*M:2*M]       
            double [:] A    = rp[2*M:3*M]       
            double [:] Ia   = rp[3*M:4*M]       
            double [:] Is   = rp[4*M:5*M]       
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
            X[i+M]   =  aa      - gE       *E[i]
            X[i+2*M] = gE* E[i] - (gAA+gAS)*A[i]
            X[i+3*M] = gAA*A[i] - gIa     *Ia[i]
            X[i+4*M] = gAS*A[i] - gIs     *Is[i]
        return

         
    def simulate(self, S0, E0, A0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='None'):
        from scipy.integrate import odeint
        
        def rhs0(rp, t):
            self.rhs(rp, t)
            self.CM = contactMatrix(t)
            return self.drpdt
            
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator. 
        u = odeint(rhs0, np.concatenate((S0, E0, A0, Ia0, Is0)), time_points, mxstep=5000000)
        #elif integrator=='odespy-vode':
        #    import odespy
        #    solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #    #solver = odespy.RKF45(rhs0)
        #    #solver = odespy.RK4(rhs0)
        #    solver.set_initial_condition(self.rp0)
        #    u, time_points = solver.solve(time_points)
        
        if filename=='None':
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS}
        else:
            from scipy.io import savemat
            savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS})
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS}
        return data
        



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
        readonly double alpha, beta, gIa, gIs, gE, gAA, gAS, fsa
        readonly double tS, tE, tA, tIa, tIs
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
    
    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives 
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gE    = parameters.get('gE')                       # recovery rate of E
        self.gAA   = parameters.get('gE')                       # rate to go from A to Ia
        self.gAS   = parameters.get('gE')                       # rate to go from A to Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter 


        self.tS    = parameters.get('tS ')                       # testing rate in S 
        self.tE    = parameters.get('tE ')                       # testing rate in E 
        self.tA    = parameters.get('tA ')                       # testing rate in A 
        self.tIa   = parameters.get('tIa')                       # testing rate in Ia
        self.tIs   = parameters.get('tIs')                       # testing rate in Is

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 5*self.M, dtype=DTYPE)           # right hand side
    
       
    cdef rhs(self, rp, tt):
        cdef: 
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double tS=self.tS, tE=self.tE, tA=self.tA, tIa=self.tIa, tIs=self.tIs
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs
            double gAA=self.gAA*self.alpha, gAS=self.gAS*(1-self.alpha)

            double [:] S    = rp[0*M:M]        
            double [:] E    = rp[1*M:2*M]       
            double [:] A    = rp[2*M:3*M]       
            double [:] Ia   = rp[3*M:4*M]       
            double [:] Is   = rp[4*M:5*M]       
            double [:] Ni   = self.Ni       
            double [:] ld   = self.lld       
            double [:,:] CM = self.CM
            double [:] X    = self.drpdt        

        for i in prange(M, nogil=True):
            bb=0
            for j in prange(M):
                 bb += beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa      - tS          *S[i]
            X[i+M]   =  aa      - (gE+tE)     *E[i]
            X[i+2*M] = gE* E[i] - (gAA+gAS+tA)*A[i]
            X[i+3*M] = gAA*A[i] - (gIa+tIa   )*Ia[i]
            X[i+4*M] = gAS*A[i] - (gIs+tIs   )*Is[i]
        return

         
    def simulate(self, S0, E0, A0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint', filename='None'):
        from scipy.integrate import odeint
        
        def rhs0(rp, t):
            self.rhs(rp, t)
            self.CM = contactMatrix(t)
            return self.drpdt
            
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator. 
        u = odeint(rhs0, np.concatenate((S0, E0, A0, Ia0, Is0)), time_points, mxstep=5000000)
        #elif integrator=='odespy-vode':
        #    import odespy
        #    solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #    #solver = odespy.RKF45(rhs0)
        #    #solver = odespy.RK4(rhs0)
        #    solver.set_initial_condition(self.rp0)
        #    u, time_points = solver.solve(time_points)
        
        if filename=='None':
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS,'tS':self.tS,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        else:
            from scipy.io import savemat
            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS,'tS':self.tS,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
            savemat(filename, {'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gAA':self.gAA,'gAS':self.gAS,'tS':self.tS,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs})
        return data
        


