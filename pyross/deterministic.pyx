import  numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t




cdef class integrators:
    """
    List of all integrator used by various deterministic models listed below
    """
    
    def simulateRHS(self, rhs0, x0, Ti, Tf, Nf, integrator, maxNumSteps):

        if integrator=='odeint':
            from scipy.integrate import odeint
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            X = odeint(rhs0, x0, time_points, mxstep=maxNumSteps) 

        elif integrator=='odespy' or integrator=='odespy-vode':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=maxNumSteps)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points) 

        elif integrator=='odespy-rkf45':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RKF45(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points) 

        elif integrator=='odespy-rk4':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RK4(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points) 

        else:
            raise Exception("Error: Integration failed! \n Please set integrator='odeint' to use the scipy odeint (Deafult). \n Use integrator='odespy-vode' to use vode from odespy (github.com/rajeshrinet/odespy). \n Use integrator='odespy-rkf45' to use RKF45 from odespy (github.com/rajeshrinet/odespy). \n Use integrator='odespy-rk4' to use RK4 from odespy (github.com/rajeshrinet/odespy). \n Alternatively, write your own integrator to evolve the system in time \n")
        return X, time_points




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(integrators):
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 3*self.M, dtype=DTYPE)           # right hand side
        
        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, rateS, lmda
            double fsa=self.fsa, gIs=self.gIs
            double [:] S    = xt[0  :M]
            double [:] Ia   = xt[M  :2*M]
            double [:] Is   = xt[2*M:3*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] dx   = self.dx

            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]                                          
            dx[i]     = -rateS - FM[i]                                # rate S  -> Ia, Is
            dx[i+M]   = alpha[i]*rateS     - gIa*Ia[i] + alpha[i]    *FM[i]      # rate Ia -> R
            dx[i+2*M] = (1-alpha[i])*rateS - gIs*Is[i] + (1-alpha[i])*FM[i]      # rate Is -> R
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx 
        
        x0 = np.concatenate((S0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(integrators):
    """
    Susceptible, Infected, Recovered, Susceptible (SIRS)
    Ia: asymptomatic
    Is: symptomatic
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics

        self.ep    = parameters.get('ep')                       # fraction of recovered who is susceptible
        sa         = parameters.get('sa')                       # daily arrival of new susceptibles
        iaa        = parameters.get('iaa')                      # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            print('sa can be a number or an array of size M')

        self.iaa   = np.zeros( self.M, dtype = DTYPE)
        if np.size(iaa)==1:
            self.iaa = iaa*np.ones(M)
        elif np.size(iaa)==M:
            self.iaa = iaa
        else:
            print('iaa can be a number or an array of size M')


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, rateS, lmda
            double fsa=self.fsa,gIs=self.gIs, ep=self.ep
            double [:] S    = xt[0  :M]
            double [:] Ia   = xt[M  :2*M]
            double [:] Is   = xt[2*M:3*M]
            double [:] Ni   = xt[3*M:4*M]
            double [:,:] CM = self.CM
            double [:] sa   = self.sa
            double [:] iaa  = self.iaa
            double [:] dx   = self.dx
            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS + sa[i] + ep*(gIa*Ia[i] + gIs*Is[i])       # rate S  -> Ia, Is and also return
            dx[i+M]   = alpha[i]*rateS - gIa*Ia[i] + iaa[i]                 # rate Ia -> R
            dx[i+2*M] = (1-alpha[i])*rateS - gIs*Is[i]                          # rate Is -> R
            dx[i+3*M] = sa[i] + iaa[i]                                 # rate of Ni
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx

        x0 = np.concatenate((S0, Ia0, Is0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(integrators):
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    """

    def __init__(self, parameters, M, Ni):
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
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, gIs=self.gIs, rateS, lmda
            double fsa=self.fsa, gE=self.gE, ce1, ce2
            double [:] S     = xt[0  :  M]
            double [:] E     = xt[  M:2*M]
            double [:] Ia    = xt[2*M:3*M]
            double [:] Is    = xt[3*M:4*M]
            double [:] Ni    = self.Ni
            double [:,:] CM  = self.CM
            double [:]   FM  = self.FM
            double [:] dx    = self.dx
            double [:] alpha = self.alpha

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]                                          
            dx[i]     = -rateS - FM[i]                                # rate S  -> E
            dx[i+M]   = rateS       - gE*  E[i] + FM[i]               # rate E  -> Ia, Is
            dx[i+2*M] = ce1*E[i] - gIa*Ia[i]                       # rate Ia -> R
            dx[i+3*M] = ce2*E[i] - gIs*Is[i]                       # rate Is -> R
        return


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx

        x0 = np.concatenate((S0, E0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI5R(integrators):
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

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # recovery rate of E class
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gIh   = parameters.get('gIh')                      # recovery rate of Is
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ih
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dx    = np.zeros( 8*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

        sa         = parameters.get('sa')                       # daily arrival of new susceptibles
        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            print('sa can be a number or an array of size M')

        hh         = parameters.get('hh')                       # hospital
        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            print('hh can be a number or an array of size M')

        cc         = parameters.get('cc')                       # ICU
        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        mm         = parameters.get('mm')                       # mortality
        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1, ce2
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :2*M]
            double [:] Ia   = xt[2*M:3*M]
            double [:] Is   = xt[3*M:4*M]
            double [:] Ih   = xt[4*M:5*M]
            double [:] Ic   = xt[5*M:6*M]
            double [:] Im   = xt[6*M:7*M]
            double [:] Ni   = xt[7*M:8*M]
            double [:,:] CM = self.CM
            
            double [:] alpha= self.alpha
            double [:] sa   = self.sa       #sa is rate of additional/removal of population by birth etc
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] dx   = self.dx

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS + sa[i]                       # rate S  -> E
            dx[i+M]   = rateS  - gE*E[i]                     # rate E  -> Ia, Is
            dx[i+2*M] = ce1*E[i] - gIa*Ia[i]              # rate Ia -> R    
            dx[i+3*M] = ce2*E[i] - gIs*Is[i]              # rate Is -> R, Ih
            dx[i+4*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic
            dx[i+5*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im 
            dx[i+6*M] = gIc*mm[i]*Ic[i]                   # rate of Im 
            dx[i+7*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return


    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx
        
        x0=np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(integrators):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gI    = parameters.get('gI')                       # recovery rate of I
        self.ki    = parameters.get('k')                        # number of stages

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( (self.ki+1)*self.M, dtype=DTYPE) # right hand side


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, ki=self.ki
            double beta=self.beta, gI=self.ki*self.gI, rateS, lmda
            double [:] S    = xt[0  :M]
            double [:] I    = xt[M  :(ki+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] dx   = self.dx

        for i in range(M):
            lmda=0
            for jj in range(ki):
                for j in range(M):
                    lmda += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS - FM[i]
            dx[i+M]   = rateS - gI*I[i] + FM[i]

            for j in range(ki-1):
                dx[i+(j+2)*M]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return


    def simulate(self, S0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):
        from scipy.integrate import odeint

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        x0=np.concatenate((S0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'k':self.ki }
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(integrators):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    See: Lloyd, Theoretical Population Biology 60, 59􏰈71 (2001), doi:10.1006􏰅tpbi.2001.1525.
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # recovery rate of E
        self.gI    = parameters.get('gI')                       # recovery rate of I
        self.ki    = parameters.get('kI')                       # number of stages
        self.ke    = parameters.get('kE')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( (self.ki + self.ke + 1)*self.M, dtype=DTYPE)           # right hand side


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, ki=self.ki, ke = self.ke
            double beta=self.beta, gI=self.ki*self.gI, rateS, lmda
            double gE = self.ke * self.gE
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :(ke+1)*M]
            double [:] I    = xt[(ke+1)*M  :(ke+ki+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] dx   = self.dx

        for i in range(M):
            lmda=0
            for jj in range(ki):
                for j in range(M):
                    lmda += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS - FM[i]

            # If there is any E stage...
            if 0 != ke :
                # People removed from S are put in E[0]
                dx[i+M+0] = rateS - gE*E[i] + FM[i]

                # Propagate cases along the E stages
                for j in range(ke - 1) :
                    dx[i + M +  (j+1)*M ] = gE * E[i+j*M] - gE * E[i+(j+1)*M]

                # Transfer cases from E[-1] to I[0]
                dx[i + (ke+1)* M + 0] = gE * E[i+(ke-1)*M] - gI * I[i]

            # However, if there aren't any E stages
            else :
                # People removed from S are put in I[0]
                dx[i + (ke+1)* M + 0] = rateS + FM[i] - gI * I[i]

            # In both cases, propagate cases along the I stages.
            for j in range(ki-1):
                dx[i+(ke+1)*M + (j+1)*M ]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return


    def simulate(self, S0, E0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):
        from scipy.integrate import odeint

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        
        x0=np.concatenate((S0, E0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'k':self.ki }
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIR(integrators):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered (SEAIR)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """

    def __init__(self, parameters, M, Ni):

        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gE    = parameters.get('gE')                       # recovery rate of E
        self.gA    = parameters.get('gA')                       # rate to go from A to Ia, Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 5*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS

            double [:] S    = xt[0*M:M]
            double [:] E    = xt[1*M:2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] dx   = self.dx
            
            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS - FM[i]                          # rate S  -> E
            dx[i+M]   =  rateS      - gE*E[i] + FM[i]           # rate E  -> A
            dx[i+2*M] = gE* E[i] - gA*A[i]                   # rate A  -> Ia, Is
            dx[i+3*M] = gAA*A[i] - gIa     *Ia[i]            # rate Ia -> R
            dx[i+4*M] = gAS*A[i] - gIs     *Is[i]            # rate Is -> R
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        x0=np.concatenate((S0, E0, A0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA}
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R(integrators):
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

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # recovery rate of E class
        self.gA    = parameters.get('gA')                       # recovery rate of A class
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gIh   = parameters.get('gIh')                      # recovery rate of Is
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ih
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        sa         = parameters.get('sa')                       # daily arrival of new susceptibles
        hh         = parameters.get('hh')                       # hospital
        cc         = parameters.get('cc')                       # ICU
        mm         = parameters.get('mm')                       # mortality

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dx    = np.zeros( 9*self.M, dtype=DTYPE)           # right hand side

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            print('sa can be a number or an array of size M')

        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            print('hh can be a number or an array of size M')

        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')


    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh, gE=self.gE, gA=self.gA
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double gAA, gAS
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Ih   = xt[5*M:6*M]
            double [:] Ic   = xt[6*M:7*M]
            double [:] Im   = xt[7*M:8*M]
            double [:] Ni   = xt[8*M:9*M]
            double [:,:] CM = self.CM

            double [:] alpha= self.alpha
            double [:] sa   = self.sa       #sa is rate of additional/removal of population by birth etc
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] dx   = self.dx

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS + sa[i]                       # rate S  -> E
            dx[i+M]   = rateS  - gE*E[i]                     # rate E  -> A
            dx[i+2*M] = gE*E[i]  - gA*A[i]                # rate A  -> I                  
            dx[i+3*M] = gAA*A[i] - gIa*Ia[i]              # rate Ia -> R
            dx[i+4*M] = gAS*A[i] - gIs*Is[i]              # rate Is -> R, Ih
            dx[i+5*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic 
            dx[i+6*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im
            dx[i+7*M] = gIc*mm[i]*Ic[i]                   # rate of Im 
            dx[i+8*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx

        x0=np.concatenate((S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(integrators):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gE    = parameters.get('gE')                       # recovery rate of E
        self.gA    = parameters.get('gA')                       # rate to go from A to Ia and Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.tE    = parameters.get('tE')                       # testing rate & contact tracing of E
        self.tA    = parameters.get('tA')                       # testing rate & contact tracing of A
        self.tIa   = parameters.get('tIa')                      # testing rate & contact tracing of Ia
        self.tIs   = parameters.get('tIs')                      # testing rate & contact tracing of Is

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 6*self.M, dtype=DTYPE)           # right hand side
        
        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')



    cdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double tE=self.tE, tA=self.tA, tIa=self.tIa, tIs=self.tIs
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS 

            double [:] S    = xt[0*M:M]
            double [:] E    = xt[1*M:2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Q    = xt[5*M:6*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] dx   = self.dx
            
            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]                          
            dx[i]     = -rateS      - FM[i]        # rate S  -> E, Q
            dx[i+M]   =  rateS      - (gE+tE)     *E[i] + FM[i]        # rate E  -> A, Q
            dx[i+2*M] = gE* E[i] - (gA+tA     )*A[i]                # rate A  -> Ia, Is, Q
            dx[i+3*M] = gAA*A[i] - (gIa+tIa   )*Ia[i]               # rate Ia -> R, Q
            dx[i+4*M] = gAS*A[i] - (gIs+tIs   )*Is[i]               # rate Is -> R, Q
            dx[i+5*M] = tE*E[i]+tA*A[i]+tIa*Ia[i]+tIs*Is[i] # rate of Q
        return                                                     


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None, maxNumSteps=100000):

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
            
        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0)) 
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return data




