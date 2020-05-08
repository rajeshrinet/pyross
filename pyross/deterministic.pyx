import  numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t




cdef class IntegratorsClass:
    """
    List of all integrator used by various deterministic models listed below
    
    Methods
    -------
    simulateRHS : Performs numerical integration
    """

    def simulateRHS(self, rhs0, x0, Ti, Tf, Nf, integrator, maxNumSteps, **kwargs):
        """
        Performs numerical integration
        
        Parameters
        ----------
        rhs0 : python function(x,t)
            Input function of current state and time x, t 
            returns dx/dt
        x0 : np.array
            Initial state vector.
        Ti : float
            Start time for integrator.
        Tf : float
            End time for integrator.
        Nf : Int
            Number of time points to evaluate at.
        integrator : string, optional
            Selects which integration method to use. The default is 'odeint'.
        maxNumSteps : int, optional
            maximum number of steps the integrator is allowed to take
            to obtain a solution. The default is 100000.
        **kwargs: optional kwargs to be passed to the IntegratorsClass

        Raises
        ------
        Exception
            If integration fails.

        Returns
        -------
        X : np.array(len(t), len(x0))
            Numerical integration solution.
        time_points : np.array
            Corresponding times at which X is evaluated at.

        """
        
        if integrator=='solve_ivp':
            from scipy.integrate import solve_ivp
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            X = solve_ivp(lambda t, xt: rhs0(xt,t), [Ti,Tf], x0, t_eval=time_points, **kwargs).y.T
        
        elif integrator=='odeint':
            from scipy.integrate import odeint
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            X = odeint(rhs0, x0, time_points, mxstep=maxNumSteps, **kwargs) 

        elif integrator=='odespy' or integrator=='odespy-vode':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=maxNumSteps)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs) 

        elif integrator=='odespy-rkf45':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RKF45(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs) 

        elif integrator=='odespy-rk4':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RK4(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs) 

        else:
            raise Exception("Error: Integration method not found! \n \
                            Please set integrator='odeint' to use the scipy's odeint (Default). \n \
                            Use integrator='odespy-vode' to use vode from odespy (github.com/rajeshrinet/odespy). \n \
                            Use integrator='odespy-rkf45' to use RKF45 from odespy (github.com/rajeshrinet/odespy). \n \
                            Use integrator='odespy-rk4' to use RK4 from odespy (github.com/rajeshrinet/odespy). \n \
                            Alternatively, write your own integrator to evolve the system in time \n")
        return X, time_points




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic

    ...

    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    S
    Ia
    Is
    R
    """
    def __init__(self, parameters, M, Ni):
        self.nClass = 3
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # recovery rate of Ia
        self.gIs   = parameters['gIs']                          # recovery rate of Is
        self.fsa   = parameters['fsa']                          # fraction of asymptomatic infectives
        alpha      = parameters['alpha'] 
            
        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 3*self.M, dtype=DTYPE)           # right hand side
        
                            
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha = alpha
        else:
            raise Exception('alpha can be a number or an array of size M')


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
            dx[i]     = -rateS - FM[i]                                           # rate S  -> Ia, Is
            dx[i+M]   = alpha[i]*rateS     - gIa*Ia[i] + alpha[i]    *FM[i]      # rate Ia -> R
            dx[i+2*M] = (1-alpha[i])*rateS - gIs*Is[i] + (1-alpha[i])*FM[i]      # rate Is -> R
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint',
                 Ti=0, seedRate=None, maxNumSteps=10000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional (DEPRICATED)
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """
        
        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx 
        
        x0 = np.concatenate((S0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 
                                'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, self.M:2*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 2*self.M:3*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        #R = self.Ni - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M]
        R = self.Ni - X.sum(axis=1)
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(IntegratorsClass):
    """
    Susceptible, Infected, Recovered, Susceptible (SIRS)
    Ia: asymptomatic
    Is: symptomatic
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            ep  : float
                fraction of recovered who become susceptable again
            sa  : float, np.array (M,)
                daily arrival of new susceptables
            iaa : float, np.array (M,)
                daily arrival of new asymptomatics
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    S
    Ia
    Is
    population
    R
    """
    

    def __init__(self, parameters, M, Ni):
        self.nClass = 3
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # recovery rate of Ia
        self.gIs   = parameters['gIs']                          # recovery rate of Is
        self.fsa   = parameters['fsa']                          # the self-isolation parameter of symptomatics
        alpha      = parameters['alpha']
        self.ep    = parameters['ep']                           # fraction of recovered who is susceptible
        sa         = parameters['sa']                           # daily arrival of new susceptibles
        iaa        = parameters['iaa']                          # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            raise Exception('sa can be a number or an array of size M')

        self.iaa   = np.zeros( self.M, dtype = DTYPE)
        if np.size(iaa)==1:
            self.iaa = iaa*np.ones(M)
        elif np.size(iaa)==M:
            self.iaa = iaa
        else:
            raise Exception('iaa can be a number or an array of size M')


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
            dx[i]     = -rateS + sa[i] + ep*(gIa*Ia[i] + gIs*Is[i])    # rate S  -> Ia, Is and also return
            dx[i+M]   = alpha[i]*rateS - gIa*Ia[i] + iaa[i]            # rate Ia -> R
            dx[i+2*M] = (1-alpha[i])*rateS - gIs*Is[i]                 # rate Is -> R
            dx[i+3*M] = sa[i] + iaa[i]                                 # rate of Ni
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                     seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : str, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx

        x0 = np.concatenate((S0, Ia0, Is0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 
                        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, self.M:2*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 2*self.M:3*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        #R = self.Ni - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M]
        R = self.Ni - X.sum(axis=1)
        return R


    def population(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            population
        """
        X = data['X'] 
        ppln  = X[:,3*self.M:4*self.M]
        return ppln 






@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(IntegratorsClass):
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            gE : float
                rate of removal from exposed individuals.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    S
    E
    A
    Ia
    Is
    R
    """

    def __init__(self, parameters, M, Ni):
        self.nClass = 4
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # recovery rate of Ia
        self.gIs   = parameters['gIs']                          # recovery rate of Is
        self.gE    = parameters['gE']                           # recovery rate of E
        self.fsa   = parameters['fsa']                          # the self-isolation parameter
        alpha      = parameters['alpha'] 


        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

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
            dx[i]     = -rateS - FM[i]                             # rate S  -> E
            dx[i+M]   = rateS       - gE*  E[i] + FM[i]            # rate E  -> Ia, Is
            dx[i+2*M] = ce1*E[i] - gIa*Ia[i]                       # rate Ia -> R
            dx[i+3*M] = ce2*E[i] - gIs*Is[i]                       # rate Is -> R
        return


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', 
                        seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposed.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx

        x0 = np.concatenate((S0, E0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,
                         'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data
    

    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'E' : Exposed population as a function of timee
        """
        X = data['X'] 
        E = X[:, self.M:2*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, 2*self.M:3*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 3*self.M:4*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        R = self.Ni - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M]
        return R




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
    
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gE : float
                rate of removal from exposeds individuals.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            gIh : float
                rate of recovery for hospitalised individuals.
            gIc : float
                rate of recovery for idividuals in intensive care.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            fh  : float
                fraction by which hospitalised individuals are isolated.
            sa : float, np.array (M,)
                daily arrival of new susceptables.
                sa is rate of additional/removal of population by birth etc
            hh : float, np.array (M,)
                fraction hospitalised from Is
            cc : float, np.array (M,)
                fraction sent to intensive care from hospitalised.
            mm : float, np.array (M,)
                mortality rate in intensive care
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(8*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    simulate
    S
    E
    Ia
    Is
    Ih
    Ic
    Im
    population
    R
    """

    def __init__(self, parameters, M, Ni):
        self.nClass = 8 -1  # only 7 input classes
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # recovery rate of E class
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gIh   = parameters['gIh']                      # recovery rate of Is
        self.gIc   = parameters['gIc']                      # recovery rate of Ih
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds
        alpha      = parameters['alpha'] 
        sa         = parameters['sa']
        hh         = parameters['hh']
        cc         = parameters['cc']
        mm         = parameters['mm']               
            
        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dx    = np.zeros( 8*self.M, dtype=DTYPE)           # right hand side

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            raise Exception('sa can be a number or an array of size M')

        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            raise Exception('hh can be a number or an array of size M')

        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            raise Exception('cc can be a number or an array of size M')

        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            raise Exception('mm can be a number or an array of size M')


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
            double [:] sa   = self.sa       
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] dx   = self.dx

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            dx[i]     = -rateS + sa[i]                    # rate S  -> E
            dx[i+M]   = rateS  - gE*E[i]                  # rate E  -> Ia, Is
            dx[i+2*M] = ce1*E[i] - gIa*Ia[i]              # rate Ia -> R, Ih    
            dx[i+3*M] = ce2*E[i] - gIs*Is[i]              # rate Is -> R, Ih
            dx[i+4*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic
            dx[i+5*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im 
            dx[i+6*M] = gIc*mm[i]*Ic[i]                   # rate of Im 
            dx[i+7*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return


    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0, 
                    integrator='odeint', seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposeds.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        Ih0 : np.array
            Initial number of hospitalized infectives.
        Ic0 : np.array
            Initial number of ICU infectives.
        Im0 : np.array
            Initial number of mortality.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx
        
        x0=np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'E' : Exposed population as a function of timee
        """
        X = data['X'] 
        E = X[:, self.M:2*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, 2*self.M:3*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 3*self.M:4*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : hospitalized population as a function of timee
        """
        X  = data['X'] 
        Ih = X[:, 4*self.M:5*self.M]
        return Ih

    
    def Ic(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : ICU hospitalized population as a function of timee
        """
        X  = data['X'] 
        Ic = X[:, 5*self.M:6*self.M]
        return Ic
    

    def Im(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : mortality as a function of timee
        """
        X  = data['X'] 
        Im = X[:, 6*self.M:7*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            population
        """
        X = data['X'] 
        ppln  = X[:,7*self.M:8*self.M]
        return ppln 


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        R = self.Ni - X.sum(axis=1)
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gI : float
                rate of removal from infectives.
            kI : int
                number of stages of infection.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array((kI + 1)*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']                         # infection rate
        self.gI    = parameters['gI']                           # recovery rate of I
        self.ki    = parameters['kI']
        self.nClass = self.ki + 1

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


    def simulate(self, S0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                 seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        I0 : np.array
            Initial number of  infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        
        x0=np.concatenate((S0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'kI':self.ki }
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(IntegratorsClass):
    """
    Susceptible, Infected, Recovered (SIkR)
    method of k-stages of I
    See: Lloyd, Theoretical Population Biology 60, 59􏰈71 (2001), doi:10.1006􏰅tpbi.2001.1525.
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gI : float
                rate of removal from infected individuals.
            gE : float
                rate of removal from exposed individuals.
            ki : int
                number of stages of infectives.
            ke : int
                number of stages of exposed. 
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array((kI = kE +1)*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']                         # infection rate
        self.gE    = parameters['gE']                           # recovery rate of E
        self.gI    = parameters['gI']                           # recovery rate of I
        self.ki    = parameters['kI']                           # number of stages
        self.ke    = parameters['kE']
        self.nClass = self.ki + self.ke + 1

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


    def simulate(self, S0, E0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', 
            seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposeds.
        I0 : np.array
            Initial number of  infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        
        x0=np.concatenate((S0, E0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'k':self.ki }
        return data




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
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            gE : float
                rate of removal from exposeds individuals.
            gA : float
                rate of removal from activated individuals.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(5*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    S
    E
    A
    Ia
    Is
    R
    """

    def __init__(self, parameters, M, Ni):
        self.nClass = 5
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # recovery rate of Ia
        self.gIs   = parameters['gIs']                          # recovery rate of Is
        self.gE    = parameters['gE']                           # recovery rate of E
        self.gA    = parameters['gA']                           # rate to go from A to Ia, Is
        self.fsa   = parameters['fsa']                          # the self-isolation parameter
        alpha      = parameters['alpha']

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 5*self.M, dtype=DTYPE)           # right hand side

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

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
            dx[i+2*M] = gE* E[i] - gA*A[i]                      # rate A  -> Ia, Is
            dx[i+3*M] = gAA*A[i] - gIa     *Ia[i]               # rate Ia -> R
            dx[i+4*M] = gAS*A[i] - gIs     *Is[i]               # rate Is -> R
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0,
             integrator='odeint', seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposeds.
        A0 : np.array
            Initial number of activateds.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
        x0=np.concatenate((S0, E0, A0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,
                    'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'E' : Exposed population as a function of timee
        """
        X = data['X'] 
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'A' : Activated population as a function of timee
        """
        X = data['X'] 
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        R = self.Ni - X.sum(axis=1)
        return R








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
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            gE : float
                rate of removal from exposeds individuals.
            gA : float
                rate of removal from activated individuals.
            gIh : float
                rate of hospitalisation of infected individuals.
            gIc : float
                rate hospitalised individuals are moved to intensive care.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(9*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    S
    E
    A
    Ia
    Is
    Ih
    Ic
    Im
    population
    R
    """

    def __init__(self, parameters, M, Ni):
        self.nClass = 9 - 1#only 8 input classes
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # recovery rate of E class
        self.gA    = parameters['gA']                       # recovery rate of A class
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gIh   = parameters['gIh']                      # recovery rate of Is
        self.gIc   = parameters['gIc']                      # recovery rate of Ih
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        sa         = parameters['sa']                       # daily arrival of new susceptibles
        hh         = parameters['hh']                       # hospital
        cc         = parameters['cc']                       # ICU
        mm         = parameters['mm']                       # mortality

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
            raise Exception('alpha can be a number or an array of size M')

        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            raise Exception('sa can be a number or an array of size M')

        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            raise Exception('hh can be a number or an array of size M')

        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            raise Exception('cc can be a number or an array of size M')

        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            raise Exception('mm can be a number or an array of size M')


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
            dx[i]     = -rateS + sa[i]                    # rate S  -> E
            dx[i+M]   = rateS  - gE*E[i]                  # rate E  -> A
            dx[i+2*M] = gE*E[i]  - gA*A[i]                # rate A  -> I                  
            dx[i+3*M] = gAA*A[i] - gIa*Ia[i]              # rate Ia -> R
            dx[i+4*M] = gAS*A[i] - gIs*Is[i]              # rate Is -> R, Ih
            dx[i+5*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic 
            dx[i+6*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im
            dx[i+7*M] = gIc*mm[i]*Ic[i]                   # rate of Im 
            dx[i+8*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0,
                 integrator='odeint', seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposeds.
        A0 : np.array
            Initial number of activateds.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        Ih0 : np.array
            Initial number of hospitalized infectives.
        Ic0 : np.array
            Initial number of ICU infectives.
        Im0 : np.array
            Initial number of mortality.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dx

        x0=np.concatenate((S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'E' : Exposed population as a function of timee
        """
        X = data['X'] 
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'A' : Activated population as a function of timee
        """
        X = data['X'] 
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : hospitalized population as a function of timee
        """
        X  = data['X'] 
        Ih = X[:, 5*self.M:6*self.M]
        return Ih

    
    def Ic(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : ICU hospitalized population as a function of timee
        """
        X  = data['X'] 
        Ic = X[:, 6*self.M:7*self.M]
        return Ic
    

    def Im(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ic' : mortality as a function of timee
        """
        X  = data['X'] 
        Im = X[:, 7*self.M:8*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            population
        """
        X = data['X'] 
        ppln  = X[:, 8*self.M:9*self.M]
        return ppln 


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        R = self.Ni - X.sum(axis=1)
        return R




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

    Attributes
    ----------
    parameters: dict
        Contains the following keys:   
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa : float
                rate of removal from asymptomatic individuals.
            gIs : float
                rate of removal from symptomatic individuals.
            gE : float
                rate of removal from exposed individuals.
            gA : float
                rate of removal from activated individuals.
            fsa : float
                fraction by which symptomatic individuals self isolate.
            tE  : float
                testing rate and contact tracing of exposeds
            tA  : float
                testing rate and contact tracing of activateds
            tIa : float
                testing rate and contact tracing of asymptomatics
            tIs : float
                testing rate and contact tracing of symptomatics
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(6*M, )
        Initial number in each compartment and class    

    Methods
    -------
    simulate 
    S
    E
    A
    Ia
    Is
    R
    Q
    """

    def __init__(self, parameters, M, Ni):
        self.nClass = 6
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gE    = parameters['gE']                       # recovery rate of E
        self.gA    = parameters['gA']                       # rate to go from A to Ia and Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.tE    = parameters['tE']                       # testing rate & contact tracing of E
        self.tA    = parameters['tA']                       # testing rate & contact tracing of A
        self.tIa   = parameters['tIa']                      # testing rate & contact tracing of Ia
        self.tIs   = parameters['tIs']                      # testing rate & contact tracing of Is
        alpha      = parameters['alpha'] 

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dx    = np.zeros( 6*self.M, dtype=DTYPE)           # right hand side
        
        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')



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
            dx[i]     = -rateS      - FM[i]                         # rate S  -> E, Q
            dx[i+M]   =  rateS      - (gE+tE)     *E[i] + FM[i]     # rate E  -> A, Q
            dx[i+2*M] = gE* E[i] - (gA+tA     )*A[i]                # rate A  -> Ia, Is, Q
            dx[i+3*M] = gAA*A[i] - (gIa+tIa   )*Ia[i]               # rate Ia -> R, Q
            dx[i+4*M] = gAS*A[i] - (gIs+tIs   )*Is[i]               # rate Is -> R, Q
            dx[i+5*M] = tE*E[i]+tA*A[i]+tIa*Ia[i]+tIs*Is[i]         # rate of Q
        return                                                     


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf, Ti=0,
                     integrator='odeint', seedRate=None, maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0 : np.array
            Initial number of susceptables.
        E0 : np.array
            Initial number of exposeds.
        A0 : np.array
            Initial number of activateds.
        Ia0 : np.array
            Initial number of asymptomatic infectives.
        Is0 : np.array
            Initial number of symptomatic infectives.
        Q0 : np.array
            Initial number of quarantineds.
        contactMatrix : python function(t)
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
        Tf : float
            Final time of integrator
        Nf : Int
            Number of time points to evaluate.
        Ti : float, optional
            Start time of integrator. The default is 0.
        integrator : TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        maxNumSteps : int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(xt, t)
            return self.dx
            
        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0)) 
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha,'beta':self.beta,'gIa':self.gIa,
                    'gIs':self.gIs,'gE':self.gE,'gA':self.gA,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population as a function of timee
        """
        X = data['X'] 
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'E' : Exposed population as a function of timee
        """
        X = data['X'] 
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'A' : Activated population as a function of timee
        """
        X = data['X'] 
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Ia' : Asymptomatics population as a function of timee
        """
        X  = data['X'] 
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def Q(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Q' : Quarantined population as a function of timee
        """
        X  = data['X'] 
        Is = X[:, 5*self.M:6*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'R' : Recovered population as a function of timee
        """
        X = data['X'] 
        R = self.Ni - X.sum(axis=1)
        return R






