import  numpy as np
cimport numpy as np
cimport cython
import pyross.utils
import warnings


DTYPE   = np.float64
from libc.stdlib cimport malloc, free


                                                                      

cdef class CommonMethods:
    """
    Parent class used for all classes listed below. 
    It includes:
    a) Integrators used by various deterministic models listed below.
    b) Method to get time series of S, etc by passing a dict of data.
    c) Method to set the contactMatrix array, CM
    """

    def simulator(self, x0, contactMatrix, Tf, Nf, integrator='odeint', 
        Ti=0, maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        
        ...

        Parameters
        ----------
        x0: np.array
            Initial state vector (number of compartment values).
            An array of size M*(model_dimension-1),
            where x0[i+j*M] should be the initial
            value of model class i of age group j.
            The removed R class must be left out.
            If Ni is dynamical, then the last M points store Ni.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.

        """

        def dxdtEval(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        if integrator=='odeint':
            from scipy.integrate import odeint
            time_points=np.linspace(Ti, Tf, Nf)   
            X = odeint(dxdtEval, x0, time_points, mxstep=maxNumSteps, **kwargs)

        elif integrator=='solve_ivp':
            from scipy.integrate import solve_ivp
            time_points=np.linspace(Ti, Tf, Nf)                                                          
            X = solve_ivp(lambda t, xt: dxdtEval(xt,t), [Ti,Tf], x0, 
                         t_eval=time_points, **kwargs).y.T

        elif integrator=='odespy' or integrator=='odespy-vode':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf)                                                          
            solver = odespy.Vode(dxdtEval, method = 'bdf', 
                    atol=1E-7, rtol=1E-6, order=5, nsteps=maxNumSteps)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        elif integrator=='odespy-rkf45':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf)                                                          
            solver = odespy.RKF45(dxdtEval)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        elif integrator=='odespy-rk4':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf)
            solver = odespy.RK4(dxdtEval)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        else:
            raise Exception("Error: Integration method not found! \n \
                            Please set integrator='odeint' to use \n \
                            the scipy.integrate's odeint (Default)\n \
                            Use integrator='odespy-vode' to use vode \
                            from odespy (github.com/rajeshrinet/odespy).\n \
                            Use integrator='odespy-rkf45' to use RKF45  \
                            from odespy (github.com/rajeshrinet/odespy).\n \
                            Use integrator='odespy-rk4' to use RK4 from \
                            odespy (github.com/rajeshrinet/odespy).     \n \
                            Alternatively, write your own integrator to \
                            evolve the system in time and store the data.\n")

        data     = {'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M}
        data_out = data.copy()
        data_out.update(self.paramList)
        return data_out

    
    cpdef set_contactMatrix(self, double t, contactMatrix):
        self.CM=contactMatrix(t)


    def S(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             S: Susceptible population time series
        """
        
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data, Ei=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             E: Exposed population time series
        """
        
        if None != Ei:
            X = data['X']  
            E = X[:, Ei[0]*self.M:Ei[1]*self.M]
        else:
            X = data['X']  
            Ei=self.readData['Ei']
            E = X[:, Ei[0]*self.M:Ei[1]*self.M]
        return E


    def A(self,  data, Ai=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             A: Activated population time series
        """
        
        if None != Ai:
            X = data['X']  
            A = X[:, Ai[0]*self.M:Ai[1]*self.M]
        else:
            X = data['X']  
            Ai=self.readData['Ai']
            A = X[:, Ai[0]*self.M:Ai[1]*self.M]
        return A


    def I(self,  data, Ii=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ia : Asymptomatics population time series
        """

        if None != Ii:
            X  = data['X']
            Ii=self.readData['Ii']
            I = X[:, Ii[0]*self.M:Ii[1]*self.M]
        else:
            X  = data['X']  
            Ii=self.readData['Ii']
            I = X[:, Ii[0]*self.M:Ii[1]*self.M]
        return I


    def Ia(self,  data, Iai=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ia : Asymptomatics population time series
        """
        
        if None != Iai:
            X  = data['X']
            Ia = X[:, Iai[0]*self.M:Iai[1]*self.M]
        else:
            X  = data['X'] 
            Iai=self.readData['Iai']
            Ia = X[:, Iai[0]*self.M:Iai[1]*self.M]
        return Ia


    def Is(self,  data, Isi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Is : symptomatics population time series
        """
        if None != Isi:
            X  = data['X']
            Is = X[:, Isi[0]*self.M:Isi[1]*self.M]
        else:
            X  = data['X']  
            Isi=self.readData['Isi']
            Is = X[:, Isi[0]*self.M:Isi[1]*self.M]
        return Is




    def R(self,  data, Rind=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             R: Removed population time series
        """
        if None != Rind:
            X = data['X']  
            R = self.population
        else:
            X = data['X']  
            Rind=self.readData['Rind']
            R = self.population
        for i in range(Rind):
            R  = R - X[:, i*self.M:(i+1)*self.M] 
        return R
   

    def Sx(self,  data, Sxi):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
            Generic compartment Sx 
        """
        X  = data['X']
        Im = X[:, Sxi[0]*self.M:Sxi[1]*self.M ]
        Sx = X[:, Sxi[0]*self.M:Sxi[1]*self.M ]
        return Sx






@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(CommonMethods):
    """
    Susceptible, Infected, Removed (SIR)
    
    * Ia: asymptomatic
    * Is: symptomatic 
    .. math::
        \dot{S_{i}}=-\lambda_{i}(t)S_{i}
    .. math::
        \dot{I}_{i}^{a} = \\alpha_{i}\lambda_{i}(t)S_{i}-\gamma_{I^{a}}I_{i}^{a},
    .. math::
        \dot{I}_{i}^{s}= \\bar{\\alpha_{i}}\lambda_{i}(t)S_{i}-\gamma_{I^{s}}I_{i}^{s},
    .. math::
        \dot{R}_{i}=\gamma_{I^{a}}I_{i}^{a}+\gamma_{I^{s}}I_{i}^{s}.
    .. math::
        \lambda_{i}(t)=\\beta\sum_{j=1}^{M}\left(C_{ij}^{a}(t)\\frac{I_{j}^{a}}{N_{j}}+C_{ij}^{s}(t)
        \\frac{I_{j}^{s}}{N_{j}}\\right),\quad i,j=1,\ldots M

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            Fraction of infected who are asymptomatic.
        beta: float, np.array (M,)
            Rate of spread of infection.
        gIa: float, np.array (M,)
            Rate of removal from asymptomatic individuals.
        gIs: float, np.array (M,)
            Rate of removal from symptomatic individuals.
        fsa: float, np.array (M,)
            Fraction by which symptomatic individuals do not self-isolate.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    
    Examples
    --------
    An example of the SIR class  

    >>> M = 1                   # SIR model with no age structure
    >>> Ni = 1000*np.ones(M)    # only one age group
    >>> N = np.sum(Ni)          # total population 
    >>> 
    >>> beta  = 0.2             # Infection rate
    >>> gIa   = 0.1             # Removal rate of asymptomatic infectives
    >>> gIs   = 0.1             # Removal rate of symptomatic infectives
    >>> alpha = 0               # Fraction of asymptomatic infectives
    >>> fsa   = 1               # self-isolation of symtomatic infectives
    >>> 
    >>> Ia0 = np.array([0])     # Intial asymptomatic infectives
    >>> Is0 = np.array([1])     # Initial symptomatic
    >>> R0  = np.array([0])     # No removed individuals initially
    >>> S0  = N-(Ia0+Is0+R0)    # S + Ia + Is + R = N
    >>> 
    >>> # there is no contact structure
    >>> def contactMatrix(t):
    >>>     return np.identity(M)
    >>> 
    >>> # duration of simulation and data file
    >>> Tf = 160;  Nt=160;
    >>> 
    >>> # instantiate model
    >>> parameters = {'alpha':alpha, 'beta':beta, 'gIa':gIa, 'gIs':gIs,'fsa':fsa}
    >>> model = pyross.deterministic.SIR(parameters, M, Ni)
    >>> 
    >>> # simulate model using two possible ways
    >>> data1 = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)                    
    >>> data2 = model.simulator(np.concatenate((S0, Ia0, Is0)), contactMatrix, Tf, Nt) 
    """


    def __init__(self, parameters, M, Ni):
        self.nClass= 3
        self.beta  = pyross.utils.age_dep_rates(parameters['beta'],  M, 'beta')
        self.gIa   = pyross.utils.age_dep_rates(parameters['gIa'],   M, 'gIa')
        self.gIs   = pyross.utils.age_dep_rates(parameters['gIs'],   M, 'gIs')
        self.fsa   = pyross.utils.age_dep_rates(parameters['fsa'],   M, 'fsa')
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], M, 'alpha')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  #Number of individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( 3*self.M, dtype=DTYPE)           # Right hand side

        self.paramList = parameters

        self.readData = {'Iai':[1,2], 'Isi':[2,3], 'Rind':3}
        self.population = self.Ni


    cpdef rhs(self, xt, tt):

        cdef:
            int N=self.N, M=self.M, i, j
            double [:] beta=self.beta, gIa=self.gIa,
            double rateS, lmda
            double [:] fsa=self.fsa, gIs=self.gIs
            double [:] S    = xt[0  :M]
            double [:] Ia   = xt[M  :2*M]
            double [:] Is   = xt[2*M:3*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt
            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0
            for j in range(M):
                 lmda += CM[i,j]*(Ia[j]+fsa[j]*Is[j])/Ni[j]
            rateS = lmda*S[i]*beta[i]
            #
            dxdt[i]     = -rateS                              # \dot S
            dxdt[i+M]   = alpha[i]*rateS     - gIa[i]*Ia[i]   # \dot Ia
            dxdt[i+2*M] = (1-alpha[i])*rateS - gIs[i]*Is[i]   # \dot Is
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint',
                 Ti=0, maxNumSteps=10000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.

        """

        x0 = np.concatenate((S0, Ia0, Is0))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(CommonMethods):
    """
    Susceptible, Infected, Removed (SIkR). Method of k-stages of I

    .. math::
        \dot{S_{i}}=-\lambda_{i}(t)S_{i},
    .. math::
        \dot{I}_{i}^{1}=k_{E}\gamma_{E}E_{i}^{k}-k_{I}\gamma_{I}I_{i}^{1},
    .. math::
        \dot{I}_{i}^{k}=k_{I}\gamma_{I}I_{i}^{(k-1)}-k_{I}\gamma_{I}I_{i}^{k}, 
    .. math::
        \dot{R}_{i}=k_{I}\gamma_{I}I_{i}^{k}.

    .. math::
        \lambda_{i}(t)=\\beta\sum_{j=1}^{M}\sum_{n=1}^{k}C_{ij}(t)\\frac{I_{j}^{n}}{N_{j}},
    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        beta: float
            Rate of spread of infection.
        gI: float
            Rate of removal from infectives.
        kI: int
            number of stages of infection.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']              # Infection rate
        self.gI    = parameters['gI']                # Removal rate of I
        self.kI    = parameters['kI']
        self.nClass= self.kI + 1

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( (self.kI+1)*self.M, dtype=DTYPE) # Right hand side

        self.paramList = parameters
        self.readData = {'Ii':[1,self.kI+1], 'Rind':self.kI+1}
        self.population = self.Ni


    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, kI=self.kI
            double beta=self.beta, gI=self.kI*self.gI, rateS, lmda
            double [:] S    = xt[0  :M]
            double [:] I    = xt[M  :(kI+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0
            for jj in range(kI):
                for j in range(M):
                    lmda += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS
            dxdt[i+M]   = rateS - gI*I[i]

            for j in range(kI-1):
                dxdt[i+(j+2)*M]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return


    def simulate(self, S0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                 maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        I0: np.array
            Initial number of  infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.

        """

        x0=np.concatenate((S0, I0))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(CommonMethods):
    """
    Susceptible, Exposed, Infected, Removed (SEIR)

    * Ia: asymptomatic
    * Is: symptomatic 
     
    .. math::
        \dot{S_{i}}=-\lambda_{i}(t)S_{i}
    .. math::
        \dot{E}_{i}=\lambda_{i}(t)S_{i}-\gamma_{E}E_{i}
    .. math::
        \dot{I}_{i}^{a} = \\alpha_{i}\gamma_{E}^{i}E_{i}-\gamma_{I^{a}}I_{i}^{a},
    .. math::
        \dot{I}_{i}^{s}= \\bar{\\alpha_{i}}\gamma_{E}^{i}E_{i}-\gamma_{I^{s}}I_{i}^{s},
    .. math::
        \dot{R}_{i}=\gamma_{I^{a}}I_{i}^{a}+\gamma_{I^{s}}I_{i}^{s}.
    .. math::
        \lambda_{i}(t)=\\beta\sum_{j=1}^{M}\left(C_{ij}^{a}(t)\\frac{I_{j}^{a}}{N_{j}}+C_{ij}^{s}(t)
        \\frac{I_{j}^{s}}{N_{j}}\\right),\quad i,j=1,\ldots M


    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            Fraction of infected who are asymptomatic.
        beta: float, np.array (M,)
            Rate of spread of infection.
        gE: float, np.array (M,)
            Rate of removal from exposed individuals.
        gIa: float, np.array (M,)
            Rate of removal from asymptomatic individuals.
        gIs: float, np.array (M,)
            Rate of removal from symptomatic individuals.
        fsa: float, np.array (M,)
            Fraction by which symptomatic individuals do not self-isolate.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 4
        self.beta = pyross.utils.age_dep_rates(parameters['beta'], M, 'beta')
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], M, 'gE')
        self.gIa = pyross.utils.age_dep_rates(parameters['gIa'], M, 'gIa')
        self.gIs = pyross.utils.age_dep_rates(parameters['gIs'], M, 'gIs')
        self.fsa = pyross.utils.age_dep_rates(parameters['fsa'], M, 'fsa')
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], M, 'alpha')

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( 4*self.M, dtype=DTYPE)           # Right hand side

        self.readData = {'Ei':[1,2], 'Iai':[2,3], 'Isi':[3,4], 'Rind':4}
        self.population = self.Ni

    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double [:] beta=self.beta, gIa=self.gIa, gIs=self.gIs,
            double rateS, lmda
            double [:] fsa=self.fsa, gE=self.gE,
            double ce1, ce2
            double [:] S     = xt[0  :  M]
            double [:] E     = xt[  M:2*M]
            double [:] Ia    = xt[2*M:3*M]
            double [:] Is    = xt[3*M:4*M]
            double [:] Ni    = self.Ni
            double [:,:] CM  = self.CM
            double [:] dxdt  = self.dxdt
            double [:] alpha = self.alpha

        for i in range(M):
            lmda=0;   ce1=gE[i]*alpha[i];  ce2=gE[i]-ce1
            for j in range(M):
                 lmda += beta[i]*CM[i,j]*(Ia[j]+fsa[j]*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS                          # \dot S
            dxdt[i+M]   = rateS       - gE[i]*  E[i]      # \dot E
            dxdt[i+2*M] = ce1*E[i] - gIa[i]*Ia[i]         # \dot Ia
            dxdt[i+3*M] = ce2*E[i] - gIs[i]*Is[i]         # \dot Is
        return


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                        maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposed.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.

        """

        x0 = np.concatenate((S0, E0, Ia0, Is0))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data







@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(CommonMethods):
    """
    Susceptible, Exposed, Infected, Removed (SEkIkR). 
    Method of k-stages of E and I
    
    .. math::
        \dot{S_{i}}=-\lambda_{i}(t)S_{i},
    .. math::
        \dot{E}_{i}^{1}=\lambda_{i}(t)S_{i}-k_{E}\gamma_{E}E_{i}^{1}
    .. math::
        \dot{E}_{i}^{k}=k_{E}\gamma_{E}E_{i}^{k-1}-k_{E}\gamma_{E}E_{i}^{k}
    .. math::
        \dot{I}_{i}^{1}=k_{E}\gamma_{E}E_{i}^{k}-k_{I}\gamma_{I}I_{i}^{1},
    .. math::
        \dot{I}_{i}^{k}=k_{I}\gamma_{I}I_{i}^{(k-1)}-k_{I}\gamma_{I}I_{i}^{k}, 
    .. math::
        \dot{R}_{i}=k_{I}\gamma_{I}I_{i}^{k}.
    .. math::
        \lambda_{i}(t)=\\beta\sum_{j=1}^{M}\sum_{n=1}^{k}C_{ij}(t)\\frac{I_{j}^{n}}{N_{j}},
    ...
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
        beta: float
            Rate of spread of infection.
        gI: float
            Rate of removal from infected individuals.
        gE: float
            Rate of removal from exposed individuals.
        kI: int
            number of stages of infectives.
        kE: int
            number of stages of exposed.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']            # Infection rate
        self.gE    = parameters['gE']              # Removal rate of E
        self.gI    = parameters['gI']              # Removal rate of I
        self.kI    = parameters['kI']              # number of stages
        self.kE    = parameters['kE']
        self.nClass= self.kI + self.kE + 1

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( (self.kI + self.kE + 1)*self.M, dtype=DTYPE)  #Right hand side

        if self.kE==0:
            raise Exception('number of E stages should be greater than zero, kE>0')
        elif self.kI==0:
            raise Exception('number of I stages should be greater than zero, kI>0')
        
        self.readData = {'Ei':[1,self.kE+1], 'Ii':[self.kE+1,self.kE+self.kI+1], 
                        'Rind':self.kE+self.kI+1}
        self.population = self.Ni



    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, kI=self.kI, kE = self.kE
            double beta=self.beta, gI=self.kI*self.gI, rateS, lmda
            double gE = self.kE * self.gE
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :(kE+1)*M]
            double [:] I    = xt[(kE+1)*M  :(kE+kI+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0
            for jj in range(kI):
                for j in range(M):
                    lmda += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS

            #Exposed class
            dxdt[i+M+0] = rateS - gE*E[i]
            for j in range(kE-1) :
                dxdt[i+M+(j+1)*M] = gE * E[i+j*M] - gE*E[i+(j+1)*M]

            #Infected
            dxdt[i + (kE+1)*M + 0] = gE*E[i+(kE-1)*M] - gI*I[i]
            for j in range(kI-1):
                dxdt[i+(kE+1)*M + (j+1)*M ]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return


    def simulate(self, S0, E0, I0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
            maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        I0: np.array
            Initial number of  infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator
        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.
        """

        x0=np.concatenate((S0, E0, I0))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data






@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(CommonMethods):
    """
    Susceptible, Exposed, Asymptomatic and infected, 
    Infected, Removed, Quarantined (SEAIRQ)

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed
    * A: asymptomatic and infectious
    * Q: quarantined 
 
    .. math::
        \dot{S_{i}}=-\lambda_{i}(t)S_{i}
    .. math::
        \dot{E}_{i}=\lambda_{i}(t)S_{i}-(\gamma_{E}+\\tau_{E})A_{i}
    .. math::
        \dot{A}_{i}=\gamma_{E}E_{i}-(\gamma_{A}+\\tau_{A})A_{i}
    .. math::
        \dot{I}_{i}^{a}=\\alpha_{i}\gamma_{A}A_{i}-(\gamma_{I^{a}}+\\tau_{I^a})I_{i}^{a},
    .. math::
        \dot{I}_{i}^{s}=\\bar{\\alpha_{i}}\gamma_{A}A_{i}-(\gamma_{I^{s}}+\\tau_{I^a})I_{i}^{s},
    .. math::
        \dot{R}_{i}=\gamma_{I^{a}}I_{i}^{a}+\gamma_{I^{s}}I_{i}^{s}.
    .. math::
        \dot{Q}_{i}=\\tau_{S}S_{i}+\\tau_{E}E_{i}+\\tau_{A}A_{i}+\\tau_{I^{s}}I_{i}^{s}+\\tau_{I^{a}}I_{i}^{a}

    .. math::
        \lambda_{i}(t)=\\beta\sum_{j=1}^{M}\left(C_{ij}^{a}\\frac{I_{j}^{a}}{N_{j}}+C_{ij}^{a}
        \\frac{A_{j}}{N_{j}}+C_{ij}^{s}\\frac{I_{j}^{s}}{N_{j}}\\right),


    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            Fraction of infected who are asymptomatic.
        beta: float, np.array (M,)
            Rate of spread of infection.
        gIa: float, np.array (M,)
            Rate of removal from asymptomatic individuals.
        gIs: float, np.array (M,)
            Rate of removal from symptomatic individuals.
        gE: float, np.array (M,)
            Rate of removal from exposed individuals.
        gA: float, np.array (M,)
            Rate of removal from activated individuals.
        fsa: float, np.array (M,)
        tE: float, np.array (M,)
            testing rate and contact tracing of exposeds
        tA: float, np.array (M,)
            testing rate and contact tracing of activateds
        tIa: float, np.array (M,)
            testing rate and contact tracing of asymptomatics
        tIs: float, np.array (M,)
            testing rate and contact tracing of symptomatics
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 6
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], M, 'alpha')
        self.beta = pyross.utils.age_dep_rates(parameters['beta'], M, 'beta')
        self.gIa = pyross.utils.age_dep_rates(parameters['gIa'], M, 'gIa')
        self.gIs = pyross.utils.age_dep_rates(parameters['gIs'], M, 'gIs')
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], M, 'gE')
        self.gA = pyross.utils.age_dep_rates(parameters['gA'], M, 'gA')
        self.fsa = pyross.utils.age_dep_rates(parameters['fsa'], M, 'fsa')

        self.tE = pyross.utils.age_dep_rates(parameters['tE'], M, 'tE')
        self.tA = pyross.utils.age_dep_rates(parameters['tA'], M, 'tA')
        self.tIa = pyross.utils.age_dep_rates(parameters['tIa'], M, 'tIa')
        self.tIs = pyross.utils.age_dep_rates(parameters['tIs'], M, 'tIs')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  
        self.Ni    = Ni

        self.paramList = parameters

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( 6*self.M, dtype=DTYPE)           # Right hand side

        self.readData = {'Ei':[1,2], 'Ai':[2,3], 'Iai':[3,4], 'Isi':[4,5], 
                        'Qi':[5,6], 'Rind':6}
        self.population = self.Ni

    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double [:] beta=self.beta
            double rateS, lmda
            double [:] tE=self.tE, tA=self.tA, tIa=self.tIa, tIs=self.tIs
            double [:] fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS

            double [:] S    = xt[0*M:M]
            double [:] E    = xt[1*M:2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Q    = xt[5*M:6*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt

            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0;   gAA=gA[i]*alpha[i];  gAS=gA[i]-gAA
            for j in range(M):
                 lmda += beta[i]*CM[i,j]*(A[j]+Ia[j]+fsa[j]*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS                                          # \dot S
            dxdt[i+M]   = rateS    - (gE[i]+tE[i])     *E[i]              # \dot E
            dxdt[i+2*M] = gE[i]* E[i] - (gA[i]+tA[i])*A[i]                # \dot A
            dxdt[i+3*M] = gAA*A[i] - (gIa[i]+tIa[i])*Ia[i]                # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - (gIs[i]+tIs[i])*Is[i]                # \dot Is
            dxdt[i+5*M] = tE[i]*E[i]+tA[i]*A[i]+tIa[i]*Ia[i]+tIs[i]*Is[i] # \dot Q
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        A0: np.array
            Initial number of activateds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Q0: np.array
            Initial number of quarantineds.
        contactMatrix: python function(t)
            The social contact matrix C_{ij} denotes the
            average number of contacts made per day by an
            individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
            contains the following keys:

            *  X : output path from integrator
            *  t : time points evaluated at,
            * 'param': input param to integrator.
        """

        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ_testing(CommonMethods):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed
    * A: asymptomatic and infectious
    * Q: quarantined 

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            Fraction of infected who are asymptomatic.
        beta: float, np.array (M,)
            Rate of spread of infection.
        gIa: float, np.array (M,)
            Rate of removal from asymptomatic individuals.
        gIs: float, np.array (M,)
            Rate of removal from symptomatic individuals.
        gE: float, np.array (M,)
            Rate of removal from exposed individuals.
        gA: float, np.array (M,)
            Rate of removal from activated individuals.
        fsa: float, np.array (M,)
            Fraction by which symptomatic individuals do not self-isolate.
        ars: float, np.array (M,)
            Fraction of population admissible for random and symptomatic tests
        kapE: float, np.array (M,)
            Fraction of positive tests for exposed individuals
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 6
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], M, 'alpha')
        self.beta = pyross.utils.age_dep_rates(parameters['beta'], M, 'beta')
        self.gIa = pyross.utils.age_dep_rates(parameters['gIa'], M, 'gIa')
        self.gIs = pyross.utils.age_dep_rates(parameters['gIs'], M, 'gIs')
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], M, 'gE')
        self.gA = pyross.utils.age_dep_rates(parameters['gA'], M, 'gA')
        self.fsa = pyross.utils.age_dep_rates(parameters['fsa'], M, 'fsa')

        self.ars = pyross.utils.age_dep_rates(parameters['ars'], M, 'ars')
        self.kapE = pyross.utils.age_dep_rates(parameters['kapE'], M, 'kapE')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)  
        self.Ni    = Ni

        self.paramList = parameters

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( 6*self.M, dtype=DTYPE)           # Right hand side

        self.readData = {'Ei':[1,2], 'Ai':[2,3], 'Iai':[3,4], 'Isi':[4,5], 
                        'Qi':[5,6], 'Rind':6}
        self.population = self.Ni
        self.testRate = None
    
    cpdef set_testRate(self, testRate):
        self.testRate = testRate


    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double [:] beta=self.beta
            double rateS, lmda, t0, tE, tA, tIa, tIs
            double [:] ars=self.ars, kapE=self.kapE
            double [:] fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS


            double [:] S    = xt[0*M:M]
            double [:] E    = xt[1*M:2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Q    = xt[5*M:6*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt

            double [:] alpha= self.alpha

            double [:] TR = self.testRate(tt)
            
        for i in range(M):
            lmda=0;   gAA=gA[i]*alpha[i];  gAS=gA[i]-gAA
            for j in range(M):
                 lmda += beta[i]*CM[i,j]*(A[j]+Ia[j]+fsa[j]*Is[j])/Ni[j]
            rateS = lmda*S[i]


            t0 = 1./(ars[i]*(Ni[i]-Q[i]-Is[i])+Is[i])
            tE = TR[i]*ars[i]*kapE[i]*t0
            tA= TR[i]*ars[i]*t0
            tIa = TR[i]*ars[i]*t0
            tIs = TR[i]*t0


            dxdt[i]     = -rateS                                      # \dot S
            dxdt[i+M]   =  rateS   - (gE[i]+tE)     *E[i]                # \dot E
            dxdt[i+2*M] = gE[i]* E[i] - (gA[i]+tA     )*A[i]                # \dot A
            dxdt[i+3*M] = gAA*A[i] - (gIa[i]+tIa   )*Ia[i]               # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - (gIs[i]+tIs   )*Is[i]               # \dot Is
            dxdt[i+5*M] = tE*E[i]+tA*A[i]+tIa*Ia[i]+tIs*Is[i]         # \dot Q

        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, testRate, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        A0: np.array
            Initial number of activateds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Q0: np.array
            Initial number of quarantineds.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        testRate: python function(t)
             Rate at which symptomatic and random individuals get tested
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
            contains the following keys:

            *  X : output path from integrator
            *  t : time points evaluated at,
            * 'param': input param to integrator.
        """

        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0))
        self.testRate = testRate
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(CommonMethods):
    """
    Susceptible, Infected, Removed, Susceptible (SIRS). 
    
    * Ia: asymptomatic
    * Is: symptomatic

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            Fraction of infected who are asymptomatic.
        beta: float
            Rate of spread of infection.
        gIa: float
            Rate of removal from asymptomatic individuals.
        gIs: float
            Rate of removal from symptomatic individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        ep  : float
            Fraction of removed who become susceptable again
        sa  : float, np.array (M,)
            daily arrival of new susceptables
        iaa: float, np.array (M,)
            daily arrival of new asymptomatics
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """


    def __init__(self, parameters, M, Ni):
        self.nClass= 3
        self.beta  = parameters['beta']               # Infection rate
        self.gIa   = parameters['gIa']                # Removal rate of Ia
        self.gIs   = parameters['gIs']                # Removal rate of Is
        self.fsa   = parameters['fsa']                # Self-isolation of symptomatics
        alpha      = parameters['alpha']
        self.ep    = parameters['ep']                 # Fraction of removed who is susceptible
        sa         = parameters['sa']                 # Daily arrival of new susceptibles
        iaa        = parameters['iaa']                # Daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)   
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dxdt  = np.zeros( 4*self.M, dtype=DTYPE)           # Right hand side

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
        
        self.paramList = parameters


    cpdef rhs(self, xt, tt):
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
            double [:] dxdt = self.dxdt
            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS + sa[i] + ep*(gIa*Ia[i] + gIs*Is[i])    # \dot S
            dxdt[i+M]   = alpha[i]*rateS - gIa*Ia[i] + iaa[i]            # \dot Ia
            dxdt[i+2*M] = (1-alpha[i])*rateS - gIs*Is[i]                 # \dot Is
            dxdt[i+3*M] = sa[i] + iaa[i]                                 # \dot Ni
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                     maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: str, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.

        """

        x0 = np.concatenate((S0, Ia0, Is0, self.Ni))
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ia : Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, self.M:2*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Is : symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 2*self.M:3*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R =  X[:,3*self.M:4*self.M] - X[:, 0:self.M] \
                - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M]
        return R


    def population(self,  data):
        """
        Parameters
        ----------
        data: Data dict

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
cdef class Model(CommonMethods):
    """
    Generic user-defined epidemic model.

    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        Contains the values for the parameters given in the model specification.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec. 
        Default: Identical mapping of the dictionary at all times. 

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "constant"  : [ ["k"] ],
                "infection" : [ ["I", "S", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "S", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'k': 1,
        }
    """

    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None, constant_CM=0):

        self.N = DTYPE(np.sum(Ni))
        self.M = DTYPE(M)
        self.Ni = np.array(Ni, dtype=DTYPE)
        self.constant_CM=constant_CM
        
        self.time_dep_param_mapping = time_dep_param_mapping
        if self.time_dep_param_mapping is not None:
            self.param_dict = parameters.copy()
            parameters = self.time_dep_param_mapping(parameters, 0)

        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.finres_terms = res[5]
        self.resource_list = res[6]
        if self.time_dep_param_mapping is None:
            self.update_model_parameters(parameters)
        else:
            self.update_time_dep_model_parameters(0)
        self.CM = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.finres_pop = np.empty( len(self.resource_list), dtype='object')  # populations for finite-resource transitions
        for i in range(len(self.resource_list)):
            ndx = self.resource_list[i][0]
            if self.parameters_length[ndx] == 1:
                self.finres_pop[i] = 0
            else:
                self.finres_pop[i] = np.zeros(self.M, dtype=DTYPE)
            
        self._lambdas = np.zeros((self.infection_terms.shape[0], M))
        self.dxdt = np.zeros(self.nClass*self.M, dtype=DTYPE)


    def update_model_parameters(self, parameters):
        if self.time_dep_param_mapping is None:
            nParams = len(self.param_keys)
            self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
            self.parameters_length = np.empty(nParams, dtype=int)
            try:
                for (i, key) in enumerate(self.param_keys):
                    param = parameters[key]
                    self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                    self.parameters_length[i] = np.size(param)
            except KeyError:
                raise Exception('The parameters passed do not contain certain keys.\
                                 The keys are {}'.format(self.param_keys))
        else:
            self.param_dict = parameters.copy()
            self.update_time_dep_model_parameters(0)

    def update_time_dep_model_parameters(self, tt):
        parameters = self.time_dep_param_mapping(self.param_dict, tt)
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        self.parameters_length = np.empty(nParams, dtype=int)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                self.parameters_length[i] = np.size(param)
        except KeyError:
            raise Exception('The parameters passed do not contain certain keys.\
                             The keys are {}'.format(self.param_keys))
            
    cpdef rhs(self, xt_arr, tt):
        cdef:
            Py_ssize_t m, n, M=self.M, i, index, nClass=self.nClass, class_index
            Py_ssize_t susceptible_index, infective_index
            Py_ssize_t reagent_index, product_index, rate_index
            Py_ssize_t resource_index, probability_index, priority_index 
            Py_ssize_t origin_index, destination_index
            int sign
            int [:, :] constant_terms=self.constant_terms, linear_terms=self.linear_terms
            int [:, :]  infection_terms=self.infection_terms
            int [:, :]  finres_terms=self.finres_terms
            np.ndarray resource_list=self.resource_list
            np.ndarray finres_pop = self.finres_pop
            double [:, :] parameters=self.parameters
            double term, frp
            double [:] xt = xt_arr
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:,:] lambdas = self._lambdas
            unsigned short nn
            unsigned short [:,:] nonzero_index_n = self.nonzero_index_n
            

        if self.time_dep_param_mapping is not None:
            self.update_time_dep_model_parameters(tt)
            parameters = self.parameters
         
        # Compute lambda
        if self.constant_terms.size > 0:
            Ni = xt_arr[(nClass-1)*M:] # update Ni

        if self.constant_CM == 1:
            for m in range(M):
                for i in range(infection_terms.shape[0]):
                    infective_index = infection_terms[i, 1]
                    lambdas[i, m] = 0
                    for n in range(1, nonzero_index_n[m, 0] + 1):
                        nn = nonzero_index_n[m, n]
                        index = nn + M*infective_index
                        if Ni[nn]>0:
                            lambdas[i, m] += CM[m,nn]*xt[index]/Ni[nn]
        else:
            for m in range(M):
                for i in range(infection_terms.shape[0]):
                    infective_index = infection_terms[i, 1]
                    lambdas[i, m] = 0
                    for n in range(M):
                        index = n + M*infective_index
                        if Ni[n]>0:
                            lambdas[i, m] += CM[m,n]*xt[index]/Ni[n]
                    
        # Calculate populations for finite resource transitions
        for i in range(len(resource_list)):
            ndx = self.resource_list[i][0]
            n_cohorts = self.parameters_length[ndx]
            if n_cohorts == 1:
                finres_pop[i] = 0
            else:
                finres_pop[i] = np.zeros(n_cohorts)
            for (class_index, priority_index) in resource_list[i][1:]:
                for m in range(M):
                    if n_cohorts == 1:
                        finres_pop[i] += xt[m + M*class_index] * parameters[priority_index, m]
                    else:
                        finres_pop[i][m] += xt[m + M*class_index] * parameters[priority_index, m]
        
        # Reset dxdt
        self.dxdt = np.zeros(nClass*M, dtype=DTYPE)
        cdef double [:] dxdt = self.dxdt

        # Compute rhs
        for m in range(M):

            if self.constant_terms.size > 0:
                for i in range(constant_terms.shape[0]):
                    rate_index = constant_terms[i, 0]
                    class_index = constant_terms[i, 1]
                    sign = constant_terms[i, 2]
                    term = parameters[rate_index, m]*sign
                    dxdt[m + M*class_index] += term
                    dxdt[m + M*(nClass-1)] += term

            for i in range(linear_terms.shape[0]):
                rate_index = linear_terms[i, 0]
                reagent_index = linear_terms[i, 1]
                product_index = linear_terms[i, 2]
                term = parameters[rate_index, m] * xt[m + M*reagent_index]
                dxdt[m + M*reagent_index] -= term
                if product_index != -1:
                    dxdt[m + M*product_index] += term

            for i in range(infection_terms.shape[0]):
                rate_index = infection_terms[i, 0]
                reagent_index = infection_terms[i, 1]
                susceptible_index = infection_terms[i, 2]
                product_index = infection_terms[i, 3]
                term = parameters[rate_index, m] * lambdas[i, m] * xt[m+M*susceptible_index]
                dxdt[m+M*susceptible_index] -= term
                if product_index != -1:
                    dxdt[m+M*product_index] += term
            
            if self.finres_terms.size > 0:
                for i in range(finres_terms.shape[0]):
                    resource_index = finres_terms[i, 0]
                    rate_index = resource_list[resource_index][0]
                    priority_index = finres_terms[i, 1]
                    probability_index = finres_terms[i, 2]
                    class_index = finres_terms[i, 3]
                    origin_index = finres_terms[i, 4]
                    destination_index = finres_terms[i, 5]

                    if np.size(finres_pop[resource_index]) == 1:
                        frp = finres_pop[resource_index]
                    else:
                        frp = finres_pop[resource_index][m]
                    if frp >= 0.5:  # only if population does not round to zero
                        term = parameters[rate_index, m] * parameters[priority_index, m] \
                               * parameters[probability_index, m] * xt[m+M*class_index] / frp
                    else:
                        term = 0

                    if origin_index != -1:
                        dxdt[m+M*origin_index] -= term
                    if destination_index != -1:
                        dxdt[m+M*destination_index] += term


    def simulate(self, x0, contactMatrix, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        x0: np.array or dict
            Initial conditions. If it is an array it should have length
            M*(model_dimension-1), where x0[i + j*M] should be the initial
            value of model class i of age group j. The removed R class
            must be left out. If it is a dict then
            it should have a key corresponding to each model class,
            with a 1D array containing the initial condition for each
            age group as value. One of the classes may be left out,
            in which case its initial values will be inferred from the
            others.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.
        """

        cdef:
            int m, n, index_n, M=self.M
            double [:,:] CM=contactMatrix(1.0)
            unsigned short [:,:] nonzero_index_n
            move_n_num = np.zeros(M, dtype=int)

        # Here nonzero elements of CM are stored for skipping these
        if self.constant_CM == 1:
            for m in range(M):
                index_n = 0
                for n in range(M):
                    if CM[m,n] > 0.0:
                        index_n += 1
                    
                move_n_num[m] = <int> index_n

            move_n_num.sort()
            max_move_n = move_n_num[M - 1]
            print("Max index n", max_move_n)
            self.nonzero_index_n = np.zeros( (self.M, max_move_n + 1), dtype=np.uint16) # the list n for non zero CM at specific m
            nonzero_index_n = self.nonzero_index_n

            for m in range(M):
                index_n = 0
                for n in range(M):
                    if CM[m,n] > 0.0:
                        nonzero_index_n[m, index_n + 1] = n
                        index_n += 1
                    
                nonzero_index_n[m,0] = index_n

        if type(x0) == list:
            x0 = np.array(x0)

        if type(x0) == np.ndarray:

            n_class_for_init = self.nClass
            if self.constant_terms.size > 0:
                n_class_for_init -= 1
            if x0.size != n_class_for_init*self.M:
                raise Exception("Initial condition x0 has the wrong dimensions. Expected x0.size=%s."
                    % ( n_class_for_init*self.M) )
        elif type(x0) == dict:
            # Check if any classes are not included in x0

            class_list = list(self.class_index_dict.keys())
            if self.constant_terms.size > 0:
                class_list.remove('Ni')

            skipped_classes = []
            for O in class_list:
                if not O in x0:
                    skipped_classes.append(O)
            if len(skipped_classes) > 0:
                raise Exception("Missing classes in initial conditions: %s" % skipped_classes)


            # Construct initial condition array
            x0_arr = np.zeros(0)

            for O in class_list:
                x0_arr = np.concatenate( [x0_arr, x0[O]] )
            x0 = x0_arr

        x0 = np.array(x0, dtype=DTYPE)

        # add Ni to x0
        if self.constant_terms.size > 0:
            x0 = np.concatenate([x0, self.Ni])

        self.paramList = self.make_parameters_dict()
        data = self.simulator(x0, contactMatrix, Tf, Nf, 
                              integrator, Ti, maxNumSteps, **kwargs)
        return data


    def make_parameters_dict(self):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict


    def model_class_data(self, model_class_key, data):
        """
        Parameters
        ----------
        data: dict
            The object returned by `simulate`.

        Returns
        -------
            The population of class `model_class_key` as a time series
        """
        X = data['X']

        if model_class_key != 'R' or 'R' in self.class_index_dict.keys():
            class_index = self.class_index_dict[model_class_key]
            Os = X[:, class_index*self.M:(class_index+1)*self.M]
        else:
            if self.constant_terms.size > 0:
                x = X[:, :(self.nClass-1)*self.M]
                x_reshaped = x.reshape((X.shape[0], (self.nClass-1), self.M))
                Os = X[:, (self.nClass-1)*self.M:] - np.sum(x_reshaped, axis=1)
            else:
                X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
                Os = self.Ni - np.sum(X_reshaped, axis=1)
        return Os


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Spp(Model):
    """
    This is a slightly more specific version of the class `Model`. 

    `Spp` is still supported for backward compatibility. 

    `Model` class is recommended over `Spp` for new users. 

    The `Spp` class works like `Model` but infection terms use a single class `S` 

    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        Contains the values for the parameters given in the model specification.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec. 
        Default: Identical mapping of the dictionary at all times. 

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "constant"  : [ ["k"] ],
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'k': 1,
        }
    """   
    
    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None, constant_CM=0):
        Xpp_model_spec = pyross.utils.Spp2Xpp(model_spec)
        super().__init__(Xpp_model_spec, parameters, M, Ni, time_dep_param_mapping=time_dep_param_mapping, constant_CM=constant_CM)
    
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SppQ(Spp):
    """User-defined epidemic model with quarantine stage.

    This is a slightly more specific version of the class `Model`. 

    `SppQ` is still supported for backward compatibility. 

    `Model` class is recommended over `SppQ` for new users. 

    To initialise the SppQ model,
    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        Contains the values for the parameters given in the model specification.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec. 
        Default: Identical mapping of the dictionary at all times. 

    Examples
    --------
    An example of model_spec and parameters for SIR class with random
    testing (without false positives/negatives) and quarantine

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            },
            "test_pos"  : [ "p_falsepos", "p_truepos", "p_falsepos"] ,
            "test_freq" : [ "tf", "tf", "tf"] 
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'p_falsepos': 0
            'p_truepos': 1
            'tf': 1
        }
    """
    
    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None):
        self.full_model_spec = pyross.utils.build_SppQ_model_spec(model_spec) 
        self.input_time_dep_param_mapping = time_dep_param_mapping
        self.testRate = None
        super().__init__(self.full_model_spec, parameters, M, Ni, time_dep_param_mapping=self.full_time_dep_param_mapping)
    
    cpdef set_testRate(self, testRate):
        self.testRate = testRate
        
    cpdef full_time_dep_param_mapping(self, input_parameters, t):
        cdef dict output_param_dict
        if self.input_time_dep_param_mapping is not None:
            output_param_dict = self.input_time_dep_param_mapping(input_parameters, t).copy()
        else:
            output_param_dict = input_parameters.copy()
        if self.testRate is not None:
            output_param_dict['tau'] = self.testRate(t)
        else:
            output_param_dict['tau'] = 0
        return output_param_dict
    
    def model_class_data(self, model_class_key, data):
        """
        Parameters
        ----------
        data: dict
            The object returned by `simulate`.

        Returns
        -------
            The population of class `model_class_key` as a time series
        """
        X = data['X']

        if model_class_key == 'Ni':
            X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
            Os = np.sum(X_reshaped, axis=1)
        elif model_class_key == 'NiQ':
            X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
            Os = np.sum(X_reshaped[:, (self.nClass//2):, :], axis=1)
        else:
            class_index = self.class_index_dict[model_class_key]
            Os = X[:, class_index*self.M:(class_index+1)*self.M]
        return Os
    
    def simulate(self, x0, contactMatrix, testRate, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Simulates a compartment model given initial conditions,
        choice of integrator and other parameters. 
        Returns the time series data and parameters in a dict. 
        Internally calls the method 'simulator' of CommonMethods
        
        ...

        Parameters
        ----------
        x0: np.array or dict
            Initial conditions. If it is an array it should have length
            M*(model_dimension-1), where x0[i + j*M] should be the initial
            value of model class i of age group j. The removed R class
            must be left out. If it is a dict then
            it should have a key corresponding to each model class,
            with a 1D array containing the initial condition for each
            age group as value. One of the classes may be left out,
            in which case its initial values will be inferred from the
            others.
        contactMatrix: python function(t)
            The social contact matrix C_{ij} denotes the
            average number of contacts made per day by an
            individual in class i with an individual in class j
        testRate: python function(t)
            The total number of PCR tests performed per day
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take.
            The default value is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
             X: output path from integrator,  t : time points evaluated at,
            'param': input param to integrator.
        """
        self.testRate = testRate
        return super().simulate(x0, contactMatrix, Tf, Nf, Ti,
                                integrator, maxNumSteps, **kwargs)
    




   
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SppSparse(Spp):
    """
    This is a slightly more specific version of the class `Model`. 

    `Spp` is still supported for backward compatibility. 

    `Model` class is recommended over `Spp` for new users. 

    The `Spp` class works like `Model` but infection terms use a single class `S` 
    ...


    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        Contains the values for the parameters given in the model specification.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec. 
        Default: Identical mapping of the dictionary at all times. 

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "constant"  : [ ["k"] ],
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'k': 1,
        }
    """

    def __init__(self, model_spec, contact_matrix0, threshold, parameters, M, Ni, time_dep_param_mapping=None, verbose=False):

        super().__init__(model_spec, parameters, M, Ni, time_dep_param_mapping)
        
        if verbose:
            print("Preprocessing sparse contact matrix\n")
        check = int(M*M/20)
        for i in range(M):
            for j in range(M):
                if verbose and (i * M + j) % check == 0:
                    print ("Processed",(i * M + j) * 100/( M * M ),"%, 1/2")
                if contact_matrix0[i,j] > threshold:
                    self.intCounter += int(1)
                    
        self.interactingMP = np.zeros((self.intCounter,2), dtype=int)
        
        counter = int(0)
        for i in range(M):
            for j in range(M):
                if verbose and (i * M + j) % check == 0:
                    print ("Processed",(i * M + j) * 100/( M * M ),"%, 2/2")
                if contact_matrix0[i,j] > threshold:
                    self.interactingMP[counter] = [int(i),int(j)]
                    counter += int(1)
            
    cpdef rhs(self, xt_arr, tt):
        cdef:
            Py_ssize_t m, n, M=self.M, i, index, nClass=self.nClass, class_index
            Py_ssize_t S_index=self.class_index_dict['S'], infective_index
            Py_ssize_t reagent_index, product_index, rate_index, morig, mpoint
            int sign
            int [:, :] constant_terms=self.constant_terms, linear_terms=self.linear_terms
            int [:, :]  infection_terms=self.infection_terms
            double [:, :] parameters=self.parameters
            double term
            double [:] xt = xt_arr
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:,:] lambdas = self._lambdas
            int [:,:] intMPs = self.interactingMP

        if self.time_dep_param_mapping is not None:
            self.update_time_dep_model_parameters(tt)
            parameters = self.parameters
         
        # Compute lambda
        if self.constant_terms.size > 0:
            Ni = xt_arr[(nClass-1)*M:] # update Ni

        for i in range(infection_terms.shape[0]):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                lambdas[i, m] = 0
            for m in range(len(intMPs)):
                morig = intMPs[m,0]
                mpoint = intMPs[m,1]
                index = mpoint + M*infective_index
                lambdas[i, morig] += CM[morig,mpoint]*xt[index]/Ni[mpoint]

        # Reset dxdt
        self.dxdt = np.zeros(nClass*M, dtype=DTYPE)
        cdef double [:] dxdt = self.dxdt

        # Compute rhs
        for m in range(M):
            if self.constant_terms.size > 0:
                for i in range(constant_terms.shape[0]):
                    rate_index = constant_terms[i, 0]
                    class_index = constant_terms[i, 1]
                    sign = constant_terms[i, 2]
                    term = parameters[rate_index, m]*sign
                    dxdt[m + M*class_index] += term
                    dxdt[m + M*(nClass-1)] += term

            for i in range(linear_terms.shape[0]):
                rate_index = linear_terms[i, 0]
                reagent_index = linear_terms[i, 1]
                product_index = linear_terms[i, 2]
                term = parameters[rate_index, m] * xt[m + M*reagent_index]
                dxdt[m + M*reagent_index] -= term
                if product_index != -1:
                    dxdt[m + M*product_index] += term

            for i in range(infection_terms.shape[0]):
                rate_index = infection_terms[i, 0]
                reagent_index = infection_terms[i, 1]
                product_index = infection_terms[i, 2]
                term = parameters[rate_index, m] * lambdas[i, m] * xt[m+M*S_index]
                dxdt[m+M*S_index] -= term
                if product_index != -1:
                    dxdt[m+M*product_index] += term
