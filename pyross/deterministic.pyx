import  numpy as np
cimport numpy as np
cimport cython
import pyross.utils
import warnings

DTYPE   = np.float
from libc.stdlib cimport malloc, free




cdef class IntegratorsClass:
    """
    List of all integrator used by various deterministic models listed below.
    """

    def simulateRHS(self, rhs0, x0, Ti, Tf, Nf, integrator, maxNumSteps, **kwargs):
        """
        Performs numerical integration

        Parameters
        ----------
        rhs0: python function(x,t)
            Input function of current state and time x, t
            returns dx/dt
        x0: np.array
            Initial state vector.
        Ti: float
            Start time for integrator.
        Tf: float
            End time for integrator.
        Nf: Int
            Number of time points to evaluate at.
        integrator: string, optional
            Selects which integration method to use. The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator is allowed to take
            to obtain a solution. The default is 100000.
        **kwargs: optional kwargs to be passed to the IntegratorsClass

        Raises
        ------
        Exception
            If integration fails.

        Returns
        -------
        X: np.array(len(t), len(x0))
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


    cpdef set_contactMatrix(self, double t, contactMatrix):
        self.CM=contactMatrix(t)


    def simulator(self, x0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        x0: np.array
            Initial number of compartment values.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data     = {'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M}
        data_out = data.copy()
        data_out.update(self.paramList)
        return data_out




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(IntegratorsClass):
    """

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
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
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 3*self.M, dtype=DTYPE)           # right hand side

        self.paramList = parameters


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
            dxdt[i]     = -rateS                                                   # \dot S
            dxdt[i+M]   = alpha[i]*rateS     - gIa[i]*Ia[i]                           # \dot Ia
            dxdt[i+2*M] = (1-alpha[i])*rateS - gIs[i]*Is[i]                           # \dot Is
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf, integrator='odeint',
                 Ti=0, maxNumSteps=10000, **kwargs):
        """
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
        maxNumSteps: int, optional (DEPRICATED)
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0 = np.concatenate((S0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                        'fsa':self.fsa, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

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
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, self.M:2*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 2*self.M:3*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R = self.Ni - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M]
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        beta: float
            rate of spread of infection.
        gI: float
            rate of removal from infectives.
        kI: int
            number of stages of infection.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']                         # infection rate
        self.gI    = parameters['gI']                           # removal rate of I
        self.kI    = parameters['kI']
        self.nClass= self.kI + 1

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( (self.kI+1)*self.M, dtype=DTYPE) # right hand side

        self.paramList = parameters


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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'kI':self.kI }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def I(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        kI = data['kI']
        X = data['X']
        I = X[:, self.M:(kI+1)*self.M]
        return I


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        kI = data['kI']

        I0 = np.zeros(self.M)
        for i in range(kI):
            I0 += X[:, (i+1)*self.M : (i+2)*self.M]
        R = self.Ni - X[:, 0:self.M] - I0
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        gE: float
            rate of removal from exposed individuals.
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
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

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
            dxdt[i]     = -rateS                                     # \dot S
            dxdt[i+M]   = rateS       - gE[i]*  E[i]                    # \dot E
            dxdt[i+2*M] = ce1*E[i] - gIa[i]*Ia[i]                       # \dot Ia
            dxdt[i+3*M] = ce2*E[i] - gIs[i]*Is[i]                       # \dot Is
        return


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
                        maxNumSteps=100000, **kwargs):
        """
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0 = np.concatenate((S0, E0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,'fsa':self.fsa,
                         'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 2*self.M:3*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 3*self.M:4*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R = self.Ni - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M]
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        beta: float
            rate of spread of infection.
        gI: float
            rate of removal from infected individuals.
        gE: float
            rate of removal from exposed individuals.
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
        self.beta  = parameters['beta']                         # infection rate
        self.gE    = parameters['gE']                           # removal rate of E
        self.gI    = parameters['gI']                           # removal rate of I
        self.kI    = parameters['kI']                           # number of stages
        self.kE    = parameters['kE']
        self.nClass= self.kI + self.kE + 1

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( (self.kI + self.kE + 1)*self.M, dtype=DTYPE)           # right hand side

        if self.kE==0:
            raise Exception('number of E stages should be greater than zero, kE>0')
        elif self.kI==0:
            raise Exception('number of I stages should be greater than zero, kI>0')


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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, I0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'kI':self.kI, 'kE':self.kE }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        kI = data['kI']
        kE = data['kE']
        X = data['X']
        E = X[:, self.M:(1+self.kE)*self.M]
        return E


    def I(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        kI = data['kI']
        kE = data['kE']
        X  = data['X']
        Is = X[:, (1+self.kE)*self.M:(1+self.kE+self.kI)*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        kI = data['kI']
        kE = data['kE']
        I0 = np.zeros(self.M)
        E0 = np.zeros(self.M)
        for i in range(kE):
            E0 += X[:, (i+1)*self.M : (i+2)*self.M]
        for i in range(kI):
            I0 += X[:, (kE+1)*self.M : (kE+1+kI)*self.M]
        R = self.Ni - X[:, 0:self.M] - I0 - E0
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkIkR(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic infected individuals.
        gIs: float
            rate of removal from symptomatic infected individuals.
        gE: float
            rate of removal from exposed individuals.
        kI: int
            number of stages of asymptomatic infectives.
        kI: int
            number of stages of symptomatic infectives.
        kE: int
            number of stages of exposed.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']                         # infection rate
        self.gE    = parameters['gE']                           # removal rate of E
        self.gIa   = parameters['gIa']                           # removal rate of Ia
        self.gIs   = parameters['gIs']                           # removal rate of Is
        self.kI    = parameters['kI']                           # number of stages
        self.fsa   = parameters['fsa']                          # the self-isolation parameter
        self.kE    = parameters['kE']
        self.nClass= self.kI + self.kI + self.kE + 1

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( (self.kI + self.kI + self.kE + 1)*self.M, dtype=DTYPE)           # right hand side

        if self.kE==0:
            raise Exception('number of E stages should be greater than zero, kE>0')
        elif self.kI==0:
            raise Exception('number of I stages should be greater than zero, kI>0')

        alpha      = parameters['alpha']                        # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha = alpha
        else:
            raise Exception('alpha can be a number or an array of size M')


    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, kI=self.kI, kE = self.kE
            double beta=self.beta, gIa=self.kI*self.gIa, rateS, lmda, ce1, ce2
            double gE=self.kE*self.gE, gIs=self.kI*self.gIs, fsa=self.fsa
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :(kE+1)*M]
            double [:] Ia   = xt[(kE+1)*M   :(kE+kI+1)*M]
            double [:] Is   = xt[(kE+kI+1)*M:(kE+kI+kI+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] dxdt = self.dxdt
            double [:] alpha = self.alpha

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for jj in range(kI):
                for j in range(M):
                    lmda += beta*CM[i,j]*(Is[j+jj*M]*fsa + Ia[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS

            #Exposed class
            dxdt[i+M+0] = rateS - gE*E[i]
            for j in range(kE - 1) :
                dxdt[i + M +  (j+1)*M ] = gE * E[i+j*M] - gE * E[i+(j+1)*M]

            #Asymptomatics class
            dxdt[i + (kE+1)*M + 0] = ce1*E[i+(kE-1)*M] - gIa*Ia[i]
            for j in range(kI-1):
                dxdt[i+(kE+1)*M + (j+1)*M ]  = gIa*Ia[i+j*M] - gIa*Ia[i+(j+1)*M]

            #Symptomatics class
            dxdt[i + (kE+kI+1)*M + 0] = ce2*E[i+(kE-1)*M] - gIs*Is[i]
            for j in range(kI-1):
                dxdt[i+(kE+kI+1)*M + (j+1)*M ]  = gIs*Is[i+j*M] - gIs*Is[i+(j+1)*M]
        return


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint',
            maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M, 'beta':self.beta,'gI':self.gI,
            'fsa':self.fsa, 'kI':self.kI, 'kE':self.kE }

        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        kE = data['kE']
        X = data['X']
        E = X[:, self.M:(1+self.kE)*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        kI = data['kI']
        kE = data['kE']
        X  = data['X']
        Ia = X[:, (1+self.kE)*self.M:(1+self.kE+self.kI)*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        kI = data['kI']
        kE = data['kE']
        X  = data['X']
        Is = X[:, (1+self.kE+self.kI)*self.M:(1+self.kE+self.kI+self.kI)*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        kI = data['kI']
        kE = data['kE']
        Ia0= np.zeros(self.M)
        Is0= np.zeros(self.M)
        E0 = np.zeros(self.M)
        for i in range(kE):
            E0 += X[:, (i+1)*self.M : (i+2)*self.M]
        for i in range(kI):
            Ia0 += X[:, (kE+1)*self.M : (kE+1+kI)*self.M]
        for i in range(kI):
            Is0 += X[:, (kE+kI+1)*self.M : (kE+1+2*kI)*self.M]
        R = self.Ni - X[:, 0:self.M] - Ia0 - Is0 - E0
        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI8R(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gE: float
            rate of removal from exposeds individuals.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gIsp: float
            rate of removal from symptomatic individuals towards buffer.
        gIh: float
            rate of removal for hospitalised individuals.
        gIhp: float
            rate of removal from hospitalised individuals towards buffer.
        gIc: float
            rate of removal for idividuals in intensive care.
        gIcp: float
            rate of removal from ICU individuals towards buffer.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        fh  : float
            fraction by which hospitalised individuals are isolated.
        sa: float, np.array (M,)
            daily arrival of new susceptables.
            sa is rate of additional/removal of population by birth etc
        hh: float, np.array (M,)
            fraction hospitalised from Is
        cc: float, np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: float, np.array (M,)
            mortality rate in intensive care
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 10  # only 7 input classes
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # removal rate of E class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIsp  = parameters['gIsp']                     # removal rate of Isp
        self.gIh   = parameters['gIh']                      # removal rate of Is
        self.gIhp  = parameters['gIhp']                     # removal rate of Ihp
        self.gIc   = parameters['gIc']                      # removal rate of Ih
        self.gIcp  = parameters['gIcp']                     # removal rate of Ixp
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds
        alpha      = parameters['alpha']                    # fraction of asymptomatics
        sa         = parameters['sa']                       # rate of additional/removal of population by birth etc
        hh         = parameters['hh']                       # fraction of infected who gets hospitalized
        cc         = parameters['cc']                       # fraction of hospitalized who endup in ICU
        mm         = parameters['mm']                       # mortality fraction from ICU
        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 11*self.M, dtype=DTYPE)           # right hand side

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


    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double gIsp=self.gIsp, gIhp=self.gIhp, gIcp=self.gIcp
            double ce1, ce2
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :2*M]
            double [:] Ia   = xt[2*M:3*M]
            double [:] Is   = xt[3*M:4*M]
            double [:] Isp  = xt[4*M:5*M]
            double [:] Ih   = xt[5*M:6*M]
            double [:] Ihp  = xt[6*M:7*M]
            double [:] Ic   = xt[7*M:8*M]
            double [:] Icp  = xt[8*M:9*M]
            double [:] Im   = xt[9*M:10*M]
            double [:] Ni   = xt[10*M:11*M]
            double [:,:] CM = self.CM

            double [:] alpha= self.alpha, balpha=1-self.alpha
            double [:] sa   = self.sa   , bsa   =1-self.sa
            double [:] hh   = self.hh   , bhh   =1-self.hh
            double [:] cc   = self.cc   , bcc   =1-self.cc
            double [:] mm   = self.mm   , bmm   =1-self.mm
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0;
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS + sa[i]                         # \dot S
            dxdt[i+M]   = rateS  - gE*E[i]                       # \dot E
            dxdt[i+2*M] = gE*alpha[i] *E[i]  - gIa*Ia[i]         # \dot Ia
            dxdt[i+3*M] = gE*balpha[i]*E[i]  - gIs*Is[i]         # \dot Is
            dxdt[i+4*M] = gIs*bhh[i]  *Is[i] - gIsp*Isp[i]       # \dot Isp
            dxdt[i+5*M] = gIs*hh[i]   *Is[i] - gIh*Ih[i]         # \dot Ih
            dxdt[i+6*M] = gIh*bcc[i]  *Ih[i] - gIhp*Ihp[i]       # \dot Ihp
            dxdt[i+7*M] = gIh*cc[i]   *Ih[i] - gIc*Ic[i]         # \dot Ic
            dxdt[i+8*M] = gIc*bmm[i]  *Ic[i] - gIcp*Icp[i]       # \dot Icp
            dxdt[i+9*M] = gIc*mm[i]   *Ic[i]                     # \dot Im
            dxdt[i+10*M]= sa[i] - gIc*mm[i]*Im[i]                # \dot Ni
        return


    def simulate(self, S0, E0, Ia0, Is0, Isp0, Ih0, Ihp0, Ic0, Icp0, Im0, contactMatrix, Tf, Nf, Ti=0,
                    integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Ih0: np.array
            Initial number of hospitalized infectives.
        Ic0: np.array
            Initial number of ICU infectives.
        Im0: np.array
            Initial number of mortality.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, Ia0, Is0, Isp0, Ih0, Ihp0, Ic0, Icp0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'fsa':self.fsa, 'fh':self.fh,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 2*self.M:3*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 3*self.M:4*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': hospitalized population time series
        """
        X  = data['X']
        Ih = X[:, 5*self.M:6*self.M]
        return Ih


    def Ic(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': ICU hospitalized population time series
        """
        X  = data['X']
        Ic = X[:, 7*self.M:8*self.M]
        return Ic


    def Im(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': mortality time series
        """
        X  = data['X']
        Im = X[:, 9*self.M:10*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            population
        """
        X = data['X']
        ppln  = X[:,10*self.M:11*self.M]
        return ppln


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
            R = N(t) - (S + E + Ia + Is + Ih + Ic)
        """
        X = data['X']
        R =  X[:, 10*self.M:11*self.M] - X[:, 0:self.M]  - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
                                                       - X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M] - X[:,6*self.M:7*self.M] \
                                                       - X[:,7*self.M:8*self.M] - X[:,8*self.M:9*self.M] - X[:,9*self.M:10*self.M]

        return R




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIR(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        gE: float
            rate of removal from exposeds individuals.
        gA: float
            rate of removal from activated individuals.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 5
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # removal rate of Ia
        self.gIs   = parameters['gIs']                          # removal rate of Is
        self.gE    = parameters['gE']                           # removal rate of E
        self.gA    = parameters['gA']                           # rate to go from A to Ia, Is
        self.fsa   = parameters['fsa']                          # the self-isolation parameter
        alpha      = parameters['alpha']

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 5*self.M, dtype=DTYPE)           # right hand side

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

    cpdef rhs(self, xt, tt):
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
            double [:] dxdt = self.dxdt

            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS                                  # \dot S
            dxdt[i+M]   =  rateS   - gE*E[i]                   # \dot E
            dxdt[i+2*M] = gE* E[i] - gA*A[i]                      # \dot A
            dxdt[i+3*M] = gAA*A[i] - gIa     *Ia[i]               # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - gIs     *Is[i]               # \dot Is
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, contactMatrix, Tf, Nf, Ti=0,
             integrator='odeint', maxNumSteps=100000, **kwargs):
        """
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt
        x0=np.concatenate((S0, E0, A0, Ia0, Is0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,'fsa':self.fsa,
                    'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             A: Activated population time series
        """
        X = data['X']
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R = self.Ni - X[:, 0:self.M] -  X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
             -X[:,4*self.M:5*self.M]
        return R


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI8R(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gE: float
            rate of removal from exposeds individuals.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gIsp: float
            rate of removal from symptomatic individuals towards buffer.
        gIh: float
            rate of removal for hospitalised individuals.
        gIhp: float
            rate of removal from hospitalised individuals towards buffer.
        gIc: float
            rate of removal for idividuals in intensive care.
        gIcp: float
            rate of removal from ICU individuals towards buffer.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        fh  : float
            fraction by which hospitalised individuals are isolated.
        sa: float, np.array (M,)
            daily arrival of new susceptables.
            sa is rate of additional/removal of population by birth etc
        hh: float, np.array (M,)
            fraction hospitalised from Is
        cc: float, np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: float, np.array (M,)
            mortality rate in intensive care
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 11
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # removal rate of E class
        self.gA    = parameters['gA']                       # removal rate of A class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIsp  = parameters['gIsp']                     # removal rate of Isp
        self.gIh   = parameters['gIh']                      # removal rate of Is
        self.gIhp  = parameters['gIhp']                     # removal rate of Ihp
        self.gIc   = parameters['gIc']                      # removal rate of Ih
        self.gIcp  = parameters['gIcp']                     # removal rate of Ixp
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds
        alpha      = parameters['alpha']                    # fraction of asymptomatics
        sa         = parameters['sa']                       # rate of additional/removal of population by birth etc
        hh         = parameters['hh']                       # fraction of infected who gets hospitalized
        cc         = parameters['cc']                       # fraction of hospitalized who endup in ICU
        mm         = parameters['mm']                       # mortality fraction from ICU

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 12*self.M, dtype=DTYPE)           # right hand side

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


    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh, gE=self.gE, gA=self.gA
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double gIsp=self.gIsp, gIhp=self.gIhp, gIcp=self.gIcp
            double ce1, ce2
            double [:] S    = xt[0  :M]
            double [:] E    = xt[M  :2*M]
            double [:] A    = xt[2*M:3*M]
            double [:] Ia   = xt[3*M:4*M]
            double [:] Is   = xt[4*M:5*M]
            double [:] Isp  = xt[5*M:6*M]
            double [:] Ih   = xt[6*M:7*M]
            double [:] Ihp  = xt[7*M:8*M]
            double [:] Ic   = xt[8*M:9*M]
            double [:] Icp  = xt[9*M:10*M]
            double [:] Im   = xt[10*M:11*M]
            double [:] Ni   = xt[11*M:12*M]
            double [:,:] CM = self.CM

            double [:] alpha= self.alpha, balpha=1-self.alpha
            double [:] sa   = self.sa   , bsa   =1-self.sa
            double [:] hh   = self.hh   , bhh   =1-self.hh
            double [:] cc   = self.cc   , bcc   =1-self.cc
            double [:] mm   = self.mm   , bmm   =1-self.mm
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0;
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]      = -rateS  + sa[i]                           # \dot S
            dxdt[i+M]    = rateS   - gE*E[i]                         # \dot E
            dxdt[i+2*M]  = gE*E[i] - gA*A[i]                         # \dot A
            dxdt[i+3*M]  = gA*alpha[i] *A[i]  - gIa *Ia[i]           # \dot Ia
            dxdt[i+4*M]  = gA*balpha[i]*A[i]  - gIs *Is[i]           # \dot Is
            dxdt[i+5*M]  = gIs*bhh[i]  *Is[i] - gIsp*Isp[i]          # \dot Isp
            dxdt[i+6*M]  = gIs*hh[i]   *Is[i] - gIh *Ih[i]           # \dot Ih
            dxdt[i+7*M]  = gIh*bcc[i]  *Ih[i] - gIhp*Ihp[i]          # \dot Ihp
            dxdt[i+8*M]  = gIh*cc[i]   *Ih[i] - gIc *Ic[i]           # \dot Ic
            dxdt[i+9*M]  = gIc*bmm[i]  *Ic[i] - gIcp*Icp[i]          # \dot Icp
            dxdt[i+10*M] = gIc*mm[i]   *Ic[i]                        # \dot Im
            dxdt[i+11*M] = sa[i]-gIc*mm[i]*Im[i]                     # \dot Ni
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Isp0, Ih0, Ihp0, Ic0, Icp0, Im0, contactMatrix, Tf, Nf, Ti=0,
                    integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Ih0: np.array
            Initial number of hospitalized infectives.
        Ic0: np.array
            Initial number of ICU infectives.
        Im0: np.array
            Initial number of mortality.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, A0, Ia0, Is0, Isp0, Ih0, Ihp0, Ic0, Icp0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'fsa':self.fsa, 'fh':self.fh,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             A: Activates population time series
        """
        X = data['X']
        E = X[:, 2*self.M:3*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': hospitalized population time series
        """
        X  = data['X']
        Ih = X[:, 6*self.M:7*self.M]
        return Ih


    def Ic(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': ICU hospitalized population time series
        """
        X  = data['X']
        Ic = X[:, 8*self.M:9*self.M]
        return Ic


    def Im(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ic': mortality time series
        """
        X  = data['X']
        Im = X[:, 10*self.M:11*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            population
        """
        X = data['X']
        ppln  = X[:,11*self.M:12*self.M]
        return ppln


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
            R = N(t) - (S + E + A+ Ia + Is + Ih + Ic)
        """
        X = data['X']
        R =  X[:, 11*self.M:12*self.M] - X[:, 0:self.M]- X[:, self.M:2*self.M]  - X[:,2*self.M:3*self.M] - X[:,3*self.M:4*self.M] \
                                                       - X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M] - X[:,6*self.M:7*self.M] \
                                                       - X[:,7*self.M:8*self.M] - X[:,8*self.M:9*self.M] - X[:,9*self.M:10*self.M] \
                                                       - X[:,10*self.M:11*self.M]

        return R








@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(IntegratorsClass):
    """
    To initialise the SEAIRQ class,

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        tE: float
            testing rate and contact tracing of exposeds
        tA: float
            testing rate and contact tracing of activateds
        tIa: float
            testing rate and contact tracing of asymptomatics
        tIs: float
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
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.paramList = parameters

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 6*self.M, dtype=DTYPE)           # right hand side


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
            dxdt[i]     = -rateS                                      # \dot S
            dxdt[i+M]   = rateS    - (gE[i]+tE[i])     *E[i]                # \dot E
            dxdt[i+2*M] = gE[i]* E[i] - (gA[i]+tA[i])*A[i]                # \dot A
            dxdt[i+3*M] = gAA*A[i] - (gIa[i]+tIa[i])*Ia[i]               # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - (gIs[i]+tIs[i])*Is[i]               # \dot Is
            dxdt[i+5*M] = tE[i]*E[i]+tA[i]*A[i]+tIa[i]*Ia[i]+tIs[i]*Is[i]         # \dot Q
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
            contains the following keys:

            * 'X': output path from integrator
            * 't': time points evaluated at,
            * 'param': input param to integrator.
        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'beta':self.beta,'gIa':self.gIa, 'fsa':self.fsa, 'gIs':self.gIs,
                     'gE':self.gE,'gA':self.gA,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
        S: np.array
            Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
        E: np.array
            Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             A: Activated population time series
        """
        X = data['X']
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R = self.Ni - X[:, 0:self.M] -  X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
             -X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M]
        return R


    def Q(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             Q: Quarantined population time series
        """
        X  = data['X']
        Is = X[:, 5*self.M:6*self.M]
        return Is



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ_testing(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        ars: float
            fraction of population admissible for random and symptomatic tests
        kapE: float
            fraction of positive tests for exposed individuals
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 6
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gE    = parameters['gE']                       # removal rate of E
        self.gA    = parameters['gA']                       # rate to go from A to Ia and Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.ars    = parameters['ars']                     # fraction of population admissible for testing
        self.kapE    = parameters['kapE']                   # fraction of positive tests for exposed

        alpha      = parameters['alpha']

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix
        self.TR    = np.zeros( self.M, dtype = DTYPE)           # test rate
        self.dxdt  = np.zeros( 6*self.M, dtype=DTYPE)           # right hand side

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')



    cpdef rhs(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda, t0, tE, tA, tIa, tIs
            double ars=self.ars, kapE=self.kapE
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
            double [:]   TR = self.TR
            double [:] dxdt = self.dxdt

            double [:] alpha= self.alpha

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]


            t0 = 1./(ars*(Ni[i]-Q[i]-Is[i])+Is[i])
            tE = TR[i]*ars*kapE*t0
            tA= TR[i]*ars*t0
            tIa = TR[i]*ars*t0
            tIs = TR[i]*t0


            dxdt[i]     = -rateS                                      # \dot S
            dxdt[i+M]   =  rateS   - (gE+tE)     *E[i]                # \dot E
            dxdt[i+2*M] = gE* E[i] - (gA+tA     )*A[i]                # \dot A
            dxdt[i+3*M] = gAA*A[i] - (gIa+tIa   )*Ia[i]               # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - (gIs+tIs   )*Is[i]               # \dot Is
            dxdt[i+5*M] = tE*E[i]+tA*A[i]+tIa*Ia[i]+tIs*Is[i]         # \dot Q

        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0, contactMatrix, testRate, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.TR = testRate(t)
            self.rhs(xt, t)
            return self.dxdt

        x0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'beta':self.beta,'gIa':self.gIa, 'fsa':self.fsa, 'gIs':self.gIs,
                     'gE':self.gE,'gA':self.gA,'ars':self.ars,'kapE':self.kapE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             A: Activated population time series
        """
        X = data['X']
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R = self.Ni - X[:, 0:self.M] -  X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
             -X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M]
        return R


    def Q(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             Q: Quarantined population time series
        """
        X  = data['X']
        Is = X[:, 5*self.M:6*self.M]
        return Is



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        ep  : float
            fraction of removed who become susceptable again
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
        self.beta  = parameters['beta']                         # infection rate
        self.gIa   = parameters['gIa']                          # removal rate of Ia
        self.gIs   = parameters['gIs']                          # removal rate of Is
        self.fsa   = parameters['fsa']                          # the self-isolation parameter of symptomatics
        alpha      = parameters['alpha']
        self.ep    = parameters['ep']                           # fraction of removed who is susceptible
        sa         = parameters['sa']                           # daily arrival of new susceptibles
        iaa        = parameters['iaa']                          # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0 = np.concatenate((S0, Ia0, Is0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                        'fsa':self.fsa, 'ep':self.ep,
                        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files

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
        data: data files

        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, self.M:2*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 2*self.M:3*self.M]
        return Is


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files

        Returns
        -------
             R: Removed population time series
        """
        X = data['X']
        R =  X[:, 3*self.M:4*self.M] - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M]
        return R


    def population(self,  data):
        """
        Parameters
        ----------
        data: data files

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
cdef class Spp(IntegratorsClass):
    """
    Parameters
    ----------
    parameters: dict
        Contains the values for the parameters given in the model specification.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, model_spec, parameters, M, Ni):

        self.N = DTYPE(np.sum(Ni))
        self.M = DTYPE(M)
        self.Ni = np.array(Ni, dtype=DTYPE)

        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.update_model_parameters(parameters)
        self.CM = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self._lambdas = np.zeros((self.infection_terms.shape[0], M))
        self.dxdt = np.zeros(self.nClass*self.M, dtype=DTYPE)


    def update_model_parameters(self, parameters):
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
        except KeyError:
            raise Exception('The parameters passed does not contain certain keys. The keys are {}'.format(self.param_keys))

    cpdef rhs(self, xt_arr, tt):
        cdef:
            Py_ssize_t m, n, M=self.M, i, index, nClass=self.nClass, class_index
            Py_ssize_t S_index=self.class_index_dict['S'], infection_index, reagent_index, product_index, rate_index
            int sign
            int [:, :] constant_terms=self.constant_terms, linear_terms=self.linear_terms, infection_terms=self.infection_terms
            double [:, :] parameters=self.parameters
            double term
            double [:] xt = xt_arr
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:,:] lambdas = self._lambdas

        # Compute lambda
        if self.constant_terms.size > 0:
            Ni = xt_arr[(nClass-1)*M:] # update Ni

        for i in range(infection_terms.shape[0]):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                lambdas[i, m] = 0
                for n in range(M):
                    index = n + M*infective_index
                    lambdas[i, m] += CM[m,n]*xt[index]/Ni[n]

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


    def simulate(self, x0, contactMatrix, Tf, Nf, Ti=0,
                     integrator='odeint', maxNumSteps=100000, **kwargs):
        """
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        data: dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.
        """

        if type(x0) == list:
            x0 = np.array(x0)

        if type(x0) == np.ndarray:
            if x0.size != self.nClass*self.M:
                raise Exception("Initial condition x0 has the wrong dimensions. Expected x0.size=%s."
                    % ( self.nClass*self.M) )
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

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        X, time_points = self.simulateRHS(rhs0, x0, Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        ## add parameters as a dictionary to this
        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M }
        param_dict = self.make_parameters_dict()
        data.update(param_dict)
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

        if model_class_key != 'R':
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
cdef class SEI5R(IntegratorsClass):
    warnings.warn('SEI5R not supported', DeprecationWarning)
    """
    DEPRECATED.

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gE: float
            rate of removal from exposeds individuals.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gIh: float
            rate of removal for hospitalised individuals.
        gIc: float
            rate of removal for idividuals in intensive care.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        fh  : float
            fraction by which hospitalised individuals are isolated.
        sa: float, np.array (M,)
            daily arrival of new susceptables.
            sa is rate of additional/removal of population by birth etc
        hh: float, np.array (M,)
            fraction hospitalised from Is
        cc: float, np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: float, np.array (M,)
            mortality rate in intensive care
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 8 -1  # only 7 input classes
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # removal rate of E class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIh   = parameters['gIh']                      # removal rate of Is
        self.gIc   = parameters['gIc']                      # removal rate of Ih
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds
        alpha      = parameters['alpha']                    # fraction of asymptomatics
        sa         = parameters['sa']                       # rate of additional/removal of population by birth etc
        hh         = parameters['hh']                       # fraction of infected who gets hospitalized
        cc         = parameters['cc']                       # fraction of hospitalized who endup in ICU
        mm         = parameters['mm']                       # mortality fraction from ICU

        self.paramList = parameters

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 8*self.M, dtype=DTYPE)           # right hand side

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


    cpdef rhs(self, xt, tt):
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
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS + sa[i]                    # \dot S
            dxdt[i+M]   = rateS  - gE*E[i]                  # \dot E
            dxdt[i+2*M] = ce1*E[i] - gIa*Ia[i]              # \dot Ia
            dxdt[i+3*M] = ce2*E[i] - gIs*Is[i]              # \dot Is
            dxdt[i+4*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # \dot Ih
            dxdt[i+5*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # \dot Ic
            dxdt[i+6*M] = gIc*mm[i]*Ic[i]                   # \dot Im
            dxdt[i+7*M] = sa[i] - gIc*mm[i]*Im[i]           # \dot Ni
        return


    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0,
                    integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Ih0: np.array
            Initial number of hospitalized infectives.
        Ic0: np.array
            Initial number of ICU infectives.
        Im0: np.array
            Initial number of mortality.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator
        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.
        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'fsa':self.fsa, 'fh':self.fh,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 2*self.M:3*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 3*self.M:4*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': hospitalized population time series
        """
        X  = data['X']
        Ih = X[:, 4*self.M:5*self.M]
        return Ih


    def Ic(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': ICU hospitalized population time series
        """
        X  = data['X']
        Ic = X[:, 5*self.M:6*self.M]
        return Ic


    def Im(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': mortality time series
        """
        X  = data['X']
        Im = X[:, 6*self.M:7*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data: data files
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
        data: data files
        Returns
        -------
             R: Removed population time series
            R = N(t) - (S + E + Ia + Is + Ih + Ic)
        """
        X = data['X']
        R =  X[:, 7*self.M:8*self.M] - X[:, 0:self.M]  - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
                                                       - X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M]

        return R



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R(IntegratorsClass):
    warnings.warn('SEAI5R not supported', DeprecationWarning)
    """
    DEPRECATED.

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        fsa: float
            fraction by which symptomatic individuals self isolate.
        gE: float
            rate of removal from exposeds individuals.
        gA: float
            rate of removal from activated individuals.
        gIh: float
            rate of hospitalisation of infected individuals.
        gIc: float
            rate hospitalised individuals are moved to intensive care.
        sa: float, np.array (M,)
            daily arrival of new susceptables.
            sa is rate of additional/removal of population by birth etc
        hh: float, np.array (M,)
            fraction hospitalised from Is
        cc: float, np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: float, np.array (M,)
            mortality rate in intensive care
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class
    """

    def __init__(self, parameters, M, Ni):
        self.nClass= 9 - 1#only 8 input classes
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # removal rate of E class
        self.gA    = parameters['gA']                       # removal rate of A class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIh   = parameters['gIh']                      # removal rate of Is
        self.gIc   = parameters['gIc']                      # removal rate of Ih
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds

        self.paramList = parameters

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        sa         = parameters['sa']                       # rate of additional/removal of population by birth etc
        hh         = parameters['hh']                       # fraction of infected who gets hospitalized
        cc         = parameters['cc']                       # fraction of hospitalized who endup in ICU
        mm         = parameters['mm']                       # mortality fraction from ICU

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # individuals in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt  = np.zeros( 9*self.M, dtype=DTYPE)           # right hand side

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


    cpdef rhs(self, xt, tt):
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
            double [:] sa   = self.sa
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] dxdt = self.dxdt

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            dxdt[i]     = -rateS + sa[i]                    # \dot S
            dxdt[i+M]   = rateS  - gE*E[i]                  # \dot E
            dxdt[i+2*M] = gE*E[i]  - gA*A[i]                # \dot A
            dxdt[i+3*M] = gAA*A[i] - gIa*Ia[i]              # \dot Ia
            dxdt[i+4*M] = gAS*A[i] - gIs*Is[i]              # \dot Is
            dxdt[i+5*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # \dot Ih
            dxdt[i+6*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # \dot Ic
            dxdt[i+7*M] = gIc*mm[i]*Ic[i]                   # \dot Im
            dxdt[i+8*M] = sa[i] - gIc*mm[i]*Im[i]           # \dot Ni
        return


    def simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, Ti=0,
                 integrator='odeint', maxNumSteps=100000, **kwargs):
        """
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
        Ih0: np.array
            Initial number of hospitalized infectives.
        Ic0: np.array
            Initial number of ICU infectives.
        Im0: np.array
            Initial number of mortality.
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
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator
        Returns
        -------
        dict
             X: output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.
        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        x0=np.concatenate((S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni))
        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data={'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                     'fsa':self.fsa, 'fh':self.fh,
                     'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
        return data


    def S(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             E: Exposed population time series
        """
        X = data['X']
        E = X[:, self.M:2*self.M]
        return E


    def A(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             A: Activated population time series
        """
        X = data['X']
        A = X[:, 2*self.M:3*self.M]
        return A


    def Ia(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ia': Asymptomatics population time series
        """
        X  = data['X']
        Ia = X[:, 3*self.M:4*self.M]
        return Ia


    def Is(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Is': symptomatics population time series
        """
        X  = data['X']
        Is = X[:, 4*self.M:5*self.M]
        return Is


    def Ih(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': hospitalized population time series
        """
        X  = data['X']
        Ih = X[:, 5*self.M:6*self.M]
        return Ih


    def Ic(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': ICU hospitalized population time series
        """
        X  = data['X']
        Ic = X[:, 6*self.M:7*self.M]
        return Ic


    def Im(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            'Ic': mortality time series
        """
        X  = data['X']
        Im = X[:, 7*self.M:8*self.M]
        return Im


    def population(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
            population
        """
        X = data['X']
        ppln = X[:, 8*self.M:9*self.M]
        return ppln


    def R(self,  data):
        """
        Parameters
        ----------
        data: data files
        Returns
        -------
             R: Removed population time series
            R = N(t) - (S + E + A + Ia + Is + Ih + Ic)
        """
        X = data['X']
        R = X[:,8*self.M:9*self.M] - X[:, 0:self.M] - X[:, self.M:2*self.M] - X[:, 2*self.M:3*self.M] - X[:, 3*self.M:4*self.M] \
                                                    - X[:,4*self.M:5*self.M] - X[:,5*self.M:6*self.M] - X[:, 6*self.M:7*self.M]
        return R
