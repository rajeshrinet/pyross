import  numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
import pyross.stochastic



cdef class control_integration:
    """
    Integrator class to implement control through changing the contact matrix
    as a function of the current state.

    Methods
    -------
    simulate_deteministic : Performs a deterministic simulation.
    """
    cdef:
        readonly int N, M
        int nClass
        double beta
        np.ndarray FM, Ni, CM, dxdt

    cdef rhs(self, rp, tt):
        return

    def simulate_deterministic(self, y0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         events_repeat=False,
                         events_subsequent=True): # only relevant of repeat_events = False
        """
        Performs detemrinistic numerical integration

        Parameters
        ----------
        y0 : np.array
            Inital state of the system.
        events : list
            List of events that the current state can satisfy to change behaviour
            of the contact matrix.
            contactMatricies
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
        Tf : float
            End time for integrator.
        Nf : Int
            Number of time points to evaluate at.
        Ti : float, optional
            Start time for integrator. The default is 0.
        seedRate : python function, optional
            Seeding of infectives. The default is None.
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO

        Returns
        -------
        y_eval : np.array(len(t), len(y0))
            Numerical integration solution.
        t_eval : np.array
            Corresponding times at which X is evaluated at.
        event_out : list
            List of events that occured during the run.

        """
        cdef:
            int cur_index_i = 0,  current_protocol_index
            double t_i, cur_t_f
            int M = self.M, i, j, N_events
            #double [:,:] CM = self.CM
            list cur_list, events_out = [], list_of_available_events
            np.ndarray cur_y0 = y0.copy()

        from scipy.integrate import solve_ivp

        def rhs0(t,rp):
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(rp, t)
            return self.dxdt

        # create a list of all events that are available
        # at the beginning  of the simulation
        N_events = len(events)
        if N_events == 1:
            list_of_available_events = []
        else:
            if events_subsequent:
                    list_of_available_events = [1]
            else:
                list_of_available_events = []
                for i in range(N_events):
                    list_of_available_events.append(i)


        t_eval = np.linspace(Ti,Tf,endpoint=True,num=Nf)
        y_eval = np.zeros([len(t_eval),self.nClass*self.M],dtype=float)

        cur_t_f = 0 # final time of current iteration
        cur_t_eval = t_eval # time interval for current iteration

        current_protocol_index = 0 # start with first contact matrix in list
        self.CM = contactMatrices[current_protocol_index]

        while (cur_t_f < Tf):
            # Ceate a list with the available events
            cur_list = []
            for j, cur_index in enumerate(list_of_available_events):
                cur_list.append(events[cur_index])
                try:
                    (cur_list[-1]).direction = events[cur_index].direction
                except:
                    pass
                (cur_list[-1]).terminal = True
            # solve dynamical equation numerically until an event occurs
            sol = solve_ivp(fun=rhs0,
                          t_span=[cur_t_eval[0], cur_t_eval[-1]],
                          y0=cur_y0,
                         t_eval=cur_t_eval,
                         method='RK45',  #'RK23', # RK45 is standard, but doesn't seem to work properly
                         events=cur_list
                         )
            # Find current event
            for i,e in enumerate(sol.t_events):
                if len(e) > 0:
                    #
                    #print('i =',i)
                    #print("list_of_available_events =",list_of_available_events)
                    current_protocol_index = list_of_available_events[i]
                    #print("current_protocol_index =",current_protocol_index)

                    if events_repeat:
                        list_of_available_events = []
                        for j in range(0,N_events):
                            if j != current_protocol_index:
                                list_of_available_events.append(j)
                    else:
                        if events_subsequent:
                            if current_protocol_index + 1 < N_events:
                                list_of_available_events = [ current_protocol_index+1 ]
                            else:
                                list_of_available_events = []
                        else:
                            if current_protocol_index in list_of_available_events:
                                list_of_available_events.remove(current_protocol_index)
                    events_out.append([sol.t_events[i][0], current_protocol_index ])
                    #print('At time {0:3.2f}, event {1} occured.'.format(sol.t_events[current_protocol_index][0],
                    #                     current_protocol_index))
                    #print("list_of_available_events =",list_of_available_events)
                    self.CM = contactMatrices[current_protocol_index]
                    break
            # Add current simulational result to trajectory
            if sol.t[-1] == Tf:
                # if no event has occured, we have just obtained
                # the rest of the time series and are finished.
                y_eval[cur_index_i:] = (sol.y).T
                break
            else:
                # if an event has occured, then we add the time series
                # up to the event and prepare an initial condition for
                # the next iteration
                cur_index_f = cur_index_i + len(sol.t)
                y_eval[cur_index_i:cur_index_f] = (sol.y).T
                #
                cur_t_f = t_eval[cur_index_i] # current final time
                cur_index_i = cur_index_f - 1 # initial index for next iteration
                # time array for next iteration:
                cur_t_eval = np.linspace( t_eval[cur_index_i], Tf,
                                          endpoint=True,
                                          num= len(t_eval[cur_index_i:]))
                # initial condition for next iteration
                cur_y0 = np.array( sol.y[:,-1] )

        return y_eval, t_eval, events_out







#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(control_integration):
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
    """
    cdef:
        double gIa, gIs, fsa
        np.ndarray alpha
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters

        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.nClass = 3             # total number of degrees of freedom we explicitly track

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( self.nClass*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

        #

    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, aa, bb
            double fsa=self.fsa, gIs=self.gIs
            double [:] S    = rp[0  :M]
            double [:] Ia   = rp[M  :2*M]
            double [:] Is   = rp[2*M:3*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt
            double [:] alpha= self.alpha


        for i in range(M):
            bb=0
            for j in range(M):
                 bb += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa - FM[i]                                # rate S  -> Ia, Is
            X[i+M]   = alpha[i]*aa     - gIa*Ia[i] + alpha[i]    *FM[i]      # rate Ia -> R
            X[i+2*M] = (1-alpha[i])*aa - gIs*Is[i] + (1-alpha[i])*FM[i]      # rate Is -> R
        return


    def simulate(self, S0, Ia0, Is0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data


        if method.lower()=='deterministic':
            y0 = np.concatenate((S0, Ia0, Is0)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data = {'X':y_eval, 't':t_eval, 'events_occured':events_out,
                      'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
            return data
        else:
            model = pyross.stochastic.SIR(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0, Ia0=Ia0, Is0=Is0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)





#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkIkR(control_integration):
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    method of k-stages of Ia, Is, E
    See: Lloyd, Theoretical Population Biology 60, 59􏰈71 (2001), doi:10.1006􏰅tpbi.2001.1525.
    Attributes
    ----------
    parameters: dict
        Contains the following keys:
            alpha : float
                fraction of infected who are asymptomatic.
            beta : float
                rate of spread of infection.
            gIa: float
                rate of removal from asymptomatic infected individuals.
            gIs: float
                rate of removal from symptomatic infected individuals.
            gE : float
                rate of removal from exposed individuals.
            kI: int
                number of stages of asymptomatic infectives.
            kI: int
                number of stages of symptomatic infectives.
            kE : int
                number of stages of exposed.
    M : int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        double gE, gIa, gIs, fsa
        int kI, kE
        np.ndarray alpha
        dict params

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters['beta']                         # infection rate
        self.gE    = parameters['gE']                           # recovery rate of E
        self.gIa   = parameters['gIa']                           # recovery rate of Ia
        self.gIs   = parameters['gIs']                           # recovery rate of Is
        self.kI    = parameters['kI']                           # number of stages
        self.fsa   = parameters['fsa']                          # the self-isolation parameter
        self.kE    = parameters['kE']
        self.nClass= self.kI + self.kI + self.kE + 1

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
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


    cdef rhs(self, xt, tt):
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



    def simulate(self, S0, E0, Ia0, Is0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data


        if method.lower()=='deterministic':
            y0 = np.concatenate((S0, E0, Ia0, Is0)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data = {'X':y_eval, 't':t_eval, 'events_occured':events_out,
                      'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                      'fsa':self.fsa, 'kI':self.kI, 'kE':self.kE ,
                        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
            return data
        else:
            raise RuntimeError("Stochastic control not yet implemented for SEkIkIkR model.")
            #model = pyross.stochastic.SEkIkIkR(self.params, self.M, self.Ni)
            #return model.simulate_events(S0=S0, Ia0=Ia0, Is0=Is0,
            #                    events=events,contactMatrices=contactMatrices,
            #                    Tf=Tf, Nf=Nf,
            #                    method=method,
            #                    nc=nc,epsilon = epsilon,
            #                    tau_update_frequency = tau_update_frequency,
            #                      events_repeat=events_repeat,
            #                      events_subsequent=events_subsequent,
            #                    seedRate=seedRate)


    def S(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'S' : Susceptible population time series
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
            'E' : Exposed population time series
        """
        kE = data['kE']
        X = data['X']
        E = X[:, self.M:(1+self.kE)*self.M]
        return E


    def Ia(self,  data):
        """
        Parameters
        ----------
        data : data files

        Returns
        -------
            'Is' : symptomatics population time series
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
        data : data files

        Returns
        -------
            'Is' : symptomatics population time series
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
        data : data files

        Returns
        -------
            'R' : Recovered population time series
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





#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIRS(control_integration):
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
    """
    cdef:
        double gIa, gIs, fsa, ep
        np.ndarray alpha, sa, iaa
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics

        self.nClass = 4

        self.ep    = parameters['ep']                       # fraction of recovered who is susceptible
        sa         = parameters['sa']                       # daily arrival of new susceptibles
        iaa        = parameters['iaa']                      # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
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


    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, aa, bb
            double fsa=self.fsa,gIs=self.gIs, ep=self.ep
            double [:] S    = rp[0  :M]
            double [:] Ia   = rp[M  :2*M]
            double [:] Is   = rp[2*M:3*M]
            double [:] Ni   = rp[3*M:4*M]
            double [:,:] CM = self.CM
            double [:] sa   = self.sa
            double [:] iaa  = self.iaa
            double [:] X    = self.dxdt
            double [:] alpha= self.alpha

        for i in range(M):
            bb=0
            for j in range(M):
                 bb += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa + sa[i] + ep*(gIa*Ia[i] + gIs*Is[i])       # rate S  -> Ia, Is and also return
            X[i+M]   = alpha[i]*aa - gIa*Ia[i] + iaa[i]                 # rate Ia -> R
            X[i+2*M] = (1-alpha[i])*aa - gIs*Is[i]                          # rate Is -> R
            X[i+3*M] = sa[i] + iaa[i]                                 # rate of Ni
        return


    def simulate(self, S0, Ia0, Is0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower()=='deterministic':
            y0 = np.concatenate((S0, Ia0, Is0)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data = {'X':y_eval, 't':t_eval, 'events_occured':events_out,
                      'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
            return data
        else:
            raise RuntimeError("Stochastic control not yet implemented for SIRS model.")
            '''
            model = pyross.stochastic.SIRS(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0, Ia0=Ia0, Is0=Is0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)
            ''';







#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(control_integration):
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
    """
    cdef:
        double gIa, gIs, gE, fsa
        np.ndarray alpha
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gE    = parameters['gE']                       # recovery rate of E
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.nClass = 4

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( 4*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, gIa=self.gIa, gIs=self.gIs, aa, bb
            double fsa=self.fsa, gE=self.gE, ce1, ce2
            double [:] S    = rp[0  :  M]
            double [:] E    = rp[  M:2*M]
            double [:] Ia   = rp[2*M:3*M]
            double [:] Is   = rp[3*M:4*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt
            double [:] alpha= self.alpha

        for i in range(M):
            bb=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 bb += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa - FM[i]                                # rate S  -> E
            X[i+M]   = aa       - gE*  E[i] + FM[i]               # rate E  -> Ia, Is
            X[i+2*M] = ce1*E[i] - gIa*Ia[i]                       # rate Ia -> R
            X[i+3*M] = ce2*E[i] - gIs*Is[i]                       # rate Is -> R
        return


    def simulate(self, S0, E0, Ia0, Is0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower() =='deterministic':
            y0 = np.concatenate((S0, E0, Ia0, Is0)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data={'X':y_eval, 't':t_eval, 'events_occured':events_out,
                'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
            return data
        else:
            model = pyross.stochastic.SEIR(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0,E0=E0, Ia0=Ia0, Is0=Is0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)



#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI5R(control_integration):
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
    """
    cdef:
        double gIa, gIs, gIh, gIc, fsa, fh, gE
        np.ndarray alpha, sa, hh, cc, mm
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # recovery rate of E class
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gIh   = parameters['gIh']                      # recovery rate of Is
        self.gIc   = parameters['gIc']                      # recovery rate of Ih
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds

        self.nClass = 8

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.dxdt = np.zeros( 8*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

        sa         = parameters['sa']                       # daily arrival of new susceptibles
        self.sa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(sa)==1:
            self.sa = sa*np.ones(M)
        elif np.size(sa)==M:
            self.sa= sa
        else:
            print('sa can be a number or an array of size M')

        hh         = parameters['hh']                       # hospital
        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            print('hh can be a number or an array of size M')

        cc         = parameters['cc']                       # ICU
        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        mm         = parameters['mm']                       # mortality
        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')


    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double fsa=self.fsa, fh=self.fh, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1, ce2
            double [:] S    = rp[0  :M]
            double [:] E    = rp[M  :2*M]
            double [:] Ia   = rp[2*M:3*M]
            double [:] Is   = rp[3*M:4*M]
            double [:] Ih   = rp[4*M:5*M]
            double [:] Ic   = rp[5*M:6*M]
            double [:] Im   = rp[6*M:7*M]
            double [:] Ni   = rp[7*M:8*M]
            double [:,:] CM = self.CM

            double [:] alpha= self.alpha
            double [:] sa   = self.sa       #sa is rate of additional/removal of population by birth etc
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] X    = self.dxdt

        for i in range(M):
            bb=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 bb += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa + sa[i]                       # rate S  -> E
            X[i+M]   = aa  - gE*E[i]                     # rate E  -> Ia, Is
            X[i+2*M] = ce1*E[i] - gIa*Ia[i]              # rate Ia -> R
            X[i+3*M] = ce2*E[i] - gIs*Is[i]              # rate Is -> R, Ih
            X[i+4*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic
            X[i+5*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im
            X[i+6*M] = gIc*mm[i]*Ic[i]                   # rate of Im
            X[i+7*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return



    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower() =='deterministic':
            y0 = np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data={'X':y_eval, 't':t_eval, 'events_occured':events_out,
                  'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,
                  'beta':self.beta,'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE}
            return data
        else:
            model = pyross.stochastic.SEI5R(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0,E0=E0, Ia0=Ia0, Is0=Is0,
                                Ih0=Ih0, Ic0=Ic0, Im0=Im0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)





#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIkR(control_integration):
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
    cdef:
        double  gI
        int ki
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gI    = parameters['gI']                       # recovery rate of I
        self.ki    = parameters['kI']                        # number of stages

        self.nClass = self.ki+1

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( (self.ki+1)*self.M, dtype=DTYPE) # right hand side


    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, ki=self.ki
            double beta=self.beta, gI=self.ki*self.gI, aa, bb
            double [:] S    = rp[0  :M]
            double [:] I    = rp[M  :(ki+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt

        for i in range(M):
            bb=0
            for jj in range(ki):
                for j in range(M):
                    bb += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa - FM[i]
            X[i+M]   = aa - gI*I[i] + FM[i]

            for j in range(ki-1):
                X[i+(j+2)*M]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return



    def simulate(self,  S0, I0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower()=='deterministic':
            y0 = np.concatenate(( S0, I0 )) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
            data={'X':y_eval, 't':t_eval, 'events_occured':events_out,
              'Ni':self.Ni, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'k':self.ki }
            return data
        else:
            model = pyross.stochastic.SIkR(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0,I0=I0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)





#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEkIkR(control_integration):
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
    cdef:
        double gE, gI, fsa
        int ki, ke
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # recovery rate of E
        self.gI    = parameters['gI']                       # recovery rate of I
        self.ki    = parameters['kI']                       # number of stages
        self.ke    = parameters['kE']

        self.nClass = self.ki + self.ke + 1

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( self.nClass*self.M, dtype=DTYPE)           # right hand side


    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, ki=self.ki, ke = self.ke
            double beta=self.beta, gI=self.ki*self.gI, aa, bb
            double gE = self.ke * self.gE
            double [:] S    = rp[0  :M]
            double [:] E    = rp[M  :(ke+1)*M]
            double [:] I    = rp[(ke+1)*M  :(ke+ki+1)*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt

        for i in range(M):
            bb=0
            for jj in range(ki):
                for j in range(M):
                    bb += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa - FM[i]

            # If there is any E stage...
            if 0 != ke :
                # People removed from S are put in E[0]
                X[i+M+0] = aa - gE*E[i] + FM[i]

                # Propagate cases along the E stages
                for j in range(ke - 1) :
                    X[i + M +  (j+1)*M ] = gE * E[i+j*M] - gE * E[i+(j+1)*M]

                # Transfer cases from E[-1] to I[0]
                X[i + (ke+1)* M + 0] = gE * E[i+(ke-1)*M] - gI * I[i]

            # However, if there aren't any E stages
            else :
                # People removed from S are put in I[0]
                X[i + (ke+1)* M + 0] = aa + FM[i] - gI * I[i]

            # In both cases, propagate cases along the I stages.
            for j in range(ki-1):
                X[i+(ke+1)*M + (j+1)*M ]   = gI*I[i+j*M] - gI*I[i+(j+1)*M]
        return



    def simulate(self,  S0, E0, I0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        y0 = np.concatenate((S0, E0, I0)) # initial condition

        if method.lower() =='deterministic':
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
        else:
            raise RuntimeError("Stochastic control not yet implemented for SEkIkR model.")

        data={'X':y_eval, 't':t_eval, 'events_occured':events_out,
            'Ni':self.Ni, 'M':self.M, 'beta':self.beta,'gI':self.gI, 'k':self.ki }
        return data




#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIR(control_integration):
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
    """
    cdef:
        double gE, gA, gIa, gIs, fsa, gIh, gIc
        np.ndarray alpha
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # recovery rate of Ia
        self.gIs   = parameters['gIs']                      # recovery rate of Is
        self.gE    = parameters['gE']                       # recovery rate of E
        self.gA    = parameters['gA']                       # rate to go from A to Ia, Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.nClass = 5

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( self.nClass*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')

    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS

            double [:] S    = rp[0*M:M]
            double [:] E    = rp[1*M:2*M]
            double [:] A    = rp[2*M:3*M]
            double [:] Ia   = rp[3*M:4*M]
            double [:] Is   = rp[4*M:5*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt

            double [:] alpha= self.alpha

        for i in range(M):
            bb=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 bb += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa - FM[i]                          # rate S  -> E
            X[i+M]   =  aa      - gE*E[i] + FM[i]           # rate E  -> A
            X[i+2*M] = gE* E[i] - gA*A[i]                   # rate A  -> Ia, Is
            X[i+3*M] = gAA*A[i] - gIa     *Ia[i]            # rate Ia -> R
            X[i+4*M] = gAS*A[i] - gIs     *Is[i]            # rate Is -> R
        return



    def simulate(self, S0, E0, A0, Ia0, Is0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,seedRate=None,
                         method='deterministic',
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        y0 = np.concatenate((S0, E0, A0, Ia0, Is0)) # initial condition

        if method.lower() =='deterministic':
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,seedRate=seedRate,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent)
        else:
            raise RuntimeError("Stochastic control not yet implemented for SEAIR model.")

        data={'X':y_eval, 't':t_eval, 'events_occured':events_out,'fsa':self.fsa,
              'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA}
        return data








#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R(control_integration):
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
    """
    cdef:
        double gE, gA, gIa, gIs, fsa, gIh, gIc, fh
        np.ndarray alpha, sa, hh, cc, mm
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters
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
        self.Ni    = Ni.copy()

        self.nClass = 9

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( self.nClass*self.M, dtype=DTYPE)           # right hand side

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


    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double fsa=self.fsa, fh=self.fh, gE=self.gE, gA=self.gA
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double gAA, gAS
            double [:] S    = rp[0  :M]
            double [:] E    = rp[M  :2*M]
            double [:] A    = rp[2*M:3*M]
            double [:] Ia   = rp[3*M:4*M]
            double [:] Is   = rp[4*M:5*M]
            double [:] Ih   = rp[5*M:6*M]
            double [:] Ic   = rp[6*M:7*M]
            double [:] Im   = rp[7*M:8*M]
            double [:] Ni   = rp[8*M:9*M]
            double [:,:] CM = self.CM

            double [:] alpha= self.alpha
            double [:] sa   = self.sa       #sa is rate of additional/removal of population by birth etc
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] X    = self.dxdt

        for i in range(M):
            bb=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 bb += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa + sa[i]                       # rate S  -> E
            X[i+M]   = aa  - gE*E[i]                     # rate E  -> A
            X[i+2*M] = gE*E[i]  - gA*A[i]                # rate A  -> I
            X[i+3*M] = gAA*A[i] - gIa*Ia[i]              # rate Ia -> R
            X[i+4*M] = gAS*A[i] - gIs*Is[i]              # rate Is -> R, Ih
            X[i+5*M] = gIs*hh[i]*Is[i] - gIh*Ih[i]       # rate Ih -> R, Ic
            X[i+6*M] = gIh*cc[i]*Ih[i] - gIc*Ic[i]       # rate Ic -> R, Im
            X[i+7*M] = gIc*mm[i]*Ic[i]                   # rate of Im
            X[i+8*M] = sa[i] - gIc*mm[i]*Im[i]           # rate of Ni
        return

    def simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,
                         method='deterministic',seedRate=None,
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower() =='deterministic':
            y0 = np.concatenate((S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,
                                  events_repeat=events_repeat,seedRate=seedRate,
                                  events_subsequent=events_subsequent)
            data = {'X':y_eval, 't':t_eval, 'events_occured':events_out,
                          'Ni':self.Ni, 'M':self.M,
                          'alpha':self.alpha, 'beta':self.beta,
                          'gIa':self.gIa,'gIs':self.gIs,
                          'gIh':self.gIh,'gIc':self.gIc,
                          'fsa':self.fsa,'fh':self.fh,
                          'gE':self.gE,'gA':self.gA,
                          'sa':self.sa,'hh':self.hh,
                          'mm':self.mm,'cc':self.cc,
                          }
            return data
        else:
            model = pyross.stochastic.SEAI5R(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0,E0=E0, A0=A0, Ia0=Ia0, Is0=Is0,
                                Ih0=Ih0, Ic0=Ic0,Im0=Im0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)




#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(control_integration):
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
    """
    cdef:
        double gIa, gIs, gE, gA, fsa
        double tE, tA, tIa, tIs
        np.ndarray alpha
        dict params

    def __init__(self, parameters, M, Ni):
        self.params = parameters

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

        self.nClass = 6

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.dxdt = np.zeros( self.nClass*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            print('alpha can be a number or an array of size M')



    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, aa, bb
            double tE=self.tE, tA=self.tA, tIa=self.tIa, tIs=self.tIs
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs, gA=self.gA
            double gAA, gAS

            double [:] S    = rp[0*M:M]
            double [:] E    = rp[1*M:2*M]
            double [:] A    = rp[2*M:3*M]
            double [:] Ia   = rp[3*M:4*M]
            double [:] Is   = rp[4*M:5*M]
            double [:] Q    = rp[5*M:6*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:]   FM = self.FM
            double [:] X    = self.dxdt

            double [:] alpha= self.alpha

        for i in range(M):
            bb=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                 bb += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            aa = bb*S[i]
            X[i]     = -aa      - FM[i]        # rate S  -> E, Q
            X[i+M]   =  aa      - (gE+tE)     *E[i] + FM[i]        # rate E  -> A, Q
            X[i+2*M] = gE* E[i] - (gA+tA     )*A[i]                # rate A  -> Ia, Is, Q
            X[i+3*M] = gAA*A[i] - (gIa+tIa   )*Ia[i]               # rate Ia -> R, Q
            X[i+4*M] = gAS*A[i] - (gIs+tIs   )*Is[i]               # rate Is -> R, Q
            X[i+5*M] = tE*E[i]+tA*A[i]+tIa*Ia[i]+tIs*Is[i] # rate of Q
        return



    def simulate(self, S0, E0, A0, Ia0, Is0, Q0,
                events, contactMatrices,
                         Tf, Nf, Ti=0,
                         method='deterministic',seedRate=None,
                         events_repeat=False,
                         events_subsequent=True,
                         int nc=30, double epsilon = 0.03,
                        int tau_update_frequency = 1):
        cdef:
            np.ndarray y_eval, t_eval, y0
            dict data

        if method.lower() == 'deterministic':
            y0 = np.concatenate((S0, E0, A0, Ia0, Is0, Q0)) # initial condition
            y_eval, t_eval, events_out = self.simulate_deterministic(y0=y0,
                                  events=events,contactMatrices=contactMatrices,
                                  Tf=Tf,Nf=Nf,Ti=Ti,
                                  events_repeat=events_repeat,seedRate=seedRate,
                                  events_subsequent=events_subsequent)
            data = {'X':y_eval, 't':t_eval, 'events_occured':events_out,
                    'fsa':self.fsa,
                    'Ni':self.Ni, 'M':self.M,'alpha':self.alpha,'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,'gE':self.gE,'gA':self.gA,
                      'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
            return data
        else:
            model = pyross.stochastic.SEAIRQ(self.params, self.M, self.Ni)
            return model.simulate_events(S0=S0,E0=E0, A0=A0,
                                Ia0=Ia0, Is0=Is0,Q0=Q0,
                                events=events,contactMatrices=contactMatrices,
                                Tf=Tf, Nf=Nf,
                                method=method,
                                nc=nc,epsilon = epsilon,
                                tau_update_frequency = tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                seedRate=seedRate)
