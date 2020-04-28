import  numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
#@cython.wraparound(False)
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
        double beta, gIa, gIs, fsa
        int N, M, k_tot, N_protocols
        np.ndarray Ni, CM, FM, drpdt, alpha
        #list events, events_, contactMatrices

    def __init__(self, parameters, M, Ni):
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.k_tot = 3             # total number of degrees of freedom we explicitly track

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.FM    = np.zeros( self.M, dtype = DTYPE)           # seed function F
        self.drpdt = np.zeros( 3*self.M, dtype=DTYPE)           # right hand side

        alpha      = parameters.get('alpha')                    # fraction of asymptomatic infectives
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
            double [:] X    = self.drpdt
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
                         Tf, Nf, Ti=0,
                        seedRate=None):
        cdef:
            int cur_index_i = 0,  current_protocol_index
            double t_i, cur_t_f
            int M = self.M, i, j
            list cur_list, events_out = []

        from scipy.integrate import solve_ivp

        def rhs0(t,rp):
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            self.rhs(rp, t)
            return self.drpdt


        t_eval = np.linspace(Ti,Tf,endpoint=True,num=Nf)
        y_eval = np.zeros([len(t_eval),self.k_tot],dtype=float)

        cur_t_f = 0 # final time of current iteration
        cur_t_eval = t_eval # time interval for current iteration
        cur_y0 = np.concatenate((S0, Ia0, Is0)) # initial condition for current iteration
        current_protocol_index = 0 # protocol we are using for current iteration
        self.CM = contactMatrices[current_protocol_index]

        while (cur_t_f < Tf):
            # Make a copy of the list of events, replacing the event
            # corresponding to the current protocol with a dummy.
            cur_list = []
            for i,e in enumerate(events):
                if i == current_protocol_index:
                    cur_list.append(lambda t, x: 1.)
                else:
                    cur_list.append(e)
                (cur_list[-1]).direction = e.direction
                (cur_list[-1]).terminal = True
            # solve dynamical equation numerically until an event occurs
            sol = solve_ivp(fun=rhs0,
                          t_span=[cur_t_eval[0], cur_t_eval[-1]],
                          y0=cur_y0,
                         t_eval=cur_t_eval,
                         method='RK23', # RK45 is standard, but doesn't seem to work properly
                         events=cur_list
                         )
            # Find current event
            for i,e in enumerate(sol.t_events):
                if len(e) > 0:
                   current_protocol_index = i
                   events_out.append([sol.t_events[current_protocol_index][0], current_protocol_index ])
                   #print('At time {0:3.2f}, event {1} occured.'.format(sol.t_events[current_protocol_index][0],
                   #                     current_protocol_index))
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

        data={'X':y_eval, 't':t_eval, 'N':self.N, 'M':self.M,'alpha':self.alpha,
        'events_occured':events_out,
        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data




#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R:
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
        self.drpdt = np.zeros( 9*self.M, dtype=DTYPE)           # right hand side

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
            double [:] X    = self.drpdt

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
                         Tf, Nf, Ti=0):
        cdef:
            int cur_index_i = 0,  current_protocol_index
            double t_i, cur_t_f
            int M = self.M, i, j
            list cur_list, events_out = []

        from scipy.integrate import solve_ivp

        def rhs0(t,rp):
            self.rhs(rp, t)
            return self.drpdt


        t_eval = np.linspace(Ti,Tf,endpoint=True,num=Nf)
        y_eval = np.zeros([len(t_eval),self.k_tot],dtype=float)

        cur_t_f = 0 # final time of current iteration
        cur_t_eval = t_eval # time interval for current iteration
        cur_y0 = np.concatenate((S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni)) # initial condition for current iteration
        current_protocol_index = 0 # protocol we are using for current iteration
        self.CM = contactMatrices[current_protocol_index]

        while (cur_t_f < Tf):
            # Make a copy of the list of events, replacing the event
            # corresponding to the current protocol with a dummy.
            cur_list = []
            for i,e in enumerate(events):
                if i == current_protocol_index:
                    cur_list.append(lambda t, x: 1.)
                else:
                    cur_list.append(e)
                (cur_list[-1]).direction = e.direction
                (cur_list[-1]).terminal = True
            # solve dynamical equation numerically until an event occurs
            sol = solve_ivp(fun=rhs0,
                          t_span=[cur_t_eval[0], cur_t_eval[-1]],
                          y0=cur_y0,
                         t_eval=cur_t_eval,
                         method='RK23', # RK45 is standard, but doesn't seem to work properly
                         events=cur_list
                         )
            # Find current event
            for i,e in enumerate(sol.t_events):
                if len(e) > 0:
                   current_protocol_index = i
                   events_out.append([sol.t_events[current_protocol_index][0], current_protocol_index ])
                   #print('At time {0:3.2f}, event {1} occured.'.format(sol.t_events[current_protocol_index][0],
                   #                     current_protocol_index))
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

        data={'X':y_eval, 't':t_eval, 'N':self.N, 'M':self.M,'alpha':self.alpha,
        'events_occured':events_out,
        'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        return data
