import  numpy as np 
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
import warnings
from scipy.special import legendre
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import nlopt

DTYPE   = np.float

cdef class CommonMethods:
    """
    Parent class used for all classes listed below. 
    It includes:
    a) Integrators used by various deterministic models listed below.
    b) Method to get time series of S, etc by passing a dict of data.
    c) Method to set the contactMatrix array, CM
    """
    
    cdef:
        readonly int N, M, kI, kE, nClass
        readonly float Ttsi, dtsi, beta_params, gI_params
        readonly np.ndarray population, beta, gI, Ni, CM
        readonly dict paramList, readData
    
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



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
# cdef class SIR(CommonMethods):
class SIR:
    """
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
        Ttsi: float
            tsi cut-off
        M: int
            Number of compartments of individual for each class.
            ==len(contactMatrix)
        Ni: np.array(M, )
            Initial number in each compartment and class
    """

    def __init__(self, parameters, beta_fun, gI_fun):
        # self.beta  = parameters['beta']  # Infection rate
        # self.gI    = parameters['gI']    # Removal rate of I
        self.kI          = parameters['kI']    # Number of compartments
        self.Ttsi        = parameters['Ttsi']  # tsi cut-off
        self.M           = parameters['M']     # Number of age classes$
        self.N           = parameters['N']     # Full population size
        self.Ni          = np.zeros( self.M, dtype=DTYPE)
        self.Ni          = parameters['Ni']    # Array of total number of people per age class

        self.beta_params = parameters['beta'] # Parameters for beta_fun
        self.gI_params   = parameters['gI']   # Parameters for gI_fun
 
        self.beta_fun = beta_fun              # beta FUNCTION
        self.gI_fun   = gI_fun                # gI FUNCTION
        
        self.nClass = self.kI + 1        # Total number of stages

        self.N     = np.sum(self.Ni)             # Population size
        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # Contact matrix C
        self.dtsi  = self.Ttsi/float(self.kI) # tsi time step

    def set_IC(self, I_list):
        """
        Set Initial Conditions from a list of infectives
        
        Argument:
        --------
        - I_list: list of tuples (int, int, float)
           Elements of the list are: (# infectives, age_class, tsi) 
        Returns:
        -------
        - S0: np.array(M)
        - I0: np.array(M*kI)
        """
        S0 = np.zeros(self.M)
        I0 = np.zeros(self.M*self.kI)
        for tup in I_list:
            if tup[2]<self.Ttsi:
                k = int(tup[2]/self.dtsi)
            else:
                k = self.kI-1
            I0[k + tup[1]*self.M] = tup[0]/float(self.N)/self.dtsi # should be a distribution
        for i in range(self.M):
            S0[i] = self.Ni[i] - np.sum([I0[k+i*self.M]*self.N*self.dtsi for k in range(self.kI)])
        S0 = S0/float(self.N)
        return S0, I0
            
    def set_beta(self):
        """
        Method to set beta(t, Ttsi, **args) as an array of size kI
        """
        beta = []
        for k in range(self.kI):
            beta.append(self.beta_fun(k*self.dtsi, self.Ttsi, self.beta_params))
        return beta

    def set_gI(self):
        """
        Method to set gI(t, Ttsi, **args) as an array of size kI
        """
        gI = []
        for k in range(self.kI):
            gI.append(self.gI_fun(k*self.dtsi, self.Ttsi, self.gI_params))
        return gI
            
        
    # cpdef trajectory(self, S, I):
    def trajectory(self, S, I):
        """
        Function to go one step forward using the deterministic integrator.
        We use a RK2/Crank-Nicolson finite difference scheme
        
        Parameters
        ----------
        S: np.array(M)
           Number of susceptibles
        I: np.array(kI*M, dtype=float)
           Infected population
       
        Returns
        -------
        Snext: np.array(M, dtype=float)
           Updated number of susceptibles
        Inext: np.array(kI*M, dtype=float)
           Updated infected population
        """
        CM = self.CM
        
        beta = self.set_beta()
        gI   = self.set_gI()
        
        Ip = np.zeros(self.kI*self.M)
        didt_p = np.zeros(self.M*(self.kI-1))
        didt_c = np.zeros(self.M*(self.kI-1))
        Inext = np.zeros(self.M*self.kI)
        lbda = np.zeros(self.M)
        lbda_p = np.zeros(self.M)
        Snext = np.zeros(self.M)
        
        dtsi = self.dtsi
        dt = dtsi # Enforce that tsi-discretisation
                  # and time-discretisation are the same!
       
        # 1. Explicit time step 
        for j in range(self.M):
            lbda[j] = 0 
            for k in range(self.kI-1):
                for l in range(self.M):
                    lbda[j] += CM[j,l]*0.5*(beta[k]*I[k+l*self.M] + beta[k+1]*I[k+1+l*self.M])*dtsi
        dsdt_p = - S*lbda 
        Sp = S + dsdt_p*dt
        Ip[0:self.M] = - dsdt_p
        for k in range(1,self.kI): # Advection
            for l in range(self.M):
                Ip[k] = I[k-1+l*self.M]
        for k in range(1, self.kI): # Recovery
            for l in range(self.M):
                didt_p[k-1+l*self.M] = - gI[k] * Ip[k+l*self.M] 
                Ip[k+l*self.M] = Ip[k+l*self.M] + didt_p[k-1+l*self.M]*dt
    
        # 2. Correction 
        for j in range(self.M):
            lbda_p[j] = 0 
            for k in range(self.kI-1):
                for l in range(self.M):
                    lbda_p[j] += 0.5*CM[j,l]*(beta[k]*I[k+l*self.M] + beta[k+1]*I[k+1+l*self.M])*dtsi
        dsdt_c = - Sp * lbda_p
        
        Snext = S + 0.5*dt*(dsdt_p + dsdt_c)
        
        Inext[0:self.M] = - dsdt_c
        for k in range(1, self.kI): # Advection
            for l in range(self.M):
                Inext[k+l*self.M] = I[k-1+l*self.M] 
        for k in range(1,self.kI):  # Recovery 
            for l in range(self.M):
                didt_c[k-1+l*self.M] = - gI[k] * Ip[k+l*self.M]
                Inext[k+l*self.M] = Inext[k+l*self.M] + 0.5*dt*(didt_p[k-1+l*self.M] + didt_c[k-1+l*self.M])
        return Snext, Inext 

    
    def simulate(self, S0, I0, contactMatrix, Tf, Ti=0):
        """
        Function to go one step forward using the deterministic integrator.
        We use a RK2/Crank-Nicolson finite difference scheme
        
        Parameters
        ----------
        S0: np.array(M)
           Initial number of susceptibles
        I0: np.array(kI*M, dtype=float)
           Initial Infected population
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j       
        Tf: float
            Final time of integrator
        Ti: float
            Initial time of integrator

        Returns
        -------
        S_traj: np.array((Nf, M), dtype=float)
           Time series of susceptibles
        I_traj: np.array((Nf, kI*M), dtype=float)
           Time series infected population
        """
    
        # Initialise:
        S = np.copy(S0)
        I = np.copy(I0)
        t = Ti
              
        S_traj = []
        I_traj = []
        t_traj = []
        
        S_traj.append(S0)
        I_traj.append(I0*self.dtsi) # distribution -> population (dimensionless)
        t_traj.append(t)
    
        ### MAIN LOOP ###
        while t<=Tf:
            self.CM = contactMatrix(t)
            Snext, Inext = self.trajectory(S, I)
            S = np.copy(Snext)
            I = np.copy(Inext)
            t += self.dtsi
            S_traj.append(Snext)
            I_traj.append(Inext*self.dtsi) # distribution -> population (dimensionless)
            t_traj.append(t)

        # Store trajectory within a dictionary
        data = {'S': S_traj,
                'I': I_traj,
                't': t_traj}
            
        return data

cdef class Simulator:
    """
    Simulator for a deterministic time-since infection model

    Methods
    -------
    solve_Predictor_Corrector: 

    solve_Galerkin: 
    """


    cdef:
        readonly dict parameters
        readonly str method
        readonly str galerkinIntegrator 
        readonly list IC 
        readonly np.ndarray phi_alpha, p_alpha

    def __init__(self, parameters, method='Predictor_Corrector', galerkinIntegrator='odeint'):
        self.parameters         = parameters
        self.method             = method
        self.galerkinIntegrator = galerkinIntegrator 



    def get_IC(self):
        parameters = self.parameters
        M = parameters['M']
        Ni = parameters['Ni']
        Nc = parameters['Nc']
        Nk = parameters['Nk']
        contactMatrix = parameters['contactMatrix']
        Cij = contactMatrix(0)
    
    
        Np = sum(Ni)            #total population size
    
        T    = parameters['T']
        Td   = parameters['Td']
        Tc   = parameters['Tc']
        tsi  = parameters['tsi']
        beta = parameters['beta']

        if M == 1:
            S_0 = np.array([1])
            I_0 = .001*np.interp(np.linspace(0, 2 ,Nk),tsi, 2**(tsi/Td))
            Ic_0 = np.zeros((Nc,M))
            IC = [S_0, I_0, Ic_0]
            return IC

        A = np.matmul(np.diag(Ni),Cij);  
        A = np.matmul(A,np.diag(1/Ni)); 
        max_eig_A = np.max(np.real(np.linalg.eigvals(A)))
        sp   = np.linspace(0,T,1000); lam = np.log(2)/Td;  #Growth rate
        rs   = max_eig_A*np.trapz(np.exp(-lam*sp)*np.interp(sp,tsi,beta),sp)
        beta = beta/rs       #now beta has been rescaled to give the correct (dimensional) doubling time
        
        #Initial susceptible is the whole population
        S_0 = Ni
    
        #get I_0 from linear stability analysis
        #find the fastest growing linear mode
        A = np.matmul(np.diag(Ni),Cij)
        A = np.matmul(A,np.diag(1/Ni))
        sp = np.linspace(0,T,1000)
        lam = np.log(2)/Td;
        A = A*np.trapz(np.exp(-lam*sp)*np.interp(sp,tsi,beta),sp)
        w, v = np.linalg.eig(-np.identity(M) + A)

        ep = 0.001/T*Np/M

        #now identify the largest eigenvalue/eigenvector...
        pos = np.where(w == np.amax(w))
        pos = pos[0][0]
        lam = T/Td*np.log(2)
        s = np.linspace(-1,1,Nk)
        I_0 = ep*np.abs(np.real(np.outer(np.exp(-lam*(s+1)),v[:,pos])))
 
        #just set Ic_0 to zero -- these numbers are too small to matter
        Ic_0 = np.zeros((Nc,M))
    
        #rescale all population sizes
        Ni = Ni/Np
        I_0 = I_0/Np*Tc  #recall that this is a number density (number infected per unit tsi)
        S_0 = S_0/Np
        if Nc > 0:
            Ic_0 = Ic_0/Np*Tc
        IC = [S_0, I_0, Ic_0]
        return IC
    
    
    
    
    
    
    def solve_Predictor_Corrector(self, tstart=0, hybrid=False):
        ''' Predictor/Corrector is a finite difference method described in the TSI report, section 2.5
             It has good properties for speed and accuracy and should be preferred in most applications
             Notable disadvantage is a lack of flexibility in time-stepping -- you must increment by
             the same time step every time.  Function evaluations at intermediate times can be found by
             interpolation.
        '''
        M  = self.parameters['M']                  
        Ni = self.parameters['Ni']                  
        Nc = self.parameters['Nc']                   
        Nk = self.parameters['Nk']                   
        Tf = self.parameters['Tf']                   
        T  = self.parameters['T']                   
        Td = self.parameters['Tc']                   
        Td = self.parameters['Td']                   
       
        tsi       = self.parameters['tsi']
        beta      = self.parameters['beta']                  
        tsi_sc    = self.parameters['tsi_sc']

        p_alpha   = self.parameters['p_alpha']
        phi_alpha = self.parameters['phi_alpha']
        Np        = np.sum(Ni)
        Ni = Ni/Np                  
        
        contactMatrix = self.parameters['contactMatrix']
        def Cij_t(t):
            return np.matmul(contactMatrix(t*Tc),np.diag(1/Ni))


        Tc = T/2 #rescaling of time, and tsi 
        #first step is to rescale beta to a value consistent with the given Td
        Cij = contactMatrix(0)
        if M > 1:
            A = np.matmul(np.diag(Ni),Cij)
            A = np.matmul(A,np.diag(1/Ni))
        else:
            A = Cij
        max_eig_A = np.max(np.real(np.linalg.eigvals(A)))
        sp = np.linspace(0,T,1000)
        lam = np.log(2)/Td;  #Growth rate
        rs = max_eig_A*np.trapz(np.exp(-lam*sp)*np.interp(sp,tsi,beta),sp)
        beta = beta/rs       #now beta has been rescaled to give the correct (dimensional) doubling time

        #nondimensionalize beta:
        beta = beta*Tc
        tsi = tsi/Tc - 1
        
        #rescale phi_alpha based on re-scaling of tsi:
        tsi_sc = tsi_sc/Tc - 1
        
        #rescale end time
        Tf = Tf/Tc
        tstart = tstart/Tc

        #rescale phi_alpha
        if Nc > 0:
            phi_alpha = phi_alpha*Tc

        S_0, I_0, Ic_0 = self.IC 
   
 
        #set up the discretization in s
        s = np.linspace(-1,1,Nk)
        h = 2/(Nk - 1)
    
        #find the timesteps
        nt = int(np.round(Tf/h)) + 1
        t = h*np.linspace(0,nt-1,nt)

        #weighted betas for numerical integration
        beta_n = h*np.interp(s,tsi,beta)
        beta_n[[0, Nk-1]] = beta_n[[0, Nk-1]]/2
    
        #weighted phi_alpha for numerical integration
        phi_alpha_n = np.zeros((Nc, Nk))
        for i in range(Nc):
            phi_alpha_n[i,:] = h*np.interp(s,tsi_sc,phi_alpha[i,:])
            phi_alpha_n[i,[0,Nk-1]] = phi_alpha_n[i,[0,Nk-1]]/2
    
        #weights for generic trapezoid integration
        w = h*np.ones(Nk)
        w[[0, Nk-1]] = w[[0, Nk-1]]/2
    
        #initialize variables
        S = S_0 + 0
        I = I_0 + 0
        Ic = Ic_0 + 0
    
        #initialize output vectors:
        S_t = np.zeros((M, nt))
        I_t = np.zeros((M, nt))
        Ic_t = np.zeros((Nc,M,nt))
    
        #set their starting values:
        S_t[:,0] = S_0
        I_t[:,0] = np.matmul(w,I_0)
        Ic_t[:,:,0] = Ic_0
    
        #initialize a few variables
        dIc_dt_e = np.zeros((Nc,M))
        dIc_dt_i = np.zeros((Nc,M))

        for i in (1 + np.arange(nt-1)):
    
            #explicit time step
            dSdt_e = -np.matmul(np.matmul(np.diag(S),Cij_t(tstart + t[i-1])),np.matmul(beta_n,I))
            Sp = S + h*dSdt_e

            for j in range(Nc):
                dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]

            I[1:Nk,:] = I[0:(Nk-1),:]
            I[0,:] = -dSdt_e

            #'implicit' step
            dSdt_i = -np.matmul(np.matmul(np.diag(Sp),Cij_t(tstart + t[i])),np.matmul(beta_n,I))
            S = S + h/2*(dSdt_e + dSdt_i)

            for j in range(Nc):
                dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]
    
            Ic = Ic + h/2*(dIc_dt_e + dIc_dt_i)
    
            I[0,:] = -dSdt_i
    
            #remember this timestep
            S_t[:,i]     = S
            I_t[:,i]     = np.matmul(w,I)
            Ic_t[:,:, i] = Ic 
    
        if not hybrid:
            return t, S_t, I_t, Ic_t
        else:
            return t, S_t, I_t, Ic_t, [S, I, Ic]



    
    def solve_Galerkin(self, atol=1e-4, rtol=1e-3, tstart=0, hybrid=False):
        '''The Galerkin method is defined in the TSI report, section 2.6
         It spectral accuracy in s and allows for adatptive timestepping in t
         For constant contact matrix, use 'odeint', otherise use 'Crank Nicolson'
             -Notable advantage over predictor/corrector is flexibility in time-stepping
             -Notable disadvantages include:
              (1) ill-suited to non-smooth dynamics (like most spectral methods)
              (2) must be solved as DAE when contact matrix is time-dependent (slow)
        
        For most practical purposes, we regard predictor/corrector as the preferred choice.
        '''
        M  = self.parameters['M']                  
        Ni = self.parameters['Ni']                   
        Nc = self.parameters['Nc']                   
        Nk = self.parameters['Nk']                   
        NL = self.parameters['NL']                   
        Tf = self.parameters['Tf']                   
        Tc = self.parameters['Tc']
        Td = self.parameters['Td']
        T  = self.parameters['T']                   
       
        tsi       = self.parameters['tsi']
        beta      = self.parameters['beta']                  
        tsi_sc    = self.parameters['tsi_sc']                  
        
        p_alpha   = self.parameters['p_alpha']
        phi_alpha = self.parameters['phi_alpha']

        galerkinIntegrator = self.galerkinIntegrator

        Np        = np.sum(Ni)
        Ni = Ni/Np                  
        
        contactMatrix = self.parameters['contactMatrix']
        def Cij_t(t):
            return np.matmul(contactMatrix(t*Tc),np.diag(1/Ni))


        Tc = T/2 #rescaling of time, and tsi 
        #first step is to rescale beta to a value consistent with the given Td
        Cij = contactMatrix(0)
        if M > 1:
            A = np.matmul(np.diag(Ni),Cij)
            A = np.matmul(A,np.diag(1/Ni))
        else:
            A = Cij
        max_eig_A = np.max(np.real(np.linalg.eigvals(A)))
        sp = np.linspace(0,T,1000)
        lam = np.log(2)/Td;  #Growth rate
        rs = max_eig_A*np.trapz(np.exp(-lam*sp)*np.interp(sp,tsi,beta),sp)
        beta = beta/rs       #now beta has been rescaled to give the correct (dimensional) doubling time
   
        #nondimensionalize beta:
        beta = beta*Tc
        tsi = tsi/Tc - 1
        
        #rescale phi_alpha based on re-scaling of tsi:
        tsi_sc = tsi_sc/Tc - 1
        
        #rescale end time
        Tf = Tf/Tc
        tstart = tstart/Tc

        #rescale phi_alpha
        if Nc > 0:
            phi_alpha = phi_alpha*Tc


        galerkinIntegrator  = self.galerkinIntegrator


        S_0, I_0, Ic_0 = self.IC
    
        #set up the discretization in s
        s = np.linspace(-1,1,1000)
        sk = np.linspace(-1,1,Nk)
    
    
        #initialize first timestep
        h = 1/Nk
    
        #weighted betas for numerical integration
        beta_n = np.zeros(NL)
        for i in range(NL):
            Pn = legendre(i)
            beta_n[i] = np.trapz(Pn(s)*np.interp(s,tsi,beta),s)
    
        #weighted phi_alpha for numerical integration
        phi_alpha_n = np.zeros((Nc, NL))
        for i in range(Nc):
            for j in range(NL):
                Pn = legendre(j)
                phi_alpha_n[i,j] = np.trapz(Pn(s)*np.interp(s,tsi_sc,phi_alpha[i,:]),s)
    
        #advection matrix
        b_nk = np.zeros((NL,NL))
        for i in range(NL):
            for j in range(NL):
                if np.mod(i + j,2) == 1 and j > i:
                    b_nk[i,j] = 2*i + 1
    
        #initialize variables
        S = S_0 + 0
        I = np.zeros((NL,M))
        for i in range(NL):
            Pn = legendre(i)
            for j in range(M):
                I[i,j] = np.trapz(Pn(s)*np.interp(s,sk,I_0[:,j]),s)*(2*i + 1)/2
    
        Ic = Ic_0 + 0
    
        #set their starting values:
        S_t = S_0
        I_t = 2*I[0]
        Ic_t = Ic_0
    
        if galerkinIntegrator == 'Crank Nicolson':
    
            #define a function to spit out time derivatives
            def get_dxdt(x, t):
                Cij = Cij_t(tstart + t)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                dxdt = np.zeros(len(x))
                for i in range(M):
                    dxdt[i] = -S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    dxdt[M + i*NL + np.array(range(NL))] = -np.matmul(b_nk,I[:,i])
                return dxdt
    
            #define a function to give 'residuals' based on current estimate of next time step
            def get_res(x, xp, t, dt):
                #Cij = Cij_t(tstart + t)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                Sp = xp[:M]
                Ip = np.transpose(np.reshape(xp[M:],(M,NL)))
                dxdt_e = get_dxdt(x,t)
                dxdt_i = get_dxdt(xp,t+dt)
                res = xp - x - dt/2*(dxdt_e + dxdt_i)
                for i in range(M):
                    res[M + i*NL + NL - 1] = np.matmul((-1.)**np.array(range(NL)),Ip[:,i]) + dxdt_i[i]
                return res
    
            #define a function to compute Jacobians.
            def get_J(x, t, dt):
                Cij = Cij_t(tstart + t + dt)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                J = np.zeros((len(x),len(x)))
                for i in range(M):
                    J[i,i] = 1 + dt/2*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    for j in range(M):
                        for k in range(NL):
                            J[i,M + j*NL + k] = dt/2*S[i]*Cij[i,j]*beta_n[k]
                for i in range(M):
                    q = np.identity(NL) + dt/2*b_nk
                    for j in range(NL - 1):
                        J[M + NL*i + j, (i*NL + M):(i*NL + M + NL)] = q[j,:]
                for i in range(M):
                    J[M + i*NL + NL - 1,i] = -np.matmul(Cij[i,:],np.matmul(beta_n,I))
                    J[M + i*NL + NL - 1,(M + i*NL):(M + i*NL + NL)] = (-1)**np.arange(NL)
                    for j in range(M):
                        J[M + i*NL + NL - 1,(M + j*NL):(M + j*NL + NL)] += - S[i]*Cij[i,j]*beta_n
                return J
    
            t_t = np.array([0])
            t = 0
    
            x = np.append(S,np.transpose(I))
            xh2 = x + 0;
            res = get_res(x, xh2, t, h)
    
            def get_next_step(x,x0,t,dt):
                xp = x0 + 0
                res = get_res(x,xp,t,dt)
                err = np.amax(np.abs(res))
                count = 0
                maxiter = 100
                while np.greater(err,0.000001) and count < maxiter:
                    J = get_J(xp,t,dt)
                    dx = np.linalg.solve(J,res)
                    xp += -dx
                    res = get_res(x, xp, t, dt)
                    err = np.amax(np.abs(res))
                    count += 1
                if count == maxiter:
                    print('solver maxed out')
                return xp
    
            while t < Tf:
                e_abs = 100
                e_rel = 100
                count = 0
                maxiter = 20
                x = xh2
    
                while (e_abs > atol or e_rel > rtol) and count < maxiter:
                    count+=1
                    if t + h > Tf:
                        h = Tf - t
                    #full step
                    xf  = get_next_step(x,x,t,h)
    
                    #half steps
                    xh1 = get_next_step(x,1/2*(x + xf),t, h/2)
                    xh2 = get_next_step(xh1, xf,t +h/2, h/2)
    
                    #compare predictions
                    err = np.abs(xh2 - xf)
                    e_abs = np.amax(err)
                    e_rel = np.amax(err/np.abs(xh2))
    
                    #adaptive time stepping
                    #reduce time step if not meeting error tolerances
                    if e_abs > atol:
                        h = 0.8*h*(atol/e_abs)**.5
                    elif e_rel > rtol:
                        h = 0.8*h*(rtol/e_rel)**.5
    
                #increment
                t_t = np.append(t_t,np.array([t + h/2, t + h]))
                t +=h
                count +=1
    
                #remember some of the results
                S_t = np.append(S_t, xh1[:M])
                S_t = np.append(S_t, xh2[:M])
                I_t = np.append(I_t, 2*xh1[M + NL*np.arange(M)])
                I_t = np.append(I_t, 2*xh2[M + NL*np.arange(M)])
                #subclasses of infecteds:
                Ii = np.transpose(np.reshape(xh1[M:],(M,NL)))
                Ie = np.transpose(np.reshape(x[M:],(M,NL)))
                dIc_dt_i = np.zeros((Nc,M))
                dIc_dt_e = np.zeros((Nc,M))
                for j in range(Nc):
                    dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],Ii)*p_alpha[j,:]
                    dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],Ie)*p_alpha[j,:]
                Ic += h/4*(dIc_dt_e + dIc_dt_i)
                if t == h:
                    Ic_t = np.concatenate(([Ic_0], [Ic]))
                else:
                    Ic_t = np.concatenate((Ic_t, [Ic]))
    
                Ie = Ii + 0
                Ii = np.transpose(np.reshape(xh2[M:],(M,NL)))
                for j in range(Nc):
                    dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],Ii)*p_alpha[j,:]
                    dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],Ie)*p_alpha[j,:]
                Ic += h/4*(dIc_dt_e + dIc_dt_i)
                Ic_t = np.concatenate((Ic_t, [Ic]))
    
                #adaptive time stepping
                #bump time step if well below threshold.
                if e_abs == 0:
                    h = 2*h
                else:
                    if e_rel/rtol >= e_abs/atol:
                        h = 0.8*h*(rtol/e_rel)**.5
                    else:
                        h = 0.8*h*(atol/e_abs)**.5
    
                if count == maxiter:
                    print('CN solver failed')
                    t = Tf
    
            #reshape the output of infecteds class:
            Ic_t_reshape = np.zeros((Nc,M,len(t_t)))
            for i in range(len(t_t)):
                Ic_t_reshape[:,:,i] = Ic_t[i,:,:]
    
            if not hybrid:
                return t_t, np.transpose(np.reshape(S_t,(len(t_t),M))), np.transpose(np.reshape(I_t,(len(t_t),M))), Ic_t_reshape
            else:
                #compute IC for next run from final point of current run.
                S_0 = xh2[:M]
                Ic_0 = Ic
                I_0 = np.zeros((Nk,M))
                I_end = np.reshape(xh2[M:],(M,NL))
                for i in range(M):
                    for j in range(NL):
                        Pn = legendre(j)
                        I_0[:,i] += Pn(sk)*I_end[i,j]
                return t_t, np.transpose(np.reshape(S_t,(len(t_t),M))), np.transpose(np.reshape(I_t,(len(t_t),M))), Ic_t_reshape, [S_0, I_0, Ic_0]
    
        elif galerkinIntegrator == 'odeint':
            #compute time derivative w. explicit treatment of BC
            def get_dxdt(x, t):
                Cij = Cij_t(tstart + t)
    
                S = x[:M]
                I = np.zeros((NL,M))
                I[:(NL-1), :] = np.transpose(np.reshape(x[M:(M + M*(NL-1))],(M,NL-1)))
    
                #solve for the highest Legendre polynomial coefficients:
                A = np.identity(M)*(-1)**(NL-1) - np.matmul(np.diag(S),Cij)*beta_n[NL-1]
                b = np.zeros(M)
                for i in range(M):
                    b[i] = S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I)) - np.sum(I[:,i]*(-1)**np.arange(NL))
                IN = np.linalg.solve(A,b)
                I[NL-1,:] = IN
    
                #compute derivs for S, I:
                dxdt = np.zeros(len(x))
                for i in range(M):
                    dxdt[i] = -S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    dxdt[M + i*(NL-1) + np.array(range(NL-1))] = -np.matmul(b_nk[:(NL-1),:],I[:,i])
    
                #compute derivs for subclasses of infecteds:
                dIc_dt = np.zeros((Nc,M))
                for i in range(Nc):
                    dIc_dt[i,:] = np.matmul(phi_alpha_n[i,:],I)*p_alpha[i,:]
                dxdt[(M + (M)*(NL-1)):] = np.reshape(dIc_dt,(1,Nc*M))
    
                return dxdt
    
            #set initial condition
            x0 = np.append(S,np.transpose(I[:(NL-1),:]))
            x0 = np.append(x0,Ic)
    
            #choose times for reporting output
            t = np.linspace(0, Tf, int(Tf*2*Nk));
            nt = len(t)
    
            #get solution
            u = odeint(get_dxdt, x0, t,rtol = rtol, atol = atol)
    
            #transform solution to get outputs
            S_t = np.transpose(u[:,:M])
            I_t = 2*np.transpose(u[:,M + (NL - 1)*np.arange(M)])
            Ic_t = np.zeros((Nc,M,nt))
            for i in range(nt):
                Ic_t[:,:,i] = np.reshape(u[i,(M + (NL-1)*(M)):],(Nc,M))
    
            if not hybrid:
                return t, S_t, I_t, Ic_t
            else:
                #compute IC for next run from final point of current run.
                S_0 = S_t[:,nt-1]
                Ic_0 = Ic_t[:,:,nt-1]
                I_0 = np.zeros((Nk,M))
                I_end = np.zeros((M, NL))
                I_end[:,:(NL-1)] = np.reshape(u[nt-1,M:(M + (NL-1)*M)],(M,NL-1))
    
                A = np.identity(M)*(-1)**(NL-1) - np.matmul(np.diag(S_t[:,nt-1]),Cij_t(t[nt - 1]))*beta_n[NL-1]
                b = np.zeros(M)
                for i in range(M):
                    b[i] = S_t[i,nt-1]*np.matmul(Cij_t(t[nt - 1])[i,:],np.matmul(beta_n,np.transpose(I_end))) - np.matmul(I_end[i,:],(-1)**np.arange(NL))
                IN = np.linalg.solve(A,b)
                I_end[:,NL-1] = IN
    
                for i in range(M):
                    for j in range(NL):
                        Pn = legendre(j)
                        I_0[:,i] += Pn(sk)*I_end[i,j]
                return t, S_t, I_t, Ic_t, [S_0, I_0, Ic_0]
        else:
            print('please choose a valid method for solving Galerkin -- Crank Nicolson or odeint')


    def solve_Predictor_CorrectorQ(self):
        ''' This is a predictor corrector method specifically for epidemics with two quarantine classes:
            (1) Voluntary quarantines -- people move in and out based on the current advice/symptoms, no concern for history
            (2) Assigned quarantines -- whether a person is quarantined or not depends on testing capabilities in the past
            Voluntary quarantines are all assumed to end before time T, whereas assigned quarantines are assumed to all last
            beyond time T.
        '''
        M  = self.parameters['M']                  
        Ni = self.parameters['Ni']                  
        Nc = self.parameters['Nc']                   
        Nk = self.parameters['Nk']                   
        Tf = self.parameters['Tf']                   
        T  = self.parameters['T']                   
        Td = self.parameters['Tc']                   
        Td = self.parameters['Td']                   
       
        tsi       = self.parameters['tsi']
        beta      = self.parameters['beta']                  
        tsi_sc    = self.parameters['tsi_sc']

        p_alpha   = self.parameters['p_alpha']
        phi_alpha = self.parameters['phi_alpha']
        Np        = np.sum(Ni)
        Ni = Ni/Np

        #unbundle quarantine parameters:
        tsiQ = self.parameters['tsiQ']
        TQ = self.parameters['TQ']
        PhiQV = self.parameters['PhiQV']
        phiQA = self.parameters['phiQA']
        Phi_QA_0 = self.parameters['Phi_QA_0']
        pQV = self.parameters['pQV_t']
        pQA = self.parameters['pQA_t']

        contactMatrixQ = self.parameters['CijQ_t']
        def CijQ_t(t):
            return np.matmul(contactMatrixQ(t*Tc),np.diag(1/Ni))
        def pQV_t(t):
            return pQV(t*Tc)
        def pQA_t(t):
            return pQA(t*Tc)
        
        contactMatrix = self.parameters['contactMatrix']
        def Cij_t(t):
            return np.matmul(contactMatrix(t*Tc),np.diag(1/Ni))


        Tc = T/2 #rescaling of time, and tsi 
        #first step is to rescale beta to a value consistent with the given Td
        Cij = contactMatrix(0)
        if M > 1:
            A = np.matmul(np.diag(Ni),Cij)
            A = np.matmul(A,np.diag(1/Ni))
        else:
            A = Cij
        max_eig_A = np.max(np.real(np.linalg.eigvals(A)))
        sp = np.linspace(0,T,1000)
        lam = np.log(2)/Td;  #Growth rate
        rs = max_eig_A*np.trapz(np.exp(-lam*sp)*np.interp(sp,tsi,beta),sp)
        beta = beta/rs       #now beta has been rescaled to give the correct (dimensional) doubling time
   
        #nondimensionalize beta:
        beta = beta*Tc
        tsi = tsi/Tc - 1
        
        #rescale phi_alpha based on re-scaling of tsi:
        tsi_sc = tsi_sc/Tc - 1
        tsiQ = tsiQ/Tc - 1
        
        #rescale end time
        Tf = Tf/Tc
        TQ = TQ/Tc

        #rescale phi_alpha
        if Nc > 0:
            phi_alpha = phi_alpha*Tc
        phiQA = phiQA*Tc

        S_0, I_0, Ic_0 = self.IC

        #scalings are done.  Begin setup.
        #set up the discretization in s
        s = np.linspace(-1,1,Nk)
        h = 2/(Nk - 1)
    
        #find the timesteps
        nt = int(np.round(Tf/h)) + 1
        t = h*np.linspace(0,nt-1,nt)
        
        #weighted betas for numerical integration
        beta_n = h*np.interp(s,tsi,beta)
        beta_n[[0, Nk-1]] = beta_n[[0, Nk-1]]/2
    
        #weighted phi_alpha for numerical integration
        phi_alpha_n = np.zeros((Nc, Nk))
        for i in range(Nc):      
            phi_alpha_n[i,:] = h*np.interp(s,tsi_sc,phi_alpha[i,:])
            phi_alpha_n[i,[0,Nk-1]] = phi_alpha_n[i,[0,Nk-1]]/2
    
        #weights for generic trapezoid integration
        w = h*np.ones(Nk)
        w[[0, Nk-1]] = w[[0, Nk-1]]/2
        
        #initialize variables
        S = S_0 + 0
        I = I_0 + 0
        Ic = Ic_0 + 0
            
        #initialize output vectors:
        S_t = np.zeros((M, nt))
        I_t = np.zeros((M, nt))
        Ic_t = np.zeros((Nc,M,nt))
        
        #set their starting values:
        S_t[:,0] = S_0    
        I_t[:,0] = np.matmul(w,I_0)
        Ic_t[:,:,0] = Ic_0

        #initialize a few variables
        dIc_dt_e = np.zeros((Nc,M))
        dIc_dt_i = np.zeros((Nc,M))
    
        #Voluntary quarantines
        Phi_QV_n = h*np.interp(s,tsiQ,PhiQV)

        #Assigned quarantines
        phi_QA_n = h*np.interp(s,tsiQ,phiQA)   #further processing needed
        Phi_QA_n = Phi_QA_0
        
        #initialize variables/lists
        IQV = np.matmul(w,np.outer(Phi_QV_n,pQV_t(0))*I_0)
        IQA = 0
        IQV_t = np.zeros((M,nt))
        IQA_t = np.zeros((M,nt))
        IQV_t[:,0] = IQV
        IQA_t[:,0] = IQA
           
    
        for i in (1 + np.arange(nt-1)):
            ######################
            # explicit time step #
            ######################
            
            IQA = Phi_QA_n*I
            IQV = np.outer(Phi_QV_n,pQV(t[i-1]))*I
            IQ = IQA + IQV
            dSdt_e = -np.matmul(np.matmul(np.diag(S), Cij_t(t[i-1])),np.matmul(beta_n,I-IQ)) + \
                     -np.matmul(np.matmul(np.diag(S),CijQ_t(t[i-1])),np.matmul(beta_n,  IQ))
               
            Sp = S + h*dSdt_e
            
            for j in range(Nc):
                dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]
            
                    
            #update assigned quarantine profile:
            dPhi_QA_dt = 1/2*(np.outer(phi_QA_n[1:],      pQA_t(t[i-1])) +\
                              np.outer(phi_QA_n[:(Nk-1)], pQA_t(t[i-0])))
            dIQA_dt = 1/2*(np.outer(phi_QA_n[1:],      pQA_t(t[i-1]))*I[1:     ,:] +\
                           np.outer(phi_QA_n[:(Nk-1)], pQA_t(t[i-0]))*I[:(Nk-1),:])
            Phi_QA_n[1:,:] = Phi_QA_n[:(Nk-1),:] + h*dPhi_QA_dt
            
            #update infected profile
            I[1:Nk,:] = I[0:(Nk-1),:]
            I[0,:] = -dSdt_e
                                                            
            ##################
            #'implicit' step #
            ##################                                                    
            IQA = Phi_QA_n*I
            IQV = np.outer(Phi_QV_n,pQV(t[i]))*I
            IQ = IQA + IQV
            dSdt_i = -np.matmul(np.matmul(np.diag(Sp), Cij_t(t[i])),np.matmul(beta_n,I-IQ)) + \
                     -np.matmul(np.matmul(np.diag(Sp),CijQ_t(t[i])),np.matmul(beta_n,  IQ))
    
            S = S + h/2*(dSdt_e + dSdt_i)
            
            for j in range(Nc):
                dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]
            
            Ic = Ic + h/2*(dIc_dt_e + dIc_dt_i)
    
            I[0,:] = -dSdt_i
            
            #remember this timestep    
            S_t[:,i]     = S    
            I_t[:,i]     = np.matmul(w,I)
            Ic_t[:,:, i] = Ic
            IQV_t[:,i]   = np.matmul(w,IQV)
            IQA_t[:,i]   = IQA_t[:,i-1] + h*np.sum(dIQA_dt,0)
        

        #back calculate the population currently under quarantine:
        tb = t[nt-1]; count = nt-1
        while (tb - h) > TQ:
            for i in range(M):
                IQA_t[i,count] += -np.interp(tb - TQ, t ,IQA_t[i,:])
            count += - 1
            tb += -h
            
        return t, S_t, I_t, Ic_t, IQV_t, IQA_t                      




    def simulate(self, IC, atol=1e-4, rtol=1e-3):
        self.IC = IC
        M  = self.parameters['M']                  
        T  = self.parameters['T']                   
        Nc = self.parameters['Nc']                   
        Nk = self.parameters['Nk']                   
        Tf = self.parameters['Tf']
        Tc = T/2                   
        
        tsi       = self.parameters['tsi']
        beta      = self.parameters['beta']                  
        beta      = self.parameters['beta']                  
        tsi_sc    = self.parameters['tsi_sc']                  
        
        method    = self.method
        galerkinIntegrator= self.galerkinIntegrator
        phi_alpha = self.parameters['phi_alpha']
        p_alpha   = self.parameters['p_alpha']                 
        
        contactMatrix = self.parameters['contactMatrix']

        if 'TQ' in self.parameters:
            t, S_t, I_t, Ic_t, IQV_t, IQA_t = self.solve_Predictor_CorrectorQ()
            data = {'t':t*T/2, 'S_t':S_t, 'I_t':I_t, 'Ic_t':Ic_t, 'IQV_t':IQV_t, 'IQA_t':IQA_t}
            return data
        
        if method == 'Predictor_Corrector':
            t, S_t, I_t, Ic_t = self.solve_Predictor_Corrector()

        elif method == 'Galerkin':
            t, S_t, I_t, Ic_t = self.solve_Galerkin(atol, rtol)
        
        elif method == 'Hybrid':
            tc = 0
            tswap = self.parameters['tswap']
            count = 0
            while tc < Tf:
                
                #run the next simulation
                if tc < tswap[count]:
                    tstep = tswap[count] - tc
                    self.parameters['Tf'] = tstep
                    sol = self.solve_Galerkin(atol, rtol, tc, True)
                else:
                    tstep = T
                    self.parameters['Tf'] = tstep
                    sol = self.solve_Predictor_Corrector(tc, True)
                    if count < len(tswap)-1:
                        count += 1
                
                #unpack results
                if count == 0 or (count == 1 and tc == 0):
                    t, S_t, I_t, Ic_t, self.IC = sol
                else:
                    t  = np.concatenate((t,tc/Tc + sol[0]))
                    nt = len(t); S_t_new = np.zeros((M,nt)); I_t_new = np.zeros((M,nt)); Ic_t_new = np.zeros((Nc,M,nt))
                    for i in range(M):
                        S_t_new[i,:] = np.append(S_t[i,:],sol[1][i,:])
                    for i in range(M):
                        I_t_new[i,:] = np.append(I_t[i,:],sol[2][i,:])
                    for i in range(Nc):
                        for j in range(M):
                            Ic_t_new[i,j,:] = np.append(Ic_t[i,j,:],sol[3][i,j,:])
                    self.IC = sol[4]
                    S_t, I_t, Ic_t = [S_t_new, I_t_new, Ic_t_new]
                
                #prepare for next loop
                tc = tc + tstep
                #print(IC_t)

        data = {'t':t*T/2, 'S_t':S_t, 'I_t':I_t, 'Ic_t':Ic_t}
        return data    
