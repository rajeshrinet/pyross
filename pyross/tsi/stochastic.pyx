import  numpy as np 
cimport numpy as np
cimport cython
# from libc.stdlib cimport malloc, free
import warnings
import random as rd
import time
import os

DTYPE = np.float

"""
GENERAL COMMENT
---------------

We do not build a general "commonMethods" class as we have only SIR class for now. 
This could be improved later if needed.

"""

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)

class SIR:
    """
    TSI-SIR class
    MC simulation with binomial tau-leaping for SIR model with time since infection 

    Parameters
    ----------
    M: int
        Number of compartments of individual for each class
        (eg. number of age class -> linear length of contactMatrix)
    parameters: dict
        Contains the following keys:
        
        kI: int
            Number of stages of infection (discretised time since infection)
        beta: np.array(M, kI)
            Infectiousness as a function of the discretised time since infection
        gamma: np.array(M, kI)
            Rate of recovery (I -> R) 

    Ni: np.array(M, )
        Initial number in each compartment and class
    """
    
    def __init__(self, parameters, beta_fun, gI_fun):
        # self.beta  = parameters['beta']  # Infection rate
        # self.gI    = parameters['gI']    # Removal rate of I
        self.kI          = parameters['kI']    # Number of compartments
        self.Ttsi        = parameters['Ttsi']  # tsi cut-off
        self.M           = parameters['M']     # Number of age classes
        self.N           = parameters['N']
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
        I0 = np.zeros((self.M,self.kI))
        for tup in I_list:
            if tup[2]<self.Ttsi:
                k = int(tup[2]/self.dtsi)
            else:
                k = self.kI-1
            I0[tup[1],k] = tup[0] # should be a distribution
        for i in range(self.M):
            S0[i] = self.Ni[i] - np.sum(I0[i,:])
        return S0, I0
        
    def set_beta(self):
        """
        Method to set beta(t, Ttsi, **args) as an array of size kI
        """
        beta = np.zeros((self.M, self.kI))
        beta_i = []
        for i in range(self.M):
            for k in range(self.kI):
                beta_i.append(self.beta_fun(k*self.dtsi, self.Ttsi, self.beta_params))
            beta[i,:] = beta_i
        return beta

    def set_gI(self):
        """
        Method to set gI(t, Ttsi, **args) as an array of size kI
        """
        gI = np.zeros((self.M, self.kI))
        gI_i = []
        for i in range(self.M):
            for k in range(self.kI):
                gI_i.append(self.gI_fun(k*self.dtsi, self.Ttsi, self.gI_params))
            gI[i,:] = gI_i
        return gI
            
    # def check_beta_gI_size(self):
    #     """
    #     method to check that beta and gI have proper sizes
    #     Parameters
    #     ----------
    #     None.

    #     Returns
    #     -------
    #     mge: string
    #        Confirmation that beta and gI have the good shape
    #     """
    #     if self.beta.shape != (self.M, self.kI):
    #         raise Exception("beta should be an np.array(M, kI)")
    #     elif self.gI.shape != (self.M, self.kI):
    #         raise Exception("gI should be an np.array(M,kI)")
    #     mge = "beta and gI have a shape (M={}, kI={})".format(self.M, self.kI)
    #     return mge
    
    def advection(self, x):
        """
        Function to do advection on a 1d vector
        I(t+dt, s) = I(t, s-dt) for s>dt
        I(t+dt, s) = 0 for s<=dt
       
        Parameter
        ---------
        x: np.array(kI) (dimension 1)
        """
        x_new = np.copy(x)
        x_new[0] = 0
        for k in range(len(x)-1):
            x_new[k+1] = x[k]
        return x_new

    def lambda_rate(self, I):
        """
        Function to compute lambda, the rate to be infected per individuals

        Parameter
        ---------
        I: np.array(M, kI)
           Current array of infected

        Return
        ------
        lbda_rate: np.array(M)
           List of the lambda_i (for each i (age class))
        """
        beta = self.set_beta()        
        lbda_rate = np.zeros(self.M)
        for i in range(self.M):
            lbda_rate_i = 0
            for j in range(self.M):
                infectiousness_j = np.sum( [ (beta[j,k]*I[j,k] + beta[j,k+1]*I[j,k+1])/(2*self.Ni[j]) for k in range(0, self.kI-1) ] )
                lbda_rate_i += self.CM[i,j]*infectiousness_j
            lbda_rate[i] = lbda_rate_i
        return lbda_rate
            
    def binomial_tau_leaping_step(self, S, I):
        """
        Do ONE time step with binomial tau-leaping for the SIR
        We use the stochastic Predictor/Corrector from Joe's notes:
        'Thoughts on stochastic SIR model'
        9 September 2020
        
        Parameters
        ----------
        S: np.array(M)
           Current number of susceptibles.           
        I: np.array(M, kI)
           Current number of infectives.
        contactMatrix: python function(t)
           The social contact matrix C_{ij} denotes the
           average number of contacts made per day by an 
           individual in class i with an individual in class j

        Returns
        -------
        Snext: np.array(M)
           Susceptibles at the next time step
        Inext: np.array(M, kI)
           Infectives at the next time step     
        """

        # Set beta and gI arrays with good size !
        beta = self.set_beta()
        gI   = self.set_gI()

        dt = self.dtsi # Enforce that tsi time step and real time step be the same !
        
        ### STEP 1: How many infected will recover ? ###
        # Compute the survival probability for each tsi
        # + Draw random numbers from binomial

        psi_I= np.zeros((self.M, self.kI))
        firings_IR = np.zeros((self.M, self.kI))
        Ipredict = np.zeros((self.M, self.kI))
        
        for i in range(self.M):
            for k in range(self.kI):
                psi_I[i,k] = np.exp(-gI[i,k]*dt)
                firings_IR[i,k] = np.random.binomial(I[i,k], 1-psi_I[i,k])
                Ipredict[i,k] = I[i,k] - firings_IR[i,k]
        for i in range(self.M): # Advection
            Ipredict[i,:] = self.advection(Ipredict[i,:])
                
        ### STEP 2: How many susceptibles will be infected ? ###
        # Compute int_0^{dt/2} lambda(u)du
        int_lbda= 0.5*(self.lambda_rate(I) + self.lambda_rate(Ipredict)) * dt
        # Compute the survival probability for each people in S
        psi_S = np.exp(-int_lbda)
        # Let's sample firings S->I
        firings_SI = np.zeros(self.M)
        Spredict = np.zeros(self.M)
        for i in range(self.M):
            firings_SI[i] = np.random.binomial(S[i], 1 - psi_S[i])
            Spredict[i] = S[i] - firings_SI[i]
        
        ### STEP 3: New infectives can infect susceptibles during dt (say, from midpoint t+dt/2) ! ###
        # Survival prob for the new infected
        psi_Snew = np.zeros(self.M)
        firings_SInew = np.zeros(self.M)
        for i in range(self.M):
            rate_Snew_i = 0
            for j in range(self.M):
                rate_Snew_i += self.CM[i,j]*(firings_SI[j]/float(self.Ni[j]))
            rate_Snew_i = beta[i,0]*rate_Snew_i     
            psi_Snew[i] = np.exp(-rate_Snew_i*(dt/2.))
            firings_SInew[i] = np.random.binomial(Spredict[i], 1 - psi_Snew[i])


        ### STEP 4: New infectives can recover during dt (say, from midpoint t+dt/2) ###
        ### This should be very unlikely for dt small, but might be non-negligeable
        ### larger dt
        psi_Inew = np.zeros(self.M)
        firings_IRnew = np.zeros(self.M)
        for i in range(self.M):
            psi_Inew[i] = np.exp(-gI[i,0]*dt/2.)
            firings_IRnew[i] = np.random.binomial(firings_SI[i], 1 - psi_Inew[i])

        
        ### STEP 5: Global update ###
        Snext = Spredict - firings_SInew
        
        Ipredict[:,0] = firings_SI + firings_SInew - firings_IRnew
        Inext = np.copy(Ipredict)

        return Snext, Inext

    def simulate(self, S0, I0, contactMatrix, Tf, Ti=0, supplied_seed=-1):
        """
        Method that runs the Monte-Carlo binomial tau-leaping for the TSI-SIR model

        Parameters
        ----------
        S0: np.array(M)
            Initial number of susceptables.
        I0: np.array(M,kI)
            Initial number of  infectives.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Ti: float, optional
            Start time of integrator. The default is 0.
        supplied_seed: float (long), optional
            Initialise the seed of the random generator.
            Default is taken from process ID and current time

        Returns
        -------
        data: dict
            Contains trajectories of S and I
        """

        ### INIT SEED ###
        # Same as Julian's "initialize_random_number_generator" from
        # "pyross/stochastic.pyx", l. 130-135 [19 Sept. 2020 version] 
        max_long = 9223372036854775807
        if supplied_seed<0:            
            seed = (abs(os.getpid()) + time.time()*1000) % max_long
        else:
            seed = supplied_seed % max_long
        rd.seed(seed)

        ### INIT VARIABLES ###
        t=Ti # initial time
        S = S0 # initial state
        I = np.copy(I0)
        self.CM = contactMatrix(t) # initial contact Matrix

        # We store the full trajectory in lists:
        S_traj = []
        S_traj.append(S)
        I_traj = []
        I_traj.append(I)
        t_traj = []
        t_traj.append(t)

        ### MAIN LOOP ###
        while t <= Tf:
            # Compute next step state
            Snext, Inext = self.binomial_tau_leaping_step(S, I)
            # Update state
            S = np.copy(Snext)
            I = np.copy(Inext)
            t += self.dtsi
            self.CM = contactMatrix(t) # update contact Matrix
            # Store data
            S_traj.append(Snext)
            I_traj.append(Inext)
            t_traj.append(t)

        # Store the trajectories within a dictionary
        data = {'S': S_traj,
                'I': I_traj,
                't': t_traj}

        return data

