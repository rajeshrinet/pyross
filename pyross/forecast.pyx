# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np
cimport cpython
#from cython.parallel import prange
DTYPE   = np.float
ctypedef np.float_t DTYPE_t
from numpy.math cimport INFINITY

import pyross.deterministic
import pyross.stochastic

from timeit import default_timer as timer
import time




cdef class SIR:
    """
    Susceptible, Infected, Removed (SIR)
    Ia: asymptomatic
    Is: symptomatic

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            cov: np.array( )
                covariance matrix for all the estimated parameters.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        self.cov = parameters.get('cov')
        #
        self.means = np.array([self.alpha,self.beta,self.gIa,self.gIs],dtype=DTYPE)
        self.cov = parameters.get('cov')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni


    def simulate(self, S0, Ia0, Is0, contactMatrix=None,
                Tf=100, Nf=101,
                Ns=1000,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True,
                ):
        """
        Parameters
        ----------
        S0: np.array(M,)
            Initial number of susceptables.
        Ia0: np.array(M,)
            Initial number of asymptomatic infectives.
        Is0: np.array(M,)
            Initial number of symptomatic infectives.
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
        """
        cdef:
            int M=self.M
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,3*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means, cov, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means,cov)
            parameters = {'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],'gIs':sample_parameters[i,3]}
            #
            if control:
                model = pyross.control.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, Ia0, Is0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                     'Ni':self.Ni, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov':self.cov,
                     'sample_parameters':sample_parameters
                      } #
        return out_dict




cdef class SIR_latent:
    """
    Susceptible, Infected, Removed (SIR)
    Ia: asymptomatic
    Is: symptomatic
    
    Latent inference class to be used when observed data is incomplete.
    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            cov: np.array( )
                Covariance matrix for all the estimated parameters.
            S0: np.array(M,)
                Estimate initial number of susceptables.
            Ia0: np.array(M,)
                Estimate initial number of asymptomatic infectives.
            Is0: np.array(M,)
                Estimate initial number of symptomatic infectives.
            cov_init : np.array((3*M, 3*M)) :
                Covariance matrix for the initial state.
                
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M, k_random
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means_params, means_init, cov_params, cov_init

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        self.cov_params = parameters.get('cov_params')
        self.cov_init = parameters.get('cov_init')
        #
        self.k_random = 4
        self.means_params = np.array([self.alpha,self.beta,self.gIa,self.gIs],dtype=DTYPE)
        self.means_init = np.zeros(3*M,dtype=DTYPE)
        self.means_init[: M] = np.array(parameters.get('S0'),dtype=DTYPE)
        self.means_init[1*M:2*M] = np.array(parameters.get('Ia0'),dtype=DTYPE)
        self.means_init[2*M:3*M] = np.array(parameters.get('Is0'),dtype=DTYPE)
        #
        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni


    def simulate(self, contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True,
                ):
        """
        Parameters
        ----------
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float
            Final time of integrator.
        Nf: Int
            Number of time points to evaluate.
        Ns: int
            Number of samples of parameters to take.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
            TODO
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
                sample_inits : List of initial state vectors tried.
        """

        cdef:
            int M=self.M
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,3*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means_params, cov_params, Ns)
        sample_inits = np.random.multivariate_normal(means_init, cov_init, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means_params,cov_params)
            while (sample_inits[i] < 0).any():
                sample_inits[i] = np.random.multivariate_normal(means_init, cov_init)
            parameters = {'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],'gIs':sample_parameters[i,3]}
            S0  = (sample_inits[i,    :M] * self.N).astype('int')
            Ia0 = (sample_inits[i, M  :2*M] * self.N).astype('int')
            Is0 = (sample_inits[i, 2*M:3*M] * self.N).astype('int')
            #
            if control:
                model = pyross.control.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, Ia0, Is0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                     'Ni':self.Ni, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov_params':self.cov_params,
                     'cov_init':self.cov_init,
                     'sample_parameters':sample_parameters,
                     'sample_inits':sample_inits
                      } #
        return out_dict


cdef class SEIR:
    """
    Susceptible, Exposed, Infected, Removed (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimated mean value of rate of removal from exposed individuals.
            cov: np.array( )
                covariance matrix for all the estimated parameters.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, fsa, gIa, gIs, gE
        readonly int k_tot
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE   = parameters.get('gE')                      # progression rate of E
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        #
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        self.means = np.array([self.alpha,self.beta,self.gIa,self.gIs,self.gE],dtype=DTYPE)
        self.cov = parameters.get('cov') #np.array([[self.cov_beta_beta,self.cov_beta_gIa],
                  #      [self.cov_beta_gIa,self.cov_gIa_gIa]],
                  #      dtype=DTYPE)

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.k_tot = 4


    def simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True,
                ):
        """
        Parameters
        ----------
        S0: np.array(M,)
            Initial number of susceptables.
        E0: np.array(M,)
            Initial number of exposed.
        Ia0: np.array(M,)
            Initial number of asymptomatic infectives.
        Is0: np.array(M,)
            Initial number of symptomatic infectives.
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
        """

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means, cov, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means,cov)
            parameters = { 'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4]}
            #
            if control:
                model = pyross.control.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, Ia0, Is0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, Ia0, Is0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                     'Ni':self.Ni, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gE':self.gE,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov':self.cov,
                     'sample_parameters':sample_parameters
                      } #
        return out_dict


cdef class SEIR_latent:
    """
    Susceptible, Exposed, Infected, Removed (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    
    Latent inference class to be used when observed data is incomplete.
    
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimated mean value of rate of removal from exposed individuals.
            cov: np.array( )
                covariance matrix for all the estimated parameters.
            S0: np.array(M,)
                Estimate initial number of susceptables.
            E0: np.array(M,)
                Estimate initial number of exposed.
            Ia0: np.array(M,)
                Estimate initial number of asymptomatic infectives.
            Is0: np.array(M,)
                Estimate initial number of symptomatic infectives.
            cov_init : np.array((3*M, 3*M)) :
                Covariance matrix for the initial state.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, fsa, gIa, gIs, gE
        readonly int k_tot, k_random
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC,
        readonly np.ndarray means_params, means_init, cov_params, cov_init

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE   = parameters.get('gE')                      # progression rate of E
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        #
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        #
        self.cov_params = parameters.get('cov_params')
        self.cov_init = parameters.get('cov_init')
        #
        self.k_random = 5
        self.k_tot = 4
        self.means_params = np.array([self.alpha,self.beta,self.gIa,self.gIs,self.gE],dtype=DTYPE)
        self.means_init = np.zeros(self.k_tot*M,dtype=DTYPE)
        self.means_init[: M] = np.array(parameters.get('S0'),dtype=DTYPE)
        self.means_init[1*M:2*M] = np.array(parameters.get('E0'),dtype=DTYPE)
        self.means_init[2*M:3*M] = np.array(parameters.get('Ia0'),dtype=DTYPE)
        self.means_init[3*M:4*M] = np.array(parameters.get('Is0'),dtype=DTYPE)
        #
        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni
        #



    def simulate(self, contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True):
        """
        Parameters
        ----------
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
                sample_inits : List of initial state vectors tried.
        """
        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means_params, cov_params, Ns)
        sample_inits = np.random.multivariate_normal(means_init, cov_init, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means_params,cov_params)
            while (sample_inits[i] < 0).any():
                sample_inits[i] = np.random.multivariate_normal(means_init, cov_init)
            parameters = { 'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4]}
            S0  = ( sample_inits[i,   :1*M] * self.N ).astype('int')
            E0  = ( sample_inits[i,1*M:2*M] * self.N ).astype('int')
            Ia0 = ( sample_inits[i,2*M:3*M] * self.N ).astype('int')
            Is0 = ( sample_inits[i,3*M:4*M] * self.N ).astype('int')
            #
            if control:
                model = pyross.control.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, Ia0, Is0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, Ia0, Is0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                     'Ni':self.Ni, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gE':self.gE,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov_params':self.cov_params,
                     'cov_init':self.cov_init,
                     'sample_parameters':sample_parameters,
                     'sample_inits':sample_inits
                      } #
        return out_dict


cdef class SEAIRQ():
    """
    Susceptible, Exposed, Infected, Removed (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    A: Asymptomatic and infectious 
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimated mean value of rate of removal from exposed individuals.
            gA: float
                Estimated mean value of rate of removal from activated individuals.
            cov: np.array( )
                covariance matrix for all the estimated parameters.
            tE  : float
                testing rate and contact tracing of exposeds
            tA  : float
                testing rate and contact tracing of activateds
            tIa: float
                testing rate and contact tracing of asymptomatics
            tIs: float
                testing rate and contact tracing of symptomatics
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, gE, gA, fsa
        readonly double tE, tA, tIa, tIs
        readonly int k_tot
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        self.gE    = parameters.get('gE')                       # progression rate from E
        self.gA   = parameters.get('gA')                        # rate to go from A to Ia
        #
        #self.tS    = parameters.get('tS')                       # testing rate in S
        self.tE    = parameters.get('tE')                       # testing rate in E
        self.tA    = parameters.get('tA')                       # testing rate in A
        self.tIa   = parameters.get('tIa')                      # testing rate in Ia
        self.tIs   = parameters.get('tIs')                      # testing rate in Is

        # the following two parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        # vector of means & covariance matrix for Gaussian distribution
        self.means = np.array([self.alpha,self.beta,
                                self.gIa,self.gIs,
                                self.gE,self.gA],dtype=DTYPE)
        self.cov = parameters.get('cov') # 11x11 matrix

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.k_tot = 6 # total number of explicit states per age group


    def simulate(self, S0, E0, A0, Ia0, Is0, Q0,
                contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True):
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
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
        """

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means, cov, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means,cov)
            parameters = { 'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4],
                        'gA':sample_parameters[i,5],
                        'tE':self.tE,
                        'tA':self.tA,
                        'tIa':self.tIa,
                        'tIs':self.tIs}
            #
            if control:
                model = pyross.control.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEAIRQ(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict={'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                'Ni':self.Ni, 'M':self.M,
                'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,
                'gE':self.gE,'gA':self.gA,
                'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs,
                'cov':self.cov,
                'sample_parameters':sample_parameters}
        return out_dict


cdef class SEAIRQ_latent():
    """
    Susceptible, Exposed, Infected, Removed (SEIR)
    Ia: asymptomatic
    Is: symptomatic
    A: Asymptomatic and infectious 
    
    Latent inference class to be used when observed data is incomplete.
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimated mean value of rate of removal from exposed individuals.
            gA: float
                Estimated mean value of rate of removal from activated individuals.
            cov: np.array( )
                covariance matrix for all the estimated parameters.
            tE  : float
                testing rate and contact tracing of exposeds
            tA  : float
                testing rate and contact tracing of activateds
            tIa: float
                testing rate and contact tracing of asymptomatics
            tIs: float
                testing rate and contact tracing of symptomatics
            S0: np.array(M,)
                Estimate initial number of susceptables.
            E0: np.array(M,)
                Estimate initial number of exposed.
            A0: np.array(M,)
                Estimate initial number of activated.
            Ia0: np.array(M,)
                Estimate initial number of asymptomatic infectives.
            Is0: np.array(M,)
                Estimate initial number of symptomatic infectives.
            Q0: np.array(M,)
                Estimate initial number of quarantined.
            cov_init : np.array((3*M, 3*M)) :
                Covariance matrix for the initial state.
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M
        readonly double alpha, beta, gIa, gIs, gE, gA, fsa
        readonly double tE, tA, tIa, tIs
        readonly int k_tot, k_random
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov
        readonly np.ndarray means_params, means_init, cov_params, cov_init

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is
        self.gE    = parameters.get('gE')                       # progression rate from E
        self.gA   = parameters.get('gA')                        # rate to go from A to Ia
        #
        #self.tS    = parameters.get('tS')                       # testing rate in S
        self.tE    = parameters.get('tE')                       # testing rate in E
        self.tA    = parameters.get('tA')                       # testing rate in A
        self.tIa   = parameters.get('tIa')                      # testing rate in Ia
        self.tIs   = parameters.get('tIs')                      # testing rate in Is

        # the following two parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.k_tot = 6 # total number of explicit states per age group
        #
        self.cov_params = parameters.get('cov_params')
        self.cov_init = parameters.get('cov_init')
        #
        self.k_random = 6
        self.means_params = np.array([self.alpha,self.beta,
                          self.gIa,self.gIs,self.gE,self.gA],dtype=DTYPE)
        self.means_init = np.zeros(self.k_tot*M,dtype=DTYPE)
        self.means_init[: M] = np.array(parameters.get('S0'),dtype=DTYPE)
        self.means_init[1*M:2*M] = np.array(parameters.get('E0'),dtype=DTYPE)
        self.means_init[2*M:3*M] = np.array(parameters.get('A0'),dtype=DTYPE)
        self.means_init[3*M:4*M] = np.array(parameters.get('Ia0'),dtype=DTYPE)
        self.means_init[4*M:5*M] = np.array(parameters.get('Is0'),dtype=DTYPE)
        self.means_init[5*M:6*M] = np.array(parameters.get('Q0'),dtype=DTYPE)

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni



    def simulate(self,
                contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True):
        """
        Parameters
        ----------
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
                sample_inits : List of initial state vectors tried.
        """

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means_params, cov_params, Ns)
        sample_inits = np.random.multivariate_normal(means_init, cov_init, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means_params,cov_params)
            while (sample_inits[i] < 0).any():
                sample_inits[i] = np.random.multivariate_normal(means_init, cov_init)
            parameters = { 'fsa':self.fsa,
                        'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4],
                        'gA':sample_parameters[i,5],
                        'tE':self.tE,
                        'tA':self.tA,
                        'tIa':self.tIa,
                        'tIs':self.tIs}
            S0 =  (sample_inits[i,: M] * self.N).astype('int')
            E0 =  (sample_inits[i,M: 2*M]* self.N).astype('int')
            A0 =  (sample_inits[i,2*M: 3*M]* self.N).astype('int')
            Ia0 =  (sample_inits[i,3*M: 4*M]* self.N).astype('int')
            Is0 =  (sample_inits[i,4*M: 5*M]* self.N).astype('int')
            Q0 =  (sample_inits[i,5*M: 6*M]* self.N).astype('int')
            #
            if control:
                model = pyross.control.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEAIRQ(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict={'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                'Ni':self.Ni, 'M':self.M,
                'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,
                'gE':self.gE,'gA':self.gA,
                'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs,
                'cov_params':self.cov_params,
                'cov_init':self.cov_init,
                'sample_parameters':sample_parameters,
                'sample_inits':sample_inits}
        return out_dict


cdef class SEAI5R():
    """
    Susceptible, Exposed, Activates, Infected, Removed (SEAIR)
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
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimate mean value of rate of removal from exposeds individuals.
            gA: float
                Estimate mean value of rate of removal from activated individuals.
            gIh: float
                Rate of hospitalisation of infected individuals.
            gIc: float
                Rate hospitalised individuals are moved to intensive care.
            cov: np.array
                covariance matrix for all the estimated parameters.
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
    Ni: np.array(9*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gE, gA, gIa, gIs, gIh, gIc, fsa, fh
        readonly int k_tot
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, sa, hh, cc, mm, means, cov

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # progression rate of E class
        self.gA    = parameters.get('gA')                       # progression rate of A class
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is

        # the following parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds
        self.gIh   = parameters.get('gIh')                      # removal rate of Ih
        self.gIc   = parameters.get('gIc')                      # removal rate of Ic
        #
        sa         = parameters.get('sa')                       # daily arrival of new susceptibles
        hh         = parameters.get('hh')                       # hospital
        cc         = parameters.get('cc')                       # ICU
        mm         = parameters.get('mm')                       # mortality
        #iaa        = parameters.get('iaa')                      # daily arrival of new asymptomatics

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

        # vector of means & covariance matrix for Gaussian distribution
        self.means = np.array([self.alpha,
                                self.beta,
                                self.gIa,self.gIs,
                                self.gE,self.gA],
                                #self.gIh,self.gIc],
                                dtype=DTYPE)
        self.cov = parameters.get('cov') # 7x7 matrix

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.k_tot = 9 # total number of explicit states per age group


    def simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True):
        """
        Parameters
        ----------
        S0: np.array(M,)
            Initial number of susceptables.
        E0: np.array(M,)
            Initial number of exposed.
        A0: np.array(M,)
            Initial number of activated.
        Ia0: np.array(M,)
            Initial number of asymptomatic infectives.
        Is0: np.array(M,)
            Initial number of symptomatic infectives.
        Ih0: np.array(M,)
            Initial number of hospitalised.
        Ic0: np.array(M,)
            Initial number in intensive care.
        Im0: np.array(M,)
            Initial number of mortalities.
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
        """

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means, cov, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means,cov)
            parameters = {'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4],
                        'gA':sample_parameters[i,5],
                        'gIh':self.gIh,'gIc':self.gIc,
                        'fsa':self.fsa,'fh':self.fh,
                        'sa':self.sa,'hh':self.hh,
                        'mm':self.mm,'cc':self.cc
                          }
            #
            if control:
                model = pyross.control.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEAI5R(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = cur_result['X'].T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                      'Ni':self.Ni, 'M':self.M,
                      'alpha':self.alpha, 'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,
                      'gIh':self.gIh,'gIc':self.gIc,
                      'fsa':self.fsa,'fh':self.fh,
                      'gE':self.gE,'gA':self.gA,
                      'sa':self.sa,'hh':self.hh,
                      'mm':self.mm,'cc':self.cc,
                      'cov':self.cov,
                      'sample_parameters':sample_parameters
                      }
        return out_dict


cdef class SEAI5R_latent():
    """
    Susceptible, Exposed, Activates, Infected, Removed (SEAIR)
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
    
    Latent inference class to be used when observed data is incomplete.
    
    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float
                Estimate mean value of fraction of infected who are asymptomatic.
            beta: float
                Estimate mean value of rate of spread of infection.
            gIa: float
                Estimate mean value of rate of removal from asymptomatic individuals.
            gIs: float
                Estimate mean value of rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
            gE: float
                Estimate mean value of rate of removal from exposeds individuals.
            gA: float
                Estimate mean value of rate of removal from activated individuals.
            gIh: float
                Rate of hospitalisation of infected individuals.
            gIc: float
                Rate hospitalised individuals are moved to intensive care.
            cov: np.array
                covariance matrix for all the estimated parameters.
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
    Ni: np.array(9*M, )
        Initial number in each compartment and class

    Methods
    -------
    simulate
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gE, gA, gIa, gIs, gIh, gIc, fsa, fh
        readonly int k_random, k_tot
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, sa, hh, cc, mm
        readonly np.ndarray means_params, means_init, cov_params, cov_init


    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # progression rate of E class
        self.gA    = parameters.get('gA')                       # progression rate of A class
        self.gIa   = parameters.get('gIa')                      # removal rate of Ia
        self.gIs   = parameters.get('gIs')                      # removal rate of Is

        # the following parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds
        self.gIh   = parameters.get('gIh')                      # removal rate of Ih
        self.gIc   = parameters.get('gIc')                      # removal rate of Ic
        #
        sa         = parameters.get('sa')                       # daily arrival of new susceptibles
        hh         = parameters.get('hh')                       # hospital
        cc         = parameters.get('cc')                       # ICU
        mm         = parameters.get('mm')                       # mortality
        #iaa        = parameters.get('iaa')                      # daily arrival of new asymptomatics

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

        self.k_tot = 8 # total number of explicit states per age group
        #
        self.cov_params = parameters.get('cov_params')
        self.cov_init = parameters.get('cov_init')
        #
        self.k_random = 6
        self.means_params = np.array([self.alpha,self.beta,
                          self.gIa,self.gIs,self.gE,self.gA],dtype=DTYPE)
        self.means_init = np.zeros(self.k_tot*M,dtype=DTYPE)
        self.means_init[: M] = np.array(parameters.get('S0'),dtype=DTYPE)
        self.means_init[1*M:2*M] = np.array(parameters.get('E0'),dtype=DTYPE)
        self.means_init[2*M:3*M] = np.array(parameters.get('A0'),dtype=DTYPE)
        self.means_init[3*M:4*M] = np.array(parameters.get('Ia0'),dtype=DTYPE)
        self.means_init[4*M:5*M] = np.array(parameters.get('Is0'),dtype=DTYPE)
        self.means_init[5*M : 6*M] = np.array(parameters.get('Ih0'),dtype=DTYPE)
        self.means_init[6*M : 7*M] = np.array(parameters.get('Ic0'),dtype=DTYPE)
        self.means_init[7*M : 8*M] = np.array(parameters.get('Im0'),dtype=DTYPE)

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni



    def simulate(self,
                contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic',
                events=[],contactMatrices=[],
                events_repeat=False,
                events_subsequent=True):
        """
        Parameters
        ----------
        contactMatrix: python function(t), optional
             The social contact matrix C_{ij} denotes the 
             average number of contacts made per day by an 
             individual in class i with an individual in class j
             The default is None.
        Tf: float, optional
            Final time of integrator. The default is 100.
        Nf: Int, optional
            Number of time points to evaluate.The default is 101,
        Ns: int, optional
            Number of samples of parameters to take. The default is 1000.
        nc: int, optional
        epsilon: np.float64, optional
            Acceptable error in leap. The default is 0.03.
        tau_update_frequency : int, optional
        verbose: bool, optional
            Verbosity of output. The default is False.
        Ti: float, optional
            Start time of integrator. The default is 0.
        method: str, optional
            Pyross integrator to use. The default is "deterministic".
        events: list of python functions, optional
            List of events that the current state can satisfy to change behaviour
            of the contact matrix. Event occurs when the value of the function
            changes sign. Event.direction determines which direction triggers
            the event, takign values {+1,-1}.
            The default is [].
        contactMatricies: list of python functions
            New contact matrix after the corresponding event occurs
            The default is [].
        events_repeat: bool, optional
            Wheither events is periodic in time. The default is false.
        events_subsequent : bool, optional
            TODO
             
        Returns
        -------
        out_dict : dict
            Dictionary containing the following keys:
                X: list
                    List of resultant trajectories
                t: list
                    List of times at which X is evaluated.
                X_mean : list
                    Mean trajectory of X
                X_std : list
                    Standard devation of trajectories of X at each time point.
                <init params> : Initial parameters passed at object instantiation.
                sample_parameters : list of parameters sampled to make trajectories.
                sample_inits : List of initial state vectors tried.
        """

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits
            bint control = False

        if (contactMatrix == None) and (len(events) == 0):
            raise RuntimeError("Neither contactMatrix function nor list of events provided.")

        if len(events) > 0:
            control = True

        sample_parameters = np.random.multivariate_normal(means_params, cov_params, Ns)
        sample_inits = np.random.multivariate_normal(means_init, cov_init, Ns)

        start_time = timer()
        for i in range(Ns):
            if verbose:
                print('Running simulation {0} of {1}'.format(i+1,Ns),end='\r')
            while (sample_parameters[i] < 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means_params,cov_params)
            while (sample_inits[i] < 0).any():
                sample_inits[i] = np.random.multivariate_normal(means_init, cov_init)
            parameters = {'alpha':sample_parameters[i,0],
                        'beta':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3],
                        'gE':sample_parameters[i,4],
                        'gA':sample_parameters[i,5],
                        'gIh':self.gIh,'gIc':self.gIc,
                        'fsa':self.fsa,'fh':self.fh,
                        'sa':self.sa,'hh':self.hh,
                        'mm':self.mm,'cc':self.cc
                          }
            #
            S0 = ( sample_inits[i, : M] * self.N).astype('int')
            E0 = (sample_inits[i, M: 2*M]* self.N).astype('int')
            A0 = (sample_inits[i, 2*M: 3*M]* self.N).astype('int')
            Ia0 = (sample_inits[i, 3*M: 4*M]* self.N).astype('int')
            Is0 = (sample_inits[i, 4*M: 5*M]* self.N).astype('int')
            Ih0 = (sample_inits[i, 5*M: 6*M]* self.N).astype('int')
            Ic0 = (sample_inits[i, 6*M: 7*M]* self.N).astype('int')
            Im0 = (sample_inits[i, 7*M: 8*M]* self.N).astype('int')
            #
            if control:
                model = pyross.control.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                              events, contactMatrices, Tf, Nf,
                              events_repeat=events_repeat,
                              events_subsequent=events_subsequent,
                              nc=nc,epsilon=epsilon,
                              tau_update_frequency=tau_update_frequency,
                              method=method)
            else:
                if method == 'deterministic':
                    model = pyross.deterministic.SEAI5R(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                                    contactMatrix,
                                    Tf, Nf)
                else:
                    model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                    cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                                  contactMatrix, Tf, Nf,
                                  nc=nc,epsilon=epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  method=method)
            #
            trajectories[i] = (cur_result['X'][:, :k_tot*M]).T
        end_time = timer()
        if verbose:
            print('Finished. Time needed for evaluation:',time.strftime('%H:%M:%S',time.gmtime( end_time-start_time)) )
        #

        mean_traj = np.mean(trajectories,axis=0)
        std_traj = np.sqrt( np.var(trajectories,axis=0) )

        out_dict = {'X':trajectories, 't':cur_result['t'],
                    'X_mean':mean_traj,'X_std':std_traj,
                      'Ni':self.Ni, 'M':self.M,
                      'alpha':self.alpha, 'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,
                      'gIh':self.gIh,'gIc':self.gIc,
                      'fsa':self.fsa,'fh':self.fh,
                      'gE':self.gE,'gA':self.gA,
                      'sa':self.sa,'hh':self.hh,
                      'mm':self.mm,'cc':self.cc,
                      'cov_params':self.cov_params,
                      'cov_init':self.cov_init,
                      'sample_parameters':sample_parameters,
                      'sample_inits':sample_inits
                      }
        return out_dict
