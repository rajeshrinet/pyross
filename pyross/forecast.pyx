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
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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


    def simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf,
                Ns,
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                verbose=False,
                method='deterministic'):

        cdef:
            int M=self.M
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,3*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters

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
            if method == 'deterministic':
                model = pyross.deterministic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                     'N':self.N, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov':self.cov,
                     'sample_parameters':sample_parameters
                      } #
        return out_dict




cdef class SIR_latent:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M, k_random
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means_params, means_init, cov_params, cov_init

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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
                method='deterministic'):

        cdef:
            int M=self.M
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,3*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ 3*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits

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
            if method == 'deterministic':
                model = pyross.deterministic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                     'N':self.N, 'M':self.M,
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
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters

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
            if method == 'deterministic':
                model = pyross.deterministic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                     'N':self.N, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gE':self.gE,
                     'gIa':self.gIa, 'gIs':self.gIs,
                     'cov':self.cov,
                     'sample_parameters':sample_parameters
                      } #
        return out_dict





cdef class SEIR_latent:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    Ia: asymptomatic
    Is: symptomatic
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits

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
            if method == 'deterministic':
                model = pyross.deterministic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEIR(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                     'N':self.N, 'M':self.M,
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
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters

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
            if method == 'deterministic':
                model = pyross.deterministic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                'N':self.N, 'M':self.M,
                'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,
                'gE':self.gE,'gA':self.gA,
                'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs,
                'cov':self.cov,
                'sample_parameters':sample_parameters}
        return out_dict





cdef class SEAIRQ_latent():
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits

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
            if method == 'deterministic':
                model = pyross.deterministic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEAIRQ(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Q0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                'N':self.N, 'M':self.M,
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
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered (SEAIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality
    S  ---> E
    E  ---> A
    A  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, R
    Ih ---> Ic, R
    Ic ---> Im, R
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is

        # the following parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds
        self.gIh   = parameters.get('gIh')                      # recovery rate of Ih
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ic
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means  = self.means
            double [:,:] cov  = self.cov
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters

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
            if method == 'deterministic':
                model = pyross.deterministic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                      'N':self.N, 'M':self.M,
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
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered (SEAIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality
    S  ---> E
    E  ---> A
    A  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, R
    Ih ---> Ic, R
    Ic ---> Im, R
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
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is

        # the following parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds
        self.gIh   = parameters.get('gIh')                      # recovery rate of Ih
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ic
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
                method='deterministic'):

        cdef:
            int M=self.M, k_tot = self.k_tot
            double [:] means_params=self.means_params, means_init=self.means_init
            double [:,:] cov_params=self.cov_params, cov_init=self.cov_init
            double start_time, end_time
            np.ndarray trajectories = np.zeros( [Ns,k_tot*M,Nf] , dtype=DTYPE)
            np.ndarray mean_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray std_traj = np.zeros([ k_tot*M, Nf] ,dtype=DTYPE)
            np.ndarray sample_parameters, sample_inits

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
            if method == 'deterministic':
                model = pyross.deterministic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf)
            elif method == 'gillespie':
                model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf,
                          method='gillespie')
            else:
                model = pyross.stochastic.SEAI5R(parameters, M, self.Ni)
                cur_result = model.simulate(S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf,
                              nc=nc,epsilon =epsilon,
                               tau_update_frequency = tau_update_frequency,
                              method='tau-leaping')
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
                      'N':self.N, 'M':self.M,
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
