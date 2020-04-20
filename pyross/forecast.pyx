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
        readonly double cov_beta_beta, cov_beta_gIa, cov_gIa_gIa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIa')                      # recovery rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        self.cov_beta_beta = parameters.get('cov_beta_beta')
        self.cov_beta_gIa = parameters.get('cov_beta_gIa')
        self.cov_gIa_gIa = parameters.get('cov_gIa_gIa')

        #
        self.means = np.array([self.beta,self.gIa],dtype=DTYPE)
        self.cov = np.array([[self.cov_beta_beta,self.cov_beta_gIa],
                        [self.cov_beta_gIa,self.cov_gIa_gIa]],
                        dtype=DTYPE)

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
            while (sample_parameters[i] <= 0).any():
                sample_parameters[i] = np.random.multivariate_normal(means,cov)
            parameters = {'alpha':self.alpha, 'beta':sample_parameters[i,0],'fsa':self.fsa,
                        'gIa':sample_parameters[i,1],'gIs':sample_parameters[i,1]}
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
                     'cov_beta_beta':self.cov_beta_beta,
                     'cov_beta_gIa':self.cov_beta_gIa,
                     'cov_gIa_gIa':self.cov_gIa_gIa,
                     'sample_parameters':sample_parameters
                      } #
        return out_dict
