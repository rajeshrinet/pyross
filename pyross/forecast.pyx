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
        self.means = np.array([self.beta,self.gIa,self.gIs],dtype=DTYPE)
        self.cov = parameters.get('cov') #np.array([[self.cov_beta_beta,self.cov_beta_gIa],
                        #[self.cov_beta_gIa,self.cov_gIa_gIa]],
                        #dtype=DTYPE)

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
            parameters = {'alpha':self.alpha, 'fsa':self.fsa,
                        'beta':sample_parameters[i,0],
                        'gIa':sample_parameters[i,1],'gIs':sample_parameters[i,2]}
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
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE   = parameters.get('gE')                      # progression rate of E
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        #
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        #
        self.means = np.array([self.beta,self.gE,self.gIa,self.gIs],dtype=DTYPE)
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
            parameters = {'alpha':self.alpha, 'fsa':self.fsa,
                        'beta':sample_parameters[i,0],
                        'gE':sample_parameters[i,1],
                        'gIa':sample_parameters[i,2],
                        'gIs':sample_parameters[i,3]}
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


cdef class SEAIRQ():
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Recovered, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, gE, gAA, gAS, fsa
        readonly double tS, tE, tA, tIa, tIs
        readonly int k_tot
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, means, cov

    def __init__(self, parameters, M, Ni):
        # these are the parameters we consider stochastic
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # progression rate from E
        self.gAA   = parameters.get('gAA')                      # rate to go from A to Ia
        self.gAS   = parameters.get('gAS')                      # rate to go from A to Is
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        #
        self.tS    = parameters.get('tS')                       # testing rate in S
        self.tE    = parameters.get('tE')                       # testing rate in E
        self.tA    = parameters.get('tA')                       # testing rate in A
        self.tIa   = parameters.get('tIa')                      # testing rate in Ia
        self.tIs   = parameters.get('tIs')                      # testing rate in Is

        # the following two parameters we consider to be exact
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives

        # vector of means & covariance matrix for Gaussian distribution
        self.means = np.array([self.beta,
                                self.gE,self.gAA,self.gAS,self.gIa,self.gIs,
                                self.tS,self.tE,self.tA,self.tIa,self.tIs],dtype=DTYPE)
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
            parameters = {'alpha':self.alpha, 'fsa':self.fsa,
                        'beta':sample_parameters[i,0],
                        'gE':sample_parameters[i,1],
                        'gAA':sample_parameters[i,2],
                        'gAS':sample_parameters[i,3],
                        'gIa':sample_parameters[i,4],
                        'gIs':sample_parameters[i,5],
                        'tS':sample_parameters[i,6],
                        'tE':sample_parameters[i,7],
                        'tA':sample_parameters[i,8],
                        'tIa':sample_parameters[i,9],
                        'tIs':sample_parameters[i,10]}
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
                'gE':self.gE,'gAA':self.gAA,
                'gAS':self.gAS,
                'tS':self.tS,'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs,
                'cov':self.cov,
                'sample_parameters':sample_parameters}
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
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # progression rate of E class
        self.gA    = parameters.get('gA')                       # progression rate of A class
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gIh   = parameters.get('gIh')                      # recovery rate of Ih
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ic

        # the following parameters we consider to be exact
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds
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
        self.means = np.array([self.beta,
                                self.gE,self.gA,
                                self.gIa,self.gIs,self.gIh,self.gIc],
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
            parameters = {'alpha':self.alpha,
                        'beta':sample_parameters[i,0],
                        'gE':sample_parameters[i,1],
                        'gA':sample_parameters[i,2],
                        'gIa':sample_parameters[i,3],
                        'gIs':sample_parameters[i,4],
                        'gIh':sample_parameters[i,5],
                        'gIc':sample_parameters[i,6],
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
