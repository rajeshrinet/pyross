# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np
cimport cpython
#from cython.parallel import prange
DTYPE   = np.float
ctypedef np.float_t DTYPE_t

import pyross.stochastic as stochastic
import pyross.deterministic as deterministic

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
            alpha: float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta: float
                rate of spread of infection.
            gIa: float
                rate of removal from asymptomatic individuals.
            gIs: float
                rate of removal from symptomatic individuals.
            fsa: float
                fraction by which symptomatic individuals do not self-isolate.
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
        readonly np.ndarray Ni
        readonly dict parameters

    def __init__(self, parameters, M, Ni):

        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)
        self.Ni    = Ni
        self.parameters = parameters

    cdef below_threshold(self,populations,thresholds):
        cdef:
            int M = self.M
        for i in range(3):
            for j in range(M):
                if (populations[i][j] < thresholds[i][j]):
                    return True
        return False

    cdef find_passing_index(self,trajectory,thresholds):
        cdef:
            int min_index = 0
            int length_of_traj = len(trajectory)
            int M = self.M
        for i in range(length_of_traj-1):
            for j in range(3):
                for k in range(M):
                    product = (trajectory[i,j*M + k] - thresholds[j][k])*(trajectory[i+1,j*M + k] - thresholds[j][k])
                    if product < 0:
                        return min_index+i+1 # index of first datapoint past threshold
        return min_index


    cpdef simulate(self, S0, Ia0, Is0, contactMatrix, long Tf,
                                dict thresholds,
                                double dt_stoch = 20,double dt_det = 300,
                                method='gillespie',
                                int nc = 30, double epsilon = 0.03,
                                int tau_update_frequency = 1,
                                ):
        cdef:
            int M=self.M
            tuple thresholds_from_below = thresholds.get('from_below')
            tuple thresholds_from_above = thresholds.get('from_above')
            tuple cur_populations = (S0, Ia0, Is0)
            int cur_t = 0
            long dt, cur_Tf, cur_Nt

        t_arr = np.arange(Tf+1,dtype=int)
        trajectory = np.zeros([Tf+1,3*M],
                              dtype=long)
        trajectory[0] = np.concatenate((S0,Ia0,Is0))

        # initialize both stochastic and deterministic simulations
        model_stoch = stochastic.SIR(self.parameters, M, self.Ni)
        model_det = deterministic.SIR(self.parameters, M, self.Ni)
        # check if initially we are above or below the threshold (from below)
        below = self.below_threshold(cur_populations,thresholds_from_below)

        while (cur_t < Tf):
            # run simulation
            if below:
                dt = long(np.round(dt_stoch))
            else:
                dt = long(np.round(dt_det))
            # duration of simulation
            cur_Tf = long( np.round( np.min([dt,(Tf-cur_t)]) ) )
            cur_Nt = long( np.round(cur_Tf+1 ) )
            #

            if below:
                cur_result = model_stoch.simulate(*cur_populations,
                                                     contactMatrix,
                                                     cur_Tf, cur_Nt,
                                                     method=method,epsilon=epsilon,
                                                     tau_update_frequency=tau_update_frequency,
                                                     )
                cur_traj = cur_result['X']
            else:
                cur_result = model_det.simulate(*cur_populations,
                                                     contactMatrix,
                                                     cur_Tf, cur_Nt)
                cur_traj = np.array(np.round(cur_result['X']),dtype=int)

            # check if we passed the threshold
            if below:
                passing_index = self.find_passing_index(cur_traj,thresholds_from_below)
            else:
                passing_index = self.find_passing_index(cur_traj,thresholds_from_above)
            if passing_index == 0: # means we have not passed through the threshold
                trajectory[cur_t+1:cur_t+cur_Nt] = cur_traj[1:]
                cur_t += cur_Tf
                cur_populations = ( cur_traj[-1,:M,],
                                   cur_traj[-1,M:2*M],
                                   cur_traj[-1,2*M:] )
            else: # means we passed through the threshold
                if below: # this means we came from below
                    below = False # now we are above
                else: # this means we came from above
                    below = True # now we are below
                trajectory[cur_t+1:cur_t+passing_index+1] = cur_traj[1:passing_index+1]
                # Note that the zero-th element of cur_traj is
                # the last datapoint from the previous simulation
                cur_t += passing_index
                cur_populations = ( cur_traj[passing_index,:M],
                                   cur_traj[passing_index,M:2*M],
                                   cur_traj[passing_index,2*M:] )
        out_dict = {'X':trajectory, 't':t_arr,
                     'Ni':self.Ni, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs,
                      'from_below':thresholds_from_below,
                      'from_above':thresholds_from_above,
                      'dt_det':dt_det,'dt_stoch':dt_stoch}
        return out_dict
