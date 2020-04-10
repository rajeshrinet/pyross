import numpy as np
cimport numpy as np
cimport cpython
#from cython.parallel import prange
DTYPE   = np.float
ctypedef np.float_t DTYPE_t
from numpy.math cimport INFINITY

cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil

from libc.stdlib cimport rand, RAND_MAX



cpdef poisson(double l = 1.):
    """
    Samples from Poisson distribution
    (This should be modified so that it can deal also with very
     large poisson parameters; in the meantime, the tau-leaping code
     samples the poisson distribution using numpy.random.poisson)
    """
    cdef:
        double random
        long n = 1
        double p = rand()/float(RAND_MAX)
        double L = exp(-l)
    while (p > L):
        n += 1
        p *= rand()/float(RAND_MAX)
    return n-1



cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M,
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC
        np.ndarray RM, rp, weights

    def __init__(self, parameters, M, Ni):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIa')                      # recovery rate of Is
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.RM = np.zeros( [3*self.M,3*self.M] , dtype=DTYPE)  # rate matrix
        self.rp = np.zeros([3*self.M],dtype=long) # state
        self.weights = np.zeros(9*self.M,dtype=DTYPE)



    cdef rate_matrix(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double alpha=self.alpha, beta=self.beta, gIa=self.gIa, aa, bb
            double fsa=self.fsa, alphab=1-self.alpha,gIs=self.gIs
            long [:] S    = rp[0  :M]
            long [:] Ia   = rp[M  :2*M]
            long [:] Is   = rp[2*M:3*M]
            double [:] Ni   = self.Ni
            double [:] ld   = self.lld
            double [:,:] CM = self.CM
            double [:,:] RM = self.RM

        for i in range(M): #, nogil=False):
            bb=0
            for j in range(M): #, nogil=False):
                 bb += beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            aa = bb*S[i]
            #
            RM[i+M,i] = alpha *aa # rate S -> Ia
            RM[i+2*M,i] = alphab *aa # rate S -> Is
            RM[i+M,i+M] = gIa*Ia[i] # rate Ia -> R
            RM[i+2*M,i+2*M] = gIs*Is[i] # rate Is -> R
        return



    cdef SSA_step(self,double time,
                      double total_rate):
        cdef:
            double [:] weights = self.weights
            long [:] rp = self.rp
            double dt, cs, t
            int M = self.M
            int I, i, j, k, max_index = 9*M
            double fRAND_MAX = float(RAND_MAX) + 1
        # draw exponentially distributed time for next reaction
        random = rand()/fRAND_MAX
        dt = -log(random) / total_rate
        t = time + dt

        # decide which reaction happens
        random = ( rand()/fRAND_MAX ) * total_rate
        cs = 0.0
        I = 0
        while cs < random and I < max_index:
            cs += weights[I]
            I += 1
        I -= 1

        # adjust population according to chosen reaction
        i = I//9
        j = (I - i*M)//3
        k = (I - i*M)%3
        if j == k:
            rp[i + M*j] -= 1
        else:
            rp[i + M*j] += 1
            rp[i + M*k] -= 1
        return t


    cdef calculate_total_reaction_rate(self):
        cdef:
            double W = 0. # total rate for next reaction to happen
            double [:,:] RM = self.RM
            double [:] weights = self.weights
            int M = self.M
            int i, j, k
        for i in range(M):
            for j in range(3):
                for k in range(3):
                    W += RM[i+j*M,i+k*M]
                    weights[i*M + 3*j + k] = RM[i+j*M,i+k*M]
        return W


    cpdef simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        if method == 'gillespie':
            return self.simulate_gillespie(S0, Ia0, Is0, contactMatrix, Tf, Nf)
        else:
            return self.simulate_tau_leaping(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)


    cpdef simulate_gillespie(self, S0, Ia0, Is0, contactMatrix, Tf, Nf):
        cdef:
            int M=self.M
            int i, j, k, I
            double t, dt, W
            #double [:] probabilities
            double [:,:] RM = self.RM
            long [:] rp = self.rp
            double [:] weights = self.weights
        max_index = 9*M*M
        t = 0
        for i in range(M):
            rp[i] = S0[i]
            rp[i+M] = Ia0[i]
            rp[i+2*M] = Is0[i]

        if Nf <= 0:
            t_arr = []
            trajectory = []
            trajectory.append((rp[:M],
                              rp[M:2*M],
                              rp[3*M:3*M]))
        else:
            t_arr = np.arange(0,int(Tf)+1,dtype=int)
            trajectory = np.zeros([Tf+1,3*M],dtype=long)
            trajectory[0] = rp
            next_writeout = 1

        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,3*M):
                W += rp[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(next_writeout,int(Tf)+1):
                        trajectory[i] = rp
                break

            # calculate current rate matrix
            self.CM = contactMatrix(t)
            self.rate_matrix(rp, t)

            # calculate total rate
            W = self.calculate_total_reaction_rate()

            # perform SSA step
            t = self.SSA_step(t,W)

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append((rp[:M],
                                  rp[M:2*M],
                                  rp[3*M:3*M]))
            else:
                while (next_writeout < t):
                    if next_writeout > Tf:
                        break
                    trajectory[next_writeout] = rp
                    next_writeout += 1

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        out_dict = {'X':out_arr, 't':t_arr,
                     'N':self.N, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs }
        return out_dict




    cpdef simulate_tau_leaping(self, S0, Ia0, Is0, contactMatrix, Tf, Nf,
                          int nc = 30, double epsilon = 0.03,
                          int tau_update_frequency = 1):
        cdef:
            int M=self.M
            int i, j, k, I, K_events
            double t, dt, W
            double [:,:] RM = self.RM
            long [:] rp = self.rp
            double [:] weights = self.weights
            double factor, cur_f
            double cur_tau
            int SSA_steps_left = 0
            int steps_until_tau_update = 0
            np.ndarray dRM = np.zeros( [3*self.M,3*self.M,3*self.M] , dtype=DTYPE)
            double verbose = 1.

        t = 0
        for i in range(M):
            rp[i] = S0[i]
            rp[i+M] = Ia0[i]
            rp[i+2*M] = Is0[i]

        if Nf <= 0:
            t_arr = []
            trajectory = []
            trajectory.append((rp[:M],
                              rp[M:2*M],
                              rp[3*M:3*M]))
        else:
            t_arr = np.arange(0,int(Tf)+1,dtype=int)
            trajectory = np.zeros([Tf+1,3*M],dtype=long)
            trajectory[0] = rp
            next_writeout = 1


        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,3*M):
                W += rp[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(next_writeout,int(Tf)+1):
                        trajectory[i] = rp
                break

            # calculate current rate matrix
            self.CM = contactMatrix(t)
            self.rate_matrix(rp, t)

            # Calculate total rate
            W = self.calculate_total_reaction_rate()

            if SSA_steps_left < 0.5:
                # check if we are below threshold
                for i in range(3*M):
                    if rp[i] > 0:
                        if rp[i] < nc:
                            SSA_steps_left = 100
                # if we are below threshold, run while-loop again
                # and switch to direct SSA algorithm
                if SSA_steps_left > 0.5:
                    continue

                if steps_until_tau_update < 0.5:
                    # Determine current timestep
                    # This is based on Eqs. (32), (33) of
                    # https://doi.org/10.1063/1.2159468   (Ref. 1)
                    #
                    # note that a single index in the above cited paper corresponds
                    # to a tuple here. In the paper, possible reactions are enumerated
                    # with a single index, we enumerate the reactions as elements of the
                    # matrix RM.
                    #
                    # evaluate Eqs. (32), (33) of Ref. 1
                    cur_tau = INFINITY
                    # iterate over species
                    for i in range(M):     #  } The tuple (i,j) here corresponds
                        for j in range(3): #  } to what is called "i" in Eqs. (32), (33)
                            cur_mu = 0.
                            cur_sig_sq = 0.
                            # current species has index I = i + j*M,
                            # and can either decay (diagonal element) or
                            # transform into J = i + k*M with k = 0,1,2 but k != j
                            for k in range(3):
                                if j == k: # decay
                                    cur_mu -= RM[i + j*M, i + k*M]
                                    cur_sig_sq += RM[i + j*M, i + k*M]
                                else: # transformation
                                    cur_mu += RM[i + j*M, i + k*M]
                                    cur_mu -= RM[i + k*M, i + j*M]
                                    cur_sig_sq += RM[i + j*M, i + k*M]
                                    cur_sig_sq += RM[i + k*M, i + j*M]
                            cur_mu = abs(cur_mu)
                            #
                            factor = epsilon*rp[i+j*M]/2.
                            if factor < 1:
                                factor = 1.
                            #
                            if cur_mu != 0:
                                cur_mu = factor/cur_mu
                            else:
                                cur_mu = INFINITY
                            if cur_sig_sq != 0:
                                cur_sig_sq = factor**2/cur_sig_sq
                            else:
                                cur_sig_sq = INFINITY
                            #
                            if cur_mu < cur_sig_sq:
                                if cur_mu < cur_tau:
                                    cur_tau = cur_mu
                            else:
                                if cur_sig_sq < cur_tau:
                                    cur_tau = cur_sig_sq
                    steps_until_tau_update = tau_update_frequency
                    #
                    # if the current timestep is less than 10/W,
                    # switch to direct SSA algorithm
                    if cur_tau < 10/W:
                        SSA_steps_left = 50
                        continue

                steps_until_tau_update -= 1
                t += cur_tau

                # draw reactions for current timestep
                for i in range(M):
                    for j in range(3):
                        for k in range(3):
                            if RM[i+j*M,i+k*M] > 0:
                                # draw poisson variable
                                #K_events = poisson( RM[i+j*M,i+k*M] * cur_tau )
                                K_events = np.random.poisson(RM[i+j*M,i+k*M] * cur_tau )
                                if j == k:
                                    rp[i + M*j] -= K_events
                                else:
                                    rp[i + M*j] += K_events
                                    rp[i + M*k] -= K_events

            else:
                # perform SSA step
                t = self.SSA_step(t,W)
                SSA_steps_left -= 1

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append((rp[:M],
                                  rp[M:2*M],
                                  rp[3*M:3*M]))
            else:
                while (next_writeout < t):
                    if next_writeout > Tf:
                        break
                    trajectory[next_writeout] = rp
                    next_writeout += 1

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        out_dict = {'X':out_arr, 't':t_arr,
                     'N':self.N, 'M':self.M,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs } #,
        return out_dict
