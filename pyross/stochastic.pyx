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
    """
    cdef:
        float random = rand()/(float(RAND_MAX)+1) * exp(l)
        long n = 0
        float cur_sum = 0.
        float cur_term = 1.
    while (cur_sum < random):
        cur_sum += cur_term
        n += 1
        cur_term *= l/float(n)
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
                int nc=20, double epsilon = 0.001,
                int tau_update_frequency = 10,
                int maximal_number_of_events = 150
                ):
        if method == 'gillespie':
            return self.simulate_gillespie(S0, Ia0, Is0, contactMatrix, Tf, Nf)
        else:
            return self.simulate_tau_leaping(S0, Ia0, Is0, contactMatrix, Tf, Nf,
                                  nc, epsilon,tau_update_frequency,
                                  maximal_number_of_events)


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
                          int nc = 20, double epsilon = 0.001,
                          int tau_update_frequency = 10,
                          int maximal_number_of_events = 150):
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
                    # This is based on Eqs. (4-6) of
                    # https://doi.org/10.1063/1.1613254   (Ref. 1)
                    #
                    # note that a single index in the above cited paper corresponds
                    # to a tuple here. In the paper, possible reactions are enumerated
                    # with a single index, we enumerate the reactions as elements of the
                    # matrix RM.
                    #
                    # calculate matrix of derivatives
                    # dRM[i,j,l] = derivative of rate for reaction (i,j) w.r.t. to N_l
                    # To estimate derivatives, we use a finite difference with
                    # step size 1.
                    for i in range(M):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3*M):
                                    dRM[i + j*M, i + k*M, l] = -RM[i + j*M, i + k*M]
                    for l in range(3*M):
                        rp[l] += 1
                        self.rate_matrix(rp, t)
                        for i in range(M):
                            for j in range(3):
                                for k in range(3):
                                    dRM[i + j*M, i + k*M, l] += RM[i + j*M, i + k*M]
                                    #RM[i + j*M, i + k*M, l] /= 1
                        rp[l] -= 1
                        self.rate_matrix(rp, t)

                    # evaluate Eqs. (4-6) of Ref. 1
                    cur_tau = INFINITY
                    factor = epsilon * W
                    for i in range(M):           # iterate      \
                        for j in range(3):       # over all      | This is j in Ref. 1
                            for k in range(3):   # reactions    /
                                #
                                cur_mu = 0.
                                cur_sig_sq = 0.
                                for i_ in range(M):             # \
                                    for j_ in range(3):         # | This is j' in Ref. 1
                                        for k_ in range(3):     # /
                                            # calculate current f_{j,j'}
                                            # Note that for our system, for every (j,j') the sum
                                            # in Ref. 1, Eq. (4), only has either one or two
                                            # nonzero elements:
                                            if j_ == k_:
                                                # for a diagonal element, the current population
                                                # decreases by one
                                                cur_f = -dRM[i + j*M, i + k*M, i_ + j_*M]
                                            else:
                                                # for an off-diagonal element, a species is
                                                # converted, so that the species numbers of the
                                                # involved species indecrease and decrease by one
                                                cur_f = dRM[i + j*M, i + k*M, i_ + j_*M]
                                                cur_f -= dRM[i + j*M, i + k*M, i_ + k_*M]
                                            #
                                            cur_mu += cur_f * RM[i_ + j_*M, i_ + k_*M]
                                            cur_sig_sq += cur_f**2 * RM[i_ + j_*M, i_ + k_*M]
                                #
                                # For the current value of j, calculate the expressions
                                # in the curly brackets in Eq. (6) of Ref. 1,
                                # and check if the values for the current reaction j are
                                # smaller than those found for all previous reactions
                                cur_mu = abs(cur_mu)
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
                    if cur_tau < 10/W:
                        # if the current timestep is less than 10/W,
                        # switch to direct SSA algorithm
                        SSA_steps_left = 100
                        continue
                    # the following if-clause is to bound the total number
                    # of reactions that occur during each step
                    # (This is ad-hoc, because sometimes the
                    #  estimated tau is too large - figure out why!)
                    elif cur_tau > maximal_number_of_events/W:
                        cur_tau = maximal_number_of_events/W

                steps_until_tau_update -= 1
                t += cur_tau

                # draw reactions for current timestep
                for i in range(M):
                    for j in range(3):
                        for k in range(3):
                            if RM[i+j*M,i+k*M] > 0:
                                # draw poisson variable
                                K_events = poisson( RM[i+j*M,i+k*M] * cur_tau )
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
