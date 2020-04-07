import numpy as np
cimport numpy as np
cimport cpython
#from cython.parallel import prange
DTYPE   = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double log(double x) nogil

from libc.stdlib cimport rand, RAND_MAX


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
        np.ndarray RM

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


    cpdef simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf):
        cdef:
            int M=self.M
            int i, j, I
            double t, dt, W, cs, max_index, random
            #double [:] probabilities
            double [:,:] RM = self.RM
            np.ndarray probabilities = np.zeros(9*M*M,dtype=float)
            float fRAND_MAX = float(RAND_MAX)

        max_index = 9*M*M
        t = 0
        rp = np.concatenate((S0,Ia0,Is0))
        rp = np.array(rp,dtype=long)

        if Nf <= 0:
            t_arr = []
            trajectory = []
            trajectory.append((*rp))
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

            W = 0. # total rate for next reaction to happen
            for i in range(3*M):
                for j in range(3*M):
                    W += RM[i,j]
                    probabilities[i*3*M + j] = RM[i,j]

            # draw exponentially distributed time for next reaction
            random = rand()/fRAND_MAX
            dt = -log(random) / W
            t = t + dt

            # decide which reaction happens
            random = ( rand()/fRAND_MAX )*W
            cs = 0.0
            I = 0
            while cs < random and I < max_index:
                cs += probabilities[I]
                I += 1
            I -= 1
            # adjust population according to chosen reaction
            i = I//(3*M)
            j = I - i*3*M
            if i == j:
                rp[i] -= 1
            else:
                rp[i] += 1
                rp[j] -= 1

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append((*rp))
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
