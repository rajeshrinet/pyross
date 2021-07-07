# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np
import os, time
cimport cpython
import pyross.utils
DTYPE   = np.float
ctypedef np.float_t DTYPE_t
from numpy.math cimport INFINITY
import warnings

cdef extern from "math.h":
    double log(double x) nogil


from libcpp.vector cimport vector

# wrapper for C++11 pseudo-random number generator
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned long seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

    cdef cppclass poisson_distribution[T]:
        poisson_distribution()
        poisson_distribution(double a)
        T operator()(mt19937 gen)

    cdef cppclass discrete_distribution[T]:
        discrete_distribution()
        discrete_distribution(vector.iterator first, vector.iterator last)
        T operator()(mt19937 gen)




cdef class stochastic_integration:
    """
    Integrators used by stochastic models:
        Gillespie and tau-leaping
    """
    cdef:
        readonly int N, M,
        readonly int nClass, nReactions, nReactions_per_agegroup
        readonly int dim_state_vec
        np.ndarray rates, overdispersions, xt, xtminus1, vectors_of_change, CM
        int overdispersion_mode
        mt19937 gen
        long seed
        dict readData


    cdef random_choice(self,weights):
        '''
        Generates random choice X from ( 0, 1, ..., len(weights) ), with
        Probability( X = i ) = weights[i] / sum(weights).

        Parameters
        ----------
        weights: 1D np.array
            Relative weights for random choice

        Returns
        -------
        X: int
            Random choice from integers in ( 0, 1, ..., len(weights) )
        '''
        cdef:
            vector[double] values = weights
            discrete_distribution[int] dd = discrete_distribution[int](\
                                              values.begin(),values.end())
        return dd(self.gen)

    cdef uniform_dist(self):
        '''
        Draws random sample X from uniform distribution on (0,1)

        Parameters
        ----------
        None

        Returns
        -------
        X: double
            Random sample from uniform distribution on (0,1)
        '''
        cdef:
            uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
        return dist(self.gen)

    cdef poisson_dist(self,double mean):
        '''
        Draws random sample X from Poisson distribution with mean "mean"

        Parameters
        ----------
        mean: double
            Mean for Poisson distribution

        Returns
        -------
        X: int
            Random sample from Poisson distribution with given mean
        '''
        cdef:
            poisson_distribution[int] dist = poisson_distribution[int](mean)
        return dist(self.gen)

    cdef initialize_random_number_generator(self,long supplied_seed=-1):
        '''
        Sets seed for random number generator.
        If negative seed is supplied, a seed will be generated based
        on process ID and current time.

        Parameters
        ----------
        supplied_seed: long
            Seed for random number generator

        Returns
        -------
        None
        '''
        cdef:
            long max_long = 9223372036854775807
        if supplied_seed < 0:
            self.seed = (abs(os.getpid()) + long(time.time()*1000)) % max_long
        else:
            self.seed = supplied_seed % max_long
        self.gen = mt19937(self.seed)



    cdef calculate_total_reaction_rate(self):
        """
        Calculates total reaction constant W = \sum_i r_i
        r_i = reaction rate for channel i


        Parameters
        ----------
        None (will be restructured in future updates)


        Modifies
        ----------
        self.weights: np.array
            Percentage chance of reaction occuring


        Returns
        -------
        W: double
            Total reaction constant
        """
        cdef:
            double W = 0. # total rate for next reaction to happen
            double [:,:] RM = self.RM
            double [:] rates = self.rates
            double [:] overdispersions = self.overdispersions
            int nReactions = self.nReactions
            int M = self.M
            int i, j, k, k_tot = self.k_tot
        for i in range(nReactions):
            W += rates[i] / np.floor(overdispersions[i])  # just to try out
        return W

    cdef rate_vector(self, xt, tt):
        return

    cdef SSA_step(self,double time,
                      double total_rate):
        """
        Gillespie Stochastic Simulation Algorithm step (SSA)
        Probabiliity of reaction occuring in time P ~ e^(-W tau)
        W = sum of all reaction rates.
        Solve to get tau =d


        Parameters
        ----------
        time: double
            Time point at which step is evaluated
        total_rate : double
            Total rate constant W.

        Returns
        -------
        X: np.array(len(t), len(x0))
            Numerical integration solution.
        time_points : np.array
            Corresponding times at which X is evaluated at.
        """
        cdef:
            double [:] rates = self.rates
            double [:] overdispersions = self.overdispersions
            long [:,:] vectors_of_change = self.vectors_of_change
            long [:] xt = self.xt
            int overdispersion_mode = self.overdispersion_mode
            double dt, t, random
            int i, j, k, counter, M = self.M
            long overdispersion_factor
            int dim_state_vec = self.dim_state_vec
            np.ndarray xt_candidate

        # draw exponentially distributed time for next reaction
        random = self.uniform_dist()
        dt = -log(random) / total_rate
        t = time + dt

        # decide which reaction happens
        i = self.random_choice(np.array(rates)/np.array(np.floor(overdispersions))) # just to try out

        # adjust population according to chosen reaction
        if overdispersion_mode == 1:
            overdispersion_factor = long ( np.floor(overdispersions[i]) )
            #
            for j in range(dim_state_vec):
                xt[j] += vectors_of_change[i,j] * overdispersion_factor
                if xt[j] < 0:
                  raise RuntimeError("Encountered negative population in SSA step")
        #
        elif overdispersion_mode == 0:
            if np.floor(overdispersions[i]) < 1.00001:
                overdispersion_factor = 1
                for j in range(dim_state_vec):
                    xt[j] += vectors_of_change[i,j] * overdispersion_factor
                    if xt[j] < 0:
                      raise RuntimeError("Encountered negative population in SSA step")
            else:
                xt_candidate = np.zeros_like( np.array(xt) ,dtype=long)
                xt_candidate[0] = -1
                counter = 0
                while (xt_candidate < 0).any():
                    overdispersion_factor = long ( self.poisson_dist( overdispersions[i] - 1. )  ) + 1
                    for j in range(dim_state_vec):
                        xt_candidate[j] = xt[j] + vectors_of_change[i,j] * overdispersion_factor
                    counter += 1
                    if counter > 1000:
                      for j in range(dim_state_vec):
                          print(xt_candidate[j])
                      raise RuntimeError("After 1000 tries, every randomly sampled overdipersion value yield negative populations" + \
                      "Try increasing threshold by increasing the " + \
                      "argument 'nc', or decreasing timestep by decreasing argument 'epsilon'")
                for j in range(dim_state_vec):
                    xt[j] = xt_candidate[j]
        else:
            raise RuntimeError("overdispersion_mode set to {0}. But only 0 and 1 are valid modes".format(overdispersion_mode))



        return t

    cpdef simulate_gillespie(self, contactMatrix, Tf, Nf):
        """
        Performs the stochastic simulation using the Gillespie algorithm.

        1. Rates for each reaction channel r_i calculated from current state.
        2. The timestep tau is chosen randomly from an exponential distribution P ~ e^(-W tau).
        3. A single reaction occurs with probablity proportional to its fractional rate constant r_i/W.
        4. The state is updated to reflect this reaction occuring and time is propagated forward by tau

        Stops if population becomes too small.


        Parameters
        ----------
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.


        Returns
        -------
        t_arr : np.array(Nf,)
            Array of time points at which the integrator was evaluated.
        out_arr : np.array
            Output path from integrator.
        """
        cdef:
            int M=self.M
            int nReactions = self.nReactions
            int dim_state_vec = self.dim_state_vec
            int i, j
            double t, dt, W, dt_out
            long [:] xt = self.xt
            double [:] rates = self.rates

        t = 0
        if Nf <= 0:
            t_arr = [t]
            trajectory = []
            trajectory.append((self.xt).copy())
        else:
            Nf -= 1 # subtract one to account for initial configuration
            dt_out = float(Tf)/float(Nf)
            t_arr = np.arange(0,Nf+1,dtype=float)*dt_out
            trajectory = np.zeros([Nf+1,dim_state_vec],dtype=long)
            trajectory[0] = xt
            next_writeout = dt_out
            j = 1

        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,dim_state_vec):
                W += xt[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = xt
                        j += 1
                break

            # calculate current rate matrix
            self.CM = contactMatrix(t)
            self.rate_vector(xt, t)

            # calculate total rate
            W = 0.
            for i in range(nReactions):
                W += rates[i]

            # if total reaction rate is zero
            if W == 0.:
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = (self.xt).copy()
                        j += 1
                break

            # perform SSA step
            t = self.SSA_step(t,W)

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append((self.xt).copy())
            else:
                while (next_writeout <= t):
                    if j > Nf:
                        break
                    trajectory[j] = (self.xt).copy()
                    next_writeout += dt_out
                    j += 1

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        return t_arr, out_arr

    cpdef check_for_event(self,double t,double t_previous,
                            events,
                            list list_of_available_events):
        """

        """

        cdef:
            long [:] xt = self.xt
            long [:] xtminus1 = self.xtminus1
            double f, f_p, direction
            int i, index_event

        for i,index_event in enumerate(list_of_available_events):
            f = events[index_event](t,xt)
            f_p = events[index_event](t_previous,xtminus1)
            # if the current event has a direction, include it
            try:
                direction = events[index_event].direction
                if direction > 0:
                    # if direction > 0, then an event means this:
                    if (f_p < 0) and (f >= 0):
                        return index_event
                elif direction < 0:
                    # if direction < 0, then an event means this:
                    if (f_p >= 0) and (f < 0):
                        return index_event
                else:
                    # if direction == 0, then any crossing through zero
                    # constitutes an event:
                    if (f_p*f <= 0):
                        return index_event
            except:
                # if no direction was given for the current event, then
                # any crossing through zero constitutes an event
                if (f_p*f <= 0):
                    return index_event
        return -1 # if no event was found, we return -1

    cpdef simulate_gillespie_events(self, events, contactMatrices,
                                Tf, Nf,
                                events_repeat=False,events_subsequent=True,
                                stop_at_event=False):
        cdef:
            int M=self.M
            int nReactions = self.nReactions
            int dim_state_vec = self.dim_state_vec
            int i, j
            double t, dt, W, dt_out, t_previous, t_last_event
            long [:] xt = self.xt
            long [:] xtminus1 = self.xtminus1
            double [:] rates = self.rates
            #
            list list_of_available_events, events_out
            int N_events, current_protocol_index, event_function_return

        t = 0
        if Nf <= 0:
            t_arr = [t]
            trajectory = []
            trajectory.append( (self.xt).copy()  )
        else:
            Nf -= 1 # subtract one to account for initial configuration
            dt_out = float(Tf)/float(Nf)
            t_arr = np.arange(0,Nf+1,dtype=float)*dt_out
            trajectory = np.zeros([Nf+1,dim_state_vec],dtype=long)
            trajectory[0] = xt
            next_writeout = dt_out
            j = 1

        # create a list of all events that are available
        # at the beginning  of the simulation
        N_events = len(events)
        if N_events == 1:
            list_of_available_events = []
        else:
            if events_subsequent:
                    list_of_available_events = [1]
            else:
                list_of_available_events = []
                for i in range(N_events):
                    list_of_available_events.append(i)
        events_out = []
        #print('list_of_available_events =',list_of_available_events)
        current_protocol_index = 0 # start with first contact matrix in list
        t_last_event = 0
        #self.CM = contactMatrices[current_protocol_index]
        #print("contactMatrices[current_protocol_index] =",contactMatrices[current_protocol_index])

        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,dim_state_vec):
                W += xt[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = xt
                        j += 1
                break

            # calculate current rate vector
            try:
                self.CM = contactMatrices[current_protocol_index](t-t_last_event)
            except:
                self.CM = contactMatrices[current_protocol_index]
            self.rate_vector(xt, t)

            # calculate total rate
            W = 0.
            for i in range(nReactions):
                W += rates[i]

            # if total reaction rate is zero
            if W == 0.:
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = (self.xt).copy()
                        j += 1
                break

            # save current state, which will become the previous state once
            # we perform the SSA step
            for i in range(dim_state_vec):
                xtminus1[i] = xt[i]

            # perform SSA step
            t_previous = t
            t = self.SSA_step(t,W)
            #print("t= {0:3.3f}\tt_p ={1:3.3f}".format(t,t_previous))

            # check for event, and update parameters if an event happened
            event_function_return = self.check_for_event(t=t,t_previous=t_previous,
                              events=events,
                            list_of_available_events=list_of_available_events)
            if event_function_return > -0.5: # this means an event has happened
                current_protocol_index = event_function_return
                #
                #print("current_protocol_index =",current_protocol_index)
                #print("list_of_available_events =",list_of_available_events)
                if events_repeat:
                    list_of_available_events = []
                    for j in range(0,N_events):
                        if j != current_protocol_index:
                            list_of_available_events.append(j)
                else:
                    if events_subsequent:
                        if current_protocol_index + 1 < N_events:
                            list_of_available_events = [ current_protocol_index+1 ]
                        else:
                            list_of_available_events = []
                    else:
                        if current_protocol_index in list_of_available_events:
                            list_of_available_events.remove(current_protocol_index)
                # add event to list, and update contact matrix
                events_out.append([t, current_protocol_index ])
                t_last_event = t
                #self.CM = contactMatrices[current_protocol_index]

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append((self.xt).copy())
            else:
                while (next_writeout <= t):
                    if j > Nf:
                        break
                    trajectory[j] = (self.xt).copy()
                    next_writeout += dt_out
                    j += 1

            # if we stop once an event has occured
            if stop_at_event:
                if len(events_out) > 0:
                    if Nf <= 0:
                        t_arr = np.array(t_arr)
                        out_arr = np.array(trajectory,dtype=long)
                    else:
                        t_arr = t_arr[:j]
                        out_arr = trajectory[:j]
                    return t_arr, out_arr, events_out

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        return t_arr, out_arr, events_out

    cdef tau_leaping_update_timestep(self,
                                    double epsilon = 0.03):
        """
        Tau leaping timestep
        This is based on Eqs. (32), (33) of
        https://doi.org/10.1063/1.2159468   (Ref. 1)

        Note that a single index in the above cited paper corresponds
        to a tuple here. In the paper, possible reactions are enumerated
        with a single index, we enumerate the reactions as elements of the
        matrix RM.

        Parameters
        ----------
        epsilon: float, optional
            The acceptable relative change of the rates during each
            tau-leaping step, as defined in Ref. 1. The default is 0.03

        Returns
        -------
        cur_tau : float
            The maximal timestep that can be taken with error < epsilon
        """
        cdef:
            int dim_state_vec = self.dim_state_vec
            int nReactions = self.nReactions
            int i, j
            double [:] rates = self.rates
            long [:,:] vectors_of_change = self.vectors_of_change
            long [:] xt = self.xt
            double cur_tau, cur_mu, cur_sig_sq
        #
        #
        # evaluate Eqs. (32), (33) of Ref. 1
        cur_tau = INFINITY
        # sum over species
        for i in range(dim_state_vec):
            cur_mu = 0.
            cur_sig_sq = 0.
            # iterate over reactions
            for j in range(nReactions):
                cur_mu += vectors_of_change[j,i] * rates[j]
                cur_sig_sq += (vectors_of_change[j,i])**2 * rates[j]
            cur_mu = abs(cur_mu)
            #
            factor = epsilon*xt[i]/2.
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
        return cur_tau

    cdef tau_leaping_update_state(self,double cur_tau):
        cdef:
            int nReactions = self.nReactions
            int dim_state_vec = self.dim_state_vec
            double [:] overdispersions = self.overdispersions
            int i, j, k, counter
            double [:] rates = self.rates
            long [:,:] vectors_of_change = self.vectors_of_change
            long [:] xt = self.xt
            long overdispersion_factor
            int overdispersion_mode = self.overdispersion_mode
            np.ndarray xt_candidate
        # Draw reactions
        for i in range(nReactions):
            if rates[i] > 0:
                K_events = self.poisson_dist( rates[i] * cur_tau )
                for k in range(K_events):
                    if overdispersion_mode == 0:
                        if np.floor(overdispersions[i]) < 1.00001:
                            overdispersion_factor = 1
                            for j in range(dim_state_vec):
                                xt[j] += vectors_of_change[i,j] * overdispersion_factor
                                if xt[j] < 0:
                                  raise RuntimeError("Encountered negative population in tau leaping step" + \
                                  "Try increasing threshold by increasing the " + \
                                  "argument 'nc', or decreasing timestep by decreasing argument 'epsilon'")
                        else:
                            xt_candidate = np.zeros_like( np.array(xt) ,dtype=long)
                            xt_candidate[0] = -1
                            counter = 0
                            while (xt_candidate < 0).any():
                                overdispersion_factor = long ( self.poisson_dist( overdispersions[i] - 1. )  ) + 1
                                for j in range(dim_state_vec):
                                    xt_candidate[j] = xt[j] + vectors_of_change[i,j] * overdispersion_factor
                                counter += 1
                                if counter > 1000:
                                  for j in range(dim_state_vec):
                                      print(xt_candidate[j])
                                  raise RuntimeError("After 1000 tries, every randomly sampled overdipersion value yield negative populations" + \
                                  "Try increasing threshold by increasing the " + \
                                  "argument 'nc', or decreasing timestep by decreasing argument 'epsilon'")
                            for j in range(dim_state_vec):
                                xt[j] = xt_candidate[j]
                    elif overdispersion_mode == 1:
                        overdispersion_factor = long ( np.floor(overdispersions[i]) )
                        #
                        for j in range(dim_state_vec):
                            xt[j] += vectors_of_change[i,j] * overdispersion_factor
                            if xt[j] < 0:
                              raise RuntimeError("Encountered negative population in tau leaping step")
        for i in range(dim_state_vec):
            if xt[i] < 0:
                raise RuntimeError("Tau leaping led to negative population. " + \
                                  "Try increasing threshold by increasing the " + \
                                  "argument 'nc'")
        return

    cpdef simulate_tau_leaping(self, contactMatrix, Tf, Nf,
                          int nc = 30, double epsilon = 0.03,
                          int tau_update_frequency = 1):
        """
        Tau leaping algorithm for producing stochastically correct trajectories
        Based on Cao et al (2006):
        https://doi.org/10.1063/1.2159468
        This method can run much faster than the Gillespie algorithm

        1. Rates for each reaction channel r_i calculated from current state.
        2. Timestep tau chosen such that \Delta r_i < epsilon \Sum r_i
        3. Number of reactions that occur in channel i ~Poisson(r_i tau)
        4. Update state by this amount

        Parameters
        ----------
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        nc: optional
            The default is 30
        epsilon: float, optional
            The acceptable relative change of the rates during each
            tau-leaping step, as defined in Cao et al. The default is 0.03
        tau_update_frequency: optional


        Returns
        -------
        t_arr : np.array(Nf,)
            Array of time points at which the integrator was evaluated.
        out_arr : np.array
            Output path from integrator.
        """
        cdef:
            int M=self.M
            int i, j
            int dim_state_vec = self.dim_state_vec
            int nReactions = self.nReactions
            double t, dt, W, dt_out
            double [:] rates = self.rates
            long [:] xt = self.xt
            double cur_tau = 0
            int SSA_steps_left = 0
            int steps_until_tau_update = 0
            double verbose = 1.

        t = 0

        if Nf <= 0:
            t_arr = [t]
            trajectory = []
            trajectory.append( (self.xt).copy()  )
        else:
            Nf -= 1 # subtract one to account for initial configuration
            dt_out = float(Tf)/float(Nf)
            t_arr = np.arange(0,Nf+1,dtype=float)*dt_out
            trajectory = np.zeros([Nf+1,dim_state_vec],dtype=long)
            trajectory[0] = xt
            next_writeout = dt_out
            j = 1
        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,dim_state_vec):
                W += xt[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = xt
                        j += 1
                break
            # calculate current rate matrix
            self.CM = contactMatrix(t)
            self.rate_vector(xt, t)

            # calculate total rate
            W = 0.
            for i in range(nReactions):
                W += rates[i]

            # if total reaction rate is zero
            if W == 0.:
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = (self.xt).copy()
                        j += 1
                break

            if SSA_steps_left < 0.5:
                # check if we are below threshold
                for i in range(dim_state_vec):
                    if xt[i] > 0:
                        if xt[i] < nc:
                            SSA_steps_left = 100
                # if we are below threshold, run while-loop again
                # and switch to direct SSA algorithm
                if SSA_steps_left > 0.5:
                    continue

                if steps_until_tau_update < 0.5:
                    #
                    cur_tau = self.tau_leaping_update_timestep(epsilon=epsilon)
                    #
                    steps_until_tau_update = tau_update_frequency
                    #
                    # if the current timestep is less than 10/W,
                    # switch to direct SSA algorithm
                    if cur_tau < 10/W:
                        SSA_steps_left = 50
                        continue

                t += cur_tau

                # draw reactions for current timestep
                self.tau_leaping_update_state(cur_tau)

            else:
                # perform SSA step
                t = self.SSA_step(t,W)
                SSA_steps_left -= 1

            steps_until_tau_update -= 1

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append( (self.xt).copy()  )
            else:
                while (next_writeout <= t):
                    if j > Nf:
                        break
                    trajectory[j] = (self.xt).copy()
                    next_writeout += dt_out
                    j += 1

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        return t_arr, out_arr

    cpdef simulate_tau_leaping_events(self,
                              events,contactMatrices,
                            Tf, Nf,
                          int nc = 30, double epsilon = 0.03,
                          int tau_update_frequency = 1,
                          events_repeat=False,events_subsequent=True,
                          stop_at_event=False):
        cdef:
            int M=self.M
            int i, j
            int dim_state_vec = self.dim_state_vec
            int nReactions = self.nReactions
            double t, dt, W, dt_out
            double [:] rates = self.rates
            long [:] xt = self.xt
            double cur_tau = 0
            double t_last_event
            int SSA_steps_left = 0
            int steps_until_tau_update = 0
            double verbose = 1.
            # needed for event-driven simulation:
            long [:] xtminus1 = self.xtminus1
            list list_of_available_events, events_out
            int N_events, current_protocol_index, event_function_return

        t = 0
        if Nf <= 0:
            t_arr = [t]
            trajectory = []
            trajectory.append( (self.xt).copy()  )
        else:
            Nf -= 1 # subtract one to account for initial configuration
            dt_out = float(Tf)/float(Nf)
            t_arr = np.arange(0,Nf+1,dtype=float)*dt_out
            trajectory = np.zeros([Nf+1,dim_state_vec],dtype=long)
            trajectory[0] = xt
            next_writeout = dt_out
            j = 1

        # create a list of all events that are available
        # at the beginning  of the simulation
        N_events = len(events)
        if N_events == 1:
            list_of_available_events = []
        else:
            if events_subsequent:
                    list_of_available_events = [1]
            else:
                list_of_available_events = []
                for i in range(N_events):
                    list_of_available_events.append(i)
        events_out = []

        current_protocol_index = 0 # start with first contact matrix in list
        t_last_event = 0
        #self.CM = contactMatrices[current_protocol_index]

        while t < Tf:
            # stop if nobody is infected
            W = 0 # number of infected people
            for i in range(M,dim_state_vec):
                W += xt[i]
            if W < 0.5: # if this holds, nobody is infected
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = xt
                        j += 1
                break

            # calculate current rate vector
            try:
                self.CM = contactMatrices[current_protocol_index](t-t_last_event)
            except:
                self.CM = contactMatrices[current_protocol_index]
            self.rate_vector(xt, t)

            # calculate total rate
            W = 0.
            for i in range(nReactions):
                W += rates[i]

            # if total reaction rate is zero
            if W == 0.:
                if Nf > 0:
                    for i in range(j,int(Nf)+1):
                        trajectory[i] = (self.xt).copy()
                        j += 1
                break

            # save current state, which will become the previous state once
            # we perform either an SSA or a tau-leaping step
            for i in range(dim_state_vec):
                xtminus1[i] = xt[i]
            t_previous = t

            # either perform tau-leaping or SSA step:
            if SSA_steps_left < 0.5:
                # check if we are below threshold for tau-leaping
                for i in range(dim_state_vec):
                    if xt[i] > 0:
                        if xt[i] < nc:
                            SSA_steps_left = 100
                # if we are below threshold, run while-loop again
                # and switch to direct SSA algorithm
                if SSA_steps_left > 0.5:
                    continue

                # update tau-leaping timestep if it is time for that
                if steps_until_tau_update < 0.5:
                    #
                    cur_tau = self.tau_leaping_update_timestep(epsilon=epsilon)
                    #
                    steps_until_tau_update = tau_update_frequency
                    #
                    # if the current timestep is less than 10/W,
                    # switch to direct SSA algorithm
                    if cur_tau < 10/W:
                        SSA_steps_left = 50
                        continue

                t += cur_tau

                # draw reactions for current timestep
                self.tau_leaping_update_state(cur_tau)

            else:
                # perform SSA step
                t = self.SSA_step(t,W)
                SSA_steps_left -= 1

            steps_until_tau_update -= 1

            # check for event, and update parameters if an event happened
            event_function_return = self.check_for_event(t=t,t_previous=t_previous,
                                events=events,
                            list_of_available_events=list_of_available_events)
            if event_function_return > -0.5: # this means an event has happened
                current_protocol_index = event_function_return
                #
                if events_repeat:
                    list_of_available_events = []
                    for j in range(0,N_events):
                        if j != current_protocol_index:
                            list_of_available_events.append(j)
                else:
                    if events_subsequent:
                        if current_protocol_index + 1 < N_events:
                            list_of_available_events = [ current_protocol_index+1 ]
                        else:
                            list_of_available_events = []
                    else:
                        if current_protocol_index in list_of_available_events:
                            list_of_available_events.remove(current_protocol_index)
                # add event to list, and update contact matrix
                events_out.append([t, current_protocol_index ])
                t_last_event = t
                #self.CM = contactMatrices[current_protocol_index]

            if Nf <= 0:
                t_arr.append(t)
                trajectory.append( (self.xt).copy()  )
            else:
                while (next_writeout <= t):
                    if j > Nf:
                        break
                    trajectory[j] = (self.xt).copy()
                    next_writeout += dt_out
                    j += 1

            # if we stop once an event has occured
            if stop_at_event:
                if len(events_out) > 0:
                    if Nf <= 0:
                        t_arr = np.array(t_arr)
                        out_arr = np.array(trajectory,dtype=long)
                    else:
                        t_arr = t_arr[:j]
                        out_arr = trajectory[:j]
                    return t_arr, out_arr, events_out

        out_arr = np.array(trajectory,dtype=long)
        t_arr = np.array(t_arr)
        return t_arr, out_arr, events_out


    def S(self,  data):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             S: Susceptible population time series
        """
        X = data['X']
        S = X[:, 0:self.M]
        return S


    def E(self,  data, Ei=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             E: Exposed population time series
        """

        if None != Ei:
            X = data['X']
            E = X[:, Ei[0]*self.M:Ei[1]*self.M]
        else:
            X = data['X']
            Ei=self.readData['Ei']
            E = X[:, Ei[0]*self.M:Ei[1]*self.M]
        return E


    def A(self,  data, Ai=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             A: Activated population time series
        """

        if None != Ai:
            X = data['X']
            A = X[:, Ai[0]*self.M:Ai[1]*self.M]
        else:
            X = data['X']
            Ai=self.readData['Ai']
            A = X[:, Ai[0]*self.M:Ai[1]*self.M]
        return A


    def I(self,  data, Ii=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ia : Asymptomatics population time series
        """

        if None != Ii:
            X  = data['X']
            Ii=self.readData['Ii']
            I = X[:, Ii[0]*self.M:Ii[1]*self.M]
        else:
            X  = data['X']
            Ii=self.readData['Ii']
            I = X[:, Ii[0]*self.M:Ii[1]*self.M]
        return I


    def Ia(self,  data, Iai=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ia : Asymptomatics population time series
        """

        if None != Iai:
            X  = data['X']
            Ia = X[:, Iai[0]*self.M:Iai[1]*self.M]
        else:
            X  = data['X']
            Iai=self.readData['Iai']
            Ia = X[:, Iai[0]*self.M:Iai[1]*self.M]
        return Ia


    def Is(self,  data, Isi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Is : symptomatics population time series
        """
        if None != Isi:
            X  = data['X']
            Is = X[:, Isi[0]*self.M:Isi[1]*self.M]
        else:
            X  = data['X']
            Isi=self.readData['Isi']
            Is = X[:, Isi[0]*self.M:Isi[1]*self.M]
        return Is


    def Isp(self,  data, Ispi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Isp : (intermediate stage between symptomatics
                   and recovered) population time series
        """
        if None != Ispi:
            X  = data['X']
            Isp = X[:, Ispi[0]*self.M:Ispi[1]*self.M]
        else:
            X  = data['X']
            Ispi=self.readData['Ispi']
            Isp = X[:, Ispi[0]*self.M:Ispi[1]*self.M]
        return Isp


    def Ih(self,  data, Ihi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ic : hospitalized population time series
        """
        if None != Ihi:
            X  = data['X']
            Ih = X[:, Ihi[0]*self.M:Ihi[1]*self.M]
        else:
            X  = data['X']
            Ihi=self.readData['Ihi']
            Ih = X[:, Ihi[0]*self.M:Ihi[1]*self.M]
        return Ih


    def Ihp(self,  data, Ihpi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ihp : (intermediate stage between symptomatics
                   and recovered) population time series
        """
        if None != Ihpi:
            X  = data['X']
            Ihp = X[:, Ihpi[0]*self.M:Ihpi[1]*self.M]
        else:
            X  = data['X']
            Ihpi=self.readData['Ihpi']
            Ihp = X[:, Ihpi[0]*self.M:Ihpi[1]*self.M]
        return Ihp


    def Ic(self,  data, Ici=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ic : ICU hospitalized population time series
        """
        if None != Ici:
            X  = data['X']
            Ic = X[:, Ici[0]*self.M:Ici[1]*self.M ]
        else:
            X  = data['X']
            Ici=self.readData['Ici']
            Ic = X[:, Ici[0]*self.M:Ici[1]*self.M ]
        return Ic


    def Icp(self,  data, Icpi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Icp : (intermediate stage between ICU
                   and recovered) population time series
        """
        if None != Icpi:
            X  = data['X']
            Icp = X[:, Icpi[0]*self.M:Icpi[1]*self.M]
        else:
            X  = data['X']
            Icpi=self.readData['Icpi']
            Icp = X[:, Icpi[0]*self.M:Icpi[1]*self.M]
        return Icp


    def Im(self,  data, Imi=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             Ic : mortality time series
        """
        if None != Imi:
            X  = data['X']
            Im = X[:, Imi[0]*self.M:Imi[1]*self.M ]
        else:
            X  = data['X']
            Imi=self.readData['Imi']
            Im = X[:, Imi[0]*self.M:Imi[1]*self.M ]
        return Im


    def R(self,  data, Rind=None):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
             R: Removed population time series
             For SEAI8R: R=N(t)-(S+E+Ia+Is+Is'+Ih+Ih'+Ic+Ic')
        """
        if None != Rind:
            X = data['X']
            R = self.population
        else:
            X = data['X']
            Rind=self.readData['Rind']
            R = self.population
        for i in range(Rind):
            R  = R - X[:, i*self.M:(i+1)*self.M]
        return R


    def Sx(self,  data, Sxi):
        """
        Parameters
        ----------
        data: Data dict

        Returns
        -------
            Generic compartment Sx
        """
        X  = data['X']
        Im = X[:, Sxi[0]*self.M:Sxi[1]*self.M ]
        Sx = X[:, Sxi[0]*self.M:Sxi[1]*self.M ]
        return Sx




cdef class SIR(stochastic_integration):
    """
    Susceptible, Infected, Removed (SIR)

    * Ia: asymptomatic
    * Is: symptomatic

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
            Fraction by which symptomatic individuals do not self-isolate.
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class

    Methods
    -------
    rate_matrix:
        Calculates the rate constant for each reaction channel.
    simulate:
        Performs stochastic numerical integration.
    """
    cdef:
        readonly double beta, gIa, gIs, fsa
        readonly np.ndarray xt0, Ni, dxtdt, lld, CC, alpha, population

    def __init__(self, parameters, M, Ni):
        cdef:
            int i
            int nRpa # short for number of reactions per age group

        self.nClass = 3
        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.nReactions_per_agegroup = 4
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha = alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()

        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            # reaction S -> Ia at age group i:
            # population of S decreases by 1, population of Ia increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = -1
            self.vectors_of_change[  i*nRpa,i+  M] = +1
            # reaction S -> Is at age group i:
            # population of S decreases by 1, population of Is increases by 1
            self.vectors_of_change[1+i*nRpa,i    ] = -1
            self.vectors_of_change[1+i*nRpa,i+2*M] = +1
            # reaction Ia -> R at age group i:
            # population of Ia decreases by 1
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            # reaction Is -> R at age group i:
            # population of Is decreases by 1
            self.vectors_of_change[3+i*nRpa,i+2*M] = -1

        self.readData = {'Iai':[1,2], 'Isi':[2,3], 'Rind':3}
        self.population = self.Ni

    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double  beta=self.beta, gIa=self.gIa, rateS, lmda
            double fsa=self.fsa, gIs=self.gIs
            long [:] S    = xt[0  :M]
            long [:] Ia   = xt[M  :2*M]
            long [:] Is   = xt[2*M:3*M]
            double [:] Ni   = self.Ni
            double [:] ld   = self.lld
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            double [:] alpha= self.alpha
            int nRpa = self.nReactions_per_agegroup

        for i in range(M): #, nogil=False):
            lmda=0
            for j in range(M): #, nogil=False):
                 lmda += beta*(CM[i,j]*Ia[j]+fsa*CM[i,j]*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            rates[  i*nRpa] = alpha[i] *rateS        # rate S -> Ia
            rates[1+i*nRpa] = (1-alpha[i]) *rateS # rate S -> Is
            rates[2+i*nRpa] = gIa*Ia[i] # rate Ia -> R
            rates[3+i*nRpa] = gIs*Is[i] # rate Is -> R
        return

    cpdef simulate(self, S0, Ia0, Is0, contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        """
        Performs the Stochastic Simulation Algorithm (SSA)


        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        contactMatrix: python function(t)
            The social contact matrix C_{ij} denotes the
            average number of contacts made per day by an
            individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        method: str, optional
            SSA to use, either 'gillespie' or 'tau_leaping'.
            The default is 'gillespie'.
        nc: TYPE, optional
        epsilon: float, optional
            The acceptable relative change of the rates during each
            tau-leaping step, as defined in Cao et al:
                    https://doi.org/10.1063/1.2159468.
            The default is 0.03
        tau_update_frequency: TYPE, optional

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'event_occured' , 'param': input param to integrator.

        """

        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = Ia0[i]
            xt[i+2*M] = Is0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix=contactMatrix,
                                     Tf= Tf, Nf= Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix=contactMatrix,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict = {'X':out_arr, 't':t_arr,
                     'Ni':self.Ni, 'M':self.M,
                     'fsa':self.fsa,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs}

        return out_dict



    cpdef simulate_events(self, S0, Ia0, Is0, events,
                contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = Ia0[i]
            xt[i+2*M] = Is0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                     'Ni':self.Ni, 'M':self.M,
                     'fsa':self.fsa,
                     'alpha':self.alpha, 'beta':self.beta,
                     'gIa':self.gIa, 'gIs':self.gIs}
        return out_dict




cdef class SIkR(stochastic_integration):
    """
    Susceptible, Infected, Removed (SIkR). Method of k-stages of I

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        beta: float
            rate of spread of infection.
        gI: float
            rate of removal from infectives.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        kI: int
            number of stages of infection.
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array((kI + 1)*M, )
        Initial number in each compartment and class
    """

    cdef:
        readonly int kI
        readonly double beta
        readonly np.ndarray xt0, Ni, dxtdt, CC, gIvec, gI, population

    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        self.kI = parameters['kI']
        self.nClass = 1 + self.kI
        self.beta  = parameters['beta']                     # infection rate

        gI    = parameters['gI']                     # removal rate of I
        self.gI    = np.zeros( self.M, dtype = DTYPE)
        if np.size(gI)==1:
            self.gI = gI*np.ones(self.kI)
        elif np.size(gI)==M:
            self.gI= gI
        else:
            print('gI can be a number or an array of size M')

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.nReactions_per_agegroup = 1 + self.kI
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()

        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            # reaction S -> I at age group i:
            # population of S decreases by 1, population of first compartment
            # of I increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = -1
            self.vectors_of_change[  i*nRpa,i+  M] = +1
            #
            # reaction I_k -> I_{k+1} at age group i:
            # population of k-th  stage by 1, population of (k+1)-th
            # stage is increases by 1
            for j in range(self.kI - 1):
                self.vectors_of_change[1+j+i*nRpa, i+(j+1)*M] = -1
                self.vectors_of_change[1+j+i*nRpa, i+(j+2)*M] = +1
            #
            # reaction I_{kI} -> R} at age group i:
            # population of last stage decreases by 1
            self.vectors_of_change[self.kI+i*nRpa,i+self.kI*M] = -1

        self.readData = {'Ii':[1,self.kI+1], 'Rind':self.kI+1}
        self.population = self.Ni


    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j, jj, kI=self.kI
            double beta=self.beta, rateS, lmda
            long [:] S    = xt[0  :M]
            long [:] I    = xt[M  :(kI+1)*M]
            double [:] gI = self.gI
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            int nRpa = self.nReactions_per_agegroup

        for i in range(M): #, nogil=False):
            lmda=0
            for jj in range(kI):
                for j in range(M):
                    lmda += beta*(CM[i,j]*I[j+jj*M])/Ni[j]
            rateS = lmda*S[i]
            #
            rates[i*nRpa] =  rateS  # rate S -> I1
            for j in range(kI-1):
                rates[1+j+i*nRpa] = kI * gI[j] * I[i+j*M] # rate I_{j} -> I_{j+1}
            rates[kI+i*nRpa] = kI * gI[kI-1] * I[i+(kI-1)*M] # rate I_{k} -> R
        return

    cpdef simulate(self, S0, I0, contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i, j
            int kI = self.kI
            long [:] xt = self.xt

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            for j in range(kI):
              xt[i+(j+1)*M] = I0[j]

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict = {'X':out_arr, 't':t_arr,
                      'Ni':self.Ni, 'M':self.M,
                       'beta':self.beta,
                      'gI':self.gI, 'kI':self.kI }
        return out_dict

    cpdef simulate_events(self, S0, I0, events,
                contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i, j
            int kI = self.kI
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            for j in range(kI):
              xt[i+(j+1)*M] = I0[j]

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                    'Ni':self.Ni, 'M':self.M,
                     'beta':self.beta,
                    'gI':self.gI, 'kI':self.kI }
        return out_dict




cdef class SEIR(stochastic_integration):
    """
    Susceptible, Exposed, Infected, Removed (SEIR)

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed

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
            Fraction by which symptomatic individuals do not self-isolate.
        gE: float
            rate of removal from exposed individuals.
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(4*M, )
        Initial number in each compartment and class
    """

    cdef:
        readonly double beta, fsa, gIa, gIs, gE
        readonly np.ndarray xt0, Ni, dxtdt, lld, CC, alpha, population

    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        self.nClass = 4
        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gE    = parameters['gE']                       # removal rate of E
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.nReactions_per_agegroup = 5
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()

        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            # reaction S -> E at age group i:
            # population of S decreases by 1, population of E increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = -1
            self.vectors_of_change[  i*nRpa,i+  M] = +1
            #
            # reaction E -> Ia at age group i:
            # population of E decreases by 1, population of Ia increases by 1
            self.vectors_of_change[1+i*nRpa,i+  M] = -1
            self.vectors_of_change[1+i*nRpa,i+2*M] = +1
            #
            # reaction E -> Is at age group i:
            # population of E decreases by 1, population of Is increases by 1
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            self.vectors_of_change[2+i*nRpa,i+3*M] = +1
            #
            # reaction Ia -> R at age group i:
            # population of Ia decreases by 1
            self.vectors_of_change[3+i*nRpa,i+2*M] = -1
            #
            # reaction Is -> R at age group i:
            # population of Ia decreases by 1
            self.vectors_of_change[4+i*nRpa,i+3*M] = -1

        self.readData = {'Ei':[1,2], 'Iai':[2,3], 'Isi':[3,4], 'Rind':4}
        self.population = self.Ni


    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double gIa=self.gIa, gIs=self.gIs
            double gE=self.gE, ce1, ce2
            double beta=self.beta, rateS, lmda
            double fsa = self.fsa
            long [:] S    = xt[0  :  M]
            long [:] E    = xt[  M:2*M]
            long [:] Ia   = xt[2*M:3*M]
            long [:] Is   = xt[3*M:4*M]
            double [:] Ni   = self.Ni
            double [:,:] CM = self.CM
            double [:] alpha = self.alpha
            double [:] rates = self.rates
            int nRpa = self.nReactions_per_agegroup


        for i in range(M): #, nogil=False):
            lmda=0;  ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                 lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            #
            rates[  i*nRpa]   =  rateS # rate S -> E
            rates[1+i*nRpa]   = ce1 * E[i] # rate E -> Ia
            rates[2+i*nRpa]   = ce2 * E[i] # rate E -> Is
            rates[3+i*nRpa]   = gIa * Ia[i] # rate Ia -> R
            rates[4+i*nRpa]   = gIs * Is[i] # rate Is -> R
        return

    cpdef simulate(self, S0, E0, Ia0, Is0, contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = Ia0[i]
            xt[i+3*M] = Is0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict = {'X':out_arr, 't':t_arr,
                      'Ni':self.Ni, 'M':self.M,
                      'alpha':self.alpha, 'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,'fsa':self.fsa,
                      'gE':self.gE}
        return out_dict

    cpdef simulate_events(self, S0, E0, Ia0, Is0, events,
                contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = Ia0[i]
            xt[i+3*M] = Is0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                    'Ni':self.Ni, 'M':self.M,
                    'alpha':self.alpha, 'beta':self.beta,
                    'gIa':self.gIa,'gIs':self.gIs,'fsa':self.fsa,
                    'gE':self.gE}
        return out_dict








cdef class SEAIRQ(stochastic_integration):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed
    * A: asymptomatic and infectious
    * Q: quarantined

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array(M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        tE  : float
            testing rate and contact tracing of exposeds
        tA  : float
            testing rate and contact tracing of activateds
        tIa: float
            testing rate and contact tracing of asymptomatics
        tIs: float
            testing rate and contact tracing of symptomatics
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(6*M, )
        Initial number in each compartment and class
    """

    cdef:
        readonly double beta, gIa, gIs, gE, gA, fsa
        readonly double tE, tA, tIa, tIs #, tS
        readonly np.ndarray xt0, Ni, dxtdt, CC, alpha, population

    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # progression rate from E
        self.gA   = parameters['gA']                      # rate to go from A to I
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        #self.tS    = parameters['tS')                       # testing rate in S
        self.tE    = parameters['tE']                      # testing rate in E
        self.tA    = parameters['tA']                       # testing rate in A
        self.tIa   = parameters['tIa']                       # testing rate in Ia
        self.tIs   = parameters['tIs']                       # testing rate in Is

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.nClass = 6
        # explicit states per age group:
        # 1. S    Susceptible
        # 2. E    Exposed
        # 3. A    Asymptomatic and infective
        # 4. Ia   Infective, asymptomatic
        # 5. Is   Infective, symptomatic
        # 6. Q    Quarantined

        self.nReactions_per_agegroup = 10
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()


        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            #
            # reaction S -> E at age group i:
            # population of S decreases by 1, population of E increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = -1
            self.vectors_of_change[  i*nRpa,i+  M] = +1
            #
            # reaction E -> A at age group i:
            self.vectors_of_change[1+i*nRpa,i+  M] = -1
            self.vectors_of_change[1+i*nRpa,i+2*M] = +1
            #
            # reaction E -> Q at age group i:
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            self.vectors_of_change[2+i*nRpa,i+5*M] = +1
            #
            # reaction A -> Ia at age group i:
            self.vectors_of_change[3+i*nRpa,i+2*M] = -1
            self.vectors_of_change[3+i*nRpa,i+3*M] = +1
            #
            # reaction A -> Is at age group i:
            self.vectors_of_change[4+i*nRpa,i+2*M] = -1
            self.vectors_of_change[4+i*nRpa,i+4*M] = +1
            #
            # reaction A -> Q at age group i:
            self.vectors_of_change[5+i*nRpa,i+2*M] = -1
            self.vectors_of_change[5+i*nRpa,i+5*M] = +1
            #
            # reaction Ia -> R at age group i:
            self.vectors_of_change[6+i*nRpa,i+3*M] = -1
            #
            # reaction Ia -> Q at age group i:
            self.vectors_of_change[7+i*nRpa,i+3*M] = -1
            self.vectors_of_change[7+i*nRpa,i+5*M] = +1
            #
            # reaction Is -> R at age group i:
            self.vectors_of_change[8+i*nRpa,i+4*M] = -1
            #
            # reaction Is -> Q at age group i:
            self.vectors_of_change[9+i*nRpa,i+4*M] = -1
            self.vectors_of_change[9+i*nRpa,i+5*M] = +1

        self.readData = {'Ei':[1,2], 'Ai':[2,3], 'Iai':[3,4], 'Isi':[4,5],
                        'Qi':[5,6], 'Rind':6}
        self.population = self.Ni

    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            #double tS=self.tS,
            double tE=self.tE, tA=self.tA, tIa=self.tIa, tIs=self.tIs
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs
            double gA=self.gA
            double gAA, gAS

            long [:] S    = xt[0*M:M]
            long [:] E    = xt[1*M:2*M]
            long [:] A    = xt[2*M:3*M]
            long [:] Ia   = xt[3*M:4*M]
            long [:] Is   = xt[4*M:5*M]
            long [:] Q    = xt[5*M:6*M]

            double [:] Ni   = self.Ni
            #
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            double [:] alpha= self.alpha
            int nRpa = self.nReactions_per_agegroup

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            # rates away from S
            rates[  i*nRpa]  = rateS  # rate S -> E
            # rates away from E
            rates[1+i*nRpa]  = gE  * E[i] # rate E -> A
            rates[2+i*nRpa]  = tE  * E[i] # rate E -> Q
            # rates away from A
            rates[3+i*nRpa]  = gAA * A[i] # rate A -> Ia
            rates[4+i*nRpa]  = gAS * A[i] # rate A -> Is
            rates[5+i*nRpa]  = tA  * A[i] # rate A -> Q
            # rates away from Ia
            rates[6+i*nRpa]  = gIa * Ia[i] # rate Ia -> R
            rates[7+i*nRpa]  = tIa * Ia[i] # rate Ia -> Q
            # rates away from Is
            rates[8+i*nRpa]  = gIs * Is[i] # rate Is -> R
            rates[9+i*nRpa]  = tIs * Is[i] # rate Is -> Q
            #
        return

    cpdef simulate(self, S0, E0, A0, Ia0, Is0, Q0,
                  contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        # write initial condition to xt
        for i in range(M):
            xt[i]     = S0[i]
            xt[i+M]   = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Q0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict={'X':out_arr, 't':t_arr,
                'Ni':self.Ni, 'M':self.M,'fsa':self.fsa,
                'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,
                'gE':self.gE,'gA':self.gA,
                'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return out_dict

    cpdef simulate_events(self, S0, E0, A0, Ia0, Is0, Q0,
                events, contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i]     = S0[i]
            xt[i+M]   = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Q0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                  'Ni':self.Ni, 'M':self.M,'fsa':self.fsa,
                  'alpha':self.alpha,'beta':self.beta,
                  'gIa':self.gIa,'gIs':self.gIs,
                  'gE':self.gE,'gA':self.gA,
                  'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return out_dict




cdef class SEAIRQ_testing(stochastic_integration):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)

    * E: exposed
    * A: Asymptomatic and infectious
    * Ia: asymptomatic
    * Is: symptomatic
    * Q: quarantined

    ...

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float, np.array(M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        ars: float
            fraction of population admissible for random and symptomatic tests
        kapE: float
            fraction of positive tests for exposed individuals
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(6*M, )
        Initial number in each compartment and class
    testRate: python function(t)
        number of tests per day and age group
    """


    cdef:
        readonly double beta, gIa, gIs, gE, gA, fsa
        readonly double ars, kapE
        readonly np.ndarray xt0, Ni, dxtdt, CC, alpha, population
        readonly object testRate


    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # progression rate from E
        self.gA    = parameters['gA']                      # rate to go from A to I
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter

        self.ars    = parameters['ars']                     # fraction of population admissible for testing
        self.kapE   = parameters['kapE']                    # fraction of positive tests for exposed


        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.testRate=None

        self.nClass = 6
        # explicit states per age group:
        # 1. S    Susceptible
        # 2. E    Exposed
        # 3. A    Asymptomatic and infective
        # 4. Ia   Infective, asymptomatic
        # 5. Is   Infective, symptomatic
        # 6. Q    Quarantined

        self.nReactions_per_agegroup = 10
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()

        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            #
            # reaction S -> E at age group i:
            # population of S decreases by 1, population of E increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = -1
            self.vectors_of_change[  i*nRpa,i+  M] = +1
            #
            # reaction E -> A at age group i:
            self.vectors_of_change[1+i*nRpa,i+  M] = -1
            self.vectors_of_change[1+i*nRpa,i+2*M] = +1
            #
            # reaction E -> Q at age group i:
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            self.vectors_of_change[2+i*nRpa,i+5*M] = +1
            #
            # reaction A -> Ia at age group i:
            self.vectors_of_change[3+i*nRpa,i+2*M] = -1
            self.vectors_of_change[3+i*nRpa,i+3*M] = +1
            #
            # reaction A -> Is at age group i:
            self.vectors_of_change[4+i*nRpa,i+2*M] = -1
            self.vectors_of_change[4+i*nRpa,i+4*M] = +1
            #
            # reaction A -> Q at age group i:
            self.vectors_of_change[5+i*nRpa,i+2*M] = -1
            self.vectors_of_change[5+i*nRpa,i+5*M] = +1
            #
            # reaction Ia -> R at age group i:
            self.vectors_of_change[6+i*nRpa,i+3*M] = -1
            #
            # reaction Ia -> Q at age group i:
            self.vectors_of_change[7+i*nRpa,i+3*M] = -1
            self.vectors_of_change[7+i*nRpa,i+5*M] = +1
            #
            # reaction Is -> R at age group i:
            self.vectors_of_change[8+i*nRpa,i+4*M] = -1
            #
            # reaction Is -> Q at age group i:
            self.vectors_of_change[9+i*nRpa,i+4*M] = -1
            self.vectors_of_change[9+i*nRpa,i+5*M] = +1

        self.readData = {'Ei':[1,2], 'Ai':[2,3], 'Iai':[3,4], 'Isi':[4,5],
                        'Qi':[5,6], 'Rind':6}
        self.population = self.Ni

    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            #double tS=self.tS,
            double t0, tE, tA, tIa, tIs
            double fsa=self.fsa, gE=self.gE, gIa=self.gIa, gIs=self.gIs
            double ars=self.ars, kapE=self.kapE
            double gA=self.gA
            double gAA, gAS

            long [:] S    = xt[0*M:M]
            long [:] E    = xt[1*M:2*M]
            long [:] A    = xt[2*M:3*M]
            long [:] Ia   = xt[3*M:4*M]
            long [:] Is   = xt[4*M:5*M]
            long [:] Q    = xt[5*M:6*M]

            double [:] Ni   = self.Ni
            #
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            double [:] alpha= self.alpha
            int nRpa = self.nReactions_per_agegroup

            double [:] TR

        if None != self.testRate :
            TR = self.testRate(tt)
        else :
            TR = np.zeros(M)


        for i in range(M):
            t0 = 1./(ars*(Ni[i]-Q[i]-Is[i])+Is[i])
            tE = TR[i]*ars*kapE*t0
            tA= TR[i]*ars*t0
            tIa = TR[i]*ars*t0
            tIs = TR[i]*t0

            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                lmda += beta*CM[i,j]*(A[j]+Ia[j]+fsa*Is[j])/Ni[j]
            rateS = lmda*S[i]
            # rates away from S
            rates[  i*nRpa]  = rateS  # rate S -> E
            # rates away from E
            rates[1+i*nRpa]  = gE  * E[i] # rate E -> A
            rates[2+i*nRpa]  = tE  * E[i] # rate E -> Q
            # rates away from A
            rates[3+i*nRpa]  = gAA * A[i] # rate A -> Ia
            rates[4+i*nRpa]  = gAS * A[i] # rate A -> Is
            rates[5+i*nRpa]  = tA  * A[i] # rate A -> Q
            # rates away from Ia
            rates[6+i*nRpa]  = gIa * Ia[i] # rate Ia -> R
            rates[7+i*nRpa]  = tIa * Ia[i] # rate Ia -> Q
            # rates away from Is
            rates[8+i*nRpa]  = gIs * Is[i] # rate Is -> R
            rates[9+i*nRpa]  = tIs * Is[i] # rate Is -> Q
            #
        return

    cpdef simulate(self, S0, E0, A0, Ia0, Is0, Q0,
                  contactMatrix, testRate, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        self.testRate=testRate

        # write initial condition to xt
        for i in range(M):
            xt[i]     = S0[i]
            xt[i+M]   = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Q0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict={'X':out_arr, 't':t_arr,
                'Ni':self.Ni, 'M':self.M,'fsa':self.fsa,
                'alpha':self.alpha,'beta':self.beta,
                'gIa':self.gIa,'gIs':self.gIs,
                'gE':self.gE,'gA':self.gA,
                'ars':self.ars,'kapE':self.kapE}
        return out_dict

    cpdef simulate_events(self, S0, E0, A0, Ia0, Is0, Q0,
                events, contactMatrices, testRate, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        self.testRate=testRate

        # write initial condition to xt
        for i in range(M):
            xt[i]     = S0[i]
            xt[i+M]   = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Q0[i]

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                  'Ni':self.Ni, 'M':self.M,'fsa':self.fsa,
                  'alpha':self.alpha,'beta':self.beta,
                  'gIa':self.gIa,'gIs':self.gIs,
                  'gE':self.gE,'gA':self.gA,
                  'tE':self.tE,'tIa':self.tIa,'tIs':self.tIs}
        return out_dict





cdef class Model(stochastic_integration):
    """
    Generic user-defined epidemic model.

    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        A dictionary containing the model parameters.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec.
        Default: Identical mapping of the dictionary at all times.

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "constant"  : [ ["k"] ],
                "infection" : [ ["I", "S", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "S", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'k': 1,
        }
    """

    cdef:
        readonly double beta, gIa, gIs, fsa
        readonly np.ndarray xt0, Ni, dxtdt, lld, CC, alpha
        np.ndarray parameters
        np.ndarray constant_terms, linear_terms, infection_terms, finres_terms, resource_list
        readonly int n_constant_terms, n_linear_terms, n_infection_terms, n_finres_terms
        np.ndarray _lambdas
        readonly np.ndarray finres_pop
        readonly np.ndarray parameters_length
        list param_keys
        readonly dict param_dict
        dict class_index_dict
        readonly object time_dep_param_mapping


    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None):
        cdef:
            int i, m
            int nRpa # short for number of reactions per age group
            int nClass, offset
            Py_ssize_t susceptible_index, infection_index
            Py_ssize_t reagent_index, product_index
            int sign, class_index


        self.N = np.sum(Ni)
        self.M = M
        self.Ni = np.array(Ni, dtype=long)

        self.time_dep_param_mapping = time_dep_param_mapping
        if self.time_dep_param_mapping is not None:
            self.param_dict = parameters.copy()
            parameters = self.time_dep_param_mapping(parameters, 0)

        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        nClass = self.nClass
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.finres_terms = res[5]
        self.resource_list = res[6]

        if self.constant_terms.size > 0:
            self.n_constant_terms = len(self.constant_terms)
        else:
            self.n_constant_terms = 0

        if self.linear_terms.size > 0:
            self.n_linear_terms = len(self.linear_terms)
        else:
            self.n_linear_terms = 0

        if self.infection_terms.size > 0:
            self.n_infection_terms = len(self.infection_terms)
        else:
            self.n_infection_terms = 0

        if self.finres_terms.size > 0:
            self.n_finres_terms = len(self.finres_terms)
        else:
            self.n_finres_terms = 0



        if self.time_dep_param_mapping is None:
            self.update_model_parameters(parameters)
        else:
            self.update_time_dep_model_parameters(0)
        self._lambdas = np.zeros((self.infection_terms.shape[0], M))

        #

        self.nReactions_per_agegroup = self.n_constant_terms + \
                    + self.n_linear_terms + \
                    + self.n_infection_terms + \
                    + self.n_finres_terms
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.finres_pop = np.empty( len(self.resource_list), dtype='object')  # populations for finite-resource transitions
        for i in range(len(self.resource_list)):
            ndx = self.resource_list[i][0]
            if self.parameters_length[ndx] == 1:
                self.finres_pop[i] = 0
            else:
                self.finres_pop[i] = np.zeros(self.M, dtype=DTYPE)
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate vector
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)  # rate vector
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()


        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for m in range(M):
            #
            for i in range(self.n_constant_terms):
                #rate_index = constant_terms[i, 0]
                class_index = self.constant_terms[i, 1]
                sign = self.constant_terms[i, 2]
                #term = parameters[rate_index, m]*sign
                #
                self.vectors_of_change[i + m*nRpa,m + M*class_index] = sign
                self.vectors_of_change[i + m*nRpa,m + M*(nClass-1)] = sign

            offset = self.n_constant_terms
            for i in range(self.n_linear_terms):
                #rate_index = linear_terms[i, 0]
                reagent_index = self.linear_terms[i, 1]
                product_index = self.linear_terms[i, 2]
                #term = parameters[rate_index, m] * xt[m + M*reagent_index]
                self.vectors_of_change[offset + i + m*nRpa,m + M*reagent_index] -= 1
                if product_index != -1:
                    self.vectors_of_change[offset + i + m*nRpa,m + M*product_index] += 1
                    #dxdt[m + M*product_index] += term

            offset += self.n_linear_terms
            for i in range(self.n_infection_terms):
                #rate_index = infection_terms[i, 0]
                reagent_index = self.infection_terms[i, 1]
                susceptible_index = self.infection_terms[i, 2]
                product_index = self.infection_terms[i, 3]
                #term = parameters[rate_index, m] * lambdas[i, m] * xt[m+M*susceptible_index]
                self.vectors_of_change[offset + i + m*nRpa,m+M*susceptible_index] -= 1
                if product_index != -1:
                    self.vectors_of_change[offset + i + m*nRpa,m+M*product_index] += 1

            offset += self.n_infection_terms
            for i in range(self.n_finres_terms):
                reagent_index = self.finres_terms[i, 4]
                product_index = self.finres_terms[i, 5]
                if reagent_index != -1:
                    self.vectors_of_change[offset + i + m*nRpa,m+M*reagent_index] -= 1
                if product_index != -1:
                    self.vectors_of_change[offset + i + m*nRpa,m+M*product_index] += 1




    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, m, n, i, j, index, rate_index, overdispersion_index
            long [:] Ni   = self.Ni
            double [:] ld   = self.lld
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            double [:] overdispersions = self.overdispersions
            int nRpa = self.nReactions_per_agegroup
            int [:, :] constant_terms=self.constant_terms
            int [:, :] linear_terms=self.linear_terms
            int [:, :] infection_terms=self.infection_terms
            int [:, :] finres_terms=self.finres_terms
            np.ndarray finres_pop = self.finres_pop
            np.ndarray resource_list=self.resource_list
            double [:, :] parameters=self.parameters
            double [:,:] lambdas = self._lambdas
            double frp
            int offset, nClass = self.nClass
            int class_index, priority_index, resource_index, probability_index

        if self.time_dep_param_mapping is not None:
            self.update_time_dep_model_parameters(tt)
            parameters = self.parameters

        # Compute lambda
        if constant_terms.size > 0:
            for i in range(M):
                Ni[i] = xt[(nClass-1)*M + i]  # update Ni

        for i in range(infection_terms.shape[0]):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                lambdas[i, m] = 0
                for n in range(M):
                    index = n + M*infective_index
                    if Ni[n]>0:
                        lambdas[i, m] += CM[m,n]*xt[index]/Ni[n]

        # Calculate populations for finite resource transitions
        for i in range(len(resource_list)):
            ndx = self.resource_list[i][0]
            n_cohorts = self.parameters_length[ndx]
            if n_cohorts == 1:
                finres_pop[i] = 0
            else:
                finres_pop[i] = np.zeros(n_cohorts)
            for (class_index, priority_index) in resource_list[i][1:]:
                for m in range(M):
                    if n_cohorts == 1:
                        finres_pop[i] += xt[m + M*class_index] * parameters[priority_index, m]
                    else:
                        finres_pop[i][m] += xt[m + M*class_index] * parameters[priority_index, m]

        for m in range(M):
            for i in range(self.n_constant_terms):
                rate_index = constant_terms[i, 0]
                overdispersion_index = constant_terms[i, 3]
                rate = parameters[rate_index, m]
                #
                rates[i + m*nRpa] = rate
                if overdispersion_index != -1:
                    overdispersions[i + m*nRpa] = parameters[overdispersion_index, m]

            offset = self.n_constant_terms
            for i in range(self.n_linear_terms):
                rate_index = linear_terms[i, 0]
                overdispersion_index = linear_terms[i, 3]
                reagent_index = linear_terms[i, 1]
                rate = parameters[rate_index, m] * xt[m + M*reagent_index]
                rates[offset + i + m*nRpa] = rate
                if overdispersion_index != -1:
                    overdispersions[offset + i + m*nRpa] = parameters[overdispersion_index, m]

            offset += self.n_linear_terms
            for i in range(self.n_infection_terms):
                rate_index = infection_terms[i, 0]
                susceptible_index = infection_terms[i, 2]
                overdispersion_index = infection_terms[i, 4]
                rate = parameters[rate_index, m] * lambdas[i, m] * xt[m+M*susceptible_index]
                rates[offset + i + m*nRpa] = rate
                if overdispersion_index != -1:
                    overdispersions[offset + i + m*nRpa] = parameters[overdispersion_index, m]

            offset += self.n_infection_terms
            for i in range(self.n_finres_terms):
                resource_index = finres_terms[i, 0]
                rate_index = resource_list[resource_index][0]
                priority_index = finres_terms[i, 1]
                probability_index = finres_terms[i, 2]
                class_index = finres_terms[i, 3]
                overdispersion_index = finres_terms[i, 6]
                if np.size(finres_pop[resource_index]) == 1:
                        frp = finres_pop[resource_index]
                else:
                        frp = finres_pop[resource_index][m]
                if frp > 0:
                    rate = parameters[rate_index, m] * parameters[priority_index, m] \
                           * parameters[probability_index, m] * xt[m+M*class_index] / frp
                else:
                    rate = 0
                rates[offset + i + m*nRpa] = rate
                if overdispersion_index != -1:
                    overdispersions[offset + i + m*nRpa] = parameters[overdispersion_index, m]
        return


    def update_model_parameters(self, parameters):
        if self.time_dep_param_mapping is None:
            nParams = len(self.param_keys)
            self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
            self.parameters_length = np.empty(nParams, dtype=int)

            try:
                for (i, key) in enumerate(self.param_keys):
                    param = parameters[key]
                    self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                    self.parameters_length[i] = np.size(param)
            except KeyError:
                raise Exception('The parameters passed do not contain certain keys.\
                                 The keys are {}'.format(self.param_keys))
        else:
            self.param_dict = parameters.copy()
            self.update_time_dep_model_parameters(0)


    def update_time_dep_model_parameters(self, tt):
        parameters = self.time_dep_param_mapping(self.param_dict, tt)
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        self.parameters_length = np.empty(nParams, dtype=int)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                self.parameters_length[i] = np.size(param)
        except KeyError:
            raise Exception('The parameters passed do not contain certain keys.\
                             The keys are {}'.format(self.param_keys))


    def make_parameters_dict(self):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    def model_class_data(self, model_class_key, data):
        """
        Parameters
        ----------
        data: dict
            The object returned by `simulate`.

        Returns
        -------
            The population of class `model_class_key` as a time series
        """
        X = data['X']

        if model_class_key != 'R' or 'R' in self.class_index_dict.keys():
            class_index = self.class_index_dict[model_class_key]
            Os = X[:, class_index*self.M:(class_index+1)*self.M]
        else:
            if self.constant_terms.size > 0:
                x = X[:, :(self.nClass-1)*self.M]
                x_reshaped = x.reshape((X.shape[0], (self.nClass-1), self.M))
                Os = X[:, (self.nClass-1)*self.M:] - np.sum(x_reshaped, axis=1)
            else:
                X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
                Os = self.Ni - np.sum(X_reshaped, axis=1)
        return Os


    def simulate(self, x0, contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        """
        Performs the Stochastic Simulation Algorithm (SSA)

        Parameters
        ----------
        x0: np.array
            Initial condition.
        contactMatrix: python function(t)
            The social contact matrix C_{ij} denotes the
            average number of contacts made per day by an
            individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        method: str, optional
            SSA to use, either 'gillespie' or 'tau_leaping'.
            The default is 'gillespie'.
        nc: TYPE, optional
        epsilon: float, optional
            The acceptable relative change of the rates during each
            tau-leaping step, as defined in Cao et al:
                  https://doi.org/10.1063/1.2159468
            The default is 0.03
        tau_update_frequency: TYPE, optional

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'event_occured' , 'param': input param to integrator.

        """

        cdef:
            int M = self.M, i, n_class_for_init
            long [:] xt = self.xt
            list class_list, skipped_classes
            dict param_dict

        if type(x0) == list:
            x0 = np.array(x0)

        if type(x0) == np.ndarray:

            n_class_for_init = self.nClass
            if self.constant_terms.size > 0:
                n_class_for_init -= 1
            if x0.size != n_class_for_init*M:
                raise Exception("Initial condition x0 has the wrong dimensions. Expected x0.size=%s."
                    % ( n_class_for_init*M) )
        elif type(x0) == dict:
            # Check if any classes are not included in x0

            class_list = list(self.class_index_dict.keys())
            if self.constant_terms.size > 0:
                class_list.remove('Ni')

            skipped_classes = []
            for O in class_list:
                if not O in x0:
                    skipped_classes.append(O)
            if len(skipped_classes) > 0:
                raise Exception("Missing classes in initial conditions: %s" % skipped_classes)

            # Construct initial condition array
            x0_arr = np.zeros(0)

            for O in class_list:
                x0_arr = np.concatenate( [x0_arr, x0[O]] )
            x0 = x0_arr

        for i in range(len(x0)):
            xt[i] = x0[i]

        # add Ni to x0
        if self.constant_terms.size > 0:
            for i in range(len(x0),self.dim_state_vec):
                xt[i] = self.Ni[i-len(x0)]
            #x0 = np.concatenate([x0, self.Ni])


        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix=contactMatrix,
                                     Tf= Tf, Nf= Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix=contactMatrix,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)

        out_dict = {'X':out_arr, 't':t_arr,
                     'Ni':self.Ni, 'M':self.M}
        param_dict = self.make_parameters_dict()
        out_dict.update(param_dict)
        return out_dict

    
cdef class Spp(Model):
    """
    This is a slightly more specific version of the class `Model`. 

    `Spp` is still supported for backward compatibility. 

    `Model` class is recommended over `Spp` for new users. 

    The `Spp` class works like `Model` but infection terms use a single class `S` 

    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        A dictionary containing the model parameters.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec.
        Default: Identical mapping of the dictionary at all times.

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "constant"  : [ ["k"] ],
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'k': 1,
        }
    """
    
    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None):
        Xpp_model_spec = pyross.utils.Spp2Xpp(model_spec)
        super().__init__(Xpp_model_spec, parameters, M, Ni, time_dep_param_mapping=time_dep_param_mapping)
    
cdef class SppQ(Spp):
    """User-defined epidemic model with quarantine stage.

    This is a slightly more specific version of the class `Model`. 

    `SppQ` is still supported for backward compatibility. 

    `Model` class is recommended over `SppQ` for new users. 

    To initialise the SppQ model,

    ...

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        A dictionary containing the model parameters.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(3*M, )
        Initial number in each compartment and class
    time_dep_param_mapping: python function, optional
        A user-defined function that takes a dictionary of time-independent parameters and time as an argument, and returns a dictionary of the parameters of model_spec.
        Default: Identical mapping of the dictionary at all times.

    Examples
    --------
    An example of model_spec and parameters for SIR class with random testing (without false positives/negatives) and quarantine

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            },
            "test_pos"  : [ "p_falsepos", "p_truepos", "p_falsepos"] ,
            "test_freq" : [ "tf", "tf", "tf"]
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1,
            'p_falsepos': 0
            'p_truepos': 1
            'tf': 1
        }
    """

    cdef:
        readonly dict full_model_spec
        readonly object input_time_dep_param_mapping
        readonly object testRate

    def __init__(self, model_spec, parameters, M, Ni, time_dep_param_mapping=None):
        self.full_model_spec = pyross.utils.build_SppQ_model_spec(model_spec)
        self.input_time_dep_param_mapping = time_dep_param_mapping
        self.testRate = None
        super().__init__(self.full_model_spec, parameters, M, Ni, time_dep_param_mapping=self.full_time_dep_param_mapping)

    cpdef set_testRate(self, testRate):
        self.testRate = testRate

    cpdef full_time_dep_param_mapping(self, input_parameters, t):
        cdef dict output_param_dict
        if self.input_time_dep_param_mapping is not None:
            output_param_dict = self.input_time_dep_param_mapping(input_parameters, t).copy()
        else:
            output_param_dict = input_parameters.copy()
        if self.testRate is not None:
            output_param_dict['tau'] = self.testRate(t)
        else:
            output_param_dict['tau'] = 0
        return output_param_dict

    def model_class_data(self, model_class_key, data):
        """
        Parameters
        ----------
        data: dict
            The object returned by `simulate`.

        Returns
        -------
            The population of class `model_class_key` as a time series
        """
        X = data['X']

        if model_class_key == 'Ni':
            X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
            Os = np.sum(X_reshaped, axis=1)
        elif model_class_key == 'NiQ':
            X_reshaped = X.reshape((X.shape[0], (self.nClass), self.M))
            Os = np.sum(X_reshaped[:, (self.nClass//2):, :], axis=1)
        else:
            class_index = self.class_index_dict[model_class_key]
            Os = X[:, class_index*self.M:(class_index+1)*self.M]
        return Os

    def simulate(self, x0, contactMatrix, testRate, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1):
        """
        Performs the Stochastic Simulation Algorithm (SSA)

        Parameters
        ----------
        x0: np.array
            Initial condition.
        contactMatrix: python function(t)
            The social contact matrix C_{ij} denotes the
            average number of contacts made per day by an
            individual in class i with an individual in class j
        testRate: python function(t)
            The total number of PCR tests performed per day
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        method: str, optional
            SSA to use, either 'gillespie' or 'tau_leaping'.
            The default is 'gillespie'.
        nc: TYPE, optional
        epsilon: float, optional
            The acceptable relative change of the rates during each
            tau-leaping step, as defined in Cao et al:
                  https://doi.org/10.1063/1.2159468
            The default is 0.03
        tau_update_frequency: TYPE, optional

        Returns
        -------
        dict
             X: output path from integrator,  t : time points evaluated at,
            'event_occured' , 'param': input param to integrator.

        """
        self.testRate = testRate
        return super().simulate(x0, contactMatrix, Tf, Nf,
                method, nc, epsilon, tau_update_frequency)


cdef class SEI5R(stochastic_integration):
    warnings.warn('SEI5R not supported', DeprecationWarning)
    """
    Susceptible, Exposed, Infected, Removed (SEIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality
    ...
    Parameters
    ----------
     parameters: dict
        Contains the following keys:
        alpha: float, np.array (M,)
            fraction of infected who are asymptomatic.
        beta: float
            rate of spread of infection.
        gE: float
            rate of removal from exposeds individuals.
        gIa: float
            rate of removal from asymptomatic individuals.
        gIs: float
            rate of removal from symptomatic individuals.
        gIh: float
            rate of removal for hospitalised individuals.
        gIc: float
            rate of removal for idividuals in intensive care.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        fh  : float
            fraction by which hospitalised individuals are isolated.
        sa: float, np.array (M,)
            daily arrival of new susceptables.
            sa is rate of additional/removal of population by birth etc
        hh: float, np.array (M,)
            fraction hospitalised from Is
        cc: float, np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: float, np.array (M,)
            mortality rate in intensive care
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(8*M, )
        Initial number in each compartment and class
    """

    cdef:
        readonly double beta, gE, gIa, gIs, gIh, gIc, fsa, fh
        readonly np.ndarray xt0, Ni, dxtdt, CC, sa, iaa, hh, cc, mm, alpha, population
        int nClass_

    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # removal rate of E class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIh   = parameters['gIh']                      # removal rate of Is
        self.gIc   = parameters['gIc']                      # removal rate of Is
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds

        sa         = parameters['sa']                       # daily arrival of new susceptibles
        hh         = parameters['hh']                       # hospital
        cc         = parameters['cc']                       # ICU
        mm         = parameters['mm']                       # mortality
        iaa        = parameters['iaa']                      # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = np.array(Ni.copy(),dtype=long)

        self.nClass = 7  # number of classes (used in unit tests)
        self.nClass_ = 8 # number of explicit classes used in this function
        # explicit states per age group:
        # 1. S    susceptibles
        # 2. E    exposed
        # 3. Ia   infectives, asymptomatic
        # 4. Is   infectives, symptomatic
        # 5. Ih   infectives, hospitalised
        # 6. Ic   infectives, in ICU
        # 7. Im   infectives, deceased
        # 8. R    Removed

        self.nReactions_per_agegroup = 11
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass_ * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

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

        self.iaa    = np.zeros( self.M, dtype = DTYPE)
        if np.size(iaa)==1:
            self.iaa = iaa*np.ones(M)
        elif np.size(iaa)==M:
            self.iaa = iaa
        else:
            print('iaa can be a number or an array of size M')

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()


        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            # birth rate
            # population of S increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = +1
            #
            # reaction S -> E at age group i:
            # population of S decreases by 1, population of E increases by 1
            self.vectors_of_change[1+i*nRpa,i    ] = -1
            self.vectors_of_change[1+i*nRpa,i+  M] = +1
            #
            # reaction E -> Ia at age group i:
            # population of E decreases by 1, population of Ia increases by 1
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            self.vectors_of_change[2+i*nRpa,i+2*M] = +1
            #
            # reaction E -> Is at age group i:
            # population of E decreases by 1, population of Is increases by 1
            self.vectors_of_change[3+i*nRpa,i+  M] = -1
            self.vectors_of_change[3+i*nRpa,i+3*M] = +1
            #
            # reaction Ia -> R at age group i:
            # population of Ia decreases by 1, population of R increases by 1
            self.vectors_of_change[4+i*nRpa,i+2*M] = -1
            self.vectors_of_change[4+i*nRpa,i+7*M] = +1
            #
            # reaction Is -> R at age group i:
            # population of Is decreases by 1, population of R increases by 1
            self.vectors_of_change[5+i*nRpa,i+3*M] = -1
            self.vectors_of_change[5+i*nRpa,i+7*M] = +1
            #
            # reaction Is -> Ih at age group i:
            self.vectors_of_change[6+i*nRpa,i+3*M] = -1
            self.vectors_of_change[6+i*nRpa,i+4*M] = +1
            #
            # reaction Ih -> R at age group i:
            self.vectors_of_change[7+i*nRpa,i+4*M] = -1
            self.vectors_of_change[7+i*nRpa,i+7*M] = +1
            #
            # reaction Ih -> Ic at age group i:
            self.vectors_of_change[8+i*nRpa,i+4*M] = -1
            self.vectors_of_change[8+i*nRpa,i+5*M] = +1
            #
            # reaction Ic -> R at age group i:
            self.vectors_of_change[9+i*nRpa,i+5*M] = -1
            self.vectors_of_change[9+i*nRpa,i+7*M] = +1
            #
            # reaction Ic -> Im at age group i:
            self.vectors_of_change[10+i*nRpa,i+5*M] = -1
            self.vectors_of_change[10+i*nRpa,i+6*M] = +1

        self.readData = {'Ei':[1,2], 'Iai':[2,3],
                        'Isi':[3,4],
                        'Ihi':[4,5],
                        'Ici':[5,6],
                        'Imi':[6,7], 'Rind':6}

    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh,  gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1, ce2
            #
            long [:] S    = xt[0  :M]
            long [:] E    = xt[M  :2*M]
            long [:] Ia   = xt[2*M:3*M]
            long [:] Is   = xt[3*M:4*M]
            long [:] Ih   = xt[4*M:5*M]
            long [:] Ic   = xt[5*M:6*M]
            long [:] Im   = xt[6*M:7*M]
            long [:] R    = xt[7*M:8*M]
            #
            long [:] Ni    = self.Ni
            #
            double [:] alpha= self.alpha
            double [:] sa   = self.sa
            double [:] iaa  = self.iaa
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            #
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            int nRpa = self.nReactions_per_agegroup

        # update Ni
        for i in range(M):
            Ni[i] = S[i] + E[i] + Ia[i] + Is[i] + Ih[i] + Ic[i] + R[i]

        for i in range(M):
            lmda=0;   ce1=gE*alpha[i];  ce2=gE-ce1
            for j in range(M):
                lmda += beta*CM[i,j]*(Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            rates[  i*nRpa]  = sa[i] # birth rate
            rates[1+i*nRpa]  =  rateS  # rate S -> E
            rates[2+i*nRpa]  = ce1 * E[i] # rate E -> Ia
            rates[3+i*nRpa]  = ce2 * E[i] # rate E -> Is
            #
            rates[4+i*nRpa]  = gIa * Ia[i] # rate Ia -> R
            #
            rates[5+i*nRpa]  = (1.-hh[i])*gIs * Is[i] # rate Is -> R
            rates[6+i*nRpa]  = hh[i]*gIs * Is[i] # rate Is -> Ih
            #
            rates[7+i*nRpa]  = (1.-cc[i])*gIh * Ih[i] # rate Ih -> R
            rates[8+i*nRpa]  = cc[i]*gIh * Ih[i] # rate Ih -> Ic
            #
            rates[9+i*nRpa]  = (1.-mm[i])*gIc * Ic[i] # rate Ic -> R
            rates[10+i*nRpa] = mm[i]*gIc * Ic[i] # rate Ic -> Im
            #
        return


    cpdef simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0,
                  contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = Ia0[i]
            xt[i+3*M] = Is0[i]
            xt[i+4*M] = Ih0[i]
            xt[i+5*M] = Ic0[i]
            xt[i+6*M] = Im0[i]
            xt[i+7*M] = self.Ni[i] - S0[i] - E0[i] - Ia0[i] - Is0[i] \
                                   - Ih0[i] - Ic0[i] - Im0[i]
            if xt[i+7*M] < 0:
                xt[i+7*M] = 0
                # It is probably better to have the error message below,
                # to warn the user if their input numbers do not match?
                #raise RuntimeError("Sum of provided initial populations for class" + \
                #    " {0} exceeds total initial population for that class\n".format(i) + \
                #    " {0} > {1}".format(xt[i+7*M],self.Ni[i]))

        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)
        # Instead of the removed population, which is stored in the last compartment,
        # we want to output the total alive population (whose knowledge is mathematically
        # equivalent to knowing the removed population).
        for i in range(M):
            out_arr[:,i+7*M] += out_arr[:,i+5*M] + out_arr[:,i+4*M] + out_arr[:,i+3*M]
            out_arr[:,i+7*M] += out_arr[:,i+2*M] + out_arr[:,i+1*M] + out_arr[:,i+0*M]


        out_dict = {'X':out_arr, 't':t_arr,
                      'Ni':self.Ni, 'M':self.M,
                      'alpha':self.alpha, 'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,
                      'gIh':self.gIh,'gIc':self.gIc,
                      'fsa':self.fsa,'fh':self.fh,
                      'gE':self.gE,
                      'sa':self.sa,'hh':self.hh,
                      'mm':self.mm,'cc':self.cc,
                      'iaa':self.iaa,
                      }
        self.population = (out_dict['X'])[:,7*self.M:8*self.M]
        return out_dict


    cpdef simulate_events(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0,
                events, contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = Ia0[i]
            xt[i+3*M] = Is0[i]
            xt[i+4*M] = Ih0[i]
            xt[i+5*M] = Ic0[i]
            xt[i+6*M] = Im0[i]
            xt[i+7*M] = self.Ni[i] - S0[i] - E0[i] - Ia0[i] - Is0[i]
            xt[i+7*M] -= Ih0[i] + Ic0[i] + Im0[i]
            if xt[i+7*M] < 0:
                xt[i+7*M] = 0
                # It is probably better to have the error message below,
                # to warn the user if their input numbers do not match?
                #raise RuntimeError("Sum of provided initial populations for class" + \
                #    " {0} exceeds total initial population for that class\n".format(i) + \
                #    " {0} > {1}".format(xt[i+7*M],self.Ni[i]))

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                    'Ni':self.Ni, 'M':self.M,
                    'alpha':self.alpha, 'beta':self.beta,
                    'gIa':self.gIa,'gIs':self.gIs,
                    'gIh':self.gIh,'gIc':self.gIc,
                    'fsa':self.fsa,'fh':self.fh,
                    'gE':self.gE,
                    'sa':self.sa,'hh':self.hh,
                    'mm':self.mm,'cc':self.cc,
                    'iaa':self.iaa,
                    }

        self.population = (out_dict['X'])[:,7*self.M:8*self.M]
        return out_dict



cdef class SEAI5R(stochastic_integration):
    warnings.warn('SEAI5R not supported', DeprecationWarning)
    """
    Susceptible, Exposed, Activates, Infected, Removed (SEAIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality
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
            Fraction by which symptomatic individuals do not self-isolate.
        gE: float
            rate of removal from exposeds individuals.
        gA: float
            rate of removal from activated individuals.
        gIh: float
            rate of hospitalisation of infected individuals.
        gIc: float
            rate hospitalised individuals are moved to intensive care.
        seed: long
            seed for pseudo-random number generator (optional).
    M: int
        Number of compartments of individual for each class.
        I.e len(contactMatrix)
    Ni: np.array(9*M, )
        Initial number in each compartment and class.
    """

    cdef:
        readonly double beta, gE, gA, gIa, gIs, gIh, gIc, fsa, fh
        readonly np.ndarray xt0, Ni, dxtdt, CC, sa, hh, cc, mm, alpha, population
        int nClass_

    def __init__(self, parameters, M, Ni):
        cdef:
            int nRpa # short for number of reactions per age group

        alpha      = parameters['alpha']                    # fraction of asymptomatic infectives
        self.beta  = parameters['beta']                     # infection rate
        self.gE    = parameters['gE']                       # progression rate of E class
        self.gA    = parameters['gA']                       # progression rate of A class
        self.gIa   = parameters['gIa']                      # removal rate of Ia
        self.gIs   = parameters['gIs']                      # removal rate of Is
        self.gIh   = parameters['gIh']                      # removal rate of Ih
        self.gIc   = parameters['gIc']                      # removal rate of Ic
        self.fsa   = parameters['fsa']                      # the self-isolation parameter of symptomatics
        self.fh    = parameters['fh']                       # the self-isolation parameter of hospitalizeds

        sa         = parameters['sa']                       # daily arrival of new susceptibles
        hh         = parameters['hh']                       # hospital
        cc         = parameters['cc']                       # ICU
        mm         = parameters['mm']                       # mortality
        #iaa        = parameters['iaa')                      # daily arrival of new asymptomatics

        self.N     = np.sum(Ni)
        self.M     = M
        #self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = np.array( Ni.copy(), dtype=long)

        self.nClass = 8  # number of classes (used in unit tests)
        self.nClass_ = 9 # number of explicit classes used in this function
        # explicit states per age group:
        # 1. S    susceptibles
        # 2. E    exposed
        # 3. A    Asymptomatic and infected
        # 4. Ia   infectives, asymptomatic
        # 5. Is   infectives, symptomatic
        # 6. Ih   infectives, hospitalised
        # 7. Ic   infectives, in ICU
        # 8. Im   infectives, deceased
        # 9. R    removed

        self.nReactions_per_agegroup = 12
        self.nReactions = self.M * self.nReactions_per_agegroup
        self.dim_state_vec = self.nClass_ * self.M

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.rates = np.zeros( self.nReactions , dtype=DTYPE)  # rate matrix
        self.overdispersions = np.ones( self.nReactions , dtype=DTYPE)
        self.xt = np.zeros([self.dim_state_vec],dtype=long) # state
        self.xtminus1 = np.zeros([self.dim_state_vec],dtype=long) # previous state
        # (for event-driven simulations)

        self.alpha    = np.zeros( self.M, dtype = DTYPE)
        if np.size(alpha)==1:
            self.alpha = alpha*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha= alpha
        else:
            raise Exception('alpha can be a number or an array of size M')

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

        # Set seed for pseudo-random number generator (if provided)
        try:
            self.initialize_random_number_generator(
                                  supplied_seed=parameters['seed'])
        except KeyError:
            self.initialize_random_number_generator()

        # Set overdispersion mode (if provided)
        try:
            self.overdispersion_mode=parameters['overdispersion_mode']
        except KeyError:
            self.overdispersion_mode = 0

        # create vectors of change for reactions
        self.vectors_of_change = np.zeros((self.nReactions,self.dim_state_vec),
                                          dtype=long)
        # self.vectors_of_change[i,j] = change in population j at reaction i
        nRpa = self.nReactions_per_agegroup
        for i in range(M):
            # birth rate
            # population of S increases by 1
            self.vectors_of_change[  i*nRpa,i    ] = +1
            #
            # reaction S -> E at age group i:
            # population of S decreases by 1, population of E increases by 1
            self.vectors_of_change[1+i*nRpa,i    ] = -1
            self.vectors_of_change[1+i*nRpa,i+  M] = +1
            #
            # reaction E -> A at age group i:
            # population of E decreases by 1, population of A increases by 1
            self.vectors_of_change[2+i*nRpa,i+  M] = -1
            self.vectors_of_change[2+i*nRpa,i+2*M] = +1
            #
            # reaction A -> Ia at age group i:
            # population of A decreases by 1, population of Ia increases by 1
            self.vectors_of_change[3+i*nRpa,i+2*M] = -1
            self.vectors_of_change[3+i*nRpa,i+3*M] = +1
            #
            # reaction A -> Is at age group i:
            # population of E decreases by 1, population of Is increases by 1
            self.vectors_of_change[4+i*nRpa,i+2*M] = -1
            self.vectors_of_change[4+i*nRpa,i+4*M] = +1
            #
            # reaction Ia -> R at age group i:
            # population of Ia decreases by 1, population of R increases by 1
            self.vectors_of_change[5+i*nRpa,i+3*M] = -1
            self.vectors_of_change[5+i*nRpa,i+8*M] = +1
            #
            # reaction Is -> R at age group i:
            # population of Is decreases by 1, population of R increases by 1
            self.vectors_of_change[6+i*nRpa,i+4*M] = -1
            self.vectors_of_change[6+i*nRpa,i+8*M] = +1
            #
            # reaction Is -> Ih at age group i:
            self.vectors_of_change[7+i*nRpa,i+4*M] = -1
            self.vectors_of_change[7+i*nRpa,i+5*M] = +1
            #
            # reaction Ih -> R at age group i:
            self.vectors_of_change[8+i*nRpa,i+5*M] = -1
            self.vectors_of_change[8+i*nRpa,i+8*M] = +1
            #
            # reaction Ih -> Ic at age group i:
            self.vectors_of_change[9+i*nRpa,i+5*M] = -1
            self.vectors_of_change[9+i*nRpa,i+6*M] = +1
            #
            # reaction Ic -> R at age group i:
            self.vectors_of_change[10+i*nRpa,i+6*M] = -1
            self.vectors_of_change[10+i*nRpa,i+8*M] = +1
            #
            # reaction Ic -> Im at age group i:
            self.vectors_of_change[11+i*nRpa,i+6*M] = -1
            self.vectors_of_change[11+i*nRpa,i+7*M] = +1

        self.readData = {'Ei':[1,2], 'Ai':[2,3], 'Iai':[3,4],
                        'Isi':[4,5],
                        'Ihi':[5,6],
                        'Ici':[6,7],
                        'Imi':[7,8], 'Rind':7}

    cdef rate_vector(self, xt, tt):
        cdef:
            int N=self.N, M=self.M, i, j
            double beta=self.beta, rateS, lmda
            double fsa=self.fsa, fh=self.fh, gE=self.gE,  gA=self.gA
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double gAA, gAS
            #
            long [:] S    = xt[0  :  M]
            long [:] E    = xt[M  :2*M]
            long [:] A    = xt[2*M:3*M]
            long [:] Ia   = xt[3*M:4*M]
            long [:] Is   = xt[4*M:5*M]
            long [:] Ih   = xt[5*M:6*M]
            long [:] Ic   = xt[6*M:7*M]
            long [:] Im   = xt[7*M:8*M]
            long [:] R    = xt[8*M:9*M]
            #
            long [:] Ni    = self.Ni
            #
            double [:] alpha= self.alpha
            double [:] sa   = self.sa
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            #
            double [:,:] CM = self.CM
            double [:] rates = self.rates
            int nRpa = self.nReactions_per_agegroup

        # update Ni
        for i in range(M):
            Ni[i] = S[i] + E[i] + A[i] + Ia[i] + Is[i] + Ih[i] + Ic[i] + R[i]

        for i in range(M):
            lmda=0;   gAA=gA*alpha[i];  gAS=gA-gAA
            for j in range(M):
                lmda += beta*CM[i,j]*(A[j] + Ia[j]+fsa*Is[j]+fh*Ih[j])/Ni[j]
            rateS = lmda*S[i]
            #
            # rates from S
            rates[  i*nRpa]  = sa[i] # birth rate
            rates[1+i*nRpa]  = rateS  # rate S -> E
            # rates from E
            rates[2+i*nRpa]  = gE * E[i] # rate E -> A
            # rates from A
            rates[3+i*nRpa]  = gAA * A[i] # rate A -> Ia
            rates[4+i*nRpa]  = gAS * A[i] # rate A -> Is
            # rates from Ia
            rates[5+i*nRpa]  = gIa * Ia[i] # rate Ia -> R
            # rates from Is
            rates[6+i*nRpa]  = (1.-hh[i])*gIs * Is[i] # rate Is -> R
            rates[7+i*nRpa]  = hh[i]*gIs * Is[i] # rate Is -> Ih
            # rate from Ih
            rates[8+i*nRpa]  = (1.-cc[i])*gIh * Ih[i] # rate Ih -> R
            rates[9+i*nRpa]  = cc[i]*gIh * Ih[i] # rate Ih -> Ic
            # rates from Ic
            rates[10+i*nRpa]  = (1.-mm[i])*gIc * Ic[i] # rate Ic -> R
            rates[11+i*nRpa]  = mm[i]*gIc * Ic[i] # rate Ic -> Im
            #
        return


    cpdef simulate(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                  contactMatrix, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt

        R_0 = self.Ni-(S0+E0+A0+Ia0+Is0+Ih0+Ic0)
        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Ih0[i]
            xt[i+6*M] = Ic0[i]
            xt[i+7*M] = Im0[i]
            xt[i+8*M] = R_0[i]
            #print(xt[i+7*M])
            if xt[i+8*M] < 0:
                raise RuntimeError("Sum of provided initial populations for class" + \
                    " {0} exceeds total initial population for that class\n".format(i) + \
                    " {0} > {1}".format(xt[i+8*M],self.Ni[i]))
        if method.lower() == 'gillespie':
            t_arr, out_arr =  self.simulate_gillespie(contactMatrix, Tf, Nf)
        else:
            t_arr, out_arr =  self.simulate_tau_leaping(contactMatrix, Tf, Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency)
        # Instead of the removed population, which is stored in the last compartment,
        # we want to output the total alive population (whose knowledge is mathematically
        # equivalent to knowing the removed population).
        for i in range(M):
            out_arr[:,i+8*M] += out_arr[:,i+6*M]
            out_arr[:,i+8*M] += out_arr[:,i+5*M] + out_arr[:,i+4*M] + out_arr[:,i+3*M]
            out_arr[:,i+8*M] += out_arr[:,i+2*M] + out_arr[:,i+1*M] + out_arr[:,i]

        out_dict = {'X':out_arr, 't':t_arr,
                      'Ni':self.Ni, 'M':self.M,
                      'alpha':self.alpha, 'beta':self.beta,
                      'gIa':self.gIa,'gIs':self.gIs,
                      'gIh':self.gIh,'gIc':self.gIc,
                      'fsa':self.fsa,'fh':self.fh,
                      'gE':self.gE,'gA':self.gA,
                      'sa':self.sa,'hh':self.hh,
                      'mm':self.mm,'cc':self.cc,
                      #'iaa':self.iaa,
                      }
        self.population = (out_dict['X'])[:,8*self.M:9*self.M]
        return out_dict



    cpdef simulate_events(self, S0, E0, A0, Ia0, Is0, Ih0, Ic0, Im0,
                events, contactMatrices, Tf, Nf,
                method='gillespie',
                int nc=30, double epsilon = 0.03,
                int tau_update_frequency = 1,
                  events_repeat=False,events_subsequent=True,
                  stop_at_event=False,
                ):
        cdef:
            int M = self.M, i
            long [:] xt = self.xt
            list events_out
            np.ndarray out_arr, t_arr

        # write initial condition to xt
        for i in range(M):
            xt[i] = S0[i]
            xt[i+M] = E0[i]
            xt[i+2*M] = A0[i]
            xt[i+3*M] = Ia0[i]
            xt[i+4*M] = Is0[i]
            xt[i+5*M] = Ih0[i]
            xt[i+6*M] = Ic0[i]
            xt[i+7*M] = Im0[i]
            xt[i+8*M] = self.Ni[i] - S0[i] - E0[i] - A0[i] - Ia0[i] - Is0[i]
            xt[i+8*M] -= Ih0[i] + Ic0[i] + Im0[i]
            #print(xt[i+7*M])
            if xt[i+8*M] < 0:
                raise RuntimeError("Sum of provided initial populations for class" + \
                    " {0} exceeds total initial population for that class\n".format(i) + \
                    " {0} > {1}".format(xt[i+8*M],self.Ni[i]))

        if method.lower() == 'gillespie':
            t_arr, out_arr, events_out =  self.simulate_gillespie_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        else:
            t_arr, out_arr, events_out =  self.simulate_tau_leaping_events(events=events,
                                  contactMatrices=contactMatrices,
                                  Tf=Tf, Nf=Nf,
                                  nc=nc,
                                  epsilon= epsilon,
                                  tau_update_frequency=tau_update_frequency,
                                  events_repeat=events_repeat,
                                  events_subsequent=events_subsequent,
                                  stop_at_event=stop_at_event)
        # Instead of the removed population, which is stored in the last compartment,
        # we want to output the total alive population (whose knowledge is mathematically
        # equivalent to knowing the removed population).
        for i in range(M):
            out_arr[:,i+8*M] += out_arr[:,i+6*M]
            out_arr[:,i+8*M] += out_arr[:,i+5*M] + out_arr[:,i+4*M] + out_arr[:,i+3*M]
            out_arr[:,i+8*M] += out_arr[:,i+2*M] + out_arr[:,i+1*M] + out_arr[:,i+0*M]

        out_dict = {'X':out_arr, 't':t_arr,  'events_occured':events_out,
                    'Ni':self.Ni, 'M':self.M,
                    'alpha':self.alpha, 'beta':self.beta,
                    'gIa':self.gIa,'gIs':self.gIs,
                    'gIh':self.gIh,'gIc':self.gIc,
                    'fsa':self.fsa,'fh':self.fh,
                    'gE':self.gE,'gA':self.gA,
                    'sa':self.sa,'hh':self.hh,
                    'mm':self.mm,'cc':self.cc,
                    #'iaa':self.iaa,
                    }
        self.population = (out_dict['X'])[:,8*self.M:9*self.M]
        return out_dict
