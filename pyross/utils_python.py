# This file contains a util function (minimization) that needs to be implemented in pure python (not cython).
# Otherwise, the p.map call does not work with the lambda function.

import numpy as np
import nlopt
import cma
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

def minimization(objective_fct, guess, bounds, global_max_iter=100, local_max_iter=100, ftol=1e-2, global_ftol_factor=10., 
                 enable_global=True, enable_local=True, cma_processes=0, cma_population=16, cma_stds=None, 
                 verbose=True, args_dict={}):
    """ Compute the global minimum of the objective function.
    
    This function computes the global minimum of `objective_fct` using a combination of a global minimisation step 
    (CMA-ES) and a local refinement step (Suplex) (both derivative free).
    
    objective_fct: callable 
        The objective function. It must be of the form fct(params, grad=0) for the use in NLopt. The parameters 
        should not be modified and `grad` can be ignored (since only derivative free algorithms are used).
    guess: numpy.array
        The initial guess.
    bounds: numpy.array
        The boundaries for the optimisation algorithm, given as a dimsx2 matrix.
    global_max_iter: int
        The maximum number of iterations for the global algorithm.
    local_max_iter: int
        The maximum number of iterations for the local algorithm.
    ftol: float
        Relative function value stopping criterion for the optimisation algorithms.
    global_ftol_factor: float
        For the global optimisation, `ftol` gets multiplied by this.
    enable_global: bool
        Enable (or disable) the global minimisation part.
    enable_local: bool
        Enable (or disable) the local minimisation part (run after the global minimiser).
    cma_processes: int
        Number of processes used in the CMA algorithm. By default, the number of CPU cores is used.
    cma_population: int
        The number of samples used in each step of the CMA algorithm. Should ideally be factor of `cma_threads`.
    cma_stds: numpy.array
        Initial standard deviation of the spread of the population for each parameter in the CMA algorithm. Ideally, 
        one should have the optimum within 3*sigma of the guessed initial value. If not specified, these values are
        chosen such that 3*sigma reaches one of the boundaries for each parameters.
    verbose: bool
        Enable output.
    args_dict: dict
        Key-word arguments that are passed to the minimisation function.
    """         
    x_result = guess
    y_result = 0
    
    # Step 1: Global optimisation
    if enable_global:
        if verbose:
            print('Starting global minimisation...')
        
        if cma_processes == 0:
            cma_processes = multiprocessing.cpu_count()
        p = Pool(cma_processes)

        options = cma.CMAOptions()
        options['bounds'] = [bounds[:, 0], bounds[:, 1]]
        options['tolfunrel'] = ftol * global_ftol_factor
        options['popsize'] = cma_population

        if cma_stds is None:
            # Standard scale: 3*sigma reaches from the guess to the closest boundary for each parameter.
            cma_stds = np.amin([bounds[:, 1] - guess, guess -  bounds[:, 0]], axis=0)
            cma_stds *= 1.0/3.0
            
        options['CMA_stds'] = cma_stds

        global_opt = cma.CMAEvolutionStrategy(guess, 1.0, options)
        iteration = 0
        while not global_opt.stop() and iteration < global_max_iter:
            positions = global_opt.ask()
            # Endless parallelisation options here. Use pool for now.
            values = p.map(lambda x: objective_fct(x, grad=0, **args_dict), positions)
            global_opt.tell(positions, values)
            if verbose:
                global_opt.disp()
            iteration += 1

        x_result = global_opt.best.x
        y_result = global_opt.best.f
        
        if verbose:
            print('Optimal value (global minimisation): ', y_result)
            print('Starting local minimisation...')
    
    # Step 2: Local refinement
    if enable_local:
        # Use derivative free local optimisation algorithm with support for boundary conditions
        # to converge to the next minimum (which is hopefully the global one).
        local_opt = nlopt.opt(nlopt.LN_NEWUOA_BOUND, guess.shape[0])
        local_opt.set_min_objective(lambda x, grad: objective_fct(x, grad, **args_dict))
        local_opt.set_lower_bounds(bounds[:,0])
        local_opt.set_upper_bounds(bounds[:,1])
        local_opt.set_ftol_rel(ftol)
        local_opt.set_maxeval(3*local_max_iter)

        x_result = local_opt.optimize(global_opt.best.x)
        y_result = local_opt.last_optimum_value()
    
        if verbose:
            print('Optimal value (local minimisation): ', y_result)
    
    return x_result, y_result
