# This file contains a util function (minimization) that needs to be implemented in pure python (not cython).
# Otherwise, the p.map call does not work with the lambda function.

import multiprocessing
import os
import numpy as np
import nlopt
import cma
import sys
import traceback
import pickle
from scipy.stats import truncnorm, lognorm
from itertools import product

try:
    # Optional support for multiprocessing in the minimization function.
    import pathos.multiprocessing as pathos_mp
except ImportError:
    pathos_mp = None

def minimization(objective_fct, guess, bounds, global_max_iter=100,
                 local_max_iter=100, ftol=1e-2, global_atol=1,
                 enable_global=True, enable_local=True, local_initial_step=None, 
                 cma_processes=0, cma_population=16, cma_stds=None,
                 cma_random_seed=None, verbose=True, tmp_file=None, load_backup_file=None, args_dict={}):
    """ Compute the global minimum of the objective function.

    This function computes the global minimum of `objective_fct` using a combination of a global minimisation step
    (CMA-ES) and a local refinement step (NEWUOA) (both derivative free).

    Parameters
    ----------
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
    local_initital_step: optional, float or np.array
        Initial step size for the local optimiser. If scalar, relative to the initial guess. 
        Default: Deterined by final state of global optimiser, or, if enable_global=False, 0.01
    ftol: float
        Relative function value stopping criterion for the optimisation algorithms.
    global_atol: float
        The absolute tolerance for global optimisation.
    enable_global: bool
        Enable (or disable) the global minimisation part.
    enable_local: bool
        Enable (or disable) the local minimisation part (run after the global minimiser).
    cma_processes: int
        Number of processes used in the CMA algorithm. By default, the number of CPU cores is used.
    cma_population: int
        The number of samples used in each step of the CMA algorithm. Should ideally be factor of `cma_processes`.
    cma_stds: numpy.array
        Initial standard deviation of the spread of the population for each parameter in the CMA algorithm. Ideally,
        one should have the optimum within 3*sigma of the guessed initial value. If not specified, these values are
        chosen such that 3*sigma reaches one of the boundaries for each parameters.
    cma_random_seed: int (between 0 and 2**32-1)
        Random seed for the optimisation algorithms. By default it is generated from numpy.random.randint.
    verbose: bool
        Enable output.
    tmp_file: optional, string
        If specified, name of a file to store the state of the optimiser as backup
    load_backup_file: optional, string
        If specified, name of a file to restore the the state of the optimiser
    args_dict: dict
        Key-word arguments that are passed to the minimisation function.

    Returns
    -------
    x_result, y_result
        Returns parameter estimate and minimal value.
    """
    x_result = guess
    y_result = 0


    # Step 1: Global optimisation
    if enable_global:
        if verbose:
            if load_backup_file is None:
                print('Starting global minimisation ...')
            else:
                print('Continuing global minimisation from backup ...')

        if not pathos_mp and cma_processes != 1:
            print('Warning: Optional dependecy for multiprocessing support `pathos` not installed.')
            print('         Switching to single processed mode (cma_processes = 1).')
        cma_processes = _get_number_processes(cma_processes)

        options = cma.CMAOptions()
        options['bounds'] = [bounds[:, 0], bounds[:, 1]]
        options['tolfun'] = global_atol
        options['popsize'] = cma_population

        if cma_stds is None:
            # Standard scale: 3*sigma reaches from the guess to the closest boundary for each parameter.
            cma_stds = np.amin([bounds[:, 1] - guess, guess -  bounds[:, 0]], axis=0)
            cma_stds *= 1.0/3.0
        options['CMA_stds'] = cma_stds

        if cma_random_seed is None:
            cma_random_seed = np.random.randint(2**32-2)
        options['seed'] = cma_random_seed
        if load_backup_file is None:
            global_opt = cma.CMAEvolutionStrategy(guess, 1.0, options)
        else:
            try:
                s = open(load_backup_file, 'rb').read() 
                global_opt = pickle.loads(s)
            except:
                print('Backup file not found or invalid, starting new optimisation')
                global_opt = cma.CMAEvolutionStrategy(guess, 1.0, options)
                
        iteration = 0
        while not global_opt.stop() and iteration < global_max_iter:
            positions = global_opt.ask()
            # Use multiprocess pool for parallelisation. This only works if this function is not in a cython file,
            # otherwise, the lambda function cannot be passed to the other processes. It also needs an external Pool
            # implementation (from `pathos.multiprocessing`) since the python internal one does not support lambda fcts.
            try:
                values = _take_global_optimisation_step(positions, objective_fct, cma_processes, **args_dict)
            except KeyboardInterrupt:
                print("Global optimisation: Interrupting global minimisation...")
                iteration = global_max_iter + 1 # break out of the optimisation loop
            except Exception as e:
                print("Exception in multiprocessing: ")
                print("Caught: {}".format(e))
                if verbose:
                    traceback.print_exc(file=sys.stdout)
                print("Will fallback to using a single core. Setting cma_processes = 1")
                cma_processes = 1
                values = _take_global_optimisation_step(positions, objective_fct, cma_processes, **args_dict)
            global_opt.tell(positions, values)
            if tmp_file is not None:
                s = global_opt.pickle_dumps()
                open(tmp_file, 'wb').write(s)
            if verbose:
                global_opt.disp()
            iteration += 1

        x_result = global_opt.best.x
        y_result = global_opt.best.f

        if verbose:
            if iteration == global_max_iter:
                print("Global optimisation: Maximum number of iterations reached.")
            print('Optimal value (global minimisation): ', y_result)
            print('Starting local minimisation...')

    # Step 2: Local refinement
    if enable_local:
        # Use derivative free local optimisation algorithm with support for boundary conditions
        # to converge to the next minimum (which is hopefully the global one).
        local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, guess.shape[0])
        local_opt.set_min_objective(lambda x, grad: objective_fct(x, grad, **args_dict))
        local_opt.set_lower_bounds(bounds[:,0])
        local_opt.set_upper_bounds(bounds[:,1])
        local_opt.set_ftol_rel(ftol)
        local_opt.set_maxeval(3*local_max_iter)

        if enable_global:
            # CMA gives us the scaling of the varialbes close to the minimum
            min_stds = global_opt.result.stds
            # These values can sometimes create initial steps outside of the boundaries (in particular if CMA is
            # only run for short times). This seems to create problems for NLopt, so we restrict the steps here
            # to be within the boundaries.
            min_stds = np.minimum(min_stds, np.amin([bounds[:, 1] - x_result, x_result -  bounds[:, 0]], axis=0))
            local_opt.set_initial_step(1/2 * min_stds)
        else:
            local_opt.set_initial_step(0.01*guess)
    
        if local_initial_step is not None:
            if type(local_initial_step) is float:
                local_opt.set_initial_step(local_initial_step*guess)
            elif (type(local_initial_step) is np.ndarray) and len(local_initial_step)==len(guess):
                local_opt.set_initial_step(local_initial_step)
            else:
                raise Exception('Wrong length of local_initial_step')

        x_result = local_opt.optimize(x_result)
        y_result = local_opt.last_optimum_value()

        if verbose:
            if local_opt.get_numevals() == 3*local_max_iter:
                print("Local optimisation: Maximum number of iterations reached.")
            print('Optimal value (local minimisation): ', y_result)

    return x_result, y_result

def _get_number_processes(procs):
    """
    If pathos_mp is not found, will return 1. If procs is 0 and pathos_mp is found, it will return number of available
    cpus. Else returns procs
    """
    if not pathos_mp:
        return 1
    elif procs == 0:
        from sys import platform
        if platform=='linux':
            return len(os.sched_getaffinity(0)) 
        else:
            return multiprocessing.cpu_count()
    else:
        return procs


def _take_global_optimisation_step(positions, objective_function, cma_processes, **kwargs):
    """
    Takes a global optimisation step either using one core, or multiprocessing
    """
    assert cma_processes > 0, "cma_processes must be bigger than 0"
    if cma_processes == 1:
        ret = [objective_function(x, grad=0, **kwargs) for x in positions]
    elif cma_processes > 1:
        # Using the pool as a context manager will make any exceptions raised kill the other processes.
        with pathos_mp.ProcessingPool(cma_processes) as pool:
            ret = pool.map(lambda x: objective_function(x, grad=0, **kwargs), positions)
    return ret


def parse_prior_fun(name, bounds, mean, std):
    if name == 'lognorm':
        return lognorm_rv(bounds, mean, std)
    elif name == 'truncnorm':
        return truncnorm_rv(bounds, mean, std)
    else:
        raise Exception('Invalid prior_fun. Choose between lognorm and truncnorm')

class Prior:
    def __init__(self, names, bounds, means, stds):
        bounds = np.array(bounds)
        means = np.array(means)
        stds = np.array(stds)
        
        self.dim = len(names)
        self.rv_names = np.unique(names)
        self.N = len(self.rv_names)
        self.masks = [np.array([name == rv_name for name in names]) for rv_name in self.rv_names]
        
        self.rvs = []
        for i in range(self.N):
            mask = self.masks[i]
            name = self.rv_names[i]
            rv = parse_prior_fun(name, bounds[mask,:], means[mask], stds[mask])
            self.rvs.append(rv)

    def logpdf(self, x):
        logpdfs = np.empty_like(x)
        for i in range(self.N):
            mask = self.masks[i]
            logpdfs[mask] = self.rvs[i].logpdf(x[mask])
        return logpdfs

    def ppf(self, x):
        ppfs = np.empty_like(x)
        for i in range(self.N):
            mask = self.masks[i]
            ppfs[...,mask] = self.rvs[i].ppf(x[...,mask])
        return ppfs

class truncnorm_rv:
    def __init__(self, bounds, mean, std):
        a = (bounds[:,0] - mean)/std
        b = (bounds[:,1] - mean)/std
        self.rv = truncnorm(a, b, loc=mean, scale=std)

    def logpdf(self, x):
        return self.rv.logpdf(x)

    def ppf(self, x):
        return self.rv.ppf(x)

class lognorm_rv:
    def __init__(self, bounds, mean, std):
        ndim = len(mean)
        
        var = std**2
        means_sq = mean**2
        scale = means_sq/np.sqrt(means_sq+var)
        s = np.sqrt(np.log(1+var/means_sq))
        self.rv = lognorm(s, scale=scale)
        self.norm = np.log(self.rv.cdf(bounds[:,1]) - self.rv.cdf(bounds[:,0]))

        # For inverse transform sampling of the truncated log-normal distribution.
        self.ppf_bounds = np.zeros((ndim, 2))
        self.ppf_bounds[:,0] = self.rv.cdf(bounds[:,0])
        self.ppf_bounds[:,1] = self.rv.cdf(bounds[:,1])
        self.ppf_bounds[:,1] = self.ppf_bounds[:,1] - self.ppf_bounds[:,0]

    def logpdf(self, x):
        return self.rv.logpdf(x) - self.norm

    def ppf(self, x):
        y = self.ppf_bounds[:,0] + x * self.ppf_bounds[:,1]
        return self.rv.ppf(y)


def hessian_finite_difference(pos, function, eps=1e-3, method="central", nprocesses=0, basis=None,
                              function_kwargs={}):
    """Forward finite-difference computation of the Hessian of a function.

    Parameters
    ----------
    pos:numpy.array(dims=1)
        Position at which the hessian is to be computed.
    function: function(numpy.array)
        Function of interest.
    pos: float or numpy.array(dims=1), optional
        Step size used for FD computation (can be parameter dependant).
    method: str
        Different options for the FD computation: "forward" or "central".
    nprocesses: int
        The number of processes used for the Hessian computation. By default, this
        chooses the number of CPU cores available.
    basis: numpy.array(dims=2), optional
        The orthogonal coordinate basis used for the finite difference scheme. Uses 
        canonical coordinates (basis=identity matrix) by default.

    Returns
    -------
    hess: numpy.array(dims=2)
        Hessian of function at pos.
    """
    k = len(pos)
    if not hasattr(eps, "__len__"):
        eps = eps*np.ones(k)
        
    if basis is None:
        basis = np.identity(k)

    procs = _get_number_processes(nprocesses)
    local_func = lambda x : function(x, **function_kwargs)

    if method == "forward":
        orig_pos = pos.copy()
        val_central = local_func(pos)

        def forward_eval(i):
            pos = orig_pos.copy()
            pos += eps[i] * basis[:,i]
            val = local_func(pos)
            return val

        if procs > 1:
            with pathos_mp.ProcessingPool(procs) as pool:
                val1 = pool.map(forward_eval, range(k))
        else:
            val1 = [forward_eval(i) for i in range(k)]

        def forward_eval2(index):
            i,j = index
            pos = orig_pos.copy()
            pos += eps[i] * basis[:,i]
            pos += eps[j] * basis[:,j]
            val2 = local_func(pos)

            hessian_entry = (val2 - val1[i] - val1[j] + val_central)/(eps[i]*eps[j])
            return hessian_entry
        
        if procs > 1:
            with pathos_mp.ProcessingPool(procs) as pool:
                hessian = pool.map(forward_eval2, product(range(k), range(k)))
        else:
            hessian = [forward_eval2(index) for index in product(range(k), range(k))]
        
        hessian = np.array(hessian).reshape((k,k))
        return basis @ (1/2 * (hessian + hessian.T)) @ basis.T

    if method == "central":
        orig_pos = pos.copy()
        index_list = []
        for i in range(k):
            for j in range(i+1):
                index_list.append((i,j))
        
        def central_eval(index):
            i,j = index
            pos = orig_pos.copy()
            pos += eps[i] * basis[:,i]
            pos += eps[j] * basis[:,j]
            val1 = local_func(pos)
            pos -= 2*eps[j] * basis[:,j]
            val2 = local_func(pos)
            pos = orig_pos.copy()
            pos -= eps[i] * basis[:,i]
            pos += eps[j] * basis[:,j]
            val3 = local_func(pos)
            pos -= 2*eps[j] * basis[:,j]
            val4 = local_func(pos)
            hessian_val = (val1 + val4 - val2 - val3) / (4*eps[i]*eps[j])
            return hessian_val
        
        if procs > 1:
            with pathos_mp.ProcessingPool(procs) as pool:
                hessian_vals = pool.map(central_eval, index_list)
        else:
            hessian_vals = [central_eval(index) for index in index_list]

        hessian = np.zeros((k,k))
        ctr = 0
        for i in range(k):
            for j in range(i+1):
                hessian[i,j] = hessian_vals[ctr]
                hessian[j,i] = hessian_vals[ctr]
                ctr += 1

        return basis @ hessian @ basis.T

    raise Exception("Finite-difference method must be 'forward' or 'central'.")

    
def eval_parallel(samples, function, nprocesses=0, function_kwargs={}):
    """Evaluate a function for many samples in parallel

    Parameters
    ----------
    samples:list
        Samples for which the function is to be computed.
    function: function(numpy.array)
        Function of interest.
    nprocesses: int
        The number of processes used for the Hessian computation. By default, this
        chooses the number of CPU cores available.

    Returns
    -------
    val: list
        Function values at samples.
    """

    procs = _get_number_processes(nprocesses)
    local_func = lambda x : function(x, **function_kwargs)

    if procs > 1:
        with pathos_mp.ProcessingPool(procs) as pool:
            val = pool.map(local_func, samples)
    else:
        val = [local_func(i) for x in samples]

    return val

