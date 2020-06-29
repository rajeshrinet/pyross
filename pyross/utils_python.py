# This file contains a util function (minimization) that needs to be implemented in pure python (not cython).
# Otherwise, the p.map call does not work with the lambda function.

import multiprocessing
import numpy as np
import nlopt
import cma

try:
    # Optional support for multiprocessing in the minimization function.
    import pathos.multiprocessing as pathos_mp
except ImportError:
    pathos_mp = None

try:
    # Optional support for nested sampling.
    import nestle
except ImportError:
    nestle = None

def minimization(objective_fct, guess, bounds, global_max_iter=100,
                local_max_iter=100, ftol=1e-2, global_atol=1,
                 enable_global=True, enable_local=True, cma_processes=0, cma_population=16, cma_stds=None,
                 cma_random_seed=None, verbose=True, args_dict={}):
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
            print('Starting global minimisation...')

        if cma_processes == 0:
            if pathos_mp:
                # Optional dependecy for multiprocessing (pathos) is installed.
                cma_processes = multiprocessing.cpu_count()
            else:
                cma_processes = 1

        if pathos_mp:
            p = pathos_mp.ProcessingPool(cma_processes)
        else:
            if cma_processes != 1:
                print('Warning: Optional dependecy for multiprocessing support `pathos` not installed.')
                print('         Switching to single processed mode (cma_processes = 1).')
                cma_processes = 1

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

        global_opt = cma.CMAEvolutionStrategy(guess, 1.0, options)
        iteration = 0
        while not global_opt.stop() and iteration < global_max_iter:
            positions = global_opt.ask()
            # Use multiprocess pool for parallelisation. This only works if this function is not in a cython file,
            # otherwise, the lambda function cannot be passed to the other processes. It also needs an external Pool
            # implementation (from `pathos.multiprocessing`) since the python internal one does not support lambda fcts.
            if cma_processes != 1:
                try:
                    values = p.map(lambda x: objective_fct(x, grad=0, **args_dict), positions)
                except:
                    # Some types of functions cannot be pickled (in particular functions that are defined in a function
                    # that is compiled with cython). This leads to an exception when trying to pass them to a different
                    # process. If this happens, we switch the algorithm to single process mode.
                    print('Warning: Running parallel optimization failed. Will switch to single-processed mode.')
                    cma_processes = 1
                    values = [objective_fct(x, 0, **args_dict) for x in positions]
            else:
                # Run the unparallelised version
                values = [objective_fct(x, 0, **args_dict) for x in positions]
            global_opt.tell(positions, values)
            if verbose:
                global_opt.disp()
            iteration += 1

        if pathos_mp:
            p.close()  # We need to close the pool, otherwise python does not terminate properly
            p.join()
            p.clear()  # If this is not set, pathos will reuse the pool we just closed, producing an error

        x_result = global_opt.best.x
        y_result = global_opt.best.f

        if verbose:
            print('Optimal value (global minimisation): ', y_result)
            print('Starting local minimisation...')

    # Step 2: Local refinement
    if enable_local:
        # Use derivative free local optimisation algorithm with support for boundary conditions
        # to converge to the next minimum (which is hopefully the global one).
        local_opt = nlopt.opt(nlopt.LN_BOBYQA, guess.shape[0])
        local_opt.set_min_objective(lambda x, grad: objective_fct(x, grad, **args_dict))
        local_opt.set_lower_bounds(bounds[:,0])
        local_opt.set_upper_bounds(bounds[:,1])
        local_opt.set_ftol_rel(ftol)
        local_opt.set_maxeval(3*local_max_iter)

        x_result = local_opt.optimize(x_result)
        y_result = local_opt.last_optimum_value()

        if verbose:
            print('Optimal value (local minimisation): ', y_result)

    return x_result, y_result


class PathosFuture:
    """Implement Future inferface needed for nestle."""
    def __init__(self, async_result, executor):
        self.executor = executor
        self.async_result = async_result

    def result(self, timeout=None):
       return self.async_result.get(timeout)

    def cancel(self):
        # We can't cancel individual computations, but we can terminate all tasks in the
        # process pool. This is the only use case of cancel in nestle, so it is safe to
        # do this here. Terminating and restarting the pool is quite slow, but typically
        # faster than waiting for all computations in the queue to finish.
        if self.executor.pool_running and not self.async_result.ready():
            self.executor.pool.terminate()
            self.executor.pool_running = False

        return True


class PathosExecutor:
    """Implement Executor interface needed for nestle."""
    def __init__(self, max_workers=None):
        # Check that pathos is installed.
        if pathos_mp is None:
            raise Exception("Multiprocessed nested sampling needs the pathos package.")

        # Check that nestle has a new enough version for the multiprocessing to work.
        if not hasattr(nestle, "FakeFuture"):
            # This will still run, although not multiprocessed. So just a warning here.
            print("Warning: The installed nestle version does not support multiprocessing. For a")
            print("parallelised version, install the current version from the github repository.")

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        self.pool = pathos_mp.ProcessingPool(max_workers)
        self.pool_running = True

    def submit(self, fn, *args, **kwargs):
        if not self.pool_running:
            self.pool.restart(True)
            self.pool_running = True

        r = self.pool.apipe(fn, *args, **kwargs)
        return PathosFuture(r, self)

    def map(self, fn, *iterables):
        if not self.pool_running:
            self.pool.restart(True)
            self.pool_running = True

        return self.pool.map(fn, *iterables)

    def shutdown(self, wait=True):
        self.pool.close()
        if wait:
            self.pool.join()
        self.pool.clear()


def nested_sampling(loglike, prior_transform, dim, queue_size, max_workers, verbose, method, npoints, maxiter, dlogz,
                    decline_factor, loglike_args, prior_transform_args):
    """Compute the log-evidence and a weighted samples of the log-likelihood using nested sampling.

    This function uses the `nestle` Python library to run nested sampling on the given log-likelihood. For a current
    version of `nestle`, the evaluations of the log-likelihood can be paralellised. This function returns the output
    of the nestle.sample function. Most of the parameters correspond to input parameters of nestle.sample. See the
    nestle documentation for details.

    Parameters
    ----------
    loglike: function
        The log-likelihood function. Keyword arguments can be passed to this function via the loglike_args parameter.
    prior_transform: function
        Return a prior sample from a random point in [0,1]^dim. Keyword arguments can be passed to this function via
        the prior_transform_args parameter.
    dim: int
        Dimension of the parameter space.
    queue_size: int
        Size of the internal queue of samples of the nested sampling algorithm. The log-likelihood of these samples
        is computed in parallel (if queue_size > 1).
    max_workers: int
        The maximal number of processes used to compute samples.
    verbose: bool
        If true, output the state of nestle.sample every 100 iterations.
    method: str
        The method used by nestle.sample ("single", "multi", "classic").
    npoints: int
        The number of active points. Larger numbers result in a more accurate evidence with higher computational cost.
    maxiter: int
        The maximum number of iterations in nestle.sample.
    dlogz: float
        Stopping threshold for the estimated error of the log-evidence. This option is mutually exclusive with `decline_factor`.
    decline_factor: float
        Stop the iteration when the weight (likelihood times prior volume) of newly saved samples has been declining for
        `decline_factor * nsamples` consecutive samples. This option is mutually exclusive with `dlogz`.
    loglike_args: dict
        Keyword parameter dictionary passed to loglike function.
    prior_transform_args: dict
        Keyword parameter dictionary passed to prior_transform function.

    Returns
    -------
    result: nestle.Result
        Result of the computation (log evidence is given by result.logz).
    """
    if nestle is None:
        raise Exception("Nested sampling needs the nestle package.")

    if verbose:
        def callback(d):
            if d["it"] % 100 == 0:
                print("Iteration {}: log_evidence = {}".format(d["it"], d["logz"]))
    else:
        callback = None

    ll = lambda params: loglike(params, **loglike_args)
    pt = lambda x: prior_transform(x, **prior_transform_args)
    if queue_size > 1:
        # Multiprocessed version.
        executor = PathosExecutor(max_workers)
        result = nestle.sample(ll, pt, dim, queue_size=queue_size, pool=executor, npoints=npoints, maxiter=maxiter,
                               dlogz=dlogz, decline_factor=decline_factor, callback=callback)
        executor.shutdown()

        return result

    # No multiprocessing.
    result = nestle.sample(ll, pt, dim, queue_size=queue_size, npoints=npoints, maxiter=maxiter,
                           dlogz=dlogz, decline_factor=decline_factor, callback=callback)
    return result
