# This file implements two methods to compute the evidence of an epidemiological model. These are an addition to nested sampling 
# which can be used to estimate the evidence and which is already implemented in `pyross.inference`. However, for complicated, 
# high-dimensional models, nested sampling can be very slow. The methods implemented here are applicable whenever MCMC sampling 
# of the posterior is feasible and should work even for sophisticated epidemiological models. 

import numpy as np
from scipy.stats import lognorm
try:
    # Support for MCMC sampling.
    import emcee
except ImportError:
    emcee = None

try:
    # Optional support for multiprocessing.
    import pathos.multiprocessing as pathos_mp
except ImportError:
    pathos_mp = None

import pyross.utils

DTYPE = np.float

def get_parameters(estimator, x, Tf, prior_dict, contactMatrix=None, generator=None, intervention_fun=None, 
                   tangent=False):
    """Process an estimator from `pyross.inference` to generate input arguments for the evidence computations 
    `pyross.evidence.evidence_smc` and `pyross.evidence.evidence_path_sampling` for estimation problems without latent 
    variables. The input has the same structure as the input of `pyross.inference.infer`, see there for a detailed 
    documentation of the arguments.

    Parameters
    ----------
    estimator: pyross.inference.SIR_type
        The estimator object of the underlying (non-latent) inference problem.
    x: 2d numpy.array
    Tf: float
    prior_dict: dict
    contactMatrix: callable, optional
    generator: pyross.contactMatrix, optional
    intervention_fun: callable, optional
    tangent: bool, optional

    Returns
    -------
    logl:
        The log-likelihood of the inference problem.
    s:
        The shape parameter of the prior lognormal distribution.
    scale:
        The scale parameter of the prior lognormal distribution.
    bounds:
        The bounds for the log-likelihood function.
    """
    # Sanity checks of the inputs
    if (contactMatrix is None) == (generator is None):
        raise Exception('Specify either a fixed contactMatrix or a generator')
    if (intervention_fun is not None) and (generator is None):
        raise Exception('Specify a generator')
    if contactMatrix is not None:
        estimator.set_contact_matrix(contactMatrix)

    # Read in parameter priors
    keys, guess, stds, bounds, \
    flat_guess_range, is_scale_parameter, scaled_guesses  \
            = pyross.utils.parse_param_prior_dict(prior_dict, estimator.M)
    s, scale = pyross.utils.make_log_norm_dist(guess, stds)

    logl = lambda params: estimator._loglikelihood(params, contactMatrix=contactMatrix, generator=generator, 
                intervention_fun=intervention_fun, keys=keys, x=x, Tf=Tf, tangent=tangent,
                is_scale_parameter=is_scale_parameter, flat_guess_range=flat_guess_range,
                scaled_guesses=scaled_guesses, bounds=bounds)
    
    return logl, s, scale, bounds


def latent_get_parameters(estimator, obs, fltr, Tf, param_priors, init_priors, contactMatrix=None, generator=None, 
                          intervention_fun=None, tangent=False):
    """Process an estimator from `pyross.inference` to generate input arguments for the evidence computations 
    `pyross.evidence.evidence_smc` and `pyross.evidence.evidence_path_sampling` for estimation problems with latent 
    variables. The input has the same structure as the input of `pyross.inference.latent_infer`, see there for a detailed 
    documentation of the arguments.

    Parameters
    ----------
    estimator: pyross.inference.SIR_type
        The estimator object of the underlying (non-latent) inference problem.
    obs:  np.array
    fltr: 2d np.array
    Tf: float
    param_priors: dict
    init_priors: dict
    contactMatrix: callable, optional
    generator: pyross.contactMatrix, optional
    intervention_fun: callable, optional
    tangent: bool, optional

    Returns
    -------
    logl:
        The log-likelihood of the inference problem.
    s:
        The shape parameter of the prior lognormal distribution.
    scale:
        The scale parameter of the prior lognormal distribution.
    bounds:
        The bounds for the log-likelihood function.
    """
    # Sanity checks of the inputs
    if (contactMatrix is None) == (generator is None):
        raise Exception('Specify either a fixed contactMatrix or a generator')
    if (intervention_fun is not None) and (generator is None):
        raise Exception('Specify a generator')
    if contactMatrix is not None:
        estimator.set_contact_matrix(contactMatrix)
    
    # Process fltr and obs
    fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

    # Read in parameter priors
    keys, param_guess, param_stds, param_bounds, param_guess_range, \
    is_scale_parameter, scaled_param_guesses \
        = pyross.utils.parse_param_prior_dict(param_priors, estimator.M)

    # Read in initial conditions priors
    init_guess, init_stds, init_bounds, init_flags, init_fltrs \
        = pyross.utils.parse_init_prior_dict(init_priors, estimator.dim, len(obs0))

    # Concatenate the flattend parameter guess with init guess
    param_length = param_guess.shape[0]
    guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
    stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
    bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

    s, scale = pyross.utils.make_log_norm_dist(guess, stds)

    logl = lambda params: estimator._loglikelihood_latent(params, generator=generator, intervention_fun=intervention_fun, 
                param_keys=keys, param_guess_range=param_guess_range, is_scale_parameter=is_scale_parameter, 
                scaled_param_guesses=scaled_param_guesses, param_length=param_length, obs=obs, fltr=fltr, Tf=Tf, obs0=obs0,
                init_flags=init_flags, init_fltrs=init_fltrs, tangent=tangent, enable_penalty=False, bounds=bounds)

    return logl, s, scale, bounds


def compute_ess(weights):
    """Compute the effective sample size of a weighted set of samples."""
    w = weights.copy() / sum(weights)
    return 1/np.sum(w**2)


def compute_cess(old_weights, weights):
    """Compute the conditional effective sample size as decribed in [Zhou, Johansen, Aston 2016]."""
    return np.sum(old_weights * weights)**2 / np.sum(old_weights * weights**2)


def resample(N, particles, logl, probs):
    """ Implements the residual resampling scheme, see for example
    [Doucet, Johansen 2008], https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
    """
    counts_det = np.floor(N * probs).astype('int')
    remaining_probs = probs - 1/N * counts_det
    remaining_probs /= np.sum(remaining_probs)
    counts_mult = np.random.multinomial(N - np.sum(counts_det), remaining_probs)
    counts = counts_det + counts_mult
    
    # Select the corresponding configurations
    ndim = particles.shape[1]
    result_particles = np.zeros((N, ndim))
    result_logl = np.zeros(N)
    index = 0
    for i in range(len(counts)):
        for _ in range(counts[i]):
            result_particles[index, :] = particles[i,:]
            result_logl[index] = logl[i]
            index += 1
        
    return result_particles, result_logl


def evidence_smc(logl, prior_s, prior_scale, bounds, npopulation=200, target_cess=0.9, min_ess=0.6, mcmc_iter=50, nprocesses=0, 
                 save_samples=True, verbose=True):
    """ Compute the evidence using an adaptive sequential Monte Carlo method.

    This function computes the model evidence of the inference problem using a sequential Monte Carlo particle method 
    starting at the prior distribution. We implement the method `SMC2` described in
    [Zhou, Johansen, Aston 2016], https://doi.org/10.1080/10618600.2015.1060885

    We start by sampling `npopulation` particles from the prior distribution with uniform weights. The target distribution of the weighted 
    set of particles gets transformed to the posterior distribution by a geometric annealing schedule. The step size is chosen adaptively
    based on the target decay rate of the effective samples size `target_cess` in each step. Once the effective sample size of the 
    weighted particles goes below `min_cess * npopulation`, we replace the weighted set of samples by a resampled, unweighted set. 
    Between each step, the particles are decorrelated and equilibreated on the current level distribution by running an MCMC chain.

    Parameters
    ----------
    logl:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_s:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_scale:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    bounds:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    npopulation: int
        The number of particles used for the SMC iteration. Higher number of particles increases the accuracy of the result.
    target_cess: float
        The target rate for the decay of the effective sample size (ess reduces by `1-target_cess` each step). Smaller
        values result in more iterations and a higher accuracy result.
    min_ess: float
        The minimal effective sample size of the system. Low effective sample size imply many particles in low probability
        regions. However, resampling adds variance so it should not be done in every step.
    mcmc_iter: int
        The number of MCMC iterations in each step of the algorithm. The number of iterations should be large enough to equilibrate
        the particles with the current distribution, higher iteration numbers will typically not result in more accurate results.
        Oftentimes, it makes more sense to increase the number of steps (via `target_cess`) instead of increasing the number of 
        iterations. This decreases the difference in distribution between consecutive steps and reduced the error of the final result. 
        This number should however be large enough to allow equal-position particles (that occur via resampling) to diverge from each 
        other.
    nprocesses: int
        The number of processes passed to the `emcee` MCMC sampler. By default, the number of physical cores is used. 
    save_samples: bool
        If true, this function returns the interal state of each MCMC iteration.
    verbose: bool
        If true, this function displays the progress of each MCMC iteration in addition to basic progress information.

    Returns
    -------
    log_evidence: float
        The estimate of the log evidence.        
    if save_samples=True:

        result_samples: list of (float, emcee.EnsembleSampler)
            The list of samplers and their corresponding step `alpha`.
    """
    if emcee is None:
        raise Exception("MCMC sampling needs optional dependency `emcee` which was not found.")

    if nprocesses == 0:
        if pathos_mp:
            # Optional dependecy for multiprocessing (pathos) is installed.
            nprocesses = pathos_mp.cpu_count()
        else:
            nprocesses = 1

    if nprocesses > 1 and pathos_mp is None:
        raise Exception("The Python package `pathos` is needed for multiprocessing.")

    if nprocesses > 1:
        mcmc_pool = pathos_mp.ProcessingPool(nprocesses)
        mcmc_map = mcmc_pool.map
    else:
        mcmc_pool = None
        mcmc_map = map

    logp = lambda params: np.sum(lognorm.logpdf(params, prior_s, scale=prior_scale))

    alpha = 0.0
    delta_alpha = 0.0001  # Initial step should be small enough to avoid float overflows.
    log_evidence = 0.0
    ndim = len(prior_s)

    # If `save_samples==True`, we save the initial position of the particles and the result of
    # each `emcee` run in this list.
    result_samples = []

    # Sample initial points (truncated log-normal within boundaries)
    ppf_bounds = np.zeros((ndim, 2))
    ppf_bounds[:,0] = lognorm.cdf(bounds[:,0], prior_s, scale=prior_scale)
    ppf_bounds[:,1] = lognorm.cdf(bounds[:,1], prior_s, scale=prior_scale)
    ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]
    points = np.random.rand(npopulation, ndim)
    y = ppf_bounds[:,0] + points * ppf_bounds[:,1]
    particles = lognorm.ppf(y, prior_s, scale=prior_scale)

    if save_samples:
        result_samples.append((alpha, particles))

    iters = 0
    weights = 1/npopulation * np.ones(npopulation)
    logl_particles = np.array(mcmc_map(logl, particles))
    while True:
        iters += 1
        
        # Find next delta_alpha value (based on target_cess):
        def comp_cess(da):
            log_incr_weights = da * logl_particles
            new_weights = log_incr_weights - np.mean(log_incr_weights)
            new_weights = weights * np.exp(new_weights)
            new_weights = new_weights / np.sum(new_weights)
            return compute_cess(weights, np.exp(log_incr_weights))
        
        da_upper = delta_alpha
        da_lower = 0.0
        # Increase da_upper until it is an upper bound
        while comp_cess(da_upper) > target_cess:
            da_lower = da_upper
            da_upper = 2*da_upper

        # Binary search for optimal delta_alpha
        delta_alpha = 1/2 * (da_upper + da_lower)
        while np.abs(comp_cess(delta_alpha) - target_cess) / target_cess > 1e-4:
            if comp_cess(delta_alpha) > target_cess:
                da_lower = delta_alpha
            else:
                da_upper = delta_alpha
            delta_alpha = 1/2 * (da_upper + da_lower)
        
        if alpha + delta_alpha > 1:
            delta_alpha = 1 - alpha
        alpha = alpha + delta_alpha
        
        # Compute reweighting from alpha -> alpha + delta_alpha:
        log_incr_weights = delta_alpha * logl_particles
        new_weights = log_incr_weights - np.mean(log_incr_weights)
        new_weights = weights * np.exp(new_weights)
        new_weights = new_weights / np.sum(new_weights)

        ess = compute_ess(new_weights)
        cess = compute_cess(weights, np.exp(log_incr_weights))
        if ess/npopulation < min_ess:
            # The effective samples size has become smaller than the minimal target. We resample the
            # the particles according to their weight to remove particles with low probability and
            # increase the population in regions with high probability.
            particles, logl_particles = resample(npopulation, particles, logl_particles, new_weights)
            log_incr_weights = delta_alpha * logl_particles
            new_weights = 1/npopulation * np.ones(npopulation)
            log_evidence += np.log(1/npopulation * np.sum(np.exp(log_incr_weights)))
        else:
            # The effective sample size is still acceptable, therefore we avoid resampling since it
            # introduces a small bias every time.
            log_evidence += np.log(np.sum(weights * np.exp(log_incr_weights)))
        print("Iteration {}: alpha = {}, ESS = {}, CESS = {}, log_evidence = {}".format(iters, alpha, ess, cess, log_evidence))
        
        if alpha == 1.0:
            # There is no need to run the chain for `alpha==1`, so we stop here.
            break

        # Allow mixing of the particles by running an MCMC chain that tagrets the current distribution. We
        # use `emcee` here to make use of the ensemble of particles. The SMC method is failry robust to
        # slow mixing here. 
        logpost = lambda param, alpha: logp(param) + alpha * logl(param)
        sampler = emcee.EnsembleSampler(npopulation, ndim, logpost, pool=mcmc_pool, kwargs={'alpha':alpha})
        emcee_state = emcee.State(particles, log_prob=list(map(logp, particles)) + alpha*logl_particles)
        sampler.run_mcmc(emcee_state, mcmc_iter, progress=verbose)

        if save_samples:
            result_samples.append((alpha, sampler))

        particles = sampler.get_last_sample().coords
        logl_particles = 1/alpha * (sampler.get_last_sample().log_prob - np.array(mcmc_map(logp, particles)))
        weights = new_weights

    if mcmc_pool is not None:
        mcmc_pool.close()
        mcmc_pool.join()
        mcmc_pool.clear()

    if save_samples:
        return log_evidence, result_samples
    
    return log_evidence


def evidence_path_sampling(logl, prior_s, prior_scale, bounds, steps, npopulation=100, mcmc_iter=1000, nprocesses=0,
                           initial_samples=10, verbose=True, extend_step_list=None, extend_sampler_list=None):
    """ Compute the evidence using path sampling (thermodynamic integration).

    This function computes posterior samples for the distributions

        p_s \propto prior * likelihood^s

    for 0<s≤1, s ∈ steps, using ensemble MCMC. The samples can be used to estimate the evidence via

        log_evidence = \int_0^1 \E_{p_s}[log_likelihood] ds

    which is know as path sampling or thermodynamic integration.

    This function starts with sampling `initial_samples * npopulation` samples from the (truncated log-normal) prior.
    Afterwards, it runs an ensemble MCMC chain with `npopulation` ensemble members for `mcmc_iter` iterations. To 
    minimise burn-in, the iteration is started with the last sample of the chain with the closest step `s` that
    has already been computed. To extend the results of this function with additional steps, provide the to-be-extended 
    result via the optional arguments `extend_step_list` and `extend_sampler_list`.

    This function only returns the step list and the corresponding samplers. To compute the evidence estimate,
    use `pyross.evidence.evidence_path_sampling_process_result`.

    Parameters
    ----------
    logl:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_s:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_scale:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    bounds:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    steps: list of float
        List of steps `s` for which the distribution `p_s` is explored using MCMC. Should be in ascending order
        and not include 0.
    npopulation: int
        The population size of the MCMC ensemble sampler (see documentation of `emcee` for details).
    mcmc_iters: int
        The number of iterations of the MCMC chain for each s ∈ steps.
    nprocesses: int
        The number of processes passed to the `emcee` MCMC sampler. By default, the number of physical cores is used.
    initial_samples: int
        Compute `initial_samples * npopulation` independent samples as the result for s = 0.
    verbose: bool
        If true, this function displays the progress of each MCMC iteration in addition to basic progress information.
    extend_step_list: list of float
        Extends the result of an earlier run of this function if this argument and `extend_sampler_list` are provided.
    extend_sampler_list: list of emcee.EnsembleSampler
        Extends the result of an earlier run of this function if this argument and `extend_step_list` are provided.

    Returns
    -------
    step_list: list of float
        The steps `s` for which `p_s` has been sampled from (including 0). Always in ascending order. 
    sampler_list: list
        The list of emcee.EnsembleSamplers (and an array of prior samples at 0). 
    """
    if emcee is None:
        raise Exception("MCMC sampling needs optional dependency `emcee` which was not found.")

    if nprocesses == 0:
        if pathos_mp:
            # Optional dependecy for multiprocessing (pathos) is installed.
            nprocesses = pathos_mp.cpu_count()
        else:
            nprocesses = 1

    if nprocesses > 1 and pathos_mp is None:
        raise Exception("The Python package `pathos` is needed for multiprocessing.")

    if nprocesses > 1:
        mcmc_pool = pathos_mp.ProcessingPool(nprocesses)
    else:
        mcmc_pool = None

    logp = lambda params: np.sum(lognorm.logpdf(params, prior_s, scale=prior_scale))

    ndim = len(prior_s)

    # Sample initial points (truncated log-normal within boundaries)
    ppf_bounds = np.zeros((ndim, 2))
    ppf_bounds[:,0] = lognorm.cdf(bounds[:,0], prior_s, scale=prior_scale)
    ppf_bounds[:,1] = lognorm.cdf(bounds[:,1], prior_s, scale=prior_scale)
    ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]
    points = np.random.rand(npopulation, ndim)
    y = ppf_bounds[:,0] + points * ppf_bounds[:,1]
    first_init_positions = lognorm.ppf(y, prior_s, scale=prior_scale)
    step_list = []
    sampler_list = []
    if extend_step_list is None:
        points = np.random.rand(initial_samples*npopulation, ndim)
        y = ppf_bounds[:,0] + points * ppf_bounds[:,1]
        ext_init_positions = lognorm.ppf(y, prior_s, scale=prior_scale)

        step_list = [0.0]
        sampler_list = [ext_init_positions]
    else:
        step_list = extend_step_list
        sampler_list = extend_sampler_list

    for i, step in enumerate(steps):
        logpost = lambda param, step: logp(param) + step * logl(param)
        
        print("step: {} ({}/{})".format(step, i+1, len(steps)))

        # Find the closest init position:
        diff_to_step = np.abs(np.array(step_list) - step)
        pos = np.argmin(diff_to_step)

        if diff_to_step[pos] == 0.0 and step != 0.0:
            # There is already an MCMC chain for this step. In this situation, we extend the existing
            # chain by mcmc_iter iterations.
            sampler_list[pos].run_mcmc(None, mcmc_iter, progress=verbose)
        else:
            # Find a good starting point for the new MCMC chain. We use the last sample of the closest
            # chain.
            if pos == 0:
                init_positions = first_init_positions
            else:
                init_positions = sampler_list[pos].get_last_sample().coords
        
            sampler = emcee.EnsembleSampler(npopulation, ndim, logpost, pool=mcmc_pool, kwargs={'step':step})        
            sampler.run_mcmc(init_positions, mcmc_iter, progress=verbose)
            
            step_list.append(step)
            sampler_list.append(sampler)

    if mcmc_pool is not None:
        mcmc_pool.close()
        mcmc_pool.join()
        mcmc_pool.clear()
        
    # Sort results by step (needed if adding intermediate steps)
    sorted_results = sorted(zip(step_list, sampler_list), key=lambda val: val[0])
    step_list = [x[0] for x in sorted_results]
    sampler_list = [x[1] for x in sorted_results]
        
    return step_list, sampler_list


def evidence_path_sampling_process_result(logl, prior_s, prior_scale, bounds, step_list, sampler_list, 
                                          burn_in=0, nprocesses=0):
    """ Compute the evidence estimate for the result of `pyross.evidence.evidence_path_sampling`.

    Parameters
    ----------
    logl:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_s:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    prior_scale:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    bounds:
        Input from `pyross.evidence.get_parameters` or `pyross.evidence.latent_get_parameters`.
    step_list: list of float
        Output of `pyross.evidence.evidence_path_sampling`. The steps `s` for which `p_s` has been 
        sampled from (including 0). Always in ascending order. 
    sampler_list: list
        Output of `pyross.evidence.evidence_path_sampling`. The list of emcee.EnsembleSamplers 
        (and an array of prior samples at 0). 
    burn_in: float or np.array
        The number of initial samples that are discarded before computing the Monte Carlo average.
    nprocesses: int
        The number of processes used to compute the prior likelihood. By default, the number of physical 
        cores is used.

    Returns
    -------
    log_evidence: float
        The estimate of the log evidence.
    vals: list of float
        The Monte Carlo average of the log-likelihood for each s s ∈ step_list.
    """
    if nprocesses == 0:
        if pathos_mp:
            # Optional dependecy for multiprocessing (pathos) is installed.
            nprocesses = pathos_mp.cpu_count()
        else:
            nprocesses = 1

    if nprocesses > 1 and pathos_mp is None:
        raise Exception("The Python package `pathos` is needed for multiprocessing.")

    if nprocesses > 1:
        mcmc_pool = pathos_mp.ProcessingPool(nprocesses)
        mcmc_map = mcmc_pool.map
    else:
        mcmc_pool = None
        mcmc_map = map

    logp = lambda params: np.sum(lognorm.logpdf(params, prior_s, scale=prior_scale))

    if np.size(burn_in) == 1:
        local_burn_in = burn_in * np.ones(len(step_list)-1, 'int')
    else:
        local_burn_in = burn_in

    vals = np.zeros(len(step_list))
    # step == 0 (special case because we sampled these positions directly from the prior):
    llike = list(mcmc_map(logl, sampler_list[0]))
    vals[0] = np.mean(llike)

    for i in range(1, len(step_list)):
        step = step_list[i]
        chain = sampler_list[i].get_chain(discard=local_burn_in[i-1], flat=True)
        log_probs = sampler_list[i].get_log_prob(discard=local_burn_in[i-1], flat=True)
        # Get likelihood values from log_probs
        llike = 1/step * (log_probs - list(map(logp, chain)))
        vals[i] = np.mean(llike)

    log_evidence = 0.0
    for i in range(1, len(step_list)):
        log_evidence += 1/2 * (step_list[i] - step_list[i-1]) * (vals[i] + vals[i-1])

    if mcmc_pool is not None:
        mcmc_pool.close()
        mcmc_pool.join()
        mcmc_pool.clear()

    return log_evidence, vals


def generate_traceplot(sampler, dims=None):
    """
    Generate a traceplot for an emcee.EnsembleSampler.

    Parameters
    ----------
    sampler: emcee.EnsembleSampler
        The sampler to plot the traceplot for.
    dims: list of int, optional
        Select the dimensions that are plotted. By default, all dimensions are selected.
    """
    if dims is None:
        dims = [i for i in range(sampler.ndim)]

    N = len(dims)
    fig, axes = plt.subplots(N, figsize=(12, N*10/8), sharex=True)
    samples = sampler.get_chain()
    for i in range(N):
        ax = axes[i]
        ax.plot(samples[:, :, dims[i]], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
    axes[-1].set_xlabel("step number")
