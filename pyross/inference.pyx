from itertools import compress
from scipy import sparse
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, approx_fprime
from scipy.stats import lognorm
from scipy.linalg import solve_triangular, rq
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import eig
from scipy.stats import multivariate_normal
from scipy.linalg.lapack import dtrtri
from scipy.linalg import cholesky
cimport numpy as np
cimport cython
from math import isclose
import time, sympy
from sympy import MutableDenseNDimArray as Array
from sympy import Inverse, tensorcontraction, tensorproduct, permutedims
from scipy import interpolate
import dill
import hashlib

try:
    # Optional support for nested sampling.
    import dynesty
except ImportError:
    dynesty = None

try:
    # Optional support for MCMC sampling.
    import emcee
except ImportError:
    emcee = None

try:
    # Optional support for multiprocessing in the minimization function.
    import pathos.multiprocessing as pathos_mp
except ImportError:
    pathos_mp = None

import pyross.deterministic
cimport pyross.deterministic
import pyross.contactMatrix
from pyross.utils_python import *
from libc.math cimport sqrt, log, INFINITY
cdef double PI = 3.14159265359


DTYPE   = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.uint8_t BOOL_t

class MaxIntegratorStepsException(Exception):
    def __init__(self, message='Maximum number of integrator steps reached'):
        super(MaxIntegratorStepsException, self).__init__(message)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR_type:
    """Parent class for inference for all SIR-type classes listed below

    All subclasses use the same functions to perform inference, which are documented below.
    """

    cdef:
        readonly Py_ssize_t nClass, M, steps, dim, vec_size
        readonly double Omega, rtol_det, rtol_lyapunov
        readonly long max_steps_det, max_steps_lyapunov, integrator_step_count
        readonly np.ndarray beta, gIa, gIs, fsa, _xm
        readonly np.ndarray alpha, fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols
        readonly str det_method, lyapunov_method
        readonly dict class_index_dict
        readonly list param_keys, _interp
        readonly object contactMatrix
        readonly bint param_mapping_enabled

    def __init__(self, parameters, nClass, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov):
        self.Omega = Omega
        self.M = M
        self.fi = fi
        assert steps >= 2, 'Number of steps must be at least 2'
        self.steps = steps
        self.set_params(parameters)
        self.det_method=det_method
        self.lyapunov_method=lyapunov_method
        self.rtol_det = rtol_det
        self.rtol_lyapunov = rtol_lyapunov
        self.max_steps_det = max_steps_det
        self.max_steps_lyapunov = max_steps_lyapunov

        self.dim = nClass*M
        self.nClass = nClass
        self.vec_size = int(self.dim*(self.dim+1)/2)
        self.CM = np.empty((M, M), dtype=DTYPE)
        self.dsigmadt = np.zeros((self.vec_size), dtype=DTYPE)
        self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.J_mat = np.empty((self.dim, self.dim), dtype=DTYPE)
        self.B_vec = np.empty((self.vec_size), dtype=DTYPE)
        self.U = np.empty((self.dim, self.dim), dtype=DTYPE)

        # preparing the indices
        self.rows, self.cols = np.triu_indices(self.dim)
        self.flat_indices = np.ravel_multi_index((self.rows, self.cols), (self.dim, self.dim))
        r, c = np.triu_indices(self.dim, k=1)
        self.flat_indices1 = np.ravel_multi_index((r, c), (self.dim, self.dim))
        self.flat_indices2 = np.ravel_multi_index((c, r), (self.dim, self.dim))

        self._xm = None
        self._interp = None

        self.param_mapping_enabled = False


    def infer_parameters(self, x, Tf, contactMatrix, prior_dict, **kwargs):
        """Infers the MAP estimates for epidemiological parameters

        Parameters
        ----------
            see `infer`

        Returns
        -------
        output: dict
            Contains the following keys for users:

            map_dict: dict
                A dictionary for MAPs. Keys are the names of the parameters and
                the corresponding values are its MAP estimates.
            -logp: float
                The value of -logp at MAP.

        Note
        ----
        This function just calls `infer` with a fixed contactMatrix function, will be deprecated.
        """


        output_dict = self.infer(x, Tf, prior_dict, contactMatrix=contactMatrix, generator=None,
                      intervention_fun=None, **kwargs)

        # match old output dictionary key names (the new implementation of infer uses the same keys as latent_infer)

        output_dict['map_dict'] = output_dict['params_dict']
        output_dict['keys'] = output_dict['param_keys']
        output_dict['flat_guess_range'] = output_dict['param_guess_range']
        output_dict['scaled_guesses'] = output_dict['scaled_param_guesses']
        output_dict['flat_map'] = output_dict['flat_params']

        return output_dict

    def nested_sampling_inference(self, x, Tf, contactMatrix, prior_dict, tangent=False, verbose=False,
                                  nprocesses=0, queue_size=None, maxiter=None, maxcall=None, dlogz=None,
                                  n_effective=None, add_live=True, sampler=None, **dynesty_args):
        """
        Run nested sampling for model parameters without latent variables.

        Note
        ----
        This function has been replaced by `pyross.inference.infer_nested_sampling` and will be deleted
        in a future version of pyross. See there for a documentation of the function parameters.
        """
        return self.infer_nested_sampling(x, Tf, prior_dict, contactMatrix=contactMatrix, generator=None,
                                          intervention_fun=None, tangent=tangent, verbose=verbose, nprocesses=nprocesses,
                                          queue_size=queue_size, maxiter=maxiter, maxcall=maxcall, dlogz=dlogz,
                                          n_effective=n_effective, add_live=add_live, sampler=sampler, **dynesty_args)

    def nested_sampling_inference_process_result(self, sampler, prior_dict):
        """
        Take the sampler generated by `nested_sampling_inference` and produce output dictionaries for further use
        in the pyross framework.

        Note
        ----
        This function has been replaced by `pyross.inference.infer_nested_sampling_process_result` and will be deleted
        in a future version of pyross. See there for a documentation of the function parameters.
        """
        result, output_samples = self.infer_nested_sampling_process_result(sampler, prior_dict, contactMatrix=self.contactMatrix,
                                                                           generator=None, intervention_fun=None)

        # Match old dictionary key names for backward compatibility.
        for out_dict in output_samples:
            out_dict['map_dict']         = out_dict.pop('params_dict')
            out_dict['keys']             = out_dict.pop('param_keys')
            out_dict['flat_guess_range'] = out_dict.pop('param_guess_range')
            out_dict['scaled_guesses']   = out_dict.pop('scaled_param_guesses')
            out_dict['flat_map']         = out_dict.pop('flat_params')

        return result, output_samples

    def mcmc_inference(self, x, Tf, contactMatrix, prior_dict, tangent=False, verbose=False, sampler=None, nwalkers=None,
                       walker_pos=None, nsamples=1000, nprocesses=0):
        """
        Sample the posterior distribution of the epidemiological parameters using ensemble MCMC.

        Note
        ----
        This function has been replaced by `pyross.inference.infer_mcmc` and will be deleted in a future version of pyross.
        See there for a documentation of the function parameters.
        """
        return self.infer_mcmc(x, Tf, prior_dict, contactMatrix=contactMatrix, generator=None, intervention_fun=None,
                               tangent=tangent, verbose=verbose, sampler=sampler, nwalkers=nwalkers,
                               walker_pos=walker_pos, nsamples=nsamples, nprocesses=nprocesses)

    def mcmc_inference_process_result(self, sampler, prior_dict, flat=True, discard=0, thin=1):
        """
        Take the sampler generated by `mcmc_inference` and produce output dictionaries for further use in the
        pyross framework.

        Note
        ----
        This function has been replaced by `pyross.inference.infer_mcmc_process_result` and will be deleted in a future version
        of pyross. See there for a documentation of the function parameters.
        """
        output_samples = self.infer_mcmc_process_result(sampler, prior_dict, contactMatrix=self.contactMatrix, generator=None,
                                                        intervention_fun=None, flat=flat, discard=discard, thin=thin)

        # Match old dictionary key names for backward compatibility.
        if flat:
            flat_sample_list = output_samples
        else:
            flat_sample_list = [item for sublist in output_samples for item in sublist]
        for out_dict in flat_sample_list:
            out_dict['map_dict']         = out_dict.pop('params_dict')
            out_dict['keys']             = out_dict.pop('param_keys')
            out_dict['flat_guess_range'] = out_dict.pop('param_guess_range')
            out_dict['scaled_guesses']   = out_dict.pop('scaled_param_guesses')
            out_dict['flat_map']         = out_dict.pop('flat_params')

        return output_samples


    def infer_control(self, x, Tf, generator, prior_dict, **kwargs):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown.

        Parameters
        ----------
            see `infer`

        Returns
        -------
        output_dict: dict
            Dictionary of MAP estimates, containing the following keys for users:

            map_dict: dict
                Dictionary for MAP estimates of the control parameters.
            -logp: float
                Value of -logp at MAP.

        Note
        ----
        This function just calls `infer` with the specified generator, will be deprecated.
        """

        output_dict = self.infer(x, Tf, prior_dict, contactMatrix=None, generator=generator,
                      **kwargs)

        # match old output dictionary key names
        del output_dict['params_dict']
        output_dict['map_dict'] = output_dict['control_params_dict']
        output_dict['keys'] = output_dict['param_keys']
        output_dict['flat_guess_range'] = output_dict['param_guess_range']
        output_dict['scaled_guesses'] = output_dict['scaled_param_guesses']

        return output_dict

    def _loglikelihood(self, params, contactMatrix=None, generator=None, intervention_fun=None, keys=None,
                       x=None, Tf=None, tangent=None,is_scale_parameter=None, flat_guess_range=None,
                       scaled_guesses=None, bounds=None, inter_steps=0, **catchall_kwargs):
        """Compute the log-likelihood of the model."""
        if bounds is not None:
            # Check that params is within bounds. If not, return -np.inf.
            if np.any(bounds[:,0] > params) or np.any(bounds[:,1] < params):
                return -np.Inf

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params, flat_guess_range,
                                                        is_scale_parameter, scaled_guesses)
        parameters, kwargs = self.fill_params_dict(keys, orig_params, return_additional_params=True)

        self.set_params(parameters)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**kwargs)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **kwargs)
        else:
            if kwargs != {}:
                raise Exception('Key error or unspecified generator')

        if tangent:
            minus_logl = self._obtain_logp_for_traj_tangent(x, Tf)
        else:
            minus_logl = self._obtain_logp_for_traj(x, Tf, inter_steps)

        return -minus_logl

    def _logposterior(self, params, prior=None, **logl_kwargs):
        """Compute the log-posterior (up to a constant) of the model."""
        logl = self._loglikelihood(params, **logl_kwargs)
        return logl + np.sum(prior.logpdf(params))

    def _infer_to_minimize(self, params, grad=0, **logp_kwargs):
        """Objective function for minimization call in infer."""
        return -self._logposterior(params, **logp_kwargs)

    def infer(self, x, Tf, prior_dict, contactMatrix=None,
              generator=None, intervention_fun=None, tangent=False,
              verbose=False, ftol=1e-6,
              global_max_iter=100, local_max_iter=100, global_atol=1.,
              enable_global=True, enable_local=True,
              cma_processes=0, cma_population=16, cma_random_seed=None):
        """
        Compute the maximum a-posteriori (MAP) estimate for all desired parameters, including control parameters, for an SIR type model
        with fully observed classes. If `generator` is specified, the lockdown is modelled by scaling the contact matrices for contact at work,
        school, and other (but not home). This function infers the scaling parameters (can be age dependent) assuming that full data
        on all classes is available (with latent variables, use `latent_infer`).

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        prior_dict: dict
            A dictionary containing priors for parameters (can include both model and intervention parameters). See examples.
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is false.
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: double
            Relative tolerance of logp
        global_max_iter: int, optional
            Number of global optimisations performed.
        local_max_iter: int, optional
            Number of local optimisation performed.
        global_atol: float
            The absolute tolerance for global optimisation.
        enable_global: bool, optional
            Set to True to enable global optimisation.
        enable_local: bool, optional
            Set to True to enable local optimisation.
        cma_processes: int, optional
            Number of parallel processes used for global optimisation.
        cma_population: int, optional
            The number of samples used in each step of the CMA algorithm.
        cma_random_seed: int (between 0 and 2**32-1)
            Random seed for the optimisation algorithms. By default it is generated from numpy.random.randint.

        Returns
        -------
        output_dict: dict
            Dictionary of MAP estimates, containing the following keys for users:

            params_dict: dict
                Dictionary for MAP estimates of the model parameters.
            control_params_dict: dict
                Dictionary for MAP estimates of the control parameters (if requested).
            -logp: float
                Value of -logp at MAP.
        Note
        ----
        This function combines the functionality of `infer_parameters` and `infer_control`,
        which will be deprecated.
        To infer model parameters only, specify a fixed `contactMatrix` function.
        To infer control parameters only, specify a `generator` and do not specify priors for model parameters.

        Examples
        --------
        An example of prior_dict to set priors for alpha and beta, where alpha
        is age dependent and we want to infer its scale parameters rather than
        each component individually. The prior distribution is assumed to be
        log-normal with the specified mean and standard deviation.

        >>> prior_dict = {
                'alpha':{
                    'mean': [0.5, 0.2],
                    'infer_scale': True,
                    'scale_factor_std': 1,
                    'scale_factor_bounds': [0.1, 10],
                    'prior_fun': 'truncnorm'
                },
                'beta':{
                    'mean': 0.02,
                    'std': 0.1,
                    'bounds': [1e-4, 1],
                    'prior_fun': 'lognorm'
                }
            }

        """


        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Read in parameter priors
        prior_names, keys, guess, stds, _, _, bounds, \
        flat_guess_range, is_scale_parameter, scaled_guesses  \
                = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)
        cma_stds = np.minimum(stds, (bounds[:, 1] - bounds[:, 0])/3)
        minimize_args = {'keys':keys, 'x':x, 'Tf':Tf,
                         'flat_guess_range':flat_guess_range,
                         'is_scale_parameter':is_scale_parameter,
                         'scaled_guesses': scaled_guesses,
                         'generator':generator, 'prior':prior,
                         'intervention_fun': intervention_fun, 'tangent': tangent}
        res = minimization(self._infer_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args, cma_random_seed=cma_random_seed)

        orig_params = pyross.utils.unflatten_parameters(res[0], flat_guess_range,
                                             is_scale_parameter, scaled_guesses)

        map_params_dict, map_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

        l_post = -res[1]
        l_prior = np.sum(prior.logpdf(res[0]))
        l_like = l_post - l_prior
        output_dict = {
            'params_dict': map_params_dict, 'flat_params': res[0], 'param_keys': keys,
            'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
            'is_scale_parameter':is_scale_parameter,
            'param_guess_range':flat_guess_range,
            'scaled_param_guesses':scaled_guesses,
            'prior': prior
        }
        if map_control_params_dict != {}:
            output_dict['control_params_dict'] = map_control_params_dict

        return output_dict

    def _nested_sampling_prior_transform(self, x, prior=None):
        return prior.ppf(x)

    def infer_nested_sampling(self, x, Tf, prior_dict, contactMatrix=None, generator=None, intervention_fun=None, tangent=False,
                              verbose=False, nprocesses=0, queue_size=None, maxiter=None, maxcall=None, dlogz=None,
                              n_effective=None, add_live=True, sampler=None, **dynesty_args):
        """
        Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        using nested sampling as implemented in the `dynesty` Python package. This function assumes that full data on
        all classes is available.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        prior_dict: dict
            A dictionary containing priors for parameters (can include both model and intervention parameters). See examples.
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is false.
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        nprocesses: int, optional
            The number of processes used for parallel evaluation of the likelihood.
        queue_size: int, optional
            Size of internal queue of likelihood values, default is nprocesses if multiprocessing is used.
        maxiter: int, optional
            The maximum number of iterations. Default is no limit.
        maxcall:int, optional
            The maximum number of calls to the likelihood function. Default no limit.
        dlogz: float, optional
            The iteration terminates if the estimated contribution of the remaining prior volume to the total evidence
            falls below this threshold. Default value is `1e-3 * (nlive - 1) + 0.01` if `add_live==True`, 0.01 otherwise.
        n_effective: float, optional
            The iteration terminates if the number of effective posterior samples reaches this values. Default is no limit.
        add_live: bool, optional
            Determines whether to add the remaining set of live points to the set of samples. Default is True.
        sampler: dynesty.NestedSampler, optional
            Continue running an instance of a nested sampler until the termination criteria are met.
        **dynesty_args:
            Arguments passed through to the construction of the dynesty.NestedSampler constructor. Relevant entries
            are (this is not comprehensive, for details see the documentation of dynesty):

            nlive: int, optional
                The number of live points. Default is 500.
            bound: {'none', 'single', 'multi', 'balls', 'cubes'}, optional
                Method used to approximately bound the prior using the current set of live points. Default is 'multi'.
            sample:  {'auto', 'unif', 'rwalk', 'rstagger', 'slice', 'rslice', 'hslice', callable}, optional
                Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.

        Returns
        -------
        sampler: dynesty.NestedSampler
            The state of the sampler after termination of the nested sampling run.
        """
        if dynesty is None:
            raise Exception("Nested sampling needs optional dependency `dynesty` which was not found.")

        if nprocesses == 0:
            if pathos_mp:
                # Optional dependecy for multiprocessing (pathos) is installed.
                nprocesses = pathos_mp.cpu_count()
            else:
                nprocesses = 1

        if queue_size is None:
            queue_size = nprocesses

        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Read in parameter priors
        prior_names, keys, guess, stds, _, _, bounds, \
        flat_guess_range, is_scale_parameter, scaled_guesses  \
                = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)

        ndim = len(guess)
        prior_transform_args = {'prior':prior}
        loglike_args = {'keys':keys, 'x':x, 'Tf':Tf, 'flat_guess_range':flat_guess_range,
                         'is_scale_parameter':is_scale_parameter, 'scaled_guesses': scaled_guesses,
                         'generator':generator, 'intervention_fun': intervention_fun, 'tangent': tangent}

        if sampler is None:
            if nprocesses > 1:
                pool = pathos_mp.ProcessingPool(nprocesses)
                sampler = dynesty.NestedSampler(self._loglikelihood, self._nested_sampling_prior_transform, ndim=ndim,
                                                logl_kwargs=loglike_args, ptform_kwargs=prior_transform_args,
                                                pool=pool, queue_size=queue_size, **dynesty_args)
            else:
                sampler = dynesty.NestedSampler(self._loglikelihood, self._nested_sampling_prior_transform, ndim=ndim,
                                                logl_kwargs=loglike_args, ptform_kwargs=prior_transform_args,
                                                **dynesty_args)
        else:
            if nprocesses > 1:
                # Restart the pool we closed at the end of the previous run.
                sampler.pool = pathos_mp.ProcessingPool(nprocesses)
                sampler.M = sampler.pool.map
            elif sampler.pool is not None:
                sampler.pool = None
                sampler.M = map

        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz, n_effective=n_effective,
                           add_live=add_live, print_progress=verbose)

        if sampler.pool is not None:
            sampler.pool.close()
            sampler.pool.join()
            sampler.pool.clear()

        return sampler

    def infer_nested_sampling_process_result(self, sampler, prior_dict,
                                             contactMatrix=None, generator=None,
                                             intervention_fun=None, **catchall_kwargs):
        """
        Take the sampler generated by `pyross.inference.infer_nested_sampling` and produce output dictionaries for
        further use in the pyross framework. See `pyross.inference.infer_nested_sampling` for description of parameters.

        Parameters
        ----------
        sampler: dynesty.NestedSampler
            The output of `pyross.inference.infer_nested_sampling`.
        prior_dict: dict
        contactMatrix: callable, optional
        generator: pyross.contactMatrix, optional
        intervention_fun: callable, optional
        **catchall_kwargs: dict
            Catched further provided arguments and ignores them.

        Returns
        -------
        result: dynesty.Result
            The result of the nested sampling iteration. Relevant entries include:

            result.logz: list
                The progression of log-evidence estimates, use result.logz[-1] for the final estimate.
        output_samples: list
            The processed weighted posterior samples.
        """
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)
        prior_names, keys, guess, stds, _, _, bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)

        result = sampler.results
        output_samples = []
        for i in range(len(result.samples)):
            sample = result.samples[i]
            weight = np.exp(result.logwt[i] - result.logz[len(result.logz)-1])
            l_like = result.logl[i]

            orig_params = pyross.utils.unflatten_parameters(sample, flat_guess_range,
                                                            is_scale_parameter, scaled_guesses)

            map_params_dict, map_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
            self.set_params(map_params_dict)

            if generator is not None:
                if intervention_fun is None:
                    self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
                else:
                    self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

            l_prior = np.sum(prior.logpdf(sample))
            l_post = l_like + l_prior
            output_dict = {
                'params_dict': map_params_dict, 'flat_params': sample, 'param_keys': keys,
                'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
                'weight':weight, 'is_scale_parameter':is_scale_parameter,
                'param_guess_range':flat_guess_range,
                'scaled_param_guesses':scaled_guesses,
                'prior': prior
            }
            if map_control_params_dict != {}:
                output_dict['control_params_dict'] = map_control_params_dict

            output_samples.append(output_dict)

        return result, output_samples

    def infer_mcmc(self, x, Tf, prior_dict, contactMatrix=None, generator=None, intervention_fun=None, tangent=False,
                   verbose=False, sampler=None, nwalkers=None, walker_pos=None, nsamples=1000, nprocesses=0):
        """
        Sample the posterior distribution of the epidemiological parameters using ensemble MCMC.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        prior_dict: dict
            A dictionary containing priors for parameters (can include both model and intervention parameters). See examples.
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        verbose: bool, optional
            Set to True to see a progress bar for the sample generation. Default is False.
        sampler: emcee.EnsembleSampler, optional
            Set to instance of the sampler (as returned by this function) to continue running the MCMC chains.
            Default is None (i.e. run a new chain).
        nwalkers:int, optional
            The number of chains in the ensemble (should be at least 2*dim). Default is 2*dim.
        walker_pos: np.array, optional
            The initial position of the walkers. If not specified, it samples random positions from the prior.
        nsamples:int, optional
            The number of samples per walker. Default is 1000.
        nprocesses: int, optional
            The number of processes used to compute the likelihood for the walkers, needs `pathos`. Default is
            the number of cpu cores if `pathos` is available, otherwise 1.

        Returns
        -------
        sampler: emcee.EnsembleSampler
            This function returns the interal state of the sampler. To look at the chain of the internal flattened parameters,
            run `sampler.get_chain()`. Use this to judge whether the chain has sufficiently converged. Either rerun
            `mcmc_inference(..., sampler=sampler)` to continue the chain or `mcmc_inference_process_result(...)` to process
            the result.

        Examples
        --------
        For the structure of `prior_dict`, see the documentation of `infer`. To start sampling the posterior,
        run for example

        >>> sampler = estimator.infer_mcmc(x, Tf, prior_dict, contactMatrix=contactMatrix, verbose=True)

        To judge the convergence of this chain, we can look at the trace plot of all the chains (for a moderate number of
        dimensions `dim`)

        >>> fig, axes = plt.subplots(dim, sharex=True)
        >>> samples = sampler.get_chain()
        >>> for i in range(dim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
        >>> axes[-1].set_xlabel("step number");

        For more detailed convergence metrics, see the documentation of `emcee`. To continue running this chain, we can
        call this function again with the sampler as argument

        >>> sampler = estimator.infer_mcmc(x, Tf, prior_dict, contactMatrix=contactMatrix, verbose=True, sampler=sampler)

        This procudes 1000 additional samples in each chain. To process the results, call `infer_mcmc_process_result`.
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

        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Read in parameter priors
        prior_names, keys, guess, stds, _, _, bounds, \
        flat_guess_range, is_scale_parameter, scaled_guesses  \
                = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)

        ndim = len(guess)
        if nwalkers is None:
            nwalkers = 2*ndim

        logpost_args = {'bounds':bounds, 'keys':keys, 'is_scale_parameter':is_scale_parameter,
                'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                'x':x, 'Tf':Tf, 'prior':prior, 'tangent':tangent, 'generator':generator,
                'intervention_fun': intervention_fun}
        if walker_pos is None:
             # If not specified, sample initial positions of walkers from prior (within bounds).
            points = np.random.rand(nwalkers, ndim)
            p0 = prior.ppf(points)
        else:
            p0 = walker_pos

        if sampler is None:
            # Start a new MCMC chain.
            if nprocesses > 1:
                mcmc_pool = pathos_mp.ProcessingPool(nprocesses)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._logposterior, kwargs=logpost_args,
                                                pool=mcmc_pool)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._logposterior, kwargs=logpost_args)

            sampler.run_mcmc(p0, nsamples, progress=verbose)
        else:
            # Continue running an existing MCMC chain.
            if nprocesses > 1:
                sampler.pool = pathos_mp.ProcessingPool(nprocesses)
            elif sampler.pool is not None:
                sampler.pool = None

            sampler.run_mcmc(None, nsamples, progress=verbose)

        if sampler.pool is not None:
            sampler.pool.close()
            sampler.pool.join()
            sampler.pool.clear()

        return sampler

    def infer_mcmc_process_result(self, sampler, prior_dict, contactMatrix=None, generator=None,
                                  intervention_fun=None, flat=True, discard=0, thin=1, **catchall_kwargs):
        """
        Take the sampler generated by `pyross.inference.infer_mcmc` and produce output dictionaries for further use in the
        pyross framework. See `pyross.inference.infer_mcmc` for additional description of parameters.

        Parameters
        ----------
        sampler: emcee.EnsembleSampler
            Output of `pyross.inference.infer_mcmc`.
        prior_dict: dict
        contactMatrix: callable, optional
        generator: pyross.contactMatrix, optional
        intervention_fun: callable, optional
        flat: bool, optional
            This decides whether to return the samples as for each chain separately (False) or as as a combined
            list (True). Default is True.
        discard: int, optional
            The number of initial samples to discard in each chain (to account for burn-in). Default is 0.
        thin: int, optional
            Thin out the chain by taking only the n-tn element in each chain. Default is 1 (no thinning).
        **catchall_kwargs: dict
            Catched further provided arguments and ignores them.

        Returns
        -------
        output_samples: list of dict (if flat=True), or list of list of dict (if flat=False)
            The processed posterior samples.
        """
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)
        prior_names, keys, guess, stds, _, _, bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)

        samples = sampler.get_chain(flat=flat, thin=thin, discard=discard)
        log_posts = sampler.get_log_prob(flat=flat, thin=thin, discard=discard)
        samples_per_chain = samples.shape[0]
        nr_chains = 1 if flat else samples.shape[1]
        if flat:
            output_samples = []
        else:
            output_samples = [[] for _ in nr_chains]

        for i in range(samples_per_chain):
            for j in range(nr_chains):
                if flat:
                    sample = samples[i,:]
                    l_post = log_posts[i]
                else:
                    sample = samples[i, j, :]
                    l_post = log_posts[i, j]
                weight = 1.0 / (samples_per_chain * nr_chains)

                orig_params = pyross.utils.unflatten_parameters(sample, flat_guess_range,
                                                                is_scale_parameter, scaled_guesses)

                map_params_dict, map_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
                self.set_params(map_params_dict)

                if generator is not None:
                    if intervention_fun is None:
                        self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
                    else:
                        self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

                l_prior = np.sum(prior.logpdf(sample))
                l_like = l_post - l_prior
                output_dict = {
                    'params_dict': map_params_dict, 'flat_params': sample, 'param_keys': keys,
                    'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
                    'weight':weight, 'is_scale_parameter':is_scale_parameter,
                    'param_guess_range':flat_guess_range,
                    'scaled_param_guesses':scaled_guesses,
                    'prior':prior
                }
                if map_control_params_dict != {}:
                    output_dict['control_params_dict'] = map_control_params_dict

                if flat:
                    output_samples.append(output_dict)
                else:
                    output_samples[j].append(output_dict)

        return output_samples


    def _mean(self, params, contactMatrix=None,
                      generator=None, intervention_fun=None,
              param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None,
                            x0=None, Tf=None, inter_steps=None):
        """Objective function for differentiation call in FIM and FIM_det."""
        param_estimates = np.copy(params)

        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)

        map_params_dict, map_control_params_dict = self.fill_params_dict(param_keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

        Nf = Tf+1

        if inter_steps:
            x0 = np.multiply(x0, self.Omega)
            xm = pyross.utils.forward_euler_integration(self._rhs0, x0, 0, Tf, Nf, inter_steps)
            xm = xm[::inter_steps]
            xm = np.divide(xm, self.Omega)
        else:
            xm = self.integrate(x0, 0, Tf, Nf)
        return np.ravel(xm[1:])


    def _cov(self, params, contactMatrix=None,
             generator=None, intervention_fun=None,
             param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None,
                            Tf=None, x0=None,
                            inter_steps=None, tangent=False):
        """Objective function for differentiation call in FIM."""
        param_estimates = np.copy(params)

        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)

        map_params_dict, map_control_params_dict = self.fill_params_dict(param_keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

        Nf = Tf+1

        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf,
                                                                  inter_steps)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, inter_steps)

        return full_cov


    def FIM(self, x, Tf, infer_result, contactMatrix=None, generator=None,
            intervention_fun=None, tangent=False, eps=None, inter_steps=100):
        """
        Computes the Fisher Information Matrix (FIM) for the MAP estimates of a stochastic SIR type model.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for numerical differentiation of the process mean and its full covariance matrix 
            with respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified, 
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. It is recommended to use a step-size greater or equal to `eps`. 
            Decreasing the step size too small can result in round-off error.
        inter_steps: int, optional
            Intermediate steps for interpolation between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make 
            computations slower. Setting `inter_steps=0` will fall back to the method accessible via `det_method` 
            for the deterministic integration. We have found that forward Euler is generally slower, but more stable 
            for derivatives with respect to parameters than the variable step size integrators used elsewhere in pyross. Default is 100. 


        Returns
        -------
        FIM: 2d numpy.array
            The Fisher Information Matrix 

        """



        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)
        infer_result_loc = infer_result.copy()
        # backwards compatibility
        if 'flat_map' in infer_result_loc:
            infer_result_loc['flat_params'] = infer_result_loc.pop('flat_map')

        flat_params = np.copy(infer_result_loc['flat_params'])

        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses']:
            kwargs[key] = infer_result_loc[key]

        x0 = x[0]

        def mean(y):
            return self._mean(y, contactMatrix=contactMatrix,
                                      generator=generator,
                              intervention_fun=intervention_fun, x0=x0,
                              Tf=Tf, inter_steps=inter_steps,
                              **kwargs)

        def covariance(y):
            return self._cov(y, contactMatrix=contactMatrix,
                                     generator=generator,
                              intervention_fun=intervention_fun, x0=x0,
                              Tf=Tf, tangent=tangent, inter_steps=inter_steps,
                              **kwargs)

        if np.all(eps == None):
            xx = infer_result_loc['flat_params']
            fx = abs(infer_result_loc['log_likelihood'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**np.divide(1,3)
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_params))
        print('eps-vector used for differentiation: ', eps)

        cov = covariance(flat_params)
        invcov = np.linalg.inv(cov)

        dim = len(flat_params)
        FIM = np.empty((dim,dim))
        dmu = []
        dcov = []

        for i in range(dim):
            dmu.append(pyross.utils.partial_derivative(mean, var=i, point=flat_params, dx=eps[i]))
            dcov.append(pyross.utils.partial_derivative(covariance,  var=i, point=flat_params, dx=eps[i]))

        for i in range(dim):
            t1 = dmu[i]@invcov@dmu[i]
            t2 = np.multiply(0.5,np.trace(invcov@dcov[i]@invcov@dcov[i]))
            FIM[i,i] = t1 + t2

        rows,cols = np.triu_indices(dim,1)

        for i,j in zip(rows,cols):
            t1 = dmu[i]@invcov@dmu[j]
            t2 = np.multiply(0.5,np.trace(invcov@dcov[i]@invcov@dcov[j]))
            FIM[i,j] = t1 + t2

        i_lower = np.tril_indices(dim,-1)
        FIM[i_lower] = FIM.T[i_lower]
        return FIM


    def FIM_det(self, x, Tf, infer_result, contactMatrix=None, generator=None,
                intervention_fun=None, eps=None, measurement_error=1e-2,
                inter_steps=100):
        """
        Computes the Fisher Information Matrix (FIM) for the MAP estimates of a deterministic (ODE based, including a constant measurement error) SIR type model.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        eps: float or numpy.array, optional
            Step size for numerical differentiation of the process mean and its full covariance matrix with 
            respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified, 
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. It is recommended to use a step-size greater or equal to `eps`. Decreasing the step size too small can result in round-off error.
        measurement_error: float, optional
            Standard deviation of measurements (uniform and independent Gaussian measurement error assumed). Default is 1e-2.
        inter_steps: int, optional
            Intermediate steps for interpolation between observations for the deterministic forward Euler integration. A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration. We have found that forward Euler is generally slower, but more stable for derivatives with respect to parameters than the variable step size integrators used elsewhere in pyross. Default is 100.
        Returns
        -------
        FIM: 2d numpy.array
            The Fisher Information Matrix
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        infer_result_loc = infer_result.copy()
        # backwards compatibility
        if 'flat_map' in infer_result_loc:
            infer_result_loc['flat_params'] = infer_result_loc.pop('flat_map')

        flat_params = np.copy(infer_result_loc['flat_params'])
        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses']:
            kwargs[key] = infer_result_loc[key]

        x0 = x[0]

        def mean(y):
            return self._mean(y, contactMatrix=contactMatrix,
                                      generator=generator,
                              intervention_fun=intervention_fun, x0=x0,
                              Tf=Tf, inter_steps=inter_steps,
                              **kwargs)

        if np.all(eps == None):
            xx = infer_result_loc['flat_params']
            fx = abs(infer_result_loc['log_likelihood'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**np.divide(1,3)
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_params))
        print('eps-vector used for differentiation: ', eps)

        sigma_sq = measurement_error*measurement_error
        cov_diag = np.repeat(sigma_sq, repeats=(int(self.dim)*(x.shape[0]-1)))
        cov = np.diag(cov_diag)
        invcov = np.linalg.inv(cov)

        dim = len(flat_params)
        FIM_det = np.empty((dim,dim))
        dmu = []

        for i in range(dim):
            dmu.append(pyross.utils.partial_derivative(mean, var=i, point=flat_params, dx=eps[i]))

        for i in range(dim):
            FIM_det[i,i] = dmu[i]@invcov@dmu[i]

        rows,cols = np.triu_indices(dim,1)

        for i,j in zip(rows,cols):
            FIM_det[i,j] = dmu[i]@invcov@dmu[j]

        i_lower = np.tril_indices(dim,-1)
        FIM_det[i_lower] = FIM_det.T[i_lower]
        return FIM_det


    def hessian(self, x, Tf, infer_result, contactMatrix=None, generator=None,
                intervention_fun=None, tangent=False, eps=None,
                fd_method="central", inter_steps=0, nprocesses=0, basis=None):
        """
        Computes the Hessian matrix for the MAP estimates of an SIR type model.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for finite differences computation of the hessian with respect to the parameters. 
            Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified,
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. For `fd_method="central"` it is recommended to use a step-size greater or equal to `eps`. Decreasing the step size too small can result in round-off error.
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central". Default is "central".
        inter_steps: int, optional
            Only used if `tangent=False`. Intermediate steps for interpolation between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration. 
            We have found that forward Euler is generally slower, but sometimes more stable for derivatives with respect to parameters 
            than the variable step size integrators used elsewhere in pyross. Default is 0.
        Returns
        -------
        hess: 2d numpy.array
            The Hessian matrix
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        flat_params = np.copy(infer_result['flat_params'])

        kwargs = {}
        kwargs['x'] = x
        kwargs['Tf'] = Tf
        kwargs['tangent'] = tangent
        for key in ['is_scale_parameter',
                    'scaled_param_guesses', 'prior']:
            kwargs[key] = infer_result[key]

        kwargs['keys'] = infer_result['param_keys']
        kwargs['flat_guess_range'] = infer_result['param_guess_range']
        kwargs['generator'] = generator
        kwargs['intervention_fun'] = intervention_fun
        kwargs['inter_steps'] = inter_steps

        if np.all(eps == None):
            xx = infer_result['flat_params']
            fx = abs(infer_result['log_posterior'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**(0.25)
        print('epsilon used for differentiation: ', eps)

        hess = hessian_finite_difference(flat_params, self._infer_to_minimize, eps, method=fd_method, nprocesses=nprocesses,
                                         basis=basis, function_kwargs=kwargs)
        return hess

    def robustness(self, FIM, FIM_det, infer_result, param_pos_1, param_pos_2,
                   range_1, range_2, resolution_1, resolution_2=None):
        """
        Robustness analysis in a two-dimensional slice of the parameter space, revealing neutral spaces as in https://doi.org/10.1073/pnas.1015814108.

        Parameters
        ----------
        FIM: 2d numpy.array
            Fisher Information matrix of a stochastic model
        FIM_det: 2d numpy.array
            Fisher information matrix of the corresponding deterministic model
        infer_result: dict
            Dictionary returned by `latent_infer`
        param_pos_1: int
            Position of 'parameter 1' in map_dict['flat_map'] for x-axis
        param_pos_2: int
            Position of 'parameter 2' in map_dict['flat_map'] for y-axis
        range_1: float
            Symmetric interval around parameter 1 for which robustness will be analysed. Absolute interval: 'parameter 1' +/- range_1
        range_2: float
            Symmetric interval around parameter 2 for which robustness will be analysed. Absolute interval: 'parameter 2' +/- range_2
        resolution_1: int
            Resolution of the meshgrid in x direction.
        resolution_2: int
            Resolution of the meshgrid in y direction. Default is resolution_2=resolution_1.
        Returns
        -------
        ff: 2d numpy.array
            shape=resolution_1 x resolution_2, meshgrid for x-axis
        ss: 2d numpy.array
            shape=resolution_1 x resolution_2, meshgrid for y-axis
        Z_sto: 2d numpy.array
            shape=resolution_1 x resolution_2, expected quadratic coefficient in the Taylor expansion of the likelihood of the stochastic model
        Z_det: 2d numpy.array
            shape=resolution_1 x resolution_2, expected quadratic coefficient in the Taylor expansion of the likelihood of the deterministic model

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> from matplotlib import cm
        >>>
        >>> # positions 0 and 1 of infer_result['flat_params'] correspond to a scale parameter for alpha, and beta, respectively.
        >>> ff, ss, Z_sto, Z_det = estimator.robustness(FIM, FIM_det, map_dict, 0, 1, 0.5, 0.01, 20)
        >>> cmap = plt.cm.PuBu_r
        >>> levels=11
        >>> colors='black'
        >>>
        >>> c = plt.contourf(ff, ss, Z_sto, cmap=cmap, levels=levels) # heat map for the stochastic coefficient
        >>> plt.contour(ff, ss, Z_sto, colors='black', levels=levels, linewidths=0.25)
        >>> plt.contour(ff, ss, Z_det, colors=colors, levels=levels) # contour plot for the deterministic model
        >>> plt.plot(infer_result['flat_params'][0], infer_result['flat_params'][1], 'o',
                    color="#A60628", markersize=6) # the MAP estimate
        >>> plt.colorbar(c)
        >>> plt.xlabel(r'$\alpha$ scale', fontsize=20, labelpad=10)
        >>> plt.ylabel(r'$\beta$', fontsize=20, labelpad=10)
        >>> plt.show()
        """
        infer_result_loc = infer_result.copy()
        # backwards compatibility
        if 'flat_map' in infer_result_loc:
            infer_result_loc['flat_params'] = infer_result_loc.pop('flat_map')

        flat_maps = infer_result_loc['flat_params']
        if resolution_2 == None:
            resolution_2 = resolution_1
        def bilinear(param_1, param_2, det=True):
            maps_temp = np.copy(flat_maps)
            maps_temp[param_pos_1] += param_1
            maps_temp[param_pos_2] += param_2
            dev = maps_temp - flat_maps
            if det:
                return -dev@FIM_det@dev
            else:
                return -dev@FIM@dev
        param_1_range = np.linspace(-range_1, range_1, resolution_1)
        param_2_range = np.linspace(-range_2, range_2, resolution_2)
        ff, ss = np.meshgrid(flat_maps[param_pos_1] + param_1_range,
                            flat_maps[param_pos_2] + param_2_range)
        Z_sto = np.zeros((len(param_1_range), len(param_2_range)))
        Z_det = np.zeros((len(param_1_range), len(param_2_range)))
        i_k = 0
        for i in param_1_range:
            j_k = 0
            for j in param_2_range:
                Z_det[i_k,j_k] = bilinear(i,j,det=True)
                Z_sto[i_k,j_k] = bilinear(i,j,det=False)
                j_k += 1
            i_k += 1
        return ff, ss, Z_sto, Z_det

    def sensitivity(self, FIM):
        """
        Computes sensitivity measures (not normalised) for
        1) each individual parameter: from the diagonal elements of the FIM
        2) incorporating parametric interactions: from the standard deviations derived from the FIM
        More on these interpretations can be found here: https://doi.org/10.1529/biophysj.104.053405
        A larger entry translates into greater anticipated model sensitivity to changes in the parameter of interest.

        Parameters
        ----------
        FIM: 2d numpy.array
            The Fisher Information Matrix

        Returns
        -------
        sensitivity_individual: numpy.array
            Sensitivity measure for individual parameters.
        sensitivity_correlated: numpy.array
            Sensitivity measure incorporating parametric interactions.
        """
        if not np.all(np.linalg.eigvalsh(FIM)>0):
            raise Exception("FIM not positive definite - check for appropriate step-size `eps` in FIM computation and/or increase `inter_steps` for a more stable result")

        sensitivity_individual = np.sqrt(np.diagonal(FIM))
        sensitivity_correlated = np.divide(1, np.sqrt(np.diagonal(np.linalg.inv(FIM))))

        return sensitivity_individual, sensitivity_correlated

    def sample_gaussian(self, N, map_estimate, cov, x, Tf, contactMatrix, prior_dict, tangent=False):
        """
        Sample `N` samples of the parameters from the Gaussian centered at the MAP estimate with specified
        covariance `cov`.

        Parameters
        ----------
        N: int
            The number of samples.
        map_estimate: dict
            The MAP estimate, e.g. as computed by `inference.infer_parameters`.
        cov: np.array
            The covariance matrix of the flat parameters.
        x:  np.array
            The full trajectory.
        Tf: float
            The total time of the trajectory.
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        prior_dict: dict
            A dictionary containing priors. See examples.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.

        Returns
        -------
        samples: list of dict
            N samples of the Gaussian distribution.
        """
        prior_names, keys, guess, stds, _, _, bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = pyross.utils.parse_param_prior_dict(prior_dict, self.M, check_length=(not self.param_mapping_enabled))
        prior = Prior(prior_names, bounds, guess, stds)
        loglike_args = {'keys':keys, 'is_scale_parameter':is_scale_parameter,
                       'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                       'x':x, 'Tf':Tf, 'tangent':tangent}

        # Sample the flat parameters.
        mean = map_estimate['flat_map']
        sample_parameters = np.random.multivariate_normal(mean, cov, N)

        samples = []
        for sample in sample_parameters:
            new_sample = map_estimate.copy()
            new_sample['flat_params'] = sample
            new_sample['map_dict'] = \
                pyross.utils.unflatten_parameters(sample, map_estimate['flat_guess_range'],
                        map_estimate['is_scale_parameter'], map_estimate['scaled_guesses'])
            l_like = self._loglike(sample, **loglike_args)
            l_prior = np.sum(prior.logpdf(sample))
            l_post = l_like + l_prior
            new_sample['log_posterior'] = l_post
            new_sample['log_prior'] = l_prior
            new_sample['log_likelihood'] = l_like
            samples.append(new_sample)

        return samples

    def evidence_laplace(self, x, Tf, infer_result, contactMatrix=None,
                         generator=None,
                intervention_fun=None, tangent=False, eps=None,
                fd_method="central", inter_steps=10):
        """
        Compute the evidence using a Laplace approximation at the MAP estimate.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for finite differences computation of the hessian with respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified, 
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. For `fd_method="central"` it is recommended to use a step-size greater or equal to `eps`. Decreasing the step size too small can result in round-off error.
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central". Default is "central".
        inter_steps: int, optional
            Only used if `tangent=False`. Intermediate steps for interpolation between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration. 
            We have found that forward Euler is generally slower, but more stable for derivatives with respect 
            to parameters than the variable step size integrators used elsewhere in pyross. Default is 10.

        Returns
        -------
        log_evidence: float
            The log-evidence computed via Laplace approximation at the MAP estimate.
        """


        logP_MAPs = infer_result['log_posterior']
        A = self.hessian(x, Tf, infer_result, contactMatrix, generator,
                         intervention_fun, tangent, eps,
                         fd_method, inter_steps)
        k = A.shape[0]

        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + 0.5*k*np.log(2*np.pi)


    def obtain_minus_log_p(self, parameters, np.ndarray x, double Tf, contactMatrix, tangent=False):
        """Computes -logp of a full trajectory
        Parameters
        ----------
        parameters: dict
            A dictionary for the model parameters.
        x: np.array
            The full trajectory.
        Tf: float
            The time duration of the trajectory.
        contactMatrix: callable
            A function that takes time (t) as an argument and returns the contactMatrix
        tangent: bool, optional
            Set to True to use tangent space inference.

        Returns
        -------
        minus_logp: float
            Value of -logp
        """

        cdef:
            double minus_log_p
            double [:, :] x_memview=x.astype('float')
        self.set_params(parameters)
        self.contactMatrix = contactMatrix
        if tangent:
            minus_logp = self._obtain_logp_for_traj_tangent(x_memview, Tf)
        else:
            minus_logp = self._obtain_logp_for_traj(x_memview, Tf)
        return minus_logp

    cdef np.ndarray _get_r_from_x(self, np.ndarray x):
        # this function will be overridden in case of extra (non-additive) compartments
        cdef:
            np.ndarray r
        r = self.fi - np.sum(x.reshape((int(self.dim/self.M), self.M)), axis=0)
        return r

    cdef double _penalty_from_negative_values(self, np.ndarray x0):
        cdef:
            double eps=0.1/self.Omega, dev
            np.ndarray R_init
        R_init = self._get_r_from_x(x0)
        dev = - (np.sum(R_init[R_init<0]) + np.sum(x0[x0<0]))
        return (dev/eps)**2 + (dev/eps)**8

    def _all_positive(self, np.ndarray x0):
        r = self._get_r_from_x(x0)
        return (x0>0).all() and (r>0).all()

    def latent_infer_parameters(self, np.ndarray obs, np.ndarray fltr, double Tf,
                            contactMatrix, param_priors, init_priors, **kwargs):
        """
        Compute the maximum a-posteriori (MAP) estimate of the parameters and the initial conditions of a SIR type model
        when the classes are only partially observed. Unobserved classes are treated as latent variables.

        Parameters
        ----------
            see `latent_infer`

        Returns
        -------
        output: dict
            Contains the following keys for users:

            map_params_dict: dict
                A dictionary for the MAP estimates for parameter values.
                The keys are the names of the parameters.
            map_x0: np.array
                The MAP estimate for the initial conditions.
            -logp: float
                The value of -logp at MAP.

        Note
        ----
        This function just calls latent_infer (with fixed `contactMatrix`), will be deprecated.
        """

        output_dict = self.latent_infer(obs, fltr, Tf, param_priors, init_priors, contactMatrix=contactMatrix, generator=None,
                            intervention_fun=None, **kwargs)

        # rename for backwards compatibility
        output_dict['map_x0'] = output_dict['x0']
        output_dict['flat_map'] = output_dict['flat_params']
        output_dict['map_params_dict'] = output_dict['params_dict']

        return output_dict


    def nested_sampling_latent_inference(self, np.ndarray obs, np.ndarray fltr, double Tf, contactMatrix, param_priors,
                                         init_priors,tangent=False, verbose=False, nprocesses=0, queue_size=None,
                                         maxiter=None, maxcall=None, dlogz=None, n_effective=None, add_live=True,
                                         sampler=None, **dynesty_args):
        """
        Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        with latent variables using nested sampling as implemented in the `dynesty` Python package.

        Note
        ----
        This function has been replaced by `pyross.inference.latent_infer_nested_sampling` and will be deleted
        in a future version of pyross. See there for a documentation of the function parameters.
        """
        return self.latent_infer_nested_sampling(obs, fltr, Tf, param_priors, init_priors, contactMatrix=contactMatrix,
                                                 generator=None, intervention_fun=None, tangent=tangent, verbose=verbose,
                                                 nprocesses=nprocesses, queue_size=queue_size, maxiter=maxiter, maxcall=maxcall,
                                                 dlogz=dlogz, n_effective=n_effective, add_live=add_live, sampler=sampler,
                                                 **dynesty_args )

    def nested_sampling_latent_inference_process_result(self, sampler, obs, fltr, param_priors, init_priors):
        """
        Take the sampler generated by `nested_sampling_latent_inference` and produce output dictionaries for further use
        in the pyross framework.

        Note
        ----
        This function has been replaced by `pyross.inference.latent_infer_nested_sampling_process_result` and will be
        deleted in a future version of pyross. See there for a documentation of the function parameters.
        """
        result, output_samples = self.latent_infer_nested_sampling_process_result(sampler, obs, fltr, param_priors, init_priors,
                                                                                  contactMatrix=self.contactMatrix)

        # Match old dictionary key names for backward compatibility.
        for out_dict in output_samples:
            out_dict['map_x0']          = out_dict.pop('x0')
            out_dict['flat_map']        = out_dict.pop('flat_params')
            out_dict['map_params_dict'] = out_dict.pop('params_dict')

        return result, output_samples


    def mcmc_latent_inference(self, np.ndarray obs, np.ndarray fltr, double Tf, contactMatrix, param_priors,
                              init_priors, tangent=False, verbose=False, sampler=None, nwalkers=None, walker_pos=None,
                              nsamples=1000, nprocesses=0):
        """
        Sample the posterior distribution of the epidemiological parameters using ensemble MCMC.

        Note
        ----
        This function has been replaced by `pyross.inference.latent_infer_mcmc` and will be deleted
        in a future version of pyross. See there for a documentation of the function parameters.
        """
        return self.latent_infer_mcmc(obs, fltr, Tf, param_priors, init_priors, contactMatrix=contactMatrix,
                                      generator=None, intervention_fun=None, tangent=tangent, verbose=verbose,
                                      sampler=sampler, nwalkers=nwalkers, walker_pos=walker_pos,
                                      nsamples=nsamples, nprocesses=nprocesses)

    def mcmc_latent_inference_process_result(self, sampler, obs, fltr, param_priors, init_priors,
                                            flat=True, discard=0, thin=1):
        """
        Take the sampler generated by `mcmc_latent_inference` and produce output dictionaries for further use
        in the pyross framework.

        Note
        ----
        This function has been replaced by `pyross.inference.latent_infer_mcmc_process_results` and will be deleted
        in a future version of pyross. See there for a documentation of the function parameters.
        """
        output_samples = self.latent_infer_mcmc_process_result(sampler, obs, fltr, param_priors, init_priors,
                                                               contactMatrix=self.contactMatrix, flat=flat,
                                                               discard=discard, thin=thin)

        # Match old dictionary key names for backward compatibility.
        if flat:
            flat_sample_list = output_samples
        else:
            flat_sample_list = [item for sublist in output_samples for item in sublist]
        for out_dict in flat_sample_list:
            out_dict['map_x0']          = out_dict.pop('x0')
            out_dict['flat_map']        = out_dict.pop('flat_params')
            out_dict['map_params_dict'] = out_dict.pop('params_dict')

        return output_samples


    def latent_infer_control(self, obs, fltr, Tf, generator, param_priors, init_priors,
                            **kwargs):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown with partially observed classes.

        Parameters
        ----------
            see `latent_infer`

        Returns
        -------
        output_dict: dict
            A dictionary containing the following keys for users:

            map_params_dict: dict
                dictionary for MAP estimates for control parameters
            map_x0: np.array
                MAP estimates for the initial conditions
            -logp: float
                Value of -logp at MAP.

        Note
        ----
        This function just calls `latent_infer` (with the specified `generator`), will be deprecated.
        """

        output_dict = self.latent_infer(obs, fltr, Tf, param_priors, init_priors,
                                        contactMatrix=None, generator=generator,
                                        **kwargs)
        del output_dict['params_dict']
        output_dict['map_params_dict']=output_dict['control_params_dict']  # Rename entry for backwards compatibility

        return output_dict

    def _loglikelihood_latent(self, params, grad=0, generator=None, intervention_fun=None, param_keys=None,
                              param_guess_range=None, is_scale_parameter=None, scaled_param_guesses=None,
                              param_length=None, obs=None, fltr=None, Tf=None, obs0=None, init_flags=None,
                              init_fltrs=None, tangent=None, smooth_penalty=False, bounds=None, inter_steps=0,
                              objective='likelihood', **catchall_kwargs):
        if bounds is not None:
            # Check that params is within bounds. If not, return -np.inf.
            if np.any(bounds[:,0] > params) or np.any(bounds[:,1] < params):
                return -np.Inf

        inits =  np.copy(params[param_length:])

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                          param_guess_range, is_scale_parameter, scaled_param_guesses)

        parameters, kwargs = self.fill_params_dict(param_keys, orig_params, return_additional_params=True)

        self.set_params(parameters)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**kwargs)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **kwargs)
        else:
            if kwargs != {}:
                raise Exception('Key error or unspecified generator')

        x0 = self._construct_inits(inits, init_flags, init_fltrs, obs0, fltr[0])
        logl = 0
        if smooth_penalty == True:
            # Steer the global optimiser away from regions with negative initial values.
            penalty = self._penalty_from_negative_values(x0)
            x0[x0<0] = 0.1/self.Omega # set to be small and positive
            logl -= penalty*fltr.shape[0]
        elif smooth_penalty == False:
            # Return -Inf if one of the initial values is negative.
            if not self._all_positive(x0):
                return -np.Inf
        # We also support `smooth_penalty == None`, which is useful for example for computing the Hessian.

        if objective == 'likelihood':
            logl += -self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent, inter_steps=inter_steps)
        elif objective == 'least_squares':
            logl += -self._obtain_square_dev_for_lat_traj(x0, obs, fltr[1:], Tf)
        elif objective == 'least_squares_diff':
            logl += -self._obtain_square_dev_for_lat_traj_diff(x0, obs, fltr[1:], Tf)
        else:
            raise Exception('Unknown objective')
            
        return logl

    def _logposterior_latent(self, params, prior=None, verbose_likelihood=False,
                             **logl_kwargs):
        logl = self._loglikelihood_latent(params, **logl_kwargs)
        logp = logl + np.sum(prior.logpdf(params))
        if verbose_likelihood:
            print(logl,logp-logl,logp)
        return logp

    def _latent_infer_to_minimize(self, params, grad=0,
                                   **logp_kwargs):
        """Objective function for minimization call in latent_infer."""
        if 'disable_penalty' in logp_kwargs:
            logp = self._logposterior_latent(params, smooth_penalty=None,  **logp_kwargs)
        else:
            logp = self._logposterior_latent(params, smooth_penalty=True,  **logp_kwargs)
        return -logp

    def latent_infer(self, np.ndarray obs, np.ndarray fltr, Tf, param_priors, init_priors,
                     contactMatrix=None, generator=None,
                     intervention_fun=None, tangent=False,
                     verbose=False, verbose_likelihood=False, ftol=1e-5, global_max_iter=100,
                     local_max_iter=100, local_initial_step=None, global_atol=1., enable_global=True,
                     enable_local=True, cma_processes=0, cma_population=16, cma_random_seed=None, 
                     objective='likelihood', alternative_guess=None, use_mode_as_guess=False, tmp_file=None, load_backup_file=None):
        """
        Compute the maximum a-posteriori (MAP) estimate for the initial conditions and all desired parameters, including control parameters,
        for a SIR type model with partially observed classes. The unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            Total time of the trajectory
        param_priors: dict
            A dictionary that specifies priors for parameters (including control parameters, if desired). See `infer` for further explanations.
        init_priors: dict
            A dictionary for priors for initial conditions. See below for examples
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is false.
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: double
            Relative tolerance of logp
        global_max_iter: int, optional
            Number of global optimisations performed.
        local_max_iter: int, optional
            Number of local optimisation performed.
        local_initital_step: optional, float or np.array
            Initial step size for the local optimiser. If scalar, relative to the initial guess. 
            Default: Deterined by final state of global optimiser, or, if enable_global=False, 0.01
        global_atol: float
            The absolute tolerance for global minimisation.
        enable_global: bool, optional
            Set to True to enable global optimisation.
        enable_local: bool, optional
            Set to True to enable local optimisation.
        cma_processes: int, optional
            Number of parallel processes used for global optimisation.
        cma_population: int, optional
            The number of samples used in each step of the CMA algorithm.
        cma_random_seed: int (between 0 and 2**32-1)
            Random seed for the optimisation algorithms. By default it is generated from numpy.random.randint.
        objective: string, optional
            Objective for the minimisation. 'likelihood' (default), 'least_square' (least squares fit w.r.t. absolute compartment values), 
            'least_squares_diff' (least squares fit w.r.t. time-differences of compartment values)
        alternative_guess: np.array, optional
            Alternative initial quess, different form the mean of the prior. 
            Array in the same format as 'flat_params' in the result dictionary of a previous optimisation run.
        use_mode_as_guess: bool, optional
            Initialise optimisation with mode instead of mean of the prior. Makes a difference for lognormal distributions. 
        tmp_file: optional, string
            If specified, name of a file to store the temporary best estimate of the global optimiser (as backup or for inspection) as numpy array file 
        load_backup_file: optional, string
            If specified, name of a file to restore the the state of the global optimiser

        Returns
        -------
        output_dict: dict
            A dictionary containing the following keys for users:

            x0: np.array
                MAP estimates for the initial conditions
            params_dict: dict
                dictionary for MAP estimates for model parameters
            control_params_dict: dict
                dictionary for MAP estimates for control parameters (if requested)
            -logp: float
                Value of -logp at MAP.

        Note
        ----
        This function combines the functionality of `latent_infer_parameters` and `latent_infer_control`,
        which will be deprecated.
        To infer model parameters only, specify a fixed `contactMatrix` function.
        To infer control parameters only, specify a `generator` and do not specify priors for model parameters.

        Examples
        --------
        Here we list three examples, one for inferring all initial conditions
        along the fastest growing linear mode, one for inferring the initial
        conditions individually and a mixed one.

        First, suppose we only observe Is out of (S, Ia, Is) and we wish to
        infer all compartmental values of S and Ia independently. For two age
        groups with population [2500, 7500],

        >>> init_priors = {
                'independent':{
                    'fltr': [True, True, True, True, False, False],
                    'mean': [2400, 7400, 50, 50],
                    'std': [200, 200, 200, 200],
                    'bounds': [[2000, 2500], [7000, 7500], [0, 400], [0, 400]]
                }
            }

        In the 'fltr' entry, we need a boolean array indicating which components
        of the full x0 = [S0[0], S0[1], Ia0[0], Ia0[1], Is0[0], Ia0[1]] array we are inferring.
        By setting fltr = [True, True, True, True, False, False], the inference algorithm
        will know that we are inferring all components of S0 and Ia0 but not Is0.
        Similar to inference for parameter values, we also assume a log-normal
        distribution for the priors for the initial conditions.

        Next, if we are happy to assume that all our initial conditions lie
        along the fastest growing linear mode and we will only infer the
        coefficient of the mode, the init_priors dict would be,

        >>> init_priors = {
                'lin_mode_coeff':{
                    'fltr': [True, True, True, True, False, False],
                    'mean': 100,
                    'std': 100,
                    'bounds': [1, 1000]
                }
            }

        Note that the 'fltr' entry is still the same as before because we still
        only want to infer S and Ia, and the initial conditions for Is is fixed
        by the observation.

        Finally, if we want to do a mixture of both (useful when some compartments
        have aligned with the fastest growing mode but others haven't), we need
        to set the init_priors to be,

        >>> init_priors = {
                'lin_mode_coeff': {
                    'fltr': [True, True, False, False, False, False],
                    'mean': 100,
                    'std': 100,
                    'bounds': [1, 1000]
                },
                'independent':{
                    'fltr': [False, False, True, True, False, False],
                    'mean': [50, 50],
                    'std': [200, 200],
                    'bounds': [0, 400], [0, 400]
                }
            }
        """

        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        param_prior_names, keys, param_mean, param_stds, param_guess, param_guess_std, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M, check_length=(not self.param_mapping_enabled))

        # Read in initial conditions priors
        init_prior_names, init_mean, init_stds, init_guess, init_guess_std, init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        mean = np.concatenate([param_mean, init_mean]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        if use_mode_as_guess:
            guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
            guess_std = np.concatenate([param_guess_std,init_guess_std]).astype(DTYPE)
        else:
            guess = mean
            guess_std =stds
            
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        prior = Prior(param_prior_names+init_prior_names, bounds, mean, stds)
        
        cma_stds = np.minimum(guess_std, (bounds[:, 1]-bounds[:, 0])/3)
        
        if alternative_guess is not None:
            guess = alternative_guess

        minimize_args = {'generator':generator, 'intervention_fun':intervention_fun,
                       'param_keys':keys, 'param_guess_range':param_guess_range,
                       'is_scale_parameter':is_scale_parameter,
                       'scaled_param_guesses':scaled_param_guesses,
                       'param_length':param_length,
                       'obs':obs, 'fltr':fltr, 'Tf':Tf, 'obs0':obs0,
                       'init_flags':init_flags, 'init_fltrs': init_fltrs,
                       'prior':prior, 'tangent':tangent, 'objective':objective, 'verbose_likelihood':verbose_likelihood}
        res = minimization(self._latent_infer_to_minimize,
                          guess, bounds, ftol=ftol,
                          global_max_iter=global_max_iter,
                          local_max_iter=local_max_iter, local_initial_step=local_initial_step, global_atol=global_atol,
                          enable_global=enable_global, enable_local=enable_local,
                          cma_processes=cma_processes,
                          cma_population=cma_population, cma_stds=cma_stds,
                          verbose=verbose, cma_random_seed=cma_random_seed,
                          args_dict=minimize_args, tmp_file=tmp_file, load_backup_file=load_backup_file)
        estimates = res[0]

        # Get the parameters (in their original structure) from the flattened parameter vector.
        param_estimates = estimates[:param_length]
        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)
        init_estimates = estimates[param_length:]


        map_params_dict, map_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)
        map_x0 = self._construct_inits(init_estimates, init_flags, init_fltrs,
                                    obs0, fltr[0])
        l_post = -res[1]
        l_prior = np.sum(prior.logpdf(res[0]))
        l_like = l_post - l_prior
        output_dict = {
            'params_dict':map_params_dict, 'x0':map_x0, 'flat_params':estimates,
            'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
            'param_keys': keys, 'param_guess_range': param_guess_range,
            'is_scale_parameter':is_scale_parameter, 'param_length':param_length,
            'scaled_param_guesses':scaled_param_guesses,
            'init_flags': init_flags, 'init_fltrs': init_fltrs,
            'prior': prior,
        }
        if map_control_params_dict != {}:
            output_dict['control_params_dict'] = map_control_params_dict

        return output_dict

    def latent_infer_nested_sampling(self, np.ndarray obs, np.ndarray fltr, Tf, param_priors, init_priors, contactMatrix=None,
                                      generator=None, intervention_fun=None, tangent=False, verbose=False, nprocesses=0,
                                      queue_size=None, maxiter=None, maxcall=None, dlogz=None, n_effective=None, add_live=True,
                                      sampler=None, **dynesty_args):
        """
        Compute the log-evidence and weighted samples for the initial conditions and all desired parameters, including control parameters,
        for a SIR type model with partially observed classes. This function uses nested sampling as implemented in the `dynesty` Python package.

        Parameters
        ----------
        obs: 2d numpy.array
            The observed trajectories with reduced number of variables
            (number of data points, (age groups * observed model classes))
        fltr: 2d numpy.array
            A matrix of shape (no. observed variables, no. total variables),
            such that obs_{ti} = fltr_{ij} * X_{tj}
        Tf: float
            Total time of the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        param_priors: dict
            A dictionary for priors for the model parameters.
            See `latent_infer` for further explanations.
        init_priors: dict
            A dictionary for priors for the initial conditions.
            See `latent_infer` for further explanations.
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        nprocesses: int, optional
            The number of processes used for parallel evaluation of the likelihood.
        queue_size: int, optional
            Size of internal queue of likelihood values, default is nprocesses if multiprocessing is used.
        maxiter: int, optional
            The maximum number of iterations. Default is no limit.
        maxcall:int, optional
            The maximum number of calls to the likelihood function. Default no limit.
        dlogz: float, optional
            The iteration terminates if the estimated contribution of the remaining prior volume to the total evidence
            falls below this threshold. Default value is `1e-3 * (nlive - 1) + 0.01` if `add_live==True`, 0.01 otherwise.
        n_effective: float, optional
            The iteration terminates if the number of effective posterior samples reaches this values. Default is no limit.
        add_live: bool, optional
            Determines whether to add the remaining set of live points to the set of samples. Default is True.
        sampler: dynesty.NestedSampler, optional
            Continue running an instance of a nested sampler until the termination criteria are met.
        **dynesty_args:
            Arguments passed through to the construction of the dynesty.NestedSampler constructor. Relevant entries
            are (this is not comprehensive, for details see the documentation of dynesty):

            nlive: int, optional
                The number of live points. Default is 500.
            bound: {'none', 'single', 'multi', 'balls', 'cubes'}, optional
                Method used to approximately bound the prior using the current set of live points. Default is 'multi'.
            sample:  {'auto', 'unif', 'rwalk', 'rstagger', 'slice', 'rslice', 'hslice', callable}, optional
                Method used to sample uniformly within the likelihood constraint, conditioned on the provided bounds.

        Returns
        -------
        sampler: dynesty.NestedSampler
            The state of the sampler after termination of the nested sampling run.
        """

        if dynesty is None:
            raise Exception("Nested sampling needs optional dependency `dynesty` which was not found.")

        if nprocesses == 0:
            if pathos_mp:
                # Optional dependecy for multiprocessing (pathos) is installed.
                nprocesses = pathos_mp.cpu_count()
            else:
                nprocesses = 1

        if queue_size is None:
            queue_size = nprocesses

        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        param_prior_names, keys, param_guess, param_stds, _, _, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M, check_length=(not self.param_mapping_enabled))

        # Read in initial conditions priors
        init_prior_names, init_guess, init_stds, _, _,init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        prior = Prior(param_prior_names+init_prior_names, bounds, guess, stds)

        ndim = len(guess)

        prior_transform_args = {'prior':prior}
        loglike_args = {'generator':generator, 'intervention_fun':intervention_fun, 'param_keys':keys,
                        'param_guess_range':param_guess_range, 'is_scale_parameter':is_scale_parameter,
                        'scaled_param_guesses':scaled_param_guesses, 'param_length':param_length, 'obs':obs,
                        'fltr':fltr, 'Tf':Tf, 'obs0':obs0, 'init_flags':init_flags, 'init_fltrs': init_fltrs,
                        'tangent':tangent, 'bounds':bounds}

        if sampler is None:
            if nprocesses > 1:
                pool = pathos_mp.ProcessingPool(nprocesses)
                sampler = dynesty.NestedSampler(self._loglikelihood_latent, self._nested_sampling_prior_transform, ndim=ndim,
                                                logl_kwargs=loglike_args, ptform_kwargs=prior_transform_args,
                                                pool=pool, queue_size=queue_size, **dynesty_args)
            else:
                sampler = dynesty.NestedSampler(self._loglikelihood_latent, self._nested_sampling_prior_transform, ndim=ndim,
                                                logl_kwargs=loglike_args, ptform_kwargs=prior_transform_args,
                                                **dynesty_args)
        else:
            if nprocesses > 1:
                # Restart the pool we closed at the end of the previous run.
                sampler.pool = pathos_mp.ProcessingPool(nprocesses)
                sampler.M = sampler.pool.map
            elif sampler.pool is not None:
                sampler.pool = None
                sampler.M = map

        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz, n_effective=n_effective,
                           add_live=add_live, print_progress=verbose)

        if sampler.pool is not None:
            sampler.pool.close()
            sampler.pool.join()
            sampler.pool.clear()

        return sampler

    def latent_infer_nested_sampling_process_result(self, sampler, obs, fltr, param_priors, init_priors, contactMatrix=None,
                                                     generator=None, intervention_fun=None, **catchall_kwargs):
        """
        Take the sampler generated by `pyross.inference.latent_infer_nested_sampling` and produce output dictionaries for
        further use in the pyross framework. See there for additional description of parameters.

        Parameters
        ----------
        sampler: dynesty.NestedSampler
            Output of `pyross.inference.latent_infer_nested_sampling`.
        obs: 2d numpy.array
        fltr: 2d numpy.array
        param_priors: dict
        init_priors: dict
        contactMatrix: callable, optional
        generator: pyross.contactMatrix, optional
        intervention_fun: callable, optional
        **catchall_kwargs: dict
            Catches further provided arguments and ignores them.

        Returns
        -------
        result: dynesty.Result
            The result of the nested sampling iteration. Relevant entries include:

            result.logz: list
                The progression of log-evidence estimates, use result.logz[-1] for the final estimate.
        output_samples: list
            The processed weighted posterior samples.
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        param_prior_names, keys, param_guess, param_stds, _, _, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M, check_length=(not self.param_mapping_enabled))

        # Read in initial conditions priors
        init_prior_names, init_guess, init_stds, _, _,init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        prior = Prior(param_prior_names+init_prior_names, bounds, guess, stds)

        result = sampler.results
        output_samples = []
        for i in range(len(result.samples)):
            sample = result.samples[i]
            weight = np.exp(result.logwt[i] - result.logz[len(result.logz)-1])
            l_like = result.logl[i]
            # Get the parameters (in their original structure) from the flattened parameter vector.
            param_sample = sample[:param_length]
            orig_params = pyross.utils.unflatten_parameters(param_sample, param_guess_range,
                                                            is_scale_parameter, scaled_param_guesses)
            init_sample = sample[param_length:]

            sample_params_dict, sample_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
            self.set_params(sample_params_dict)

            if generator is not None:
                if intervention_fun is None:
                    self.contactMatrix = generator.constant_contactMatrix(**sample_control_params_dict)
                else:
                    self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **sample_control_params_dict)

            sample_x0 = self._construct_inits(init_sample, init_flags, init_fltrs, obs0, fltr[0])
            l_prior = np.sum(prior.logpdf(sample))
            l_post = l_prior + l_like
            output_dict = {
                'params_dict':sample_params_dict, 'x0':sample_x0, 'flat_params':sample,
                'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
                'weight':weight, 'param_keys': keys, 'param_guess_range': param_guess_range,
                'is_scale_parameter':is_scale_parameter, 'param_length':param_length,
                'scaled_param_guesses':scaled_param_guesses,
                'init_flags': init_flags, 'init_fltrs': init_fltrs,
                'prior':prior
            }

            if sample_control_params_dict != {}:
                output_dict['control_params_dict'] = sample_control_params_dict

            output_samples.append(output_dict)

        return result, output_samples

    def latent_infer_mcmc(self, np.ndarray obs, np.ndarray fltr, Tf, param_priors, init_priors, contactMatrix=None, generator=None,
                          intervention_fun=None, tangent=False, verbose=False, sampler=None, nwalkers=None, walker_pos=None,
                          nsamples=1000, nprocesses=0):
        """ Sample the posterior distribution of the initial conditions and all desired parameters, including control parameters, using
        ensemble MCMC. This requires the optional dependency `emcee`.

        Parameters
        ----------
        obs: 2d numpy.array
            The observed trajectories with reduced number of variables
            (number of data points, (age groups * observed model classes))
        fltr: 2d numpy.array
            A matrix of shape (no. observed variables, no. total variables),
            such that obs_{ti} = fltr_{ij} * X_{tj}
        Tf: float
            Total time of the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        param_priors: dict
            A dictionary for priors for the model parameters.
            See `latent_infer_parameters` for further explanations.
        init_priors: dict
            A dictionary for priors for the initial conditions.
            See `latent_infer_parameters` for further explanations.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        verbose: bool, optional
            Set to True to see a progress bar for the sample generation. Default is False.
        sampler: emcee.EnsembleSampler, optional
            Set to instance of the sampler (as returned by this function) to continue running the MCMC chains.
            Default is None (i.e. run a new chain).
        nwalkers:int, optional
            The number of chains in the ensemble (should be at least 2*dim). Default is 2*dim.
        walker_pos: np.array, optional
            The initial position of the walkers. If not specified, the function samples random positions from the prior.
        nsamples:int, optional
            The number of samples per walker. Default is 1000.
        nprocesses: int, optional
            The number of processes used to compute the likelihood for the walkers, needs `pathos` for values > 1.
            Default is the number of cpu cores if `pathos` is available, otherwise 1.

        Returns
        -------
        sampler: emcee.EnsembleSampler
            This function returns the state of the sampler. To look at the chain of the internal flattened parameters,
            run `sampler.get_chain()`. Use this to judge whether the chain has sufficiently converged. Either rerun
            `latent_infer_mcmc(..., sampler=sampler)` to continue the chain or `latent_infer_mcmc_process_result(...)`
            to process the result.

        Examples
        --------
        For the structure of the model input parameters, in particular `param_priors, init_priors`, see the documentation
        of `latent_infer`. To start sampling the posterior, run for example

        >>> sampler = estimator.latent_infer_mcmc(obs, fltr, Tf, param_priors, init_priors, contactMatrix=contactMatrix,
                                                  verbose=True)

        To judge the convergence of this chain, we can look at the trace plot of all the chains (for a moderate number of
        dimensions `dim`)

        >>> fig, axes = plt.subplots(dim, sharex=True)
        >>> samples = sampler.get_chain()
        >>> for i in range(dim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
        >>> axes[-1].set_xlabel("step number");

        For more detailed convergence metrics, see the documentation of `emcee`. To continue running this chain, we can
        call this function again with the sampler as argument

        >>> sampler = estimator.latent_infer_mcmc(obs, fltr, Tf, param_priors, init_priors, contactMatrix=contactMatrix,
                                                  verbose=True, sampler=sampler)

        This procudes 1000 additional samples in each chain. To process the results, call
        `pyross.inference.latent_infer_mcmc_process_result`.
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

        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        param_prior_names, keys, param_guess, param_stds, _, _, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M, check_length=(not self.param_mapping_enabled))

        # Read in initial conditions priors
        init_prior_names, init_guess, init_stds, _, _,init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        prior = Prior(param_prior_names+init_prior_names, bounds, guess, stds)

        ndim = len(guess)

        if nwalkers is None:
            nwalkers = 2*ndim

        logpost_args = {'generator':generator, 'intervention_fun':intervention_fun, 'bounds':bounds, 'param_keys':keys,
                        'param_guess_range':param_guess_range, 'is_scale_parameter':is_scale_parameter,
                        'scaled_param_guesses':scaled_param_guesses, 'param_length':param_length, 'obs':obs,
                        'fltr':fltr, 'Tf':Tf, 'obs0':obs0, 'init_flags':init_flags, 'init_fltrs': init_fltrs,
                        'prior':prior, 'tangent':tangent}

        if walker_pos is None:
             # If not specified, sample initial positions of walkers from prior (within bounds).
            points = np.random.rand(nwalkers, ndim)
            p0 = prior.ppf(points)
        else:
            p0 = walker_pos

        if sampler is None:
            # Start a new MCMC chain.
            if nprocesses > 1:
                mcmc_pool = pathos_mp.ProcessingPool(nprocesses)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._logposterior_latent,
                                                kwargs=logpost_args, pool=mcmc_pool)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self._logposterior_latent,
                                                kwargs=logpost_args)

            sampler.run_mcmc(p0, nsamples, progress=verbose)
        else:
            # Continue running an existing MCMC chain.
            if nprocesses > 1:
                # Restart the pool we closed at the end of the previous run.
                sampler.pool = pathos_mp.ProcessingPool(nprocesses)
            elif sampler.pool is not None:
                # If the user decided to not have multiprocessing in a subsequent run, we need
                # to reset the pool in the emcee.EnsembleSampler.
                sampler.pool = None

            sampler.run_mcmc(None, nsamples, progress=verbose)

        if sampler.pool is not None:
            sampler.pool.close()
            sampler.pool.join()
            sampler.pool.clear()

        return sampler

    def latent_infer_mcmc_process_result(self, sampler, obs, fltr, param_priors, init_priors, contactMatrix=None,
                                         generator=None, intervention_fun=None, flat=True, discard=0, thin=1,
                                         **catchall_kwargs):
        """
        Take the sampler generated by `pyross.inference.latent_infer_mcmc` and produce output dictionaries for
        further use in the pyross framework.

        Parameters
        ----------
        sampler: emcee.EnsembleSampler
            Output of `mcmc_latent_inference`.
        obs: 2d numpy.array
        fltr: 2d numpy.array
        param_priors: dict
        init_priors: dict
        contactMatrix: callable, optional
        generator: pyross.contactMatrix, optional
        intervention_fun: callable, optional
        flat: bool, optional
            This decides whether to return the samples as for each chain separately (False) or as as a combined
            list (True). Default is True.
        discard: int, optional
            The number of initial samples to discard in each chain (to account for burn-in). Default is 0.
        thin: int, optional
            Thin out the chain by taking only the n-tn element in each chain. Default is 1 (no thinning).
        **catchall_kwargs: dict
            Catches further provided arguments and ignores them.

        Returns
        -------
        output_samples: list of dict (if flat=True), or list of list of dict (if flat=False)
            The processed posterior samples.
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        param_prior_names, keys, param_guess, param_stds, _, _, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M, check_length=(not self.param_mapping_enabled))

        # Read in initial conditions priors
        init_prior_names, init_guess, init_stds, _, _,init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        prior = Prior(param_prior_names+init_prior_names, bounds, guess, stds)

        samples = sampler.get_chain(flat=flat, thin=thin, discard=discard)
        log_posts = sampler.get_log_prob(flat=flat, thin=thin, discard=discard)
        samples_per_chain = samples.shape[0]
        nr_chains = 1 if flat else samples.shape[1]
        if flat:
            output_samples = []
        else:
            output_samples = [[] for _ in nr_chains]

        for i in range(samples_per_chain):
            for j in range(nr_chains):
                if flat:
                    sample = samples[i,:]
                    l_post = log_posts[i]
                else:
                    sample = samples[i, j, :]
                    l_post = log_posts[i]
                weight = 1.0 / (samples_per_chain * nr_chains)
                param_sample = sample[:param_length]
                orig_params = pyross.utils.unflatten_parameters(param_sample, param_guess_range,
                                                                is_scale_parameter, scaled_param_guesses)
                init_sample = sample[param_length:]

                sample_params_dict, sample_control_params_dict = self.fill_params_dict(keys, orig_params, return_additional_params=True)
                self.set_params(sample_params_dict)

                if generator is not None:
                    if intervention_fun is None:
                        self.contactMatrix = generator.constant_contactMatrix(**sample_control_params_dict)
                    else:
                        self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **sample_control_params_dict)

                sample_x0 = self._construct_inits(init_sample, init_flags, init_fltrs, obs0, fltr[0])
                l_prior = np.sum(prior.logpdf(sample))
                l_like = l_post - l_prior
                output_dict = {
                    'params_dict':sample_params_dict, 'x0':sample_x0, 'flat_params':sample,
                    'log_posterior':l_post, 'log_prior':l_prior, 'log_likelihood':l_like,
                    'weight':weight, 'param_keys': keys, 'param_guess_range': param_guess_range,
                    'is_scale_parameter':is_scale_parameter, 'param_length':param_length,
                    'scaled_param_guesses':scaled_param_guesses,
                    'init_flags': init_flags, 'init_fltrs': init_fltrs,
                    'prior':prior
                }

                if sample_control_params_dict != {}:
                    output_dict['control_params_dict'] = sample_control_params_dict

                if flat:
                    output_samples.append(output_dict)
                else:
                    output_samples[j].append(output_dict)

        return output_samples

    def _latent_mean(self, params, contactMatrix=None,
                      generator=None, intervention_fun=None,
              param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None, param_length=None,
                            obs=None, fltr=None, Tf=None, obs0=None,
                            inter_steps=None,
                            init_flags=None, init_fltrs=None):
        """Objective function for differentiation call in latent_FIM and latent_FIM_det."""
        param_estimates = np.copy(params[:param_length])

        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)

        init_estimates =  np.copy(params[param_length:])

        map_params_dict, map_control_params_dict = self.fill_params_dict(param_keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

        fltr_ = fltr[1:]
        Nf=fltr_.shape[0]+1
        full_fltr = sparse.block_diag(fltr_)

        x0 = self._construct_inits(init_estimates, init_flags, init_fltrs, obs0, fltr[0])

        if inter_steps:
            x0 = np.multiply(x0, self.Omega)
            xm = pyross.utils.forward_euler_integration(self._rhs0, x0, 0, Tf, Nf, inter_steps)
            xm = xm[::inter_steps]
            xm = np.divide(xm, self.Omega)
        else:
            xm = self.integrate(x0, 0, Tf, Nf)
        xm_red = full_fltr@(np.ravel(xm[1:]))
        return xm_red

    def _latent_cov(self, params, contactMatrix=None,
             generator=None, intervention_fun=None,
             param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None, param_length=None,
                            obs=None, fltr=None, Tf=None, obs0=None,
                            inter_steps=None,
                            init_flags=None, init_fltrs=None, tangent=False):
        """Objective function for differentiation call in latent_FIM."""
        param_estimates = np.copy(params[:param_length])

        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)

        init_estimates =  np.copy(params[param_length:])

        map_params_dict, map_control_params_dict = self.fill_params_dict(param_keys, orig_params, return_additional_params=True)
        self.set_params(map_params_dict)

        if generator is not None:
            if intervention_fun is None:
                self.contactMatrix = generator.constant_contactMatrix(**map_control_params_dict)
            else:
                self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **map_control_params_dict)

        x0 = self._construct_inits(init_estimates, init_flags, init_fltrs, obs0, fltr[0])
        fltr_ = fltr[1:]
        Nf=fltr_.shape[0]+1
        full_fltr = sparse.block_diag(fltr_)

        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf,
                                                                  inter_steps)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, inter_steps)

        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        return cov_red

    def latent_FIM(self, obs, fltr, Tf, infer_result, contactMatrix=None,
                   generator=None,
                   intervention_fun=None, tangent=False, eps=None,
                   inter_steps=100):
        """
        Computes the Fisher Information Matrix (FIM) of the stochastic model for the initial conditions and all desired parameters, including control parameters, for a SIR type model with partially observed classes. The unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for numerical differentiation of the process mean and its full covariance matrix with 
            respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified,
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. It is recommended to use a step-size greater or equal to `eps`. Decreasing the step size too small can result in round-off error.
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will 
            make computations slower. Setting `inter_steps=0` will fall back to the method accessible via 
            `det_method` for the deterministic integration. We have found that forward Euler is generally 
            slower, but more stable for derivatives with respect to parameters than the variable step 
            size integrators used elsewhere in pyross. Default is 100.
        Returns
        -------
        FIM: 2d numpy.array
            The Fisher Information Matrix
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        infer_result_loc = infer_result.copy()
        # backwards compatibility
        if 'flat_map' in infer_result_loc:
            infer_result_loc['flat_params'] = infer_result_loc.pop('flat_map')

        flat_params = np.copy(infer_result_loc['flat_params'])

        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs']:
            kwargs[key] = infer_result_loc[key]

        def mean(y):
            return self._latent_mean(y, contactMatrix=contactMatrix,
                                      generator=generator,
                              intervention_fun=intervention_fun, obs=obs,
                              fltr=fltr, Tf=Tf, obs0=obs0,
                              inter_steps=inter_steps,
                              **kwargs)

        def covariance(y):
            return self._latent_cov(y, contactMatrix=contactMatrix,
                                     generator=generator,
                              intervention_fun=intervention_fun, obs=obs,
                              fltr=fltr, Tf=Tf, obs0=obs0, tangent=tangent,
                              inter_steps=inter_steps,
                              **kwargs)

        if np.all(eps == None):
            xx = infer_result_loc['flat_params']
            fx = abs(infer_result_loc['log_likelihood'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**np.divide(1,3)
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_params))
        print('eps-vector used for differentiation: ', eps)

        cov = covariance(flat_params)
        invcov = np.linalg.inv(cov)

        dim = len(flat_params)
        FIM = np.empty((dim,dim))
        dmu = []
        dcov = []

        for i in range(dim):
            dmu.append(pyross.utils.partial_derivative(mean, var=i, point=flat_params, dx=eps[i]))
            dcov.append(pyross.utils.partial_derivative(covariance,  var=i, point=flat_params, dx=eps[i]))

        for i in range(dim):
            t1 = dmu[i]@invcov@dmu[i]
            t2 = np.multiply(0.5,np.trace(invcov@dcov[i]@invcov@dcov[i]))
            FIM[i,i] = t1 + t2

        rows,cols = np.triu_indices(dim,1)

        for i,j in zip(rows,cols):
            t1 = dmu[i]@invcov@dmu[j]
            t2 = np.multiply(0.5,np.trace(invcov@dcov[i]@invcov@dcov[j]))
            FIM[i,j] = t1 + t2

        i_lower = np.tril_indices(dim,-1)
        FIM[i_lower] = FIM.T[i_lower]
        return FIM

    def latent_FIM_det(self, obs, fltr, Tf, infer_result,
                       contactMatrix=None, generator=None,
                       intervention_fun=None,
                       eps=None, measurement_error=1e-2, inter_steps=100):
        """
        Computes the Fisher Information Matrix (FIM) of the deterministic model (ODE based, including a constant measurement error) for the initial conditions and all desired parameters, including control parameters, for a SIR type model with partially observed classes. The unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        eps: float or numpy.array, optional
            Step size for numerical differentiation of the process mean and its full covariance matrix with 
            respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. 
            If not specified, 
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. It is recommended to use a step-size greater or equal to `eps`. Decreasing the step size too small can result in round-off error.
        measurement_error: float, optional
            Standard deviation of measurements (uniform and independent Gaussian measurement error assumed). Default is 1e-2.
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will 
            make computations slower. Setting inter_steps=0 will fall back to the method accessible via 
            det_method for the deterministic integration. We have found that forward Euler is generally slower, 
            but more stable for derivatives with respect to parameters than the variable step size integrators 
            used elsewhere in pyross. Default is 100.

        Returns
        -------
        FIM_det: 2d numpy.array
            The Fisher Information Matrix
        """



        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        infer_result_loc = infer_result.copy()
        # backwards compatibility
        if 'flat_map' in infer_result_loc:
            infer_result_loc['flat_params'] = infer_result_loc.pop('flat_map')

        flat_params = np.copy(infer_result_loc['flat_params'])
        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs']:
            kwargs[key] = infer_result_loc[key]

        def mean(y):
            return self._latent_mean(y, contactMatrix=contactMatrix,
                                      generator=generator,
                              intervention_fun=intervention_fun, obs=obs,
                              fltr=fltr, Tf=Tf, obs0=obs0,
                              inter_steps=inter_steps,
                              **kwargs)

        if np.all(eps == None):
            xx = infer_result_loc['flat_params']
            fx = abs(infer_result_loc['log_likelihood'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**np.divide(1,3)
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_params))
        print('eps-vector used for differentiation: ', eps)

        fltr_ = fltr[1:]
        sigma_sq = measurement_error*measurement_error
        cov_diag = np.repeat(sigma_sq, repeats=(int(self.dim)*(fltr_.shape[0])))
        cov = np.diag(cov_diag)
        full_fltr = sparse.block_diag(fltr_)
        cov_red = full_fltr@cov@np.transpose(full_fltr)
        invcov = np.linalg.inv(cov_red)

        dim = len(flat_params)
        FIM_det = np.empty((dim,dim))
        dmu = []

        for i in range(dim):
            dmu.append(pyross.utils.partial_derivative(mean, var=i, point=flat_params, dx=eps[i]))

        for i in range(dim):
            FIM_det[i,i] = dmu[i]@invcov@dmu[i]

        rows,cols = np.triu_indices(dim,1)

        for i,j in zip(rows,cols):
            FIM_det[i,j] = dmu[i]@invcov@dmu[j]

        i_lower = np.tril_indices(dim,-1)
        FIM_det[i_lower] = FIM_det.T[i_lower]
        return FIM_det

    def latent_hessian(self, obs, fltr, Tf, infer_result, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False,
                       eps=None, fd_method="central", inter_steps=0, nprocesses=0, basis=None):
        """
        Computes the Hessian matrix for the initial conditions and all desired parameters, including control parameters, for a SIR type model with partially observed classes. The unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for finite differences computation of the hessian with respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. If not specified,
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

           is used. For `fd_method="central"` it is recommended to use a step-size greater or equal to `eps`. 
           Decreasing the step size too small can result in round-off error.
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central". Default is "central".
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration. 
            We have found that forward Euler is generally slower, but sometimes more stable for derivatives with respect 
            to parameters than the variable step size integrators used elsewhere in pyross. Default is 0.
        Returns
        -------
        hess: 2d numpy.array
            The Hessian matrix
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        flat_params = np.copy(infer_result['flat_params'])

        kwargs = {}
        kwargs['obs'] = obs
        kwargs['fltr'] = fltr
        kwargs['Tf'] = Tf
        kwargs['obs0'] = obs0
        kwargs['tangent'] = tangent
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs', 'prior']:
            kwargs[key] = infer_result[key]

        kwargs['generator']=generator
        kwargs['intervention_fun']=intervention_fun
        kwargs['inter_steps']=inter_steps
        kwargs['disable_penalty']=None

        if np.all(eps == None):
            xx = infer_result['flat_params']
            fx = abs(infer_result['log_posterior'])
            eps = 100 * xx * np.divide(np.spacing(fx),fx)**(0.25)
            #eps = 10.*np.spacing(flat_params)**(0.25)
        print('epsilon used for differentiation: ', eps)

        hess = hessian_finite_difference(flat_params, self._latent_infer_to_minimize, eps, method=fd_method, nprocesses=nprocesses, 
                                         basis=basis, function_kwargs=kwargs)
        return hess
    
    
    def sample_latent(self, obs, fltr, Tf, infer_result, flat_params_list, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False, inter_steps=0, nprocesses=0):
        """
        Samples the posterior and prior 

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            The total time of the trajectory.
        infer_result: dict
            Dictionary returned by latent_infer
        flat_params_list: list of np.array's
            Parameters for which the prior and posterior are sampled
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration.
        nprocesses: int, optional
            The number of processes used to compute the likelihood for the walkers, needs `pathos`. Default is
            the number of cpu cores if `pathos` is available, otherwise 1.

        Returns
        -------
        posterior: np.array
            posterior evaluated along the 1d slice
        prior: np.array
            prior evaluated along the 1d slice
        
        """
        # Sanity checks of the inputs
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)

        # Process fltr and obs
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        flat_params = np.copy(infer_result['flat_params'])

        kwargs = {}
        kwargs['obs'] = obs
        kwargs['fltr'] = fltr
        kwargs['Tf'] = Tf
        kwargs['obs0'] = obs0
        kwargs['tangent'] = tangent
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs', 'prior']:
            kwargs[key] = infer_result[key]

        kwargs['generator']=generator
        kwargs['intervention_fun']=intervention_fun
        kwargs['inter_steps']=inter_steps
        kwargs['disable_penalty']=None


        posterior = eval_parallel(flat_params_list, self._latent_infer_to_minimize, nprocesses=nprocesses, function_kwargs=kwargs)
        prior = [ np.sum(infer_result['prior'].logpdf(s)) for s in flat_params_list]
        
        return -np.array(posterior), np.array(prior)
    

    def latent_param_slice(self, obs, fltr, Tf, infer_result, pos, direction, scale, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False, inter_steps=0, nprocesses=0):
        """
        Samples the posterior and prior along a one-dimensional slice of the parameter space

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            The total time of the trajectory.
        infer_result: dict
            Dictionary returned by latent_infer
        pos: np.array
            Position in parameter space around which the parameter slice is computed
        direction: np.array
            Direction in parameter space in which the parameter slice is computed    
        scale: np.array
            Values by which the direction vector is scaled. Points evaluated are pos + scale * direction
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration.
        nprocesses: int, optional
            The number of processes used to compute the likelihood for the walkers, needs `pathos`. Default is
            the number of cpu cores if `pathos` is available, otherwise 1.

        Returns
        -------
        posterior: np.array
            posterior evaluated along the 1d slice
        prior: np.array
            prior evaluated along the 1d slice
        
        """


        samples = [ pos + s*direction for s in scale]
      
        return self.sample_latent(obs, fltr, Tf, infer_result, samples, contactMatrix,
                       generator, intervention_fun, tangent, inter_steps, nprocesses)
    
    def sample_gaussian_latent(self, N, obs, fltr, Tf, infer_result, invcov, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False, inter_steps=0, allow_negative=False, nprocesses=0):
        """
        Sample `N` samples of the parameters from the Gaussian centered at the MAP estimate with specified
        covariance `cov`.
        
        Parameters
        ----------
        N: int
            The number of samples.
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            The total time of the trajectory.
        infer_result: dict
            Dictionary returned by latent_infer
        invcov: np.array
            The inverse covariance matrix of the flat parameters.
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        allow_negative: bool, optional
            Allow negative values of the sample parameters. If False, samples with negative paramters values are discarded 
            and additional samples are drawn until the specified number `N` of samples is reached. Default is False.
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make computations slower. 
            Setting `inter_steps=0` will fall back to the method accessible via `det_method` for the deterministic integration.
        nprocesses: int, optional
            The number of processes used to compute the likelihood for the walkers, needs `pathos`. Default is
            the number of cpu cores if `pathos` is available, otherwise 1.

        Returns
        -------
        samples: list of np.array's
            N samples of the Gaussian distribution (flat parameters).
        posterior: np.array
            posterior evaluated along the 1d slice
        prior: np.array
            prior evaluated along the 1d slice
        
        """

        
        mean = infer_result['flat_params']
        
        chol = cholesky(invcov, lower=False)
        L = dtrtri(chol, lower=0)[0]
        
        uninormal=multivariate_normal(cov=np.eye(len(mean)))
        samples=[]
        xlist=uninormal.rvs(1000000)
        ndx=0
        for i in range(N):
            while True:
                x=xlist[ndx]
                ndx=ndx+1
                if ndx>=len(xlist):
                    xlist=uninormal.rvs(len(xlist))
                    ndx=0
                s=(L@x.T).T + mean
                if np.min(s)>0 or allow_negative:
                    break
            samples.append(s)
        
        posterior, prior = self.sample_latent(obs, fltr, Tf, infer_result, samples, contactMatrix,
                                           generator, intervention_fun, tangent, inter_steps, nprocesses)
        
        return samples, posterior, prior
    


    def latent_evidence_laplace(self, obs, fltr, Tf, infer_result, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False,
                       eps=None, fd_method="central", inter_steps=100):
        """
        Compute the evidence using a Laplace approximation at the MAP estimate for a SIR type model with partially observed classes. The unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            Total time of the trajectory
        infer_result: dict
            Dictionary returned by latent_infer
        contactMatrix: callable, optional
            A function that returns the contact matrix at time t (input). If specified, control parameters are not inferred.
            Either a contactMatrix or a generator must be specified.
        generator: pyross.contactMatrix, optional
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
            Either a contactMatrix or a generator must be specified.
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO), where aW, aS and aO are (2, M) arrays.
            The contact matrices are then rescaled as :math:`aW[0]_i CW_{ij} aW[1]_j` etc.
            If not set, assume intervention that's constant in time.
            See `contactMatrix.constant_contactMatrix` for details on the keyword parameters.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for finite differences computation of the hessian with respect to the parameters. Must be either a scalar, or an array of length `len(infer_result['flat_params'])`. If not specified, 
            
            .. code-block:: python

               eps = 100*infer_result['flat_params'] 
                     *numpy.divide(numpy.spacing(infer_result['log_likelihood']),
                     infer_result['log_likelihood'])**(0.25) 

            is used. For `fd_method="central"` it is recommended to use a step-size greater or equal to `eps`. 
            Decreasing the step size too small can result in round-off error.
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central". Default is "central".
        inter_steps: int, optional
            Intermediate steps between observations for the deterministic forward Euler integration. 
            A higher number of intermediate steps will improve the accuracy of the result, but will make 
            computations slower. Setting `inter_steps=0` will fall back to the method accessible via `det_method` 
            for the deterministic integration. We have found that forward Euler is generally slower, 
            but more stable for derivatives with respect to parameters than the variable step size integrators used elsewhere in pyross. Default is 100.
        Returns
        -------
        log_evidence: float
            The log-evidence computed via Laplace approximation at the MAP estimate.
        """
        logP_MAPs = infer_result['log_posterior']
        A = self.latent_hessian(obs, fltr, Tf, infer_result, contactMatrix,
                generator, intervention_fun, tangent, eps, fd_method,
                inter_steps)
        k = A.shape[0]

        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + 0.5*k*np.log(2*np.pi)


    def minus_logp_red(self, parameters, np.ndarray x0, np.ndarray obs,
                            np.ndarray fltr, double Tf, contactMatrix, tangent=False, objective='likelihood'):
        """Computes -logp for a latent trajectory

        Parameters
        ----------
        parameters: dict
            A dictionary of parameter values, same as the ones required for initialisation.
        x0: numpy.array
            Initial conditions
        obs: numpy.array
            The observed trajectory without the initial datapoint
        fltr: boolean sequence or array
            True for observed and False for unobserved.
            e.g. if only Is is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            The total time of the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.

        Returns
        -------
        minus_logp: float
            -log(p) for the observed trajectory with the given parameters and initial conditions
        """

        cdef double minus_log_p
        cdef Py_ssize_t nClass=int(self.dim/self.M)
        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # check that x0 is consistent with obs0
        x0_obs = fltr[0].dot(x0)
        if not np.allclose(x0_obs, obs0):
            print('x0 not consistent with obs0. '
                  'Using x0 in the calculation of logp...')
        self.set_params(parameters)
        if objective == 'likelihood':
            minus_logp = self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent)
        elif objective == 'least_squares':
            minus_logp = self._obtain_square_dev_for_lat_traj(x0, obs, fltr[1:], Tf)
        elif objective == 'least_squares_diff':
            minus_logp = self._obtain_square_dev_for_lat_traj_diff(x0, obs, fltr[1:], Tf)
        else:
            raise Exception('Unknown objective')
        return minus_logp

    def sample_endpoints(self, obs, fltr, Tf, infer_result, nsamples, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False,
                       inter_steps=100):
        cdef Py_ssize_t i
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)
        x0 = infer_result['x0'].copy()
        fltr, obs, _ = pyross.utils.process_latent_data(fltr, obs)
        self.set_params(infer_result['params_dict'])
        mean, cov, null_space, known_space = self._mean_cov_for_lat_endpoint(x0, obs, fltr[1:], Tf)
        partial_inits = np.random.multivariate_normal(mean, cov, nsamples)
        inits = (null_space.T@partial_inits.T).T + known_space
        for i in range(nsamples):
            while not self._all_positive(inits[i]):
                partial_inits = np.random.multivariate_normal(mean, cov)
                inits[i] = (null_space.T@partial_inits) + known_space
        return inits

    def sample_trajs(self, obs, fltr, Tf, infer_result, nsamples, contactMatrix=None,
                       generator=None, intervention_fun=None, tangent=False,
                       inter_steps=100, require_positive=True):
        cdef Py_ssize_t i, Nf=obs.shape[0]
        self._process_contact_matrix(contactMatrix, generator, intervention_fun)
        x0 = infer_result['x0'].copy()
        fltr = pyross.utils.process_fltr(fltr, Nf)
        self.set_params(infer_result['params_dict'])
        mean, cov, full_null_space, known_space = self._mean_cov_for_lat_traj(x0, obs[1:], fltr[1:], Tf)
        trajs = np.full((nsamples, (Nf-1), self.dim), -1, dtype=DTYPE)
        if require_positive:
            for i in range(nsamples):
                while not all(map(self._all_positive, trajs[i])):
                    partial_trajs = np.random.multivariate_normal(mean, cov)
                    trajs[i] = (full_null_space.T@partial_trajs + known_space).reshape((Nf-1, self.dim))
        else:
            partial_trajs = np.random.default_rng().multivariate_normal(mean, cov, nsamples, method='eigh')
            for i in range(nsamples):
                trajs[i] = (full_null_space.T@partial_trajs[i] + known_space).reshape((Nf-1, self.dim))
        return trajs


    def get_mean_inits(self, init_priors, np.ndarray obs0, np.ndarray fltr0):
        """Construct full initial conditions from the prior dict

        Parameters
        ----------
        init_priors: dict
            A dictionary for priors for initial conditions.
            Same as the `init_priors` passed to `latent_infer`.
            In this function, only takes the mean.
        obs0: numpy.array
            Observed initial conditions.
        fltr0: numpy.array
            Filter for the observed initial conditions.

        Returns
        -------
        x0: numpy.array
            Full initial conditions.
        """
        _, init_mean, _, _, _, _,init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))
        x0 = self._construct_inits(init_mean, init_flags, init_fltrs, obs0, fltr0)
        return x0

    cpdef find_fastest_growing_lin_mode(self, double t):
        cdef:
            np.ndarray [DTYPE_t, ndim=1] x0
            Py_ssize_t S_index, M=self.M
        # assume no infected at the start and compute eig vecs for the infectious species
        x0 = np.zeros((self.dim), dtype=DTYPE)
        assert 'S' in self.class_index_dict, 'for fastest growing mode, a class "S" needs to be specified'
        S_index = self.class_index_dict['S']
        x0[S_index*M:(S_index+1)*M] = self.fi
        self.compute_jacobian_and_b_matrix(x0, t,
                                           b_matrix=False, jacobian=True)
        sign, eigvec = pyross.utils.largest_real_eig(self.J_mat)
        if not sign: # if eigval not positive, just return the zero state
            return np.zeros(self.dim)
        else:
            if eigvec[S_index*M] > 0:
                eigvec = - eigvec
            return eigvec/np.linalg.norm(eigvec, ord=1)

    def set_lyapunov_method(self, lyapunov_method, rtol=None, max_steps=0):
        """Sets the method used for deterministic integration for the SIR_type model

        Parameters
        ----------
        lyapunov_method: str
            The name of the integration method. Choose between 'LSODA', 'RK45', 'RK2', 'RK4', and 'euler'.
        rtol: double, optional
            relative tolerance of the integrator (default 1e-3)
        max_steps: int
            Maximum number of integration steps (total) for the integrator. Default: unlimited (represented as 0)
            Parameters for which the integrator reaches max_steps are disregarded by the optimiser.
        """
        if lyapunov_method not in ['LSODA', 'RK45', 'RK2', 'RK4', 'euler']:
            raise Exception('{} not implemented. Choose between LSODA, RK45, RK2, RK4, and euler'.format(lyapunov_method))
        self.lyapunov_method=lyapunov_method
        if rtol is not None:
            self.rtol_lyapunov = rtol
        if max_steps is not None:
            self.max_steps_lyapunov = max_steps

    def set_det_method(self, det_method, rtol=None, max_steps=None):
        """Sets the method used for deterministic integration for the SIR_type model

        Parameters
        ----------
        det_method: str
            The name of the integration method. Choose between 'LSODA' and 'RK45'.
        rtol: double, optional
            relative tolerance of the integrator (default 1e-3)
        max_steps: int, optional
            Maximum number of integration steps (total) for the integrator. Default: unlimited (represented as 0)
            Parameters for which the integrator reaches max_steps are disregarded by the optimiser.
        """
        if det_method not in ['LSODA', 'RK45']:
            raise Exception('{} not implemented. Choose between LSODA and RK45'.format(det_method))
        self.det_method=det_method
        if rtol is not None:
            self.rtol_det = rtol
        if max_steps is not None:
            self.max_steps_det = max_steps


    def set_det_model(self, parameters):
        """
        Sets the internal deterministic model with given epidemiological parameters

        Parameters
        ----------
        parameters: dict
            A dictionary of parameter values, same as the ones required for initialisation.
        """
        raise NotImplementedError("Please Implement set_det_model in subclass")

    def set_contact_matrix(self, contactMatrix):
        """
        Sets the internal contact matrix

        Parameters
        ----------
        contactMatrix: callable
            A function that returns the contact matrix given time, with call
            signature contactMatrix(t).
        """
        self.contactMatrix = contactMatrix

    def make_params_dict(self):
        raise NotImplementedError("Please Implement make_params_dict in subclass")

    def fill_params_dict(self, keys, params, return_additional_params=False):
        """Returns a full dictionary for epidemiological parameters with some changed values

        Parameters
        ----------
        keys: list of String
            A list of names of parameters to be changed.
        params: numpy.array of list
            An array of the same size as keys for the updated value.
        return_additional_params: boolean, optional (default = False)
            Handling of parameters that are not model parameters (e.g. control parameters). False: raise exception, True: return second dictionary with other parameters

        Returns
        -------
        full_parameters: dict
            A dictionary of epidemiological parameters.
            For parameter names specified in `keys`, set the values to be the ones in `params`;
            for the others, use the values stored in the class.
        """
        full_parameters = self.make_params_dict()
        others = {}
        for (i, k) in enumerate(keys):
            if k in self.param_keys:
                full_parameters[k] = params[i]
            elif return_additional_params:
                others[k] = params[i]
            else:
                raise Exception('{} is not a parameter of the model'.format(k))
        if return_additional_params:
            return full_parameters, others
        else:
            return full_parameters

    def set_params(self, parameters):
        """Sets epidemiological parameters used for evaluating -log(p)

        Parameters
        ----------
        parameters: dict
            A dictionary containing all epidemiological parameters.
            Same keys as the one used to initialise the class.

        Notes
        -----
        Can use `fill_params_dict` to generate the full dictionary if only a few parameters are changed
        """
        self.set_det_model(parameters)
        self.beta = pyross.utils.age_dep_rates(parameters['beta'], self.M, 'beta')
        self.gIa = pyross.utils.age_dep_rates(parameters['gIa'], self.M, 'gIa')
        self.gIs = pyross.utils.age_dep_rates(parameters['gIs'], self.M, 'gIs')
        self.fsa = pyross.utils.age_dep_rates(parameters['fsa'], self.M, 'fsa')
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], self.M, 'alpha')

    def _construct_inits(self, init_guess, flags, fltrs, obs0, fltr0):
        cdef:
            np.ndarray [DTYPE_t, ndim=1] x0
            np.ndarray [DTYPE_t, ndim=2] F
            Py_ssize_t start=0
        x0 = obs0
        F = fltr0
        if flags[0]: # lin mode
            coeff = init_guess[0]
            x0 = np.concatenate((x0, fltrs[0]@self._lin_mode_inits(coeff)))
            F = np.concatenate((F, fltrs[0]), axis=0)
            start += 1
        if flags[1]: # independent guesses
            x0 = np.concatenate((x0, init_guess[start:]))
            F = np.concatenate((F, fltrs[1]), axis=0)
        assert np.linalg.matrix_rank(F) == self.dim, 'Conflicts in initial conditions'
        return np.linalg.solve(F, x0)

    def _lin_mode_inits(self, double coeff):
        cdef double [:] v, x0, fi=self.fi
        v = self.find_fastest_growing_lin_mode(0)
        v = np.multiply(v, coeff)
        x0 = np.zeros((self.dim), dtype=DTYPE)
        x0[:self.M] = fi
        return np.add(x0, v)

    cdef double _obtain_logp_for_traj(self, double [:, :] x, double Tf,
                                     Py_ssize_t inter_steps=0):
        cdef:
            double log_p = 0
            double [:] xi, xf, dev
            double [:, :] cov, xm, _xm
            Py_ssize_t i, Nf=x.shape[0], steps=self.steps
            double [:] time_points = np.linspace(0, Tf, Nf)
        for i in range(Nf-1):
            xi = x[i]
            xf = x[i+1]
            ti = time_points[i]
            tf = time_points[i+1]
            if inter_steps:
                xi = np.multiply(xi, self.Omega)
                _xm = pyross.utils.forward_euler_integration(self._rhs0, xi,
                                                             ti, tf,
                                                             steps, inter_steps)
                _xm = np.divide(_xm, self.Omega)
                self._xm = np.copy(_xm)
                self._interp = []
                times = np.linspace(ti, tf, inter_steps*steps)
                for i in range(_xm.shape[1]):
                    self._interp.append(interpolate.interp1d(times, _xm[:,i],
                                                             kind='linear'))
                xm = _xm[::inter_steps]
                sol = self.interpolate_euler
            else:
                xm, sol = self.integrate(xi, ti, tf, steps, dense_output=True)
            self.integrator_step_count = 0
            cov = self._estimate_cond_cov(sol, ti, tf)
            dev = np.subtract(xf, xm[steps-1])
            log_p += self._log_cond_p(dev, cov)
        return -log_p

    cdef double _obtain_logp_for_lat_traj(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf, tangent=False,
                                         Py_ssize_t inter_steps=0):
        cdef:
            Py_ssize_t reduced_dim=obs_flattened.shape[0], Nf=fltr.shape[0]+1
            double [:, :] xm
            double [:] xm_red, dev
            np.ndarray[DTYPE_t, ndim=2] cov_red, full_cov
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf, inter_steps=inter_steps)
        else:
            try:
                xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, inter_steps=inter_steps)
            except MaxIntegratorStepsException:
                return np.Inf
        full_fltr = sparse.block_diag(fltr)
        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        cov_red_inv_dev, ldet = pyross.utils.solve_symmetric_close_to_singular(cov_red, dev)
        log_p = -np.dot(dev, cov_red_inv_dev)*(self.Omega/2.)
        log_p -= (ldet-reduced_dim*log(self.Omega))/2. + (reduced_dim/2.)*log(2.*PI)
        log_p -= reduced_dim*np.log(self.Omega)
        return -log_p
    
    
    cdef double _obtain_square_dev_for_lat_traj(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf):
        cdef:
            Py_ssize_t reduced_dim=obs_flattened.shape[0], Nf=fltr.shape[0]+1
            double [:, :] xm
            double [:] xm_red, dev
            
        xm = self.integrate(x0, 0, Tf, Nf, dense_output=False,
                                           max_step=self.steps*Nf)
        xm = xm[1:]
        full_fltr = sparse.block_diag(fltr)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        sqdev = np.sum(np.square(dev))
        return sqdev

    cdef double _obtain_square_dev_for_lat_traj_diff(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf):
        cdef:
            Py_ssize_t reduced_dim=obs_flattened.shape[0], Nf=fltr.shape[0]+1
            double [:, :] xm
            double [:] xm_red, dev
            
        xm = self.integrate(x0, 0, Tf, Nf, dense_output=False,
                                           max_step=self.steps*Nf)
        xm = np.diff(xm,axis=0)
        full_fltr = sparse.block_diag(fltr)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        sqdev = np.sum(np.square(dev)/(xm_red+np.ones(reduced_dim)))
        return sqdev
    
    def _mean_cov_for_lat_endpoint(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf, tangent=False, Py_ssize_t inter_steps=0):
        cdef:
            Py_ssize_t Nf=fltr.shape[0]+1, reduced_dim=obs_flattened.shape[0]
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf, inter_steps=inter_steps)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, inter_steps=inter_steps)
        last_fltr = fltr[Nf-2]
        last_obs = obs_flattened[reduced_dim-last_fltr.shape[0]:]
        null_space, known_space = self._split_spaces(last_fltr, last_obs)

        last_full_fltr = np.vstack((fltr[Nf-2], null_space))
        full_fltr = sparse.block_diag([*fltr[:Nf-2], last_full_fltr])
        cov_red = full_fltr@full_cov@(full_fltr.T)
        xm_red = full_fltr@np.ravel(xm)

        xm_known = xm_red[:reduced_dim]
        dev=np.subtract(obs_flattened, xm_known)
        invcov = np.linalg.inv(cov_red)
        cov_last_red = np.linalg.inv(invcov[reduced_dim:, reduced_dim:])
        xm_last_red = xm_red[reduced_dim:] - cov_last_red@invcov[reduced_dim:, :reduced_dim]@dev
        return xm_last_red, cov_last_red/self.Omega, null_space, known_space

    def _mean_cov_for_lat_traj(self, double [:] x0, np.ndarray obs, np.ndarray fltr,
                                            double Tf, tangent=False, Py_ssize_t inter_steps=0):
        cdef:
            Py_ssize_t Nf=fltr.shape[0]+1, i, dim=self.dim
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf, inter_steps=inter_steps)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, inter_steps=inter_steps)
        known_spaces = np.empty((Nf-1, dim), dtype=DTYPE)
        null_spaces = []
        full_fltrs = []

        for i in range(Nf-1):
            null_space, known_spaces[i] = self._split_spaces(fltr[i], obs[i])
            null_spaces.append(null_space)
            full_fltrs.append(fltr[i])
      
        full_fltr_mat = sparse.block_diag(full_fltrs)
        full_null_space = sparse.block_diag(null_spaces)
        
        full_cov11 = full_null_space@full_cov@(full_null_space.T)
        full_cov12 = full_null_space@full_cov@(full_fltr_mat.T)
        full_cov22 = full_fltr_mat@full_cov@(full_fltr_mat.T)
        
        xm_known  = full_fltr_mat@np.ravel(xm)
        xm_null  = full_null_space@np.ravel(xm)
        
        obs_flattened = pyross.utils.process_obs(obs, Nf-1)
        dev=np.subtract(obs_flattened, xm_known)
        tmp = full_cov12@np.linalg.inv(full_cov22)
        cov_red = np.subtract(full_cov11, tmp@(full_cov12.T))
        xm_red = xm_null + tmp@dev
        return xm_red, cov_red/self.Omega, full_null_space, known_spaces.flatten()

    def _split_spaces(self, fltr, obs):
        m, n = fltr.shape
        r, q = rq(fltr)
        null_space = q[:n-m]
        known_space = (q[n-m:]).T  @ solve_triangular(r[:, n-m:], obs)
        return null_space, known_space


    cdef double _obtain_logp_for_traj_tangent(self, double [:, :] x, double Tf):
        cdef:
            double [:, :] dx, cov
            double [:] xt, time_points, dx_det
            double dt, logp, t
            Py_ssize_t i, Nf=x.shape[0]
        time_points = np.linspace(0, Tf, Nf)
        dt = time_points[2]
        dx = np.gradient(x, axis=0)*2
        logp = 0
        for i in range(1, Nf-1):
            xt = x[i]
            t = time_points[i]
            self.det_model.set_contactMatrix(t, self.contactMatrix)
            self.det_model.rhs(np.multiply(xt, self.Omega), t)
            dx_det = np.multiply(dt/self.Omega, self.det_model.dxdt)
            self.compute_jacobian_and_b_matrix(xt, t)
            cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
            dev = np.subtract(dx[i], dx_det)
            logp += self._log_cond_p(dev, cov)
        return -logp

    cdef double _log_cond_p(self, double [:] x, double [:, :] cov):
        cdef:
            double [:] invcov_x
            double log_cond_p
            double det
        invcov_x, ldet = pyross.utils.solve_symmetric_close_to_singular(cov, x)
        log_cond_p = - np.dot(x, invcov_x)*(self.Omega/2) - (self.dim/2)*log(2*PI)
        log_cond_p -= (ldet - self.dim*log(self.Omega))/2
        log_cond_p -= self.dim*np.log(self.Omega)
        return log_cond_p

    cdef _estimate_cond_cov(self, object sol, double t1, double t2):
        cdef:
            double [:] cov_vec, sigma0=np.zeros((self.vec_size), dtype=DTYPE)
            double [:, :] cov

        def rhs(t, sig):
            x = sol(t)/self.Omega # sol is an ODESolver obj for extensive variables
            self.compute_jacobian_and_b_matrix(x, t, b_matrix=True, jacobian=True)
            self._compute_dsigdt(sig)
            self.integrator_step_count += 1
            if self.max_steps_lyapunov != 0 and self.integrator_step_count > self.max_steps_lyapunov:
                raise MaxIntegratorStepsException()
            return self.dsigmadt

        cov_vec = self._solve_lyapunov_type_eq(rhs, sigma0, t1, t2, self.steps)
        cov = self.convert_vec_to_mat(cov_vec)
        return cov

    cpdef obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf, Py_ssize_t inter_steps=0):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:, :] _xm=np.empty((inter_steps*Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf
            double [:, :] cond_cov, cov, temp
            double [:, :, :, :] full_cov
            double ti, tf
        if inter_steps:
            x0 = np.multiply(x0, self.Omega)
            _xm = pyross.utils.forward_euler_integration(self._rhs0, x0,
                                                                0, Tf,
                                                                Nf, inter_steps)
            _xm = np.divide(_xm, self.Omega)
            self._xm = np.copy(_xm)
            self._interp = []
            times = np.linspace(0, Nf, inter_steps*Nf)
            for i in range(dim):
                self._interp.append(interpolate.interp1d(times, _xm[:,i],
                                                          kind='linear'))
            xm = _xm[::inter_steps]
            sol = self.interpolate_euler
        else:
            xm, sol = self.integrate(x0, 0, Tf, Nf, dense_output=True,
                                           max_step=self.steps*Nf)
        cov = np.zeros((dim, dim), dtype=DTYPE)
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        self.integrator_step_count = 0
        for i in range(Nf-1):
            ti = time_points[i]
            tf = time_points[i+1]
            cond_cov = self._estimate_cond_cov(sol, ti, tf)
            self._obtain_time_evol_op(sol, ti, tf)
            cov = np.add(self.U@cov@self.U.T, cond_cov)
            full_cov[i, :, i, :] = cov
            if i>0:
                for j in range(0, i):
                    temp = full_cov[j, :, i-1, :]@self.U.T
                    full_cov[j, :, i, :] = temp
                    full_cov[i, :, j, :] = temp.T
        # returns mean and cov for all but first (fixed!) time point
        return xm[1:], np.reshape(full_cov, ((Nf-1)*dim, (Nf-1)*dim))

    cpdef interpolate_euler(self, t):
        ip = []
        for i in range(self.dim):
            ip.append(self._interp[i](t))
        return np.asarray(ip).T

    cpdef obtain_full_mean_cov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf, Py_ssize_t inter_steps=0):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xt
            double [:, :] cov, cond_cov, U, J_dt, temp
            double [:, :, :, :] full_cov
            double t, dt=time_points[1]
        if inter_steps:
            x0 = np.multiply(x0, self.Omega)
            xm = pyross.utils.forward_euler_integration(self._rhs0, x0, 0, Tf,
                                                        Nf, inter_steps)
            xm = xm[::inter_steps]
            xm = np.divide(xm, self.Omega)
        else:
            xm = self.integrate(x0, 0, Tf, Nf, max_step=self.steps*Nf)
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        cov = np.zeros((dim, dim), dtype=DTYPE)
        for i in range(Nf-1):
            t = time_points[i]
            xt = xm[i]
            self.compute_jacobian_and_b_matrix(xt, t, b_matrix=True, jacobian=True)
            cond_cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
            J_dt = np.multiply(dt, self.J_mat)
            U = np.add(np.identity(dim), J_dt)
            cov = np.add(np.dot(np.dot(U, cov), U.T), cond_cov)
            full_cov[i, :, i, :] = cov
            if i>0:
                for j in range(0, i):
                    temp = np.dot(full_cov[j, :, i-1, :], U.T)
                    full_cov[j, :, i, :] = temp
                    full_cov[i, :, j, :] = temp.T
        return xm[1:], np.reshape(full_cov, ((Nf-1)*dim, (Nf-1)*dim)) # returns mean and cov for all but first (fixed!) time point

    cpdef _rhs0(self, t, xt):
        self.det_model.set_contactMatrix(t, self.contactMatrix)
        self.det_model.rhs(xt, t)
        return self.det_model.dxdt

    cpdef _obtain_time_evol_op_2(self, sol, double t1, double t2):
        cdef:
            double [:, :] U=self.U
            double [:] xi, xf
            double epsilon=1./self.Omega
            Py_ssize_t i, j, steps=self.steps
        if isclose(t1, t2):
            U = np.eye(self.dim)
        else:
            xi = sol(t1)/self.Omega
            xf = sol(t2)/self.Omega
            for i in range(self.dim):
                xi[i] += epsilon
                pos = self.integrate(xi, t1, t2, steps)[steps-1]
                for j in range(self.dim):
                    U[j, i] = (pos[j]-xf[j])/(epsilon)
                xi[i] -= epsilon

    def _obtain_time_evol_op(self, sol, double t1, double t2):
        cdef:
            Py_ssize_t steps=self.steps

        def rhs(t, U_vec):
            xt = sol(t)/self.Omega
            self.compute_jacobian_and_b_matrix(xt, t, b_matrix=False, jacobian=True)
            U_mat = np.reshape(U_vec, (self.dim, self.dim))
            dUdt = np.dot(self.J_mat, U_mat)
            self.integrator_step_count += 1
            if self.max_steps_lyapunov != 0 and self.integrator_step_count > self.max_steps_lyapunov:
                raise MaxIntegratorStepsException()
            return np.ravel(dUdt)

        if isclose(t1, t2): ## float precision
            self.U = np.eye(self.dim)
        else:
            U0 = np.identity((self.dim)).flatten()
            U_vec = self._solve_lyapunov_type_eq(rhs, U0, t1, t2, steps)
            self.U = np.reshape(U_vec, (self.dim, self.dim))

    def _solve_lyapunov_type_eq(self, rhs, M0, t1, t2, steps):
        if self.lyapunov_method=='euler':
            sol_vec = pyross.utils.forward_euler_integration(rhs, M0, t1, t2, steps)[steps-1]
        elif self.lyapunov_method=='RK45':
            res = solve_ivp(rhs, (t1, t2), M0, method='RK45', t_eval=np.array([t2]), first_step=(t2-t1)/steps, max_step=(t2-t1)/steps, rtol=self.rtol_lyapunov)
            sol_vec = res.y[:, 0]
        elif self.lyapunov_method=='LSODA':
            res = solve_ivp(rhs, (t1, t2), M0, method='LSODA', t_eval=np.array([t2]), first_step=(t2-t1)/steps, max_step=(t2-t1)/steps, rtol=self.rtol_lyapunov)
            sol_vec = res.y[:, 0]
        elif self.lyapunov_method=='RK2':
            sol_vec = pyross.utils.RK2_integration(rhs, M0, t1, t2, steps)[steps-1]
        elif self.lyapunov_method=='RK4':
            sol_vec = pyross.utils.RK4_integration(rhs, M0, t1, t2, steps)[steps-1]
        else:
            raise Exception("Error: lyapunov method not found. Use set_lyapunov_method to change the method")
        return sol_vec

    cdef _compute_dsigdt(self, double [:] sig):
        cdef:
            Py_ssize_t i, j
            double [:] dsigdt=self.dsigmadt, B_vec=self.B_vec, linear_term_vec
            double [:, :] sigma_mat
            np.ndarray[DTYPE_t, ndim=2] linear_term
        sigma_mat = self.convert_vec_to_mat(sig)
        linear_term = np.dot(self.J_mat, sigma_mat) + np.dot(sigma_mat, (self.J_mat).T)
        linear_term_vec = linear_term[(self.rows, self.cols)]
        for i in range(self.vec_size):
            dsigdt[i] = B_vec[i] + linear_term_vec[i]

    cpdef convert_vec_to_mat(self, double [:] cov):
        cdef:
            double [:, :] cov_mat
            Py_ssize_t i, j, count=0, dim=self.dim
        cov_mat = np.empty((dim, dim), dtype=DTYPE)
        for i in range(dim):
            cov_mat[i, i] = cov[count]
            count += 1
            for j in range(i+1, dim):
                cov_mat[i, j] = cov[count]
                cov_mat[j, i] = cov[count]
                count += 1
        return cov_mat

    def _process_contact_matrix(self, contactMatrix, generator, intervention_fun):
        if (contactMatrix is None) == (generator is None):
            raise Exception('Specify either a fixed contactMatrix or a generator')
        if (intervention_fun is not None) and (generator is None):
            raise Exception('Specify a generator')
        if contactMatrix is not None:
            self.contactMatrix = contactMatrix


    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                             b_matrix=True, jacobian=False):
        raise NotImplementedError("Please Implement compute_jacobian_and_b_matrix in subclass")

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps,
                  dense_output=False, max_step=100000):
        """An light weight integrate method similar to `simulate` in pyross.deterministic

        Parameters
        ----------
        x0: np.array
            Initial state of the given model
        t1: float
            Initial time of integrator
        t2: float
            Final time of integrator
        steps: int
            Number of time steps for numerical integrator evaluation.
        max_step: int, optional
            The maximum allowed step size of the integrator.

        Returns
        -------
        sol: np.array
            The state of the system evaulated at the time point specified. Only used if det_method is set to 'solve_ivp'.
        """

        def rhs0(double t, double [:] xt):
            self.det_model.set_contactMatrix(t, self.contactMatrix)
            self.det_model.rhs(xt, t)
            self.integrator_step_count += 1
            if self.max_steps_det != 0 and self.integrator_step_count > self.max_steps_det:
                raise MaxIntegratorStepsException()
            return self.det_model.dxdt

        x0 = np.multiply(x0, self.Omega)
        time_points = np.linspace(t1, t2, steps)
        self.integrator_step_count = 0
        res = solve_ivp(rhs0, [t1,t2], x0, method=self.det_method,
                        t_eval=time_points, dense_output=dense_output,
                        max_step=max_step, rtol=self.rtol_det)
        y = np.divide(res.y.T, self.Omega)

        if dense_output:
            return y, res.sol
        else:
            return y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(SIR_type):
    """
    Susceptible, Infected, Removed (SIR)

    * Ia: asymptomatic
    * Is: symptomatic

    To initialise the SIR class,

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float
            Ratio of asymptomatic carriers
        beta: float
            Infection rate upon contact
        gIa: float
            Recovery rate for asymptomatic
        gIs: float
            Recovery rate for symptomatic
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
    M: int
        Number of age groups
    fi: float numpy.array
        Number of people in each age group divided by Omega.
    Omega: float, optional
        System size parameter, e.g. total population. Default to 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-4)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0). 
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    
    """
    cdef readonly pyross.deterministic.SIR det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-4, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0):
        self.param_keys = ['alpha', 'beta', 'gIa', 'gIs', 'fsa']
        super().__init__(parameters, 3, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov)
        self.class_index_dict = {'S':0, 'Ia':1, 'Is':2}
        self.set_det_model(parameters)

    def set_det_model(self, parameters):
        self.det_model = pyross.deterministic.SIR(parameters, self.M, self.fi*self.Omega)

    def infection_indices(self):
        return [1, 2]

    def make_params_dict(self):
        parameters = {'alpha':self.alpha, 'beta':self.beta, 'gIa':self.gIa, 'gIs':self.gIs, 'fsa':self.fsa}
        return parameters

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                                b_matrix=True, jacobian=False):
        cdef:
            double [:] s, Ia, Is
            Py_ssize_t M=self.M
        s = x[0:M]
        Ia = x[M:2*M]
        Is = x[2*M:3*M]
        self.CM = self.contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        if b_matrix:
            self.noise_correlation(s, Ia, Is, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double [:] fsa=self.fsa, beta=self.beta
            double [:] fi=self.fi
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta[m]*CM[m,n]*(Ia[n]+fsa[n]*Is[n])/fi[n]

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double [:] gIa=self.gIa, gIs=self.gIs, fsa=self.fsa, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = alpha[m]*l[m]
            J[2, m, 0, m] = balpha[m]*l[m]
            for n in range(M):
                J[0, m, 1, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 2, n] = -s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
                J[1, m, 1, n] = alpha[m]*s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 2, n] = alpha[m]*s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
                J[2, m, 1, n] = balpha[m]*s[m]*beta[m]*CM[m, n]/fi[n]
                J[2, m, 2, n] = balpha[m]*s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
            J[1, m, 1, m] -= gIa[m]
            J[2, m, 2, m] -= gIs[m]
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double [:] gIa=self.gIa, gIs=self.gIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - alpha[m]*l[m]*s[m]
            B[1, m, 1, m] = alpha[m]*l[m]*s[m] + gIa[m]*Ia[m]
            B[0, m, 2, m] = - balpha[m]*l[m]*s[m]
            B[2, m, 2, m] = balpha[m]*l[m]*s[m] + gIs[m]*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEIR(SIR_type):
    """
    Susceptible, Exposed, Infected, Removed (SEIR)

    * Ia: asymptomatic
    * Is: symptomatic

    To initialise the SEIR class,

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float or np.array(M)
            Fraction of infected who are asymptomatic.
        beta: float
            Rate of spread of infection.
        gIa: float
            Rate of removal from asymptomatic individuals.
        gIs: float
            Rate of removal from symptomatic individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        gE: float
            rate of removal from exposed individuals.
    M: int
        Number of age groups
    fi: float numpy.array
        Number of people in each compartment divided by Omega
    Omega: float, optional
        System size, e.g. total population. Default is 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    """

    cdef:
        readonly np.ndarray gE
        readonly pyross.deterministic.SEIR det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0):
        self.param_keys = ['alpha', 'beta', 'gE', 'gIa', 'gIs', 'fsa']
        super().__init__(parameters, 4, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov)
        self.class_index_dict = {'S':0, 'E':1, 'Ia':2, 'Is':3}
        self.set_det_model(parameters)

    def infection_indices(self):
        return [1, 2, 3]

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], self.M, 'gE')

    def set_det_model(self, parameters):
        self.det_model = pyross.deterministic.SEIR(parameters, self.M, self.fi*self.Omega)

    def make_params_dict(self):
        parameters = {'alpha':self.alpha, 'beta':self.beta, 'gIa':self.gIa,
                            'gIs':self.gIs, 'gE':self.gE, 'fsa':self.fsa}
        return parameters

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                            b_matrix=True, jacobian=False):
        cdef:
            double [:] s, e, Ia, Is
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        Ia = x[2*M:3*M]
        Is = x[3*M:4*M]
        self.CM = self.contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        if b_matrix:
            self.noise_correlation(s, e, Ia, Is, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double [:] fsa=self.fsa, beta=self.beta, fi=self.fi
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta[m]*CM[m,n]*(Ia[n]+fsa[n]*Is[n])/fi[n]

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double [:] gIa=self.gIa, gIs=self.gIs, gE=self.gE, fsa=self.fsa, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE[m]
            J[2, m, 1, m] = alpha[m]*gE[m]
            J[2, m, 2, m] = - gIa[m]
            J[3, m, 1, m] = balpha[m]*gE[m]
            J[3, m, 3, m] = - gIs[m]
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 3, n] = -s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
                J[1, m, 2, n] = s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 3, n] = s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double [:] gIa=self.gIa, gIs=self.gIs, gE=self.gE
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + gE[m]*e[m]
            B[1, m, 2, m] = -alpha[m]*gE[m]*e[m]
            B[1, m, 3, m] = -balpha[m]*gE[m]*e[m]
            B[2, m, 2, m] = alpha[m]*gE[m]*e[m]+gIa[m]*Ia[m]
            B[3, m, 3, m] = balpha[m]*gE[m]*e[m]+gIs[m]*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ(SIR_type):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed
    * A: asymptomatic and infectious
    * Q: quarantined

    To initialise the SEAIRQ class,

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float or np.array(M)
            Fraction of infected who are asymptomatic.
        beta: float
            Rate of spread of infection.
        gIa: float
            Rate of removal from asymptomatic individuals.
        gIs: float
            Rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        tE: float
            testing rate and contact tracing of exposeds
        tA: float
            testing rate and contact tracing of activateds
        tIa: float
            testing rate and contact tracing of asymptomatics
        tIs: float
            testing rate and contact tracing of symptomatics
    M: int
        Number of compartments
    fi: float numpy.array
        Number of people in each compartment divided by Omega.
    Omega: float, optional
        System size, e.g. total population. Default is 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    """

    cdef:
        readonly np.ndarray gE, gA, tE, tA, tIa, tIs
        readonly pyross.deterministic.SEAIRQ det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0):
        self.param_keys = ['alpha', 'beta', 'gE', 'gA', \
                           'gIa', 'gIs', 'fsa', \
                           'tE', 'tA', 'tIa', 'tIs']
        super().__init__(parameters, 6, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov)
        self.class_index_dict = {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}
        self.set_det_model(parameters)

    def infection_indices(self):
        return [1, 2, 3, 4]


    def set_det_model(self, parameters):
        self.det_model = pyross.deterministic.SEAIRQ(parameters, self.M, self.fi*self.Omega)

    def make_params_dict(self):
        parameters = {'alpha':self.alpha,
                      'beta':self.beta,
                      'gIa':self.gIa,
                      'gIs':self.gIs,
                      'gE':self.gE,
                      'gA':self.gA,
                      'fsa': self.fsa,
                      'tS': 0,
                      'tE': self.tE,
                      'tA': self.tA,
                      'tIa': self.tIa,
                      'tIs': self.tIs
                      }
        return parameters

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                        b_matrix=True, jacobian=False):
        cdef:
            double [:] s, e, a, Ia, Is, Q
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        self.CM = self.contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
        if b_matrix:
            self.noise_correlation(s, e, a, Ia, Is, q, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] a, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double [:] fsa=self.fsa, beta=self.beta, fi=self.fi
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta[m]*CM[m,n]*(Ia[n]+a[n]+fsa[n]*Is[n])/fi[n]

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double [:] gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double [:] tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE[m] - tE[m]
            J[2, m, 1, m] = gE[m]
            J[2, m, 2, m] = - gA[m] - tA[m]
            J[3, m, 2, m] = alpha[m]*gA[m]
            J[3, m, 3, m] = - gIa[m] - tIa[m]
            J[4, m, 2, m] = balpha[m]*gA[m]
            J[4, m, 4, m] = -gIs[m] - tIs[m]
            J[5, m, 1, m] = tE[m]
            J[5, m, 2, m] = tA[m]
            J[5, m, 3, m] = tIa[m]
            J[5, m, 4, m] = tIs[m]
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 3, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 4, n] = -s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
                J[1, m, 2, n] = s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 3, n] = s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 4, n] = s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double [:] beta=self.beta, gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA
            double [:] tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + (gE[m]+tE[m])*e[m]
            B[1, m, 2, m] = -gE[m]*e[m]
            B[2, m, 2, m] = gE[m]*e[m]+(gA[m]+tA[m])*a[m]
            B[2, m, 3, m] = -alpha[m]*gA[m]*a[m]
            B[2, m, 4, m] = -balpha[m]*gA[m]*a[m]
            B[3, m, 3, m] = alpha[m]*gA[m]*a[m]+(gIa[m]+tIa[m])*Ia[m]
            B[4, m, 4, m] = balpha[m]*gA[m]*a[m] + (gIs[m]+tIs[m])*Is[m]
            B[1, m, 5, m] = -tE[m]*e[m]
            B[2, m, 5, m] = -tA[m]*a[m]
            B[3, m, 5, m] = -tIa[m]*Ia[m]
            B[4, m, 5, m] = -tIs[m]*Is[m]
            B[5, m, 5, m] = tE[m]*e[m]+tA[m]*a[m]+tIa[m]*Ia[m]+tIs[m]*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAIRQ_testing(SIR_type):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious

    * Ia: asymptomatic
    * Is: symptomatic
    * E: exposed
    * A: asymptomatic and infectious
    * Q: quarantined

    To initialise the SEAIRQ class,

    Parameters
    ----------
    parameters: dict
        Contains the following keys:

        alpha: float or np.array(M)
            Fraction of infected who are asymptomatic.
        beta: float
            Rate of spread of infection.
        gIa: float
            Rate of removal from asymptomatic individuals.
        gIs: float
            Rate of removal from symptomatic individuals.
        gE: float
            rate of removal from exposed individuals.
        gA: float
            rate of removal from activated individuals.
        fsa: float
            Fraction by which symptomatic individuals do not self-isolate.
        ars : float
            fraction of population admissible for random and symptomatic tests
        kapE : float
            fraction of positive tests for exposed individuals
    testRate: python function
        number of tests per day and age group
    M: int
        Number of compartments
    fi: float numpy.array
        Number of people in each age group divided by Omega.
    Omega: float, optional
        System size, e.g. total population. Default is 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    """

    cdef:
        readonly np.ndarray gE, gA, ars, kapE
        readonly object testRate
        readonly pyross.deterministic.SEAIRQ_testing det_model

    def __init__(self, parameters, testRate, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0):
        self.param_keys = ['alpha', 'beta', 'gE', 'gA', \
                           'gIa', 'gIs', 'fsa', \
                           'ars', 'kapE']
        super().__init__(parameters, 6, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov)
        self.testRate=testRate
        self.class_index_dict = {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}
        self.make_det_model(parameters)

    def infection_indices(self):
        return (1, 2, 3, 4)

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], self.M, 'gE')
        self.gA = pyross.utils.age_dep_rates(parameters['gA'], self.M, 'gA')
        self.ars = pyross.utils.age_dep_rates(parameters['ars'], self.M, 'ars')
        self.kapE = pyross.utils.age_dep_rates(parameters['kapE'], self.M, 'kapE')


    def set_testRate(self, testRate):
        self.testRate=testRate

    def make_det_model(self, parameters):
        self.det_model = pyross.deterministic.SEAIRQ_testing(parameters, self.M, self.fi*self.Omega)
        self.det_model.set_testRate(self.testRate)

    def make_params_dict(self):
        parameters = {'alpha':self.alpha,
                      'beta':self.beta,
                      'gIa':self.gIa,
                      'gIs':self.gIs,
                      'gE':self.gE,
                      'gA':self.gA,
                      'fsa': self.fsa,
                      'ars': self.ars,
                      'kapE': self.kapE
                      }
        return parameters

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                            b_matrix=True, jacobian=False):
        cdef:
            double [:] s, e, a, Ia, Is, Q, TR
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        self.CM = self.contactMatrix(t)
        TR=self.testRate(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
        if b_matrix:
            self.noise_correlation(s, e, a, Ia, Is, q, l, TR)
        if jacobian:
            self.jacobian(s, e, a, Ia, Is, q, l, TR)

    cdef fill_lambdas(self, double [:] a, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double [:] fsa=self.fsa, beta=self.beta, fi=self.fi
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta[m]*CM[m,n]*(Ia[n]+a[n]+fsa[n]*Is[n])/fi[n]

    cdef jacobian(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l, double [:] TR):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double Omega = self.Omega
            double [:] gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double [:] ars=self.ars, kapE=self.kapE, beta=self.beta
            double t0, tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            t0 = 1./(ars[m]*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars[m]*kapE[m]*t0/Omega
            tA= TR[m]*ars[m]*t0/Omega
            tIa = TR[m]*ars[m]*t0/Omega
            tIs = TR[m]*t0/Omega

            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 3, n] = -s[m]*beta[m]*CM[m, n]/fi[n]
                J[0, m, 4, n] = -s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
                J[1, m, 2, n] = s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 3, n] = s[m]*beta[m]*CM[m, n]/fi[n]
                J[1, m, 4, n] = s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE[m] - tE
            J[1, m, 4, m] += (1-ars[m])*tE*t0*e[m]
            J[1, m, 5, m] = -ars[m]*tE*t0*e[m]
            J[2, m, 1, m] = gE[m]
            J[2, m, 2, m] = - gA[m] - tA
            J[2, m, 4, m] = (1-ars[m])*tA*t0*a[m]
            J[2, m, 5, m] = - ars[m]*tA*t0*a[m]
            J[3, m, 2, m] = alpha[m]*gA[m]
            J[3, m, 3, m] = - gIa[m] - tIa
            J[3, m, 4, m] = (1-ars[m])*tIa*t0*Ia[m]
            J[3, m, 5, m] = - ars[m]*tIa*t0*Ia[m]
            J[4, m, 2, m] = balpha[m]*gA[m]
            J[4, m, 4, m] = - gIs[m] - tIs + (1-ars[m])*tIs*t0*Is[m]
            J[4, m, 5, m] = - ars[m]*tIs*t0*Is[m]
            J[5, m, 1, m] = tE
            J[5, m, 2, m] = tA
            J[5, m, 3, m] = tIa
            J[5, m, 4, m] = tIs - (1-ars[m])*t0*(tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m])
            J[5, m, 5, m] = ars[m]*t0*(tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m])
        self.J_mat = self.J.reshape((dim, dim))


    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l, double [:] TR):
        cdef:
            Py_ssize_t m, M=self.M
            double Omega=self.Omega
            double [:] beta=self.beta, gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA
            double [:] ars=self.ars, kapE=self.kapE
            double tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            t0 = 1./(ars[m]*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars[m]*kapE[m]*t0/Omega
            tA= TR[m]*ars[m]*t0/Omega
            tIa = TR[m]*ars[m]*t0/Omega
            tIs = TR[m]*t0/Omega

            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + (gE[m]+tE)*e[m]
            B[1, m, 2, m] = -gE[m]*e[m]
            B[2, m, 2, m] = gE[m]*e[m]+(gA[m]+tA)*a[m]
            B[2, m, 3, m] = -alpha[m]*gA[m]*a[m]
            B[2, m, 4, m] = -balpha[m]*gA[m]*a[m]
            B[3, m, 3, m] = alpha[m]*gA[m]*a[m]+(gIa[m]+tIa)*Ia[m]
            B[4, m, 4, m] = balpha[m]*gA[m]*a[m] + (gIs[m]+tIs)*Is[m]
            B[1, m, 5, m] = -tE*e[m]
            B[2, m, 5, m] = -tA*a[m]
            B[3, m, 5, m] = -tIa*Ia[m]
            B[4, m, 5, m] = -tIs*Is[m]
            B[5, m, 5, m] = tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Model(SIR_type):
    """
    Generic user-defined epidemic model.

    To initialise the Model,

    Parameters
    ----------
    model_spec: dict
        A dictionary specifying the model. See `Examples`.
    parameters: dict
        A dictionary containing the model parameters.
        All parameters can be float if not age-dependent, and np.array(M,) if age-dependent
    M: int
        Number of age groups.
    fi: np.array(M) or list
        Fraction of each age group.
    Omega: int
        Total population.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    parameter_mapping: python function, optional
        A user-defined function that maps the dictionary the parameters used for inference to a dictionary of parameters used in model_spec. Default is an identical mapping.
    time_dep_param_mapping: python function, optional
        As parameter_mapping, but time-dependent. The user-defined function takes time as a second argument.

    See `SIR_type` for a table of all the methods

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
        readonly np.ndarray constant_terms, linear_terms, infection_terms, finres_terms, resource_list
        readonly np.ndarray model_parameters
        readonly np.ndarray model_parameters_length
        readonly np.ndarray finres_pop
        readonly pyross.deterministic.Model det_model
        readonly dict model_spec
        readonly dict param_dict
        readonly list model_param_keys
        readonly object parameter_mapping
        readonly object time_dep_param_mapping



    def __init__(self, model_spec, parameters, M, fi, Omega=1, steps=4,
                                    det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0,
                                    parameter_mapping=None, time_dep_param_mapping=None):
        if parameter_mapping is not None and time_dep_param_mapping is not None:
            raise Exception('Specify either parameter_mapping or time_dep_param_mapping')
        self.parameter_mapping = parameter_mapping
        self.time_dep_param_mapping = time_dep_param_mapping
        self.param_keys = list(parameters.keys())
        if parameter_mapping is not None:
            self.model_param_keys = list(parameter_mapping(parameters).keys())
        elif time_dep_param_mapping is not None:
            self.param_dict = parameters.copy()
            self.model_param_keys = list(time_dep_param_mapping(parameters, 0).keys())
        else:
            self.model_param_keys = self.param_keys.copy()
        self.model_spec=model_spec
        res = pyross.utils.parse_model_spec(model_spec, self.model_param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.finres_terms = res[5]
        self.resource_list = res[6]
        super().__init__(parameters, self.nClass, M, fi, Omega, steps, det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov)
        if self.parameter_mapping is not None:
            parameters = self.parameter_mapping(parameters)
            self.param_mapping_enabled = True
        if self.time_dep_param_mapping is not None:
            self.det_model = pyross.deterministic.Model(model_spec, parameters, M, fi*Omega, time_dep_param_mapping=time_dep_param_mapping)
            self.param_mapping_enabled = True
        else:
            self.det_model = pyross.deterministic.Model(model_spec, parameters, M, fi*Omega)
        
        self.finres_pop = np.empty( len(self.resource_list), dtype='object')  # populations for finite-resource transitions
        for i in range(len(self.resource_list)):
            ndx = self.resource_list[i][0]
            if self.model_parameters_length[ndx] == 1:
                self.finres_pop[i] = 0
            else:
                self.finres_pop[i] = np.zeros(self.M, dtype=DTYPE)

    def infection_indices(self):
        cdef Py_ssize_t a = 100
        indices = set()
        linear_terms_indices = list(range(self.linear_terms.shape[0]))

        # Find all the infection terms
        for term in self.infection_terms:
            infective_index = term[1]
            indices.add(infective_index)

        # Find all the terms that turn into infection terms
        a = 100
        while a > 0:
            a = 0
            temp = linear_terms_indices.copy()
            for i in linear_terms_indices:
                product_index = self.linear_terms[i, 2]
                if product_index in indices:
                    a += 1
                    indices.add(self.linear_terms[i, 1])
                    temp.remove(i)
            linear_terms_indices = temp
        return list(indices)

    def set_params(self, parameters):
        if self.det_model is not None:
            self.set_det_model(parameters)
        nParams = len(self.param_keys)
        self.param_dict = parameters.copy()
        if self.parameter_mapping is not None:
            model_parameters = self.parameter_mapping(parameters)
            nParams = len(self.model_param_keys)
            self.model_parameters = np.empty((nParams, self.M), dtype=DTYPE)
            self.model_parameters_length = np.empty(nParams, dtype=int)
            try:
                for (i, key) in enumerate(self.model_param_keys):
                    param = model_parameters[key]
                    self.model_parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                    self.model_parameters_length[i] = np.size(param)
            except KeyError:
                raise Exception('The parameters returned by parameter_mapping(...) do not contain certain keys. The keys are {}'.format(self.model_param_keys))
        elif self.time_dep_param_mapping is not None:
            self.set_time_dep_model_parameters(0)
        else:
            self.model_parameters = np.empty((nParams, self.M), dtype=DTYPE)
            self.model_parameters_length = np.empty(nParams, dtype=int)
            try:
                for (i, key) in enumerate(self.param_keys):
                    param = parameters[key]
                    self.model_parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                    self.model_parameters_length[i] = np.size(param)
            except KeyError:
                raise Exception('The parameters passed do not contain certain keys. The keys are {}'.format(self.param_keys))

    def set_time_dep_model_parameters(self, tt):
        model_parameters = self.time_dep_param_mapping(self.param_dict, tt)
        nParams = len(self.model_param_keys)
        self.model_parameters = np.empty((nParams, self.M), dtype=DTYPE)
        self.model_parameters_length = np.empty(nParams, dtype=int)
        
        try:
            for (i, key) in enumerate(self.model_param_keys):
                param = model_parameters[key]
                self.model_parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
                self.model_parameters_length[i] = np.size(param)
        except KeyError:
            raise Exception('The parameters passed do not contain certain keys.\
                             The keys are {}'.format(self.param_keys))

    def set_det_model(self, parameters):
        if self.parameter_mapping is not None:
            self.det_model.update_model_parameters(self.parameter_mapping(parameters))
        else:
            self.det_model.update_model_parameters(parameters)


    def make_params_dict(self):
        param_dict = self.param_dict.copy()
        return param_dict

    cdef np.ndarray _get_r_from_x(self, np.ndarray x):
        cdef:
            np.ndarray r
            np.ndarray xrs=x.reshape(int(self.dim/self.M), self.M)
        if 'R' in self.class_index_dict.keys():
            r = xrs[self.class_index_dict['R'],:]
        elif self.constant_terms.size > 0:
            r = xrs[-1,:] - np.sum(xrs[:-1,:], axis=0)
        else:
            r = self.fi - np.sum(xrs, axis=0)
        return r

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                            b_matrix=True, jacobian=False):
        cdef:
            Py_ssize_t nClass=self.nClass, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fi=self.fi
        self.CM = self.contactMatrix(t)
        if self.time_dep_param_mapping is not None:
            self.set_time_dep_model_parameters(t)
        if self.constant_terms.size > 0:
            fi = x[(nClass-1)*M:]
        self.fill_lambdas(x, l)
        self.fill_finres_pop(x)
        if b_matrix:
            self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
            self.noise_correlation(x, l)
        if jacobian:
            self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
            self.jacobian(x, l)

    cdef fill_lambdas(self, double [:] x, double [:, :] l):
        cdef:
            double [:, :] CM=self.CM
            int [:, :] infection_terms=self.infection_terms
            double infection_rate
            double [:] fi=self.fi
            Py_ssize_t m, n, i, infective_index, index, M=self.M, num_of_infection_terms=infection_terms.shape[0]
        for i in range(num_of_infection_terms):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                for n in range(M):
                    index = n + M*infective_index
                    if fi[n]>0:
                        l[i, m] += CM[m,n]*x[index]/fi[n]

    cdef fill_finres_pop(self, double [:] x):
        # Calculate populations for finite resource transitions
        cdef:
            Py_ssize_t class_index, priority_index, m, i
        for i in range(len(self.resource_list)):
            ndx = self.resource_list[i][0]
            n_cohorts = self.model_parameters_length[ndx]
            if n_cohorts == 1:
                self.finres_pop[i] = 0
            else:
                self.finres_pop[i] = np.zeros(n_cohorts)
            for (class_index, priority_index) in self.resource_list[i][1:]:
                for m in range(self.M):
                    if n_cohorts == 1:
                        self.finres_pop[i] += x[m + self.M*class_index] * self.model_parameters[priority_index, m]
                    else:
                        self.finres_pop[i][m] += x[m + self.M*class_index] * self.model_parameters[priority_index, m]

    cdef jacobian(self, double [:] x, double [:, :] l):
        cdef:
            Py_ssize_t i, m, n, M=self.M, dim=self.dim
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, susceptible_index
            Py_ssize_t resource_index, priority_index, probability_index, class_index, res_class_index, res_priority_index
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
            double [:, :] parameters=self.model_parameters
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            int [:, :] finres_terms=self.finres_terms
            np.ndarray resource_list=self.resource_list
            np.ndarray finres_pop = self.finres_pop
            double [:] rate
            double term, term2, frp
            double [:] fi=self.fi

        # infection terms
        for i in range(infection_terms.shape[0]):
            rate_index = infection_terms[i, 0]
            infective_index = infection_terms[i, 1]
            susceptible_index = infection_terms[i, 2]
            product_index = infection_terms[i, 3]
                
            rate = parameters[rate_index]
            for m in range(M):
                J[susceptible_index, m, susceptible_index, m] -= rate[m]*l[i, m]
                if product_index>-1:
                    J[product_index, m, susceptible_index, m] += rate[m]*l[i, m]
                for n in range(M):
                    J[susceptible_index, m, infective_index, n] -= x[susceptible_index*M+m]*rate[m]*CM[m, n]/fi[n]
                    if product_index>-1:
                        J[product_index, m, infective_index, n] += x[susceptible_index*M+m]*rate[m]*CM[m, n]/fi[n]

        # linear terms
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                J[reagent_index, m, reagent_index, m] -= rate[m]
                if product_index>-1:
                    J[product_index, m, reagent_index, m] += rate[m]

        # finite-resource terms
        if finres_terms.size > 0:
            for i in range(finres_terms.shape[0]):
                resource_index = finres_terms[i, 0]
                rate_index = resource_list[resource_index][0]
                priority_index = finres_terms[i, 1]
                probability_index = finres_terms[i, 2]
                class_index = finres_terms[i, 3]
                reagent_index = self.finres_terms[i, 4]
                product_index = self.finres_terms[i, 5]
                for m in range(M):
                    if np.size(finres_pop[resource_index]) == 1:
                        frp = finres_pop[resource_index]
                    else:
                        frp = finres_pop[resource_index][m]
                    if frp > 0.5 / self.Omega:
                        term = parameters[rate_index, m] * parameters[priority_index, m] \
                               * parameters[probability_index, m] / (frp * self.Omega)
                    else:
                        term = 0
                    if reagent_index>-1:
                        J[reagent_index, m, class_index, m] -= term
                    if product_index>-1:
                        J[product_index, m, class_index, m] += term
                    if frp > 0:
                        term *= - x[class_index*M+m] / frp
                    for (res_class_index, res_priority_index) in resource_list[resource_index][1:]:
                        if np.size(finres_pop[resource_index]) == 1:
                            for n in range(M):
                                term2 = term * parameters[res_priority_index, n]
                                if reagent_index>-1:
                                    J[reagent_index, m, res_class_index, n] -= term2
                                if product_index>-1:
                                    J[product_index, m, res_class_index, n] += term2
                        else:
                            term2 = term * parameters[res_priority_index, m]
                            if reagent_index>-1:
                                J[reagent_index, m, res_class_index, m] -= term2
                            if product_index>-1:
                                J[product_index, m, res_class_index, m] += term2
                                

        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] x, double [:, :] l):
        cdef:
            Py_ssize_t i, m, n, M=self.M, nClass=self.nClass, class_index
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, overdispersion_index, susceptible_index
            Py_ssize_t resource_index, priority_index, probability_index
            double [:, :, :, :] B=self.B
            double [:, :] CM=self.CM
            double [:, :] parameters=self.model_parameters
            int [:, :] constant_terms=self.constant_terms
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            int [:, :] finres_terms=self.finres_terms
            np.ndarray resource_list=self.resource_list
            np.ndarray finres_pop = self.finres_pop
            double frp
            double [:] s, reagent, rate, overdispersion
            double Omega=self.Omega
        

        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                overdispersion_index = constant_terms[i, 3]
                rate = parameters[rate_index]
                if overdispersion_index == -1:
                    overdispersion = np.ones(M)
                else:
                    overdispersion = parameters[overdispersion_index]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]*overdispersion[m]/Omega
                    B[nClass-1, m, nClass-1, m] += rate[m]*overdispersion[m]/Omega

        for i in range(infection_terms.shape[0]):

                            
            rate_index = infection_terms[i, 0]
            infective_index = infection_terms[i, 1]
            susceptible_index = infection_terms[i, 2]
            product_index = infection_terms[i, 3]
            overdispersion_index = infection_terms[i, 4]
            rate = parameters[rate_index]
            s = x[susceptible_index*M:(susceptible_index+1)*M]
            if overdispersion_index == -1:
                overdispersion = np.ones(M)
            else:
                overdispersion = parameters[overdispersion_index]
            for m in range(M):
                B[susceptible_index, m, susceptible_index, m] += rate[m]*overdispersion[m]*l[i, m]*s[m]
                if product_index>-1:
                    B[susceptible_index, m, product_index, m] -=  rate[m]*overdispersion[m]*l[i, m]*s[m]
                    B[product_index, m, product_index, m] += rate[m]*overdispersion[m]*l[i, m]*s[m]
                    B[product_index, m, susceptible_index, m] -= rate[m]*overdispersion[m]*l[i, m]*s[m]

        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            overdispersion_index = linear_terms[i, 3]
            reagent = x[reagent_index*M:(reagent_index+1)*M]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            if overdispersion_index == -1:
                overdispersion = np.ones(M)
            else:
                overdispersion = parameters[overdispersion_index]
            for m in range(M): # only fill in the upper triangular form
                B[reagent_index, m, reagent_index, m] += rate[m]*overdispersion[m]*reagent[m]
                if product_index>-1:
                    B[product_index, m, product_index, m] += rate[m]*overdispersion[m]*reagent[m]
                    B[reagent_index, m, product_index, m] += -rate[m]*overdispersion[m]*reagent[m]
                    B[product_index, m, reagent_index, m] += -rate[m]*overdispersion[m]*reagent[m]

        if finres_terms.size > 0:
            for i in range(finres_terms.shape[0]):
                resource_index = finres_terms[i, 0]
                rate_index = resource_list[resource_index][0]
                priority_index = finres_terms[i, 1]
                probability_index = finres_terms[i, 2]
                class_index = finres_terms[i, 3]
                reagent_index = finres_terms[i, 4]
                product_index = finres_terms[i, 5]
                overdispersion_index = finres_terms[i, 6]
                if overdispersion_index == -1:
                    overdispersion = np.ones(M)
                else:
                    overdispersion = parameters[overdispersion_index]
                for m in range(M):
                    if np.size(finres_pop[resource_index]) == 1:
                        frp = finres_pop[resource_index]
                    else:
                        frp = finres_pop[resource_index][m]
                    term = parameters[rate_index, m] * parameters[priority_index, m] \
                           * parameters[probability_index, m] * overdispersion[m] * x[class_index*M+m] / (frp * self.Omega)
                    if reagent_index>-1:
                        B[reagent_index, m, reagent_index, m] += term
                        if product_index>-1:
                            B[reagent_index, m, product_index, m] -= term
                            B[product_index, m, reagent_index, m] -= term
                    if product_index>-1:
                        B[product_index, m, product_index, m] += term

        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

        
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
        Number of age groups.
    fi: np.array(M) or list
        Fraction of each age group.
    Omega: int
        Total population.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    parameter_mapping: python function, optional
        A user-defined function that maps the dictionary the parameters used for inference to a dictionary of parameters used in model_spec. Default is an identical mapping.
    time_dep_param_mapping: python function, optional
        As parameter_mapping, but time-dependent. The user-defined function takes time as a second argument.

    See `SIR_type` for a table of all the methods

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

    def __init__(self, model_spec, parameters, M, fi, Omega=1, steps=4,
                                    det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0,
                                    parameter_mapping=None, time_dep_param_mapping=None):
        Xpp_model_spec = pyross.utils.Spp2Xpp(model_spec)
        super().__init__(Xpp_model_spec, parameters, M, fi, Omega=Omega, steps=steps,
                                    det_method=det_method, lyapunov_method=lyapunov_method, rtol_det=rtol_det, rtol_lyapunov=rtol_lyapunov, max_steps_det=max_steps_det, 
                                    max_steps_lyapunov=max_steps_lyapunov, parameter_mapping=parameter_mapping, time_dep_param_mapping=time_dep_param_mapping)
        
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
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
    testRate: python function
        number of tests per day and age group
    M: int
        Number of age groups.
    fi: np.array(M) or list
        Fraction of each age group.
    Omega: int
        Total population.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        For robustness, set steps to be large, lyapunov_method='LSODA'.
        For speed, set steps to be small (~4), lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA' and 'RK45'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2', 'RK4' and 'euler'. Default is 'LSODA'.
    rtol_det: float, optional
        relative tolerance for the deterministic integrator (default 1e-3)
    rtol_lyapunov: float, optional
        relative tolerance for the Lyapunov-type integrator (default 1e-3)
    max_steps_det: int, optional
        Maximum number of integration steps (total) for the deterministic integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_det are disregarded by the optimiser.
    max_steps_lyapunov: int, optional
        Maximum number of integration steps (total) for the Lyapunov-type integrator. Default: unlimited (represented as 0)
        Parameters for which the integrator reaches max_steps_lyapunov are disregarded by the optimiser.
    parameter_mapping: python function, optional
        A user-defined function that maps the dictionary the parameters used for inference to a dictionary of parameters used in model_spec. Default is an identical mapping.
    time_dep_param_mapping: python function, optional
        As parameter_mapping, but time-dependent. The user-defined function takes time as a second argument.

    See `SIR_type` for a table of all the methods

    Examples
    --------
    An example of model_spec and parameters for SIR class with random
    testing (without false positives/negatives) and quarantine

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
        readonly object input_param_mapping
        readonly object testRate
        
    def __init__(self, model_spec, parameters, testRate, M, fi, Omega=1, steps=4,
                                    det_method='LSODA', lyapunov_method='LSODA', rtol_det=1e-3, rtol_lyapunov=1e-3, max_steps_det=0, max_steps_lyapunov=0, parameter_mapping=None, time_dep_param_mapping=None):
        if parameter_mapping is not None and time_dep_param_mapping is not None:
            raise Exception('Specify either parameter_mapping or time_dep_param_mapping')
        self.full_model_spec = pyross.utils.build_SppQ_model_spec(model_spec) 
        self.input_time_dep_param_mapping = time_dep_param_mapping
        self.input_param_mapping = parameter_mapping
        self.testRate = testRate
        super().__init__(self.full_model_spec, parameters, M, fi, Omega, steps,
                                    det_method, lyapunov_method, rtol_det, rtol_lyapunov, max_steps_det, max_steps_lyapunov, parameter_mapping=None, time_dep_param_mapping=self.full_time_dep_param_mapping)
        
    
    cpdef full_time_dep_param_mapping(self, input_parameters, t):
        cdef dict output_param_dict
        if self.input_time_dep_param_mapping is not None:
            output_param_dict = self.input_time_dep_param_mapping(input_parameters, t).copy()
        elif self.input_param_mapping is not None:
            output_param_dict = self.input_param_mapping(input_parameters).copy()   
        else:
            output_param_dict = input_parameters.copy()
        if self.testRate is not None:
            output_param_dict['tau'] = self.testRate(t)
        else:
            output_param_dict['tau'] = 0
        return output_param_dict
    
    def set_testRate(self, testRate):
        self.testRate = testRate
        self.set_det_model(self.param_dict)
        
