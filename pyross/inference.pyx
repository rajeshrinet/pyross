from itertools import compress
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, approx_fprime
from scipy.stats import lognorm
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.misc import derivative
cimport numpy as np
cimport cython
import time

try:
    # Optional support for nested sampling.
    import nestle
except ImportError:
    nestle = None

try: ## optional support for symbolic differentiation
    import sympy
except ImportError:
    sympy=None

import pyross.deterministic
cimport pyross.deterministic
import pyross.contactMatrix
from pyross.utils_python import minimization
from libc.math cimport sqrt, log, INFINITY
cdef double PI = 3.14159265359


DTYPE   = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.uint8_t BOOL_t

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR_type:
    '''Parent class for inference for all SIR-type classes listed below

    All subclasses use the same functions to perform inference, which are documented below.
    '''

    cdef:
        readonly Py_ssize_t nClass, M, steps, dim, vec_size
        readonly double N
        readonly np.ndarray beta, gIa, gIs, fsa
        readonly np.ndarray alpha, fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols
        readonly str det_method, lyapunov_method


    def __init__(self, parameters, nClass, M, fi, N, steps, det_method, lyapunov_method):
        self.N = N
        self.M = M
        self.fi = fi
        if steps < 4:
            raise Exception('Steps must be at least 4 for internal spline interpolation.')
        self.steps = steps
        self.set_params(parameters)
        self.det_method=det_method
        self.lyapunov_method=lyapunov_method

        self.dim = nClass*M
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

    def _infer_parameters_to_minimize(self, params, grad=0, keys=None, is_scale_parameter=None, scaled_guesses=None,
                               flat_guess_range=None, eps=None, x=None, Tf=None, Nf=None,
                               contactMatrix=None, s=None, scale=None, tangent=None):
        """Objective function for minimization call in infer_parameters."""
        # Restore parameters from flattened parameters
        orig_params = self._unflatten_parameters(params, flat_guess_range, is_scale_parameter, scaled_guesses)

        parameters = self.fill_params_dict(keys, orig_params)
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        if tangent:
            minus_logp = self.obtain_log_p_for_traj_tangent_space(x, Tf, Nf, model, contactMatrix)
        else:
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def symbolic_test(self):
        if sympy == None:
            pass
        else:
            x  =  sympy.symbols('x')
            expr = sympy.sin(x)
            diff = sympy.diff(expr, x)
            return diff, float(x.subs(x, 2))

    def infer_parameters(self, keys, guess, stds, bounds, np.ndarray x,
                        double Tf, Py_ssize_t Nf, contactMatrix,
                        tangent=False,
                        infer_scale_parameter=False, verbose=False, full_output=False,
                        ftol=1e-6, eps=1e-5, global_max_iter=100, local_max_iter=100, global_atol=1,
                        enable_global=True, enable_local=True, cma_processes=0, cma_population=16, cma_stds=None):
        '''Compute the maximum a-posteriori (MAP) estimate of the parameters of the SIR type model.
        This function assumes that full data on all classes is available (with latent variables, use SIR_type.latent_inference).

        IN DEVELOPMENT: Parameters that support age-dependent values can be inferred age-dependently by setting the guess
        to a numpy.array of self.M initial values. By default, each age-dependent parameter is inferred independently.
        If the relation of the different parameters is known, a scale factor of the initial guess can be inferred instead
        by setting infer_scale_parameter to True for each age-dependent parameter where this is wanted. Note that
        computing hessians for age-dependent rates is not yet supported. This functionality might be changed in the
        future without warning.

        Parameters
        ----------
        keys: list
            A list of names for parameters to be inferred
        guess: numpy.array or list
            Prior expectation (and initial guess) for the parameter values. Age-dependent
            rates can be inferred by supplying a guess that is an array instead a single float.
        stds: numpy.array
            Standard deviations for the log normal prior of the parameters
        bounds: 2d numpy.array
            Bounds for the parameters (number of parameters x 2)
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        infer_scale_parameter: bool or list of bools (size: number of age-dependenly specified parameters)
            Decide if age-dependent parameters are supposed to be inferred separately (default) or if a scale parameter
            for the guess should be inferred. This can be set either globally for all age-dependent parameters or for each
            age-dependent parameter individually
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: float
            Relative tolerance of logp
        eps: float
            Disallow parameters closer than `eps` to the boundary (to avoid numerical instabilities).
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
        cma_stds: int, optional
            The standard deviation used in cma global optimisation. If not specified, `cma_stds` is set to `stds`.

        Returns
        -------
        estimates: numpy.array
            The MAP parameter estimate
        '''
        # Transfer the guesses, stds, ... which can contain arrays as entries for age-dependend rates to a flat vector for inference.
        flat_guess, flat_stds, flat_bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = self._flatten_parameters(guess, stds, bounds, infer_scale_parameter)
        s, scale = pyross.utils.make_log_norm_dist(flat_guess, flat_stds)

        if cma_stds is None:
            # Use prior standard deviations here
            flat_cma_stds = flat_stds
        else:
            flat_cma_stds = np.zeros(len(flat_guess))
            for i in range(len(guess)):
                flat_cma_stds[flat_guess_range[i]] = cma_stds[i]

        minimize_args={'keys':keys, 'is_scale_parameter':is_scale_parameter, 'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                       'eps':eps, 'x':x, 'Tf':Tf, 'Nf':Nf, 'contactMatrix':contactMatrix, 's':s, 'scale':scale, 'tangent':tangent}
        res = minimization(self._infer_parameters_to_minimize, flat_guess, flat_bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=flat_cma_stds, verbose=verbose, args_dict=minimize_args)
        params = res[0]
        # Get the parameters (in their original structure) from the flattened parameter vector.
        orig_params = self._unflatten_parameters(params, flat_guess_range, is_scale_parameter, scaled_guesses)
        if full_output:
            return np.array(orig_params), res[1]
        else:
            return np.array(orig_params)


    def nested_sampling_inference(self, keys, guess, stds, np.ndarray x, double Tf, Py_ssize_t Nf, contactMatrix,
                                  bounds=None, tangent=False, infer_scale_parameter=False, verbose=False,
                                  npoints=100, method='single', max_iter=1000, dlogz=None, decline_factor=None):
        '''Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        using nested sampling as implemented in the `nestle` Python package. This function assumes that full data on
        all classes is available.

        This function provides a computational alterantive to `log_G_evidence` and `infer_parameters`. It does not use
        the Laplace approximation to compute the evidence and, in addition,  returns a set of representative samples that can
        be used to compute a posterior mean estimate (insted of the MAP estimate). This approach approach is much more resource
        intensive and typically only viable for small models or tangent space inference.

        Parameters
        ----------
        keys: list
            A list of names for parameters to be inferred
        guess: numpy.array or list
            Prior expectation (and initial guess) for the parameter values. Age-dependent
            rates can be inferred by supplying a guess that is an array instead a single float.
        stds: numpy.array
            Standard deviations for the log normal prior of the parameters
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        bounds: np.array(len(guess), 2), optional
            Bound the prior within the values specified by this. This can be used to avoid sampling the posterior
            in regions where the solution is numerically unstable (e.g. parameters close to 0). Any bound introduces
            a bias to the result, therefore one must make sure that the blocked regions are negligible.
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        infer_scale_parameter: bool or list of bools (size: number of age-dependenly specified parameters)
            Decide if age-dependent parameters are supposed to be inferred separately (default) or if a scale parameter
            for the guess should be inferred. This can be set either globally for all age-dependent parameters or for each
            age-dependent parameter individually
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        npoints: int
            Argument of `nestle.sample`. The number of active points used in the nested sampling algorithm. The higher the
            number the more accurate and expensive is the evidence computation.
        method: str
            Nested sampling method used int `nestle.sample`, see their documentation. Default is `single`, for multimodel posteriors,
            use `multi`.
        max_iter: int
            Maximum number of iterations of the nested sampling algorithm.
        dlogz: float, optional
            Stopping threshold for the estimated error of the log-evidence. This option is mutually exclusive with `decline_factor`.
        decline_factor: float, optional
            Stop the iteration when the weight (likelihood times prior volume) of newly saved samples has been declining for
            `decline_factor * nsamples` consecutive samples. This option is mutually exclusive with `dlogz`.

        Returns
        -------
        log_evidence:
            The nested sampling estimate of the log-evidence.
        (samples, weights):
            A set of weighted samples approximating the posterios distribution.
        '''

        if nestle is None:
            raise Exception("Nested sampling needs optional dependency `nestle` which was not found.")

        enable_bounds = True
        if bounds is None:
            enable_bounds = False
            bounds = np.zeros((len(guess), 2))

        flat_guess, flat_stds, flat_bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = self._flatten_parameters(guess, stds, bounds, infer_scale_parameter)
        s, scale = pyross.utils.make_log_norm_dist(flat_guess, flat_stds)

        k = len(flat_guess)
        # We sample the prior by inverse transform sampling from the unite cube. To implement bounds (if supplied)
        # we shrink the unit cube according the the provided bounds in each dimension.
        ppf_bounds = np.zeros((k, 2))
        if enable_bounds:
            ppf_bounds[:,0] = lognorm.cdf(flat_bounds[:,0], s, scale=scale)
            ppf_bounds[:,1] = lognorm.cdf(flat_bounds[:,1], s, scale=scale)
            ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]
        else:
            ppf_bounds[:, 1] = 1.0


        def prior_transform(x):
            # Tranform into bounded region
            y = ppf_bounds[:,0] + x * ppf_bounds[:,1]
            return lognorm.ppf(y, s, scale=scale)

        def to_sample(params):
            params_unflat = self._unflatten_parameters(params, flat_guess_range, is_scale_parameter, scaled_guesses)
            parameters = self.fill_params_dict(keys, params_unflat)
            logP = -self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix, tangent=False)
            logP += np.sum(lognorm.logpdf(params, s, scale=scale))
            return logP

        if verbose:
            def callback(d):
                if d["it"] % 100 == 0:
                    print("Iteration {}: log_evidence = {}".format(d["it"], d["logz"]))
        else:
            callback=None

        result = nestle.sample(to_sample, prior_transform, k, method=method, npoints=npoints, maxiter=max_iter,
                               dlogz=dlogz, decline_factor=decline_factor, callback=callback)

        log_evidence = result.logz

        unflattened_samples = []
        for sample in result.samples:
            sample_unflat = self._unflatten_parameters(sample, flat_guess_range, is_scale_parameter, scaled_guesses)
            unflattened_samples.append(sample)
        weighted_samples = (unflattened_samples, result.weights)

        return log_evidence, weighted_samples


    def _infer_control_to_minimize(self, params, grad=0, keys=None, bounds=None, x=None, Tf=None, Nf=None, generator=None,
                                   intervention_fun=None, tangent=None, s=None, scale=None):
        """Objective function for minimization call in infer_control."""
        parameters = self.make_params_dict()
        model =self.make_det_model(parameters)
        kwargs = {k:params[i] for (i, k) in enumerate(keys)}
        if intervention_fun is None:
            contactMatrix=generator.constant_contactMatrix(**kwargs)
        else:
            contactMatrix=generator.intervention_custom_temporal(intervention_fun, **kwargs)
        if tangent:
            minus_logp = self.obtain_log_p_for_traj_tangent_space(x, Tf, Nf, model, contactMatrix)
        else:
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp


    def infer_control(self, keys, guess, stds, x, Tf, Nf, generator, bounds,
                      intervention_fun=None, tangent=False,
                      verbose=False, ftol=1e-6,
                      global_max_iter=100, local_max_iter=100, global_atol=1., enable_global=True,
                      enable_local=True, cma_processes=0, cma_population=16, cma_stds=None):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown. The lockdown is modelled by scaling the contact matrices for contact at work, school, and other
        (but not home) uniformly in all age groups. This function infers the scaling parameters assuming that full data
        on all classes is available (with latent variables, use SIR_type.latent_infer_control).

        Parameters
        ----------
        guess: numpy.array
            Prior expectation (and initial guess) for the control parameter values
        stds: numpy.array
            Standard deviations for the log normal prior of the control parameters
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        generator: pyross.contactMatrix
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
        bounds: 2d numpy.array
            Bounds for the parameters (number of parameters x 2).
            Note that the upper bound must be smaller than the absolute physical upper bound minus epsilon
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`, where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO). If not set, assume intervention that's constant in time and infer (aW, aS, aO).
        tangent: bool, optional
            Set to True to use tangent space inference. Default is false.
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: double
            Relative tolerance of logp
        eps: double
            Disallow paramters closer than `eps` to the boundary (to avoid numerical instabilities).
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
        cma_stds: int, optional
            The standard deviation used in cma global optimisation. If not specified, `cma_stds` is set to `stds`.

        Returns
        -------
        res: numpy.array
            MAP estimate of the control parameters
        """
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        if cma_stds is None:
            # Use prior standard deviations here
            cma_stds = stds

        minimize_args = {'keys':keys, 'bounds':bounds, 'x':x, 'Tf':Tf, 'Nf':Nf, 'generator':generator, 's':s, 'scale':scale,
                          'intervention_fun': intervention_fun, 'tangent': tangent}
        res = minimization(self._infer_control_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        return res[0]


    def compute_hessian(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3, tangent=False,
                        infer_scale_parameter=False, fd_method="central"):
        '''
        Computes the Hessian of the MAP estimate.

        Parameters
        ----------
        keys: list
            A list of parameter names that are inferred
        maps: numpy.array
            MAP estimates
        prior_mean: numpy.array
            The mean of the prior (should be the same as "guess" for infer_parameters)
        prior_stds: numpy.array
            The standard deviations of the prior (same as "stds" for infer_parameters)
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        contactMatrix: callable
            A function that takes time (t) as an argument and returns the contactMatrix
        eps: float or numpy.array, optional
            The step size of the Hessian calculation, default=1e-3
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        hess: 2d numpy.array
            The Hessian
        '''
        bounds = np.zeros((len(maps), 2)) # This does not matter here.
        flat_maps, _, _, flat_maps_range, is_scale_parameter, scaled_maps \
            = self._flatten_parameters(maps, prior_stds, bounds, infer_scale_parameter)
        flat_prior_mean, flat_prior_stds, _, _, _, _ \
            = self._flatten_parameters(prior_mean, prior_stds, bounds, infer_scale_parameter)

        s, scale = pyross.utils.make_log_norm_dist(flat_prior_mean, flat_prior_stds)
        def minuslogP(y):
            y_unflat = self._unflatten_parameters(y, flat_maps_range, is_scale_parameter, scaled_maps)
            parameters = self.fill_params_dict(keys, y_unflat)
            minuslogp = self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix, tangent=tangent)
            minuslogp -= np.sum(lognorm.logpdf(y, s, scale=scale))
            return minuslogp

        hess = pyross.utils.hessian_finite_difference(flat_maps, minuslogP, eps, method=fd_method)
        return hess
    
    def FIM(self, param_keys, init_fltr, maps, obs, fltr, Tf, Nf,
            contactMatrix, dx=None, tangent=False, infer_scale_parameter=False):
        '''
        Computes the Fisher Information Matrix (FIM) of the model.
        Parameters
        ----------
        param_keys: list
            A list of parameters to be inferred.
        init_fltr: boolean array
            True for initial conditions to be inferred.
            Shape = (nClass*M)
            Total number of True = total no. of variables - total no. of observed
        maps: numpy.array
            MAP parameter and initial condition estimate (computed for example with SIR_type.latent_inference).
        obs: numpy.array
            The observed data with the initial datapoint
        fltr: boolean sequence or array
            True for observed and False for unobserved.
            e.g. if only `Is` is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        contactMatrix: callable
            A function that takes time (t) as an argument and returns the contactMatrix
        dx: float, optional
            Step size for numerical differentiation of the process mean and its full covariance matrix with respect 
            to the parameters. If not specified, the square root of the machine epsilon for the smallest entry on the 
            diagonal of the covariance matrix is chosen. Decreasing the step size too small can result in round-off error.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is false.
        infer_scale_parameter: bool or list of bools (size: number of age-dependenly specified parameters)
            Decide if age-dependent parameters are supposed to be inferred separately (default) or if a scale parameter
            for the guess should be inferred. This can be set either globally for all age-dependent parameters or for each
            age-dependent parameter individually
        Returns
        -------
        FIM: 2d numpy.array
            The Fisher Information Matrix
        '''
        fltr0 = fltr # Assume time-independent filter for now
        obs0 = obs[0]
        
        param_dim = len(param_keys)
        
        bounds = np.zeros((len(maps), 2)) # This does not matter here
        prior_stds = np.zeros((len(maps))) # This does not matter here
            
        flat_maps, _,_, flat_maps_range, is_scale_parameter, scaled_maps \
            =self._flatten_parameters(maps, prior_stds, bounds, infer_scale_parameter)
        
        def partial_derivative(func, var, point, dx):
            args = point[:]
            def wraps(x):
                args[var] = x
                return func(args)
            return derivative(wraps, point[var], dx=dx)
        
        def _mean(y):
            y_unflat = self._unflatten_parameters(y, flat_maps_range, is_scale_parameter,
                                                  scaled_maps)
            inits = np.copy(y_unflat[param_dim:])
            x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
            parameters = self.fill_params_dict(param_keys, y_unflat)
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            if tangent:
                xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, 
                                                                       Nf, model, 
                                                                       contactMatrix)
            else:
                xm, full_cov = self.obtain_full_mean_cov(x0, Tf, 
                                                         Nf, model, 
                                                         contactMatrix)
            return np.ravel(xm) 
        
        def _cov(y):
            y_unflat = self._unflatten_parameters(y, flat_maps_range, is_scale_parameter,
                                                  scaled_maps)
            inits = np.copy(y_unflat[param_dim:])
            x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
            parameters = self.fill_params_dict(param_keys, y_unflat)
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            if tangent:
                xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, 
                                                                       Nf, model, 
                                                                       contactMatrix)
            else:
                xm, full_cov = self.obtain_full_mean_cov(x0, Tf, 
                                                         Nf, model, 
                                                         contactMatrix)
            return full_cov
        
        def _invcov(y):
            y_unflat = self._unflatten_parameters(y, flat_maps_range, is_scale_parameter,
                                                  scaled_maps)
            inits = np.copy(y_unflat[param_dim:])
            x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
            parameters = self.fill_params_dict(param_keys, y_unflat)
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            if tangent:
                full_invcov = self.obtain_full_invcov_tangent_space(x0, Tf,
                                                                     Nf, model,
                                                                     contactMatrix)
            else:
                full_invcov = self.obtain_full_invcov(x0, Tf,
                                                       Nf, model,
                                                       contactMatrix)
            return full_invcov
        
        cov = _cov(flat_maps)
        if dx == None:
            dx = np.sqrt(np.spacing(np.amin(np.abs(np.diagonal(cov)))))
            
        invcov = _invcov(flat_maps)
        
        dim = len(flat_maps)
        FIM = np.zeros((dim,dim))
        
        rows,cols = np.triu_indices(dim)
        for i,j in zip(rows,cols):
            dmu_i = partial_derivative(_mean, var=i, point=flat_maps, dx=dx)
            dmu_j = partial_derivative(_mean, var=j, point=flat_maps, dx=dx)
            dcov_i = partial_derivative(_cov, var=i, point=flat_maps, dx=dx)
            dcov_j = partial_derivative(_cov, var=j, point=flat_maps, dx=dx)
            t1 = dmu_i@cov@dmu_j
            t2 = np.multiply(0.5,np.trace(invcov@dcov_i@invcov@dcov_j))
            FIM[i,j] = t1 + t2
        i_lower = np.tril_indices(dim,-1)
        FIM[i_lower] = FIM.T[i_lower]
        return FIM

    def error_bars(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3,
                   tangent=False, infer_scale_parameter=False, fd_method="central"):
        hessian = self.compute_hessian(keys, maps, prior_mean, prior_stds, x,Tf,Nf,contactMatrix,eps,
                                       tangent, infer_scale_parameter, fd_method=fd_method)
        return np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    def log_G_evidence(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3,
                       tangent=False, infer_scale_parameter=False, fd_method="central"):
        """Compute the evidence using a Laplace approximation at the MAP estimate."""
        cdef double logP_MAPs
        cdef Py_ssize_t k

        bounds = np.zeros((len(maps), 2)) # Create dummy bounds to pass to flatten function.
        flat_maps, _, _, _, _, _ \
            = self._flatten_parameters(maps, prior_stds, bounds, infer_scale_parameter)
        flat_prior_mean, flat_prior_stds, _, _, _, _ \
            = self._flatten_parameters(prior_mean, prior_stds, bounds, infer_scale_parameter)

        s, scale = pyross.utils.make_log_norm_dist(flat_prior_mean, flat_prior_stds)
        parameters = self.fill_params_dict(keys, maps)
        logP_MAPs = -self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
        logP_MAPs += np.sum(lognorm.logpdf(flat_maps, s, scale=scale))
        k = flat_prior_mean.shape[0]
        A = self.compute_hessian(keys, maps, prior_mean, prior_stds, x,Tf,Nf,contactMatrix,eps, tangent,
                                 infer_scale_parameter, fd_method=fd_method)
        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + k/2*np.log(2*np.pi)

    def obtain_minus_log_p(self, parameters, double [:, :] x, double Tf, int Nf, contactMatrix, tangent=False):
        cdef double minus_log_p
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        if tangent:
            minus_logp = self.obtain_log_p_for_traj_tangent_space(x, Tf, Nf, model, contactMatrix)
        else:
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        return minus_logp

    def _latent_infer_parameters_to_minimize(self, params, grad=0, param_keys=None, init_fltr=None,
                                            is_scale_parameter=None, scaled_guesses=None,
                                            flat_guess_range=None, flat_param_guess_size=None,
                                            obs=None, fltr=None, Tf=None, Nf=None, contactMatrix=None,
                                            s=None, scale=None, obs0=None, fltr0=None, tangent=None):
        """Objective function for minimization call in laten_inference."""
        inits =  np.copy(params[flat_param_guess_size:])

        # Restore parameters from flattened parameters
        orig_params = self._unflatten_parameters(params[:flat_param_guess_size], flat_guess_range, is_scale_parameter, scaled_guesses)

        parameters = self.fill_params_dict(param_keys, orig_params)
        self.set_params(parameters)
        model = self.make_det_model(parameters)

        x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
        penalty = self._penalty_from_negative_values(x0)
        x0[x0<0] = 0.1/self.N # set to be small and positive

        minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))

        # add penalty for being negative
        minus_logp += penalty*Nf

        return minus_logp

    cdef double _penalty_from_negative_values(self, np.ndarray x0):
        cdef:
            double eps=0.1/self.N, dev
            np.ndarray R_init, R_dev, x0_dev
        R_init = self.fi - np.sum(x0.reshape((int(self.dim/self.M), self.M)), axis=0)
        dev = - (np.sum(R_init[R_init<0]) + np.sum(x0[x0<0]))
        return (dev/eps)**2 + (dev/eps)**8


    def latent_infer_parameters(self, param_keys, np.ndarray init_fltr, np.ndarray guess, np.ndarray stds,
                            np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, contactMatrix, np.ndarray bounds,
                            tangent=False, infer_scale_parameter=False,
                            verbose=False, full_output=False, double ftol=1e-5,
                            global_max_iter=100, local_max_iter=100, global_atol=1,
                            enable_global=True, enable_local=True, cma_processes=0,
                            cma_population=16, cma_stds=None, np.ndarray obs0=None, np.ndarray fltr0=None):
        """
        Compute the maximum a-posteriori (MAP) estimate of the parameters and the initial conditions of a SIR type model
        when the classes are only partially observed. Unobserved classes are treated as latent variables.

        Parameters
        ----------
        param_keys: list
            A list of parameters to be inferred.
        init_fltr: boolean array
            True for initial conditions to be inferred.
            Shape = (nClass*M)
            Total number of True = total no. of variables - total no. of observed
        guess: numpy.array or list
            Prior expectation for the parameter values listed, and prior for initial conditions.
            Expect of length len(param_keys)+ (total no. of variables - total no. of observed).
            Age-dependent rates can be inferred by supplying a guess that is an array instead a single float.
        stds: numpy.array
            Standard deviations for the log normal prior.
        obs: 2d numpy.array
            The observed trajectories with reduced number of variables
            (number of data points, (age groups * observed model classes))
        fltr: 2d numpy.array
            A matrix of shape (no. observed variables, no. total variables),
            such that obs_{ti} = fltr_{ij} * X_{tj}
        Tf: float
            Total time of the trajectory
        Nf: int
            Total number of data points along the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        bounds: 2d numpy.array
            Bounds for the parameters + initial conditions
            ((number of parameters + number of initial conditions) x 2).
            Better bounds makes it easier to find the true global minimum.
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        infer_scale_parameter: bool or list of bools (size: number of age-dependenly specified parameters)
            Decide if age-dependent parameters are supposed to be inferred separately (default) or if a scale parameter
            for the guess should be inferred. This can be set either globally for all age-dependent parameters or for each
            age-dependent parameter individually
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: float, optional
            Relative tolerance
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
        cma_stds: int, optional
            The standard deviation used in cma global optimisation. If not specified, `cma_stds` is set to `stds`.
        obs0: numpy.array, optional
            Observed initial condition, if more detailed than obs[0]
        fltr0: 2d numpy.array, optional
            Matrix filter for obs0

        Returns
        -------
        params: nested list
            MAP estimate of paramters (nested if some parameters are age dependent) and initial values of the classes.
        """
        cdef:
            Py_ssize_t param_dim = len(param_keys)

        fltr = pyross.utils.process_fltr(fltr, Nf)
        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless specified otherwise
            obs0=obs[0]
            fltr0=fltr[0]

        obs = pyross.utils.process_obs(obs[1:], Nf-1)
        fltr = fltr[1:]

        assert int(np.sum(init_fltr)) == self.dim - fltr0.shape[0]
        assert len(guess) == param_dim + int(np.sum(init_fltr)), 'len(guess) must equal to total number of params + inits to be inferred'

        # Transfer the parameter parts of guess, stds, ... which can contain arrays as entries for age-dependent rates to a flat list
        flat_param_guess, flat_param_stds, flat_param_bounds, flat_param_guess_range, is_scale_parameter, scaled_param_guesses \
            = self._flatten_parameters(guess[:param_dim], stds[:param_dim], bounds[:param_dim], infer_scale_parameter)

        # Concatenate the flattend parameter guess with init guess
        init_guess = guess[param_dim:]
        init_stds = stds[param_dim:]
        init_bounds = bounds[param_dim:]
        flat_guess = np.concatenate([flat_param_guess, init_guess]).astype(DTYPE)
        flat_stds = np.concatenate([flat_param_stds,init_stds]).astype(DTYPE)
        flat_bounds = np.concatenate([flat_param_bounds, init_bounds], axis=0).astype(DTYPE)

        s, scale = pyross.utils.make_log_norm_dist(flat_guess, flat_stds)

        if cma_stds is None:
            # Use prior standard deviations here
            flat_cma_stds = flat_stds
        else:
            flat_cma_stds_params = np.zeros(len(flat_param_guess))
            cma_stds_init = cma_stds[param_dim:]
            for i in range(param_dim):
                flat_cma_stds_params[flat_param_guess_range[i]] = cma_stds[i]
            flat_cma_stds = np.concatenate([flat_cma_stds_params, cma_stds_init])

        minimize_args = {'param_keys':param_keys, 'init_fltr':init_fltr,
                        'is_scale_parameter':is_scale_parameter,
                        'scaled_guesses':scaled_param_guesses, 'flat_guess_range':flat_param_guess_range,
                        'flat_param_guess_size':len(flat_param_guess),
                         'obs':obs, 'fltr':fltr, 'Tf':Tf, 'Nf':Nf, 'contactMatrix':contactMatrix,
                         's':s, 'scale':scale, 'obs0':obs0, 'fltr0':fltr0, 'tangent':tangent}

        res = minimization(self._latent_infer_parameters_to_minimize, flat_guess, flat_bounds, ftol=ftol,
                           global_max_iter=global_max_iter, local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=flat_cma_stds, verbose=verbose, args_dict=minimize_args)

        estimates = res[0]

        # Get the parameters (in their original structure) from the flattened parameter vector.
        flat_param_estimates = estimates[:len(flat_param_guess)]
        orig_params = self._unflatten_parameters(flat_param_estimates, flat_param_guess_range, is_scale_parameter,
                                                 scaled_param_guesses)
        orig_params = [*orig_params, *estimates[len(flat_param_guess):]]

        if full_output:
            return np.array(orig_params), res[1]
        else:
            return np.array(orig_params)


    def nested_sampling_latent_inference(self, param_keys, np.ndarray init_fltr, np.ndarray guess, np.ndarray stds,
                                         np.ndarray obs, np.ndarray fltr, double Tf, Py_ssize_t Nf, contactMatrix,
                                         bounds=None, np.ndarray obs0=None, np.ndarray fltr0=None, tangent=False,
                                         infer_scale_parameter=False, verbose=False,
                                         return_samples=True, npoints=100, method='single', max_iter=1000, dlogz=None,
                                         decline_factor=None):
        '''Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        with latent variables using nested sampling as implemented in the `nestle` Python package.

        This function provides a computational alterantive to `latent_infer_parameters`. It computes an estimate of the evidence and,
        in addition, returns a set of representative samples that can be used to compute a posterior mean estimate (insted of the MAP
        estimate). This approach approach is much more resource intensive and typically only viable for small models or tangent space inference.

        Parameters
        ----------
        param_keys: list
            A list of parameters to be inferred.
        init_fltr: boolean array
            True for initial conditions to be inferred.
            Shape = (nClass*M)
            Total number of True = total no. of variables - total no. of observed
        guess: numpy.array or list
            Prior expectation for the parameter values listed, and prior for initial conditions.
            Expect of length len(param_keys)+ (total no. of variables - total no. of observed).
            Age-dependent rates can be inferred by supplying a guess that is an array instead a single float.
        stds: numpy.array
            Standard deviations for the log normal prior.
        obs: 2d numpy.array
            The observed trajectories with reduced number of variables
            (number of data points, (age groups * observed model classes))
        fltr: 2d numpy.array
            A matrix of shape (no. observed variables, no. total variables),
            such that obs_{ti} = fltr_{ij} * X_{tj}
        Tf: float
            Total time of the trajectory
        Nf: int
            Total number of data points along the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        bounds: np.array(len(guess), 2), optional
            Bound the prior within the values specified by this array. This can be used to avoid sampling the posterior
            in regions where the solution is numerically unstable (e.g. parameters close to 0). Any bound introduces
            a bias to the result, therefore one must make sure that the blocked regions are negligible.
        obs0: numpy.array, optional
            Observed initial condition, if more detailed than obs[0]
        fltr0: 2d numpy.array, optional
            Matrix filter for obs0
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        infer_scale_parameter: bool or list of bools (size: number of age-dependenly specified parameters)
            Decide if age-dependent parameters are supposed to be inferred separately (default) or if a scale parameter
            for the guess should be inferred. This can be set either globally for all age-dependent parameters or for each
            age-dependent parameter individually
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        npoints: int
            Argument of `nestle.sample`. The number of active points used in the nested sampling algorithm. The higher the
            number the more accurate and expensive is the evidence computation.
        method: str
            Nested sampling method used int `nestle.sample`, see their documentation. Default is `single`, for multimodel posteriors,
            use `multi`.
        max_iter: int
            Maximum number of iterations of the nested sampling algorithm.
        dlogz: float, optional
            Stopping threshold for the estimated error of the log-evidence. This option is mutually exclusive with `decline_factor`.
        decline_factor: float, optional
            Stop the iteration when the weight (likelihood times prior volume) of newly saved samples has been declining for
            `decline_factor * nsamples` consecutive samples. This option is mutually exclusive with `dlogz`.

        Returns
        -------
        log_evidence:
            The nested sampling estimate of the log-evidence.
        (samples, weights):
            A set of weighted samples approximating the posterios distribution.
        '''

        if nestle is None:
            raise Exception("Nested sampling needs optional dependency `nestle` which was not found.")

        param_dim = len(param_keys)

        enable_bounds = True
        if bounds is None:
            enable_bounds = False
            bounds = np.zeros((len(guess), 2))

        fltr = pyross.utils.process_fltr(fltr, Nf)

        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless specified otherwise
            obs0=obs[0]
            fltr0=fltr[0]
        obs = pyross.utils.process_obs(obs[1:], Nf-1)
        fltr = fltr[1:]

        assert int(np.sum(init_fltr)) == self.dim - fltr0.shape[0]
        assert len(guess) == param_dim + int(np.sum(init_fltr)), 'len(guess) must equal to total number of params + inits to be inferred'

        # Transfer the parameter parts of guess, stds, ... which can contain arrays as entries for age-dependent rates to a flat list
        flat_param_guess, flat_param_stds, flat_param_bounds, flat_param_guess_range, is_scale_parameter, scaled_param_guesses \
            = self._flatten_parameters(guess[:param_dim], stds[:param_dim], bounds[:param_dim], infer_scale_parameter)
        flat_param_guess_size = len(flat_param_guess)

        # Concatenate the flattend parameter guess with init guess
        init_guess = guess[param_dim:]
        init_stds = stds[param_dim:]
        init_bounds = bounds[param_dim:]
        flat_guess = np.concatenate([flat_param_guess, init_guess]).astype(DTYPE)
        flat_stds = np.concatenate([flat_param_stds,init_stds]).astype(DTYPE)
        flat_bounds = np.concatenate([flat_param_bounds, init_bounds], axis=0).astype(DTYPE)

        s, scale = pyross.utils.make_log_norm_dist(flat_guess, flat_stds)

        k = len(flat_guess)
        ppf_bounds = np.zeros((k, 2))
        if enable_bounds:
            ppf_bounds[:,0] = lognorm.cdf(flat_bounds[:,0], s, scale=scale)
            ppf_bounds[:,1] = lognorm.cdf(flat_bounds[:,1], s, scale=scale)
            ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]
        else:
            ppf_bounds[:,1] = 1.0

        def prior_transform(x):
            # Tranform into bounded region
            y = ppf_bounds[:,0] + x * ppf_bounds[:,1]
            return lognorm.ppf(y, s, scale=scale)

        def to_sample(params):
            # Todo: replace this by latent_infer_parameters minimisation function.
            inits =  np.copy(params[flat_param_guess_size:])

            # Restore parameters from flattened parameters
            params_unflat = self._unflatten_parameters(params[:flat_param_guess_size], flat_param_guess_range, is_scale_parameter,
                                                       scaled_param_guesses)

            parameters = self.fill_params_dict(param_keys, params_unflat)
            self.set_params(parameters)
            model = self.make_det_model(parameters)

            x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)

            minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent)
            minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))

            return -minus_logp

        if verbose:
            def callback(d):
                if d["it"] % 100 == 0:
                    print("Iteration {}: log_evidence = {}".format(d["it"], d["logz"]))
        else:
            callback=None

        result = nestle.sample(to_sample, prior_transform, k, method=method, npoints=npoints, maxiter=max_iter,
                               dlogz=dlogz, decline_factor=decline_factor, callback=callback)

        log_evidence = result.logz

        unflattened_samples = []
        for sample in result.samples:
            sample_unflat = self._unflatten_parameters(sample[:flat_param_guess_size], flat_param_guess_range, is_scale_parameter,
                                                        scaled_param_guesses)
            sample_unflat = [*sample_unflat, *sample[flat_param_guess_size:]]
            unflattened_samples.append(np.array(sample))
        weighted_samples = (unflattened_samples, result.weights)

        return log_evidence, weighted_samples


    def _latent_infer_control_to_minimize(self, params, grad = 0, keys=None, bounds=None, generator=None, x0=None,
                                          obs=None, fltr=None, Tf=None, Nf=None,
                                          intervention_fun=None, tangent=None,
                                          s=None, scale=None):
        """Objective function for minimization call in latent_infer_control."""
        parameters = self.make_params_dict()
        model = self.make_det_model(parameters)
        kwargs = {k:params[i] for (i, k) in enumerate(keys)}
        if intervention_fun is None:
            contactMatrix = generator.constant_contactMatrix(**kwargs)
        else:
            contactMatrix = generator.intervention_custom_temporal(intervention_fun, **kwargs)
        minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent=tangent)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def latent_infer_control(self, keys, np.ndarray guess, np.ndarray stds, np.ndarray x0, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, generator, np.ndarray bounds,
                            intervention_fun=None, tangent=False,
                            verbose=False, double ftol=1e-5, global_max_iter=100,
                            local_max_iter=100, global_atol=1., enable_global=True, enable_local=True,
                            cma_processes=0, cma_population=16, cma_stds=None, full_output=False):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown with partially observed classes. The unobserved classes are treated as latent variables. The lockdown
        is modelled by scaling the contact matrices for contact at work, school, and other (but not home) uniformly in
        all age groups. This function infers the scaling parameters.

        Parameters
        ----------
        keys: list
            A list of keys for the control parameters to be inferred.
        guess: numpy.array
            Prior expectation (and initial guess) for the control parameter values.
        stds: numpy.array
            Standard deviations for the log normal prior of the control parameters
        x0: numpy.array
            Initial conditions.
        obs:
            Observed trajectory (number of data points x (age groups * observed model classes)).
        fltr: boolean sequence or array
            True for observed and False for unobserved classes.
            e.g. if only Is is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            Total time of the trajectory
        Nf: float
            Number of data points along the trajectory
        generator: pyross.contactMatrix
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
        bounds: 2d numpy.array
            Bounds for the parameters (number of parameters x 2).
            Note that the upper bound must be smaller than the absolute physical upper bound minus epsilon
        intervention_fun: callable, optional
            The calling signature is `intervention_func(t, **kwargs)`, where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO). If not set, assume intervention that's constant in time and infer (aW, aS, aO).
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
            The absolute tolerance for global minimisation.
        enable_global: bool, optional
            Set to True to enable global optimisation.
        enable_local: bool, optional
            Set to True to enable local optimisation.
        cma_processes: int, optional
            Number of parallel processes used for global optimisation.
        cma_population: int, optional
            The number of samples used in each step of the CMA algorithm.
        cma_stds: int, optional
            The standard deviation used in cma global optimisation. If not specified, `cma_stds` is set to `stds`.
        full_output: bool, optional
            Set to True to return full minimization output

        Returns
        -------
        params: numpy.array
            MAP estimate of control parameters
        y_result: float (returned if full_output is True)
            logp for MAP estimates

        """
        fltr = pyross.utils.process_fltr(fltr[1:], Nf-1)
        obs = pyross.utils.process_obs(obs[1:], Nf-1)
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)

        if cma_stds is None:
            # Use prior standard deviations here
            cma_stds = stds

        minimize_args = {'keys':keys, 'bounds':bounds, 'generator':generator, 'x0':x0, 'obs':obs, 'fltr':fltr, 'Tf':Tf,
                         'Nf':Nf, 's':s, 'scale':scale, 'intervention_fun':intervention_fun, 'tangent': tangent}
        res = minimization(self._latent_infer_control_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)
        params = res[0]

        if full_output == True:
            return res
        else:
            return params


    def compute_hessian_latent(self, param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
                               tangent=False, infer_scale_parameter=False, eps=1.e-3, obs0=None, fltr0=None, fd_method="central"):
        '''Computes the Hessian over the parameters and initial conditions.

        Parameters
        ----------
        maps: numpy.array
            MAP parameter and initial condition estimate (computed for example with SIR_type.latent_inference).
        obs: numpy.array
            The observed data with the initial datapoint
        fltr: boolean sequence or array
            True for observed and False for unobserved.
            e.g. if only `Is` is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            Total time of the trajectory
        Nf: int
            Total number of data points along the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        eps: float or numpy.array, optional
            Step size in the calculation of the Hessian.
        obs0: numpy.array, optional
            Observed initial condition, if more detailed than obs[0]
        fltr0: 2d numpy.array, optional
            Matrix filter for obs0
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        hess: numpy.array
            The Hessian over (flat) parameters and initial conditions.
        '''
        param_dim = len(param_keys)
        fltr = pyross.utils.process_fltr(fltr, Nf)

        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless
            # specified otherwise
            obs0=obs[0]
            fltr0=fltr[0]

        obs = pyross.utils.process_obs(obs[1:], Nf-1)
        fltr = fltr[1:]

        bounds = np.zeros((len(maps), 2)) # This does not matter here
        flat_maps, _, _, flat_maps_range, is_scale_parameter, scaled_maps \
            = self._flatten_parameters(maps, prior_stds, bounds, infer_scale_parameter)
        flat_prior_mean, flat_prior_stds, _, _, _, _ \
            = self._flatten_parameters(prior_mean, prior_stds, bounds, infer_scale_parameter)

        s, scale = pyross.utils.make_log_norm_dist(flat_prior_mean, flat_prior_stds)

        def minuslogP(y):
            y_unflat = self._unflatten_parameters(y, flat_maps_range, is_scale_parameter, scaled_maps)
            inits =  np.copy(y_unflat[param_dim:])
            x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
            parameters = self.fill_params_dict(param_keys, y_unflat)
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            minuslogp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent)
            minuslogp -= np.sum(lognorm.logpdf(y, s, scale=scale))
            return minuslogp

        hess = pyross.utils.hessian_finite_difference(flat_maps, minuslogP, eps, method=fd_method)
        return hess

    def error_bars_latent(self, param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
                          tangent=False, infer_scale_parameter=False, eps=1.e-3, obs0=None, fltr0=None, fd_method="central"):
        hessian = self.compute_hessian_latent(param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf,
                                              contactMatrix, tangent, infer_scale_parameter, eps, obs0, fltr0, fd_method=fd_method)
        return np.sqrt(np.diagonal(np.linalg.inv(hessian)))


    def log_G_evidence_latent(self, param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
                              tangent=False, infer_scale_parameter=False, eps=1.e-3, obs0=None, fltr0=None, fd_method="central"):
        """Compute the evidence using a Laplace approximation at the MAP estimate."""
        cdef double logP_MAPs
        cdef Py_ssize_t k

        param_dim = len(param_keys)
        fltr = pyross.utils.process_fltr(fltr, Nf)

        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless specified otherwise
            obs0=obs[0]
            fltr0=fltr[0]

        bounds = np.zeros((len(maps), 2))  # Create dummy bounds to pass to flatten function.
        flat_maps, _, _, _, _, _ \
            = self._flatten_parameters(maps, prior_stds, bounds, infer_scale_parameter)
        flat_prior_mean, flat_prior_stds, _, _, _, _ \
            = self._flatten_parameters(prior_mean, prior_stds, bounds, infer_scale_parameter)

        s, scale = pyross.utils.make_log_norm_dist(flat_prior_mean, flat_prior_stds)
        inits =  np.copy(maps[param_dim:])
        x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
        parameters = self.fill_params_dict(param_keys, maps)
        logP_MAPs = -self.minus_logp_red(parameters, x0, obs[1:], fltr[1:], Tf, Nf, contactMatrix)
        logP_MAPs += np.sum(lognorm.logpdf(flat_maps, s, scale=scale))

        k = flat_prior_mean.shape[0]
        A = self.compute_hessian_latent(param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf,
                                        contactMatrix, tangent, infer_scale_parameter, eps, obs0, fltr0, fd_method=fd_method)
        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + k/2*np.log(2*np.pi)

    def minus_logp_red(self, parameters, double [:] x0, np.ndarray obs,
                            np.ndarray fltr, double Tf, int Nf, contactMatrix, tangent=False):
        '''Computes -logp for a latent trajectory

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
        Nf: int
            The total number of datapoints
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.

        Returns
        -------
        minus_logp: float
            -log(p) for the observed trajectory with the given parameters and initial conditions
        '''

        cdef double minus_log_p
        cdef Py_ssize_t nClass=int(self.dim/self.M)
        if np.any(np.sum(np.reshape(x0, (nClass, self.M)), axis=0) > self.fi):
            raise Exception("the sum over x0 is larger than fi")

        fltr = pyross.utils.process_fltr(fltr, Nf-1)
        obs = pyross.utils.process_obs(obs, Nf-1)
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent)
        return minus_logp

    def set_lyapunov_method(self, lyapunov_method):
        '''Sets the method used for deterministic integration for the SIR_type model

        Parameters
        ----------
        det_method: str
            The name of the integration method. Choose between 'LSODA', 'RK45', 'RK2' and 'euler'.
        '''
        if lyapunov_method not in ['LSODA', 'RK45', 'RK2', 'euler']:
            raise Exception('det_method not implemented. Please see our documentation for the available options')
        self.lyapunov_method=lyapunov_method

    def set_det_method(self, det_method):
        '''Sets the method used for deterministic integration for the SIR_type model

        Parameters
        ----------
        det_method: str
            The name of the integration method. Choose between 'LSODA', 'RK45', 'RK2' and 'euler'.
        '''
        if det_method not in ['LSODA', 'RK45', 'RK2', 'euler']:
            raise Exception('det_method not implemented. Please see our documentation for the available options')
        self.det_method=det_method

    def make_det_model(self, parameters):
        '''Returns a determinisitic model of the same epidemiological class and same parameters

        Parameters
        ----------
        parameters: dict
            A dictionary of parameter values, same as the ones required for initialisation.

        Returns
        -------
        det_model: a class in pyross.deterministic
            A determinisitic model of the same epidemiological class and same parameters
        '''
        pass # to be implemented in subclass

    def make_params_dict(self, params=None):
        pass # to be implemented in subclass

    def fill_params_dict(self, keys, params):
        '''Returns a full dictionary for epidemiological parameters with some changed values

        Parameters
        ----------
        keys: list of String
            A list of names of parameters to be changed.
        params: numpy.array of list
            An array of the same size as keys for the updated value.

        Returns
        -------
        full_parameters: dict
            A dictionary of epidemiological parameters.
            For parameter names specified in `keys`, set the values to be the ones in `params`;
            for the others, use the values stored in the class.
        '''
        full_parameters = self.make_params_dict()
        for (i, k) in enumerate(keys):
            full_parameters[k] = params[i]
        return full_parameters

    def fill_initial_conditions(self, np.ndarray partial_inits, double [:] obs_inits,
                                        np.ndarray init_fltr, np.ndarray fltr):
        '''Returns the full initial condition given partial initial conditions and the observed data

        Parameters
        ----------
        partial_inits: 1d np.array
            Partial initial conditions.
        obs_inits: 1d np.array
            The observed initial conditions.
        init_fltr: 1d np.array
            A vector boolean fltr that yields the partial initis given full initial conditions.
        fltr: 2d np.array
            A matrix fltr that yields the observed data from full data. Same as the one used for latent_infer_parameters.

        Returns
        -------
        x0: 1d np.array
            The full initial condition.
        '''

        cdef:
            np.ndarray x0=np.empty(self.dim, dtype=DTYPE)
            double [:] z, unknown_inits, partial_inits_memview=partial_inits.astype(DTYPE)
        z = np.subtract(obs_inits, np.dot(fltr[:, init_fltr], partial_inits_memview))
        unknown_inits = np.linalg.solve(fltr[:, np.invert(init_fltr)], z)
        x0[init_fltr] = partial_inits_memview
        x0[np.invert(init_fltr)] = unknown_inits
        return x0

    def set_params(self, parameters):
        '''Sets epidemiological parameters used for evaluating -log(p)

        Parameters
        ----------
        parameters: dict
            A dictionary containing all epidemiological parameters.
            Same keys as the one used to initialise the class.

        Notes
        -----
        Can use `fill_params_dict` to generate the full dictionary if only a few parameters are changed
        '''

        self.beta = pyross.utils.age_dep_rates(parameters['beta'], self.M, 'beta')
        self.gIa = pyross.utils.age_dep_rates(parameters['gIa'], self.M, 'gIa')
        self.gIs = pyross.utils.age_dep_rates(parameters['gIs'], self.M, 'gIs')
        self.fsa = pyross.utils.age_dep_rates(parameters['fsa'], self.M, 'fsa')
        self.alpha = pyross.utils.age_dep_rates(parameters['alpha'], self.M, 'alpha')


    cdef double obtain_log_p_for_traj(self, double [:, :] x, double Tf, int Nf, model, contactMatrix):
        cdef:
            double log_p = 0
            double [:] time_points = np.linspace(0, Tf, Nf)
            double [:] xi, xf, dev, mean
            double [:, :] cov
            Py_ssize_t i
        for i in range(Nf-1):
            xi = x[i]
            xf = x[i+1]
            ti = time_points[i]
            tf = time_points[i+1]
            mean, cov = self.estimate_cond_mean_cov(xi, ti, tf, model, contactMatrix)
            dev = np.subtract(xf, mean)
            log_p += self.log_cond_p(dev, cov)
        return -log_p

    cdef double obtain_log_p_for_traj_matrix_fltr(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf, Py_ssize_t Nf, model, contactMatrix, tangent=False):
        cdef:
            Py_ssize_t reduced_dim=fltr.shape[0]*fltr.shape[1]
            double [:, :] xm
            double [:] xm_red, dev
            np.ndarray[DTYPE_t, ndim=2] cov_red, full_cov
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf, model, contactMatrix)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, model, contactMatrix)
        full_fltr = sparse.block_diag(fltr)
        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        cov_red_inv_dev, ldet = pyross.utils.solve_symmetric_close_to_singular(cov_red, dev)
        log_p = -np.dot(dev, cov_red_inv_dev)*(self.N/2)
        log_p -= (ldet-reduced_dim*log(self.N))/2 + (reduced_dim/2)*log(2*PI)
        return -log_p

    cdef double obtain_log_p_for_traj_tangent_space(self, double [:, :] x, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            double [:, :] dx, cov
            double [:] xt, time_points, dx_det
            double dt, logp, t
            Py_ssize_t i
        time_points = np.linspace(0, Tf, Nf)
        dt = time_points[2]
        dx = np.gradient(x, axis=0)*2
        logp = 0
        for i in range(1, Nf-1):
            xt = x[i]
            t = time_points[i]
            dx_det, cov = self.estimate_dx_and_cov(xt, t, dt, model, contactMatrix)
            dev = np.subtract(dx[i], dx_det)
            logp += self.log_cond_p(dev, cov)
        return -logp

    cdef double log_cond_p(self, double [:] x, double [:, :] cov):
        cdef:
            double [:] invcov_x
            double log_cond_p
            double det
        invcov_x, ldet = pyross.utils.solve_symmetric_close_to_singular(cov, x)
        log_cond_p = - np.dot(x, invcov_x)*(self.N/2) - (self.dim/2)*log(2*PI)
        log_cond_p -= (ldet - self.dim*log(self.N))/2
        return log_cond_p

    cdef estimate_cond_mean_cov(self, double [:] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] cov_array
            double [:] cov, sigma0=np.zeros((self.vec_size), dtype=DTYPE)
            double [:, :] x, cov_mat
            Py_ssize_t steps = self.steps
            double [:] time_points = np.linspace(t1, t2, steps)
        x = self.integrate(x0, t1, t2, steps, model, contactMatrix)
        spline = make_interp_spline(time_points, x)

        def rhs(t, sig):
            self.CM = contactMatrix(t)
            self.lyapunov_fun(t, sig, spline)
            return self.dsigmadt

        if self.lyapunov_method=='euler':
            cov_array = pyross.utils.forward_euler_integration(rhs, sigma0, t1, t2, steps)
            cov = cov_array[steps-1]
        elif self.lyapunov_method=='RK45':
            res = solve_ivp(rhs, (t1, t2), sigma0, method='RK45', t_eval=np.array([t2]), first_step=(t2-t1)/steps, max_step=steps)
            cov = res.y[0]
        elif self.lyapunov_method=='LSODA':
            res = solve_ivp(rhs, (t1, t2), sigma0, method='LSODA', t_eval=np.array([t2]), first_step=(t2-t1)/steps, max_step=steps)
            cov = res.y[0]
        elif self.lyapunov_method=='RK2':
            cov_array = pyross.utils.RK2_integration(rhs, sigma0, t1, t2, steps)
            cov = cov_array[steps-1]
        else:
            raise Exception("Error: lyapunov method not found. Use set_lyapunov_method to change the method")

        cov_mat = self.convert_vec_to_mat(cov)
        return x[steps-1], cov_mat

    cdef estimate_dx_and_cov(self, double [:] xt, double t, double dt, model, contactMatrix):
        cdef:
            double [:] dx_det
            double [:, :] cov
        model.set_contactMatrix(t, contactMatrix)
        model.rhs(np.multiply(xt, self.N), t)
        dx_det = np.multiply(dt/self.N, model.dxdt)
        self.compute_tangent_space_variables(xt, t, contactMatrix)
        cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
        return dx_det, cov

    cpdef obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf
            double [:, :] cond_cov, cov, temp
            double [:, :, :, :] full_cov
            double ti, tf
        xm = self.integrate(x0, 0, Tf, Nf, model, contactMatrix, method='LSODA')
        cov = np.zeros((dim, dim), dtype=DTYPE)
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        for i in range(Nf-1):
            ti = time_points[i]
            tf = time_points[i+1]
            xi = xm[i]
            xf, cond_cov = self.estimate_cond_mean_cov(xi, ti, tf, model,
                                                    contactMatrix)
            self.obtain_time_evol_op(xi, xf, ti, tf, model, contactMatrix)
            cov = np.add(self.U@cov@self.U.T, cond_cov)
            full_cov[i, :, i, :] = cov
            if i>0:
                for j in range(0, i):
                    temp = full_cov[j, :, i-1, :]@self.U.T
                    full_cov[j, :, i, :] = temp
                    full_cov[i, :, j, :] = temp.T
        # returns mean and cov for all but first (fixed!) time point
        return xm[1:], np.reshape(full_cov, ((Nf-1)*dim, (Nf-1)*dim))

    cpdef obtain_full_invcov(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf
            double [:, :] cov
            np.ndarray[DTYPE_t, ndim=2] invcov, temp
            double ti, tf
        xm = self.integrate(x0, 0, Tf, Nf, model, contactMatrix, method='LSODA')
        full_cov_inv=[[None]*(Nf-1) for i in range(Nf-1)]
        for i in range(Nf-1):
            ti = time_points[i]
            tf = time_points[i+1]
            xi = xm[i]
            xf = xm[i+1]
            _, cov = self.estimate_cond_mean_cov(xi, ti, tf, model, contactMatrix)
            self.obtain_time_evol_op(xi, xf, ti, tf, model, contactMatrix)
            invcov=np.linalg.inv(cov)
            full_cov_inv[i][i]=invcov
            if i>0:
                temp = invcov@self.U
                full_cov_inv[i-1][i-1] += np.transpose(self.U)@temp
                full_cov_inv[i-1][i]=-np.transpose(self.U)@invcov
                full_cov_inv[i][i-1]=-temp
        full_cov_inv=sparse.bmat(full_cov_inv, format='csc').todense()
        return full_cov_inv # returns invcov for all but first (fixed!) time point

    cpdef obtain_full_mean_cov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xt
            double [:, :] cov, cond_cov, U, J_dt, temp
            double [:, :, :, :] full_cov
            double t, dt=time_points[1]
        xm = self.integrate(x0, 0, Tf, Nf, model, contactMatrix, method='LSODA')
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        cov = np.zeros((dim, dim), dtype=DTYPE)
        for i in range(Nf-1):
            t = time_points[i]
            xt = xm[i]
            self.compute_tangent_space_variables(xt, t, contactMatrix, jacobian=True)
            cond_cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
            if False in np.isfinite(self.B_vec):
                print(np.array(xt), t, self.B_vec)
            J_dt = np.multiply(dt, self.J_mat)
            U = np.add(np.identity(dim), J_dt)
            cov = np.dot(np.dot(U, cov), U.T)
            cov = np.add(cov, cond_cov)
            full_cov[i, :, i, :] = cov
            if i>0:
                for j in range(0, i):
                    temp = np.dot(full_cov[j, :, i-1, :], U.T)
                    full_cov[j, :, i, :] = temp
                    full_cov[i, :, j, :] = temp.T
        return xm[1:], np.reshape(full_cov, ((Nf-1)*dim, (Nf-1)*dim)) # returns mean and cov for all but first (fixed!) time point

    cpdef obtain_full_invcov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xt, B_vec=self.B_vec
            double [:, :] cov, U, J_dt, J_mat=self.J_mat
            np.ndarray[DTYPE_t, ndim=2] invcov, temp
            double t, dt=time_points[1]
        xm = self.integrate(x0, 0, Tf, Nf, model, contactMatrix, method='LSODA')
        full_cov_inv=[[None]*(Nf-1) for i in range(Nf-1)]
        for i in range(Nf-1):
            t = time_points[i]
            xt = xm[i]
            self.compute_tangent_space_variables(xt, t, contactMatrix, jacobian=True)
            cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
            J_dt = np.multiply(dt, self.J_mat)
            U = np.add(np.identity(dim), J_dt)
            invcov=np.linalg.inv(cov)
            full_cov_inv[i][i]=invcov
            if i>0:
                temp = invcov@U
                full_cov_inv[i-1][i-1] += np.transpose(U)@temp
                full_cov_inv[i-1][i]=-np.transpose(U)@invcov
                full_cov_inv[i][i-1]=-temp
        full_cov_inv=sparse.bmat(full_cov_inv, format='csc').todense()
        return full_cov_inv # returns invcov for all but first (fixed!) time point

    cdef obtain_time_evol_op(self, double [:] x0, double [:] xf, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] U=self.U
            double epsilon=1./self.N
            Py_ssize_t i, j, steps=self.steps
        for i in range(self.dim):
            x0[i] += epsilon
            pos = self.integrate(x0, t1, t2, steps, model, contactMatrix)[steps-1]
            for j in range(self.dim):
                U[j, i] = (pos[j]-xf[j])/(epsilon)
            x0[i] -= epsilon

    cdef compute_dsigdt(self, double [:] sig):
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

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        pass # to be implemented in subclasses

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        pass # to be implemented in subclass


    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, method=None, maxNumSteps=100000):
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
        model: pyross model
            Model to integrate (pyross.deterministic.SIR etc)
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        maxNumSteps:
            The maximum number of steps taken by the integrator.

        Returns
        -------
        sol: np.array
            The state of the system evaulated at the time point specified. Only used if det_method is set to 'solve_ivp'.
        """
        def rhs0(t, xt):
            model.set_contactMatrix(t, contactMatrix)
            model.rhs(np.multiply(xt, self.N), t)
            return model.dxdt/self.N

        if method is None:
            method = self.det_method
        if method=='LSODA':
            time_points = np.linspace(t1, t2, steps)
            sol = solve_ivp(rhs0, [t1,t2], x0, method='LSODA', t_eval=time_points, max_step=maxNumSteps).y.T
        elif method=='RK45':
            time_points = np.linspace(t1, t2, steps)
            sol = solve_ivp(rhs0, [t1,t2], x0, method='RK45', t_eval=time_points, max_step=maxNumSteps).y.T
        elif method=='euler':
            sol = pyross.utils.forward_euler_integration(rhs0, x0, t1, t2, steps)
        elif method=='RK2':
            sol = pyross.utils.RK2_integration(rhs0, x0, t1, t2, steps)
        else:
            print(method)
            raise Exception("Error: method not found. use set_det_method to reset, or pass in a valid method")
        return sol

    def _flatten_parameters(self, guess, stds, bounds, infer_scale_parameter):
        # Deal with age-dependent rates: Transfer the supplied guess to a flat guess where the age dependent rates are either listed
        # as multiple parameters (infer_scale_parameter is False) or replaced by a scaling factor with initial value 1.0
        # (infer_scale_parameter is True).
        age_dependent = np.array([hasattr(g, "__len__") for g in guess], dtype=np.bool)  # Select all guesses with more than 1 entry
        n_age_dep = np.sum(age_dependent)
        if not hasattr(infer_scale_parameter, "__len__"):
            # infer_scale_parameter can be either set for all age-dependent parameters or individually
            infer_scale_parameter = np.array([infer_scale_parameter]*n_age_dep, dtype=np.bool)
        is_scale_parameter = np.zeros(len(guess), dtype=np.bool)
        k = 0
        for j in range(len(guess)):
            if age_dependent[j]:
                is_scale_parameter[j] = infer_scale_parameter[k]
                k += 1

        n_scaled_age_dep = np.sum(infer_scale_parameter)
        flat_guess_size  = len(guess) - n_age_dep + self.M * (n_age_dep - n_scaled_age_dep) + n_scaled_age_dep

        # Define a new flat guess and a list of slices that correspond to the intitial guess
        flat_guess       = np.zeros(flat_guess_size)
        flat_stds        = np.zeros(flat_guess_size)
        flat_bounds      = np.zeros((flat_guess_size, 2))
        flat_guess_range = []  # Indicates the position(s) in flat_guess that each parameter corresponds to
        scaled_guesses   = []  # Store the age-dependent guesses where we infer a scale parameter in this list
        i = 0; j = 0
        while i < flat_guess_size:
            if age_dependent[j] and is_scale_parameter[j]:
                flat_guess[i]    = 1.0          # Initial guess for the scaling parameter
                flat_stds[i]     = stds[j]      # Assume that suitable std. deviation for scaling factor and bounds are
                flat_bounds[i,:] = bounds[j,:]  # provided by the user (only one bound for age-dependent parameters possible).
                scaled_guesses.append(guess[j])
                flat_guess_range.append(i)
                i += 1
            elif age_dependent[j]:
                flat_guess[i:i+self.M]    = guess[j]
                flat_stds[i:i+self.M]     = stds[j]
                flat_bounds[i:i+self.M,:] = bounds[j,:]
                flat_guess_range.append(list(range(i, i+self.M)))
                i += self.M
            else:
                flat_guess[i]    = guess[j]
                flat_stds[i]     = stds[j]
                flat_bounds[i,:] = bounds[j,:]
                flat_guess_range.append(i)
                i += 1
            j += 1

        return flat_guess, flat_stds, flat_bounds, flat_guess_range, is_scale_parameter, scaled_guesses

    def _unflatten_parameters(self, params, flat_guess_range, is_scale_parameter, scaled_guesses):
        # Restore parameters from flattened parameters
        orig_params = []
        k=0
        for j in range(len(flat_guess_range)):
            if is_scale_parameter[j]:
                orig_params.append(np.array([params[flat_guess_range[j]]*val for val in scaled_guesses[k]]))
                k += 1
            else:
                orig_params.append(params[flat_guess_range[j]])

        return orig_params


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
            The fraction of symptomatic people who are self-isolating
    M: int
        Number of age groups
    fi: float numpy.array
        Fraction of each age group
    N: int
        Total population
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    """

    def __init__(self, parameters, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        super().__init__(parameters, 3, M, fi, N, steps, det_method, lyapunov_method)

    def make_det_model(self, parameters):
        return pyross.deterministic.SIR(parameters, self.M, self.fi*self.N)

    def make_params_dict(self, params=None):
        if params is None:
            parameters = {'alpha':self.alpha, 'beta':self.beta, 'gIa':self.gIa, 'gIs':self.gIs, 'fsa':self.fsa}
        else:
            parameters = {'alpha':params[0], 'beta':params[1], 'gIa':params[2], 'gIs':params[3], 'fsa':self.fsa}
        return parameters

    def get_init_keys_dict(self):
        return {'S':0, 'Ia':1, 'Is':2}

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, s, Ia, Is
            Py_ssize_t M=self.M
        x = spline(t)
        s = x[0:M]
        Ia = x[M:2*M]
        Is = x[2*M:3*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        self.jacobian(s, l)
        self.noise_correlation(s, Ia, Is, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            double [:] s, Ia, Is
            Py_ssize_t M=self.M
        s = x[0:M]
        Ia = x[M:2*M]
        Is = x[2*M:3*M]
        self.CM = contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
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
            Fraction by which symptomatic individuals self isolate.
        gE: float
            rate of removal from exposed individuals.
    M: int
        Number of age groups
    fi: float numpy.array
        Fraction of each age group
    N: int
        Total population
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    """

    cdef:
        readonly np.ndarray gE

    def __init__(self, parameters, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        super().__init__(parameters, 4, M, fi, N, steps, det_method, lyapunov_method)

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], self.M, 'gE')

    def make_det_model(self, parameters):
        return pyross.deterministic.SEIR(parameters, self.M, self.fi*self.N)


    def make_params_dict(self, params=None):
        if params is None:
            parameters = {'alpha':self.alpha, 'beta':self.beta, 'gIa':self.gIa,
                            'gIs':self.gIs, 'gE':self.gE, 'fsa':self.fsa}
        else:
            parameters = {'alpha':params[0], 'beta':params[1], 'gIa':params[2],
                            'gIs':params[3], 'gE': params[4], 'fsa':self.fsa}
        return parameters

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'Ia':2, 'Is':3}


    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, s, e, Ia, Is
            Py_ssize_t M=self.M
        x = spline(t)
        s = x[0:M]
        e = x[M:2*M]
        Ia = x[2*M:3*M]
        Is = x[3*M:4*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        self.jacobian(s, l)
        self.noise_correlation(s, e, Ia, Is, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            double [:] s, e, Ia, Is
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        Ia = x[2*M:3*M]
        Is = x[3*M:4*M]
        self.CM = contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
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
                J[2, m, 3, n] = s[m]*beta[m]*CM[m, n]*fsa[n]/fi[n]
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
            fraction by which symptomatic individuals self isolate.
        tE: float
            testing rate and contact tracing of exposeds
        tA: float
            testing rate and contact tracing of activateds
        tIa: float
            testing rate and contact tracing of asymptomatics
        tIs: float
            testing rate and contact tracing of symptomatics
    M: int
        Number of age groups.
    fi: float numpy.array
        Fraction of each age group.
    N: int
        Total population.
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    """

    cdef:
        readonly np.ndarray gE, gA, tE, tA, tIa, tIs

    def __init__(self, parameters, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        super().__init__(parameters, 6, M, fi, N, steps, det_method, lyapunov_method)

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], self.M, 'gE')
        self.gA = pyross.utils.age_dep_rates(parameters['gA'], self.M, 'gA')
        self.tE = pyross.utils.age_dep_rates(parameters['tE'], self.M, 'tE')
        self.tA = pyross.utils.age_dep_rates(parameters['tA'], self.M, 'tA')
        self.tIa = pyross.utils.age_dep_rates(parameters['tIa'], self.M, 'tIa')
        self.tIs = pyross.utils.age_dep_rates(parameters['tIs'], self.M, 'tIs')

    def make_det_model(self, parameters):
        return pyross.deterministic.SEAIRQ(parameters, self.M, self.fi*self.N)

    def make_params_dict(self, params=None):
        if params is None:
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
        else:
            parameters = {'alpha':params[0],
                          'beta':params[1],
                          'gIa':params[2],
                          'gIs':params[3],
                          'gE': params[4],
                          'gA': params[5],
                          'fsa': self.fsa,
                          'tS': 0,
                          'tE': self.tE,
                          'tA': self.tA,
                          'tIa': self.tIa,
                          'tIs': self.tIs
                          }
        return parameters

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, s, e, a, Ia, Is, Q
            Py_ssize_t M=self.M
        x = spline(t)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
        self.jacobian(s, l)
        self.noise_correlation(s, e, a, Ia, Is, q, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            double [:] s, e, a, Ia, Is, Q
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        self.CM = contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
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
            fraction by which symptomatic individuals self isolate.
        ars : float
            fraction of population admissible for random and symptomatic tests
        kapE : float
            fraction of positive tests for exposed individuals
    testRate: python function
        number of tests per day and age group
    M: int
        Number of age groups.
    fi: float numpy.array
        Fraction of each age group.
    N: int
        Total population.
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    """

    cdef:
        readonly np.ndarray gE, gA, ars, kapE
        readonly object testRate

    def __init__(self, parameters, testRate, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        super().__init__(parameters, 6, M, fi, N, steps, det_method, lyapunov_method)
        self.testRate=testRate

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}


    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = pyross.utils.age_dep_rates(parameters['gE'], self.M, 'gE')
        self.gA = pyross.utils.age_dep_rates(parameters['gA'], self.M, 'gA')
        self.ars = pyross.utils.age_dep_rates(parameters['ars'], self.M, 'ars')
        self.kapE = pyross.utils.age_dep_rates(parameters['kapE'], self.M, 'kapE')


    def set_testRate(self, testRate):
        self.testRate=testRate

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, method=None, maxNumSteps=100000):
        model.set_testRate(self.testRate)
        return super().integrate(x0, t1, t2, steps, model, contactMatrix, maxNumSteps=maxNumSteps, method=method)

    def make_det_model(self, parameters):
        det_model = pyross.deterministic.SEAIRQ_testing(parameters, self.M, self.fi*self.N)
        det_model.set_testRate(self.testRate)
        return det_model


    def make_params_dict(self, params=None):
        if params is None:
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
        else:
            parameters = {'alpha':params[0],
                          'beta':params[1],
                          'gIa':params[2],
                          'gIs':params[3],
                          'gE': params[4],
                          'gA': params[5],
                          'fsa': self.fsa,
                          'ars': self.ars,
                          'kapE': self.kapE
                          }
        return parameters

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, s, e, a, Ia, Is, Q, TR
            Py_ssize_t M=self.M
        x = spline(t)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        TR=self.testRate(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
        self.jacobian(s, e, a, Ia, Is, q, l, TR)
        self.noise_correlation(s, e, a, Ia, Is, q, l, TR)
        self.compute_dsigdt(sig)


    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            double [:] s, e, a, Ia, Is, Q, TR
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        self.CM = contactMatrix(t)
        TR=self.testRate(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
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
            double N = self.N
            double [:] gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double [:] ars=self.ars, kapE=self.kapE, beta=self.beta
            double t0, tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            t0 = 1./(ars[m]*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars[m]*kapE[m]*t0/N
            tA= TR[m]*ars[m]*t0/N
            tIa = TR[m]*ars[m]*t0/N
            tIs = TR[m]*t0/N

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
            double N=self.N
            double [:] beta=self.beta, gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA
            double [:] ars=self.ars, kapE=self.kapE
            double tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            t0 = 1./(ars[m]*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars[m]*kapE[m]*t0/N
            tA= TR[m]*ars[m]*t0/N
            tIa = TR[m]*ars[m]*t0/N
            tIs = TR[m]*t0/N

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
cdef class Spp(SIR_type):
    """User-defined epidemic model.

    To initialise the Spp model,

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
    N: int
        Total population.
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.


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

    cdef:
        readonly np.ndarray constant_terms, linear_terms, infection_terms
        readonly np.ndarray parameters
        readonly list param_keys
        readonly dict class_index_dict
        readonly pyross.deterministic.Spp det_model


    def __init__(self, model_spec, parameters, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        super().__init__(parameters, self.nClass, M, fi, N, steps, det_method, lyapunov_method)
        self.det_model = pyross.deterministic.Spp(model_spec, parameters, M, fi*N)


    def set_params(self, parameters):
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
        except KeyError:
            raise Exception('The parameters passed does not contain certain keys. The keys are {}'.format(self.param_keys))

    def make_det_model(self, parameters):
        # small hack to make this class work with SIR_type
        self.det_model.update_model_parameters(parameters)
        return self.det_model


    def make_params_dict(self, params=None):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, fi=self.fi
            Py_ssize_t nClass=self.nClass, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
        x = spline(t)
        cdef double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
        if self.constant_terms.size > 0:
            fi = x[(nClass-1)*M:]
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.fill_lambdas(x, l)
        self.jacobian(x, l)
        self.noise_correlation(x, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            Py_ssize_t nClass=self.nClass, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fi=self.fi
        self.CM = contactMatrix(t)
        if self.constant_terms.size > 0:
            fi = x[(nClass-1)*M:]
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.fill_lambdas(x, l)
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
                    l[i, m] += CM[m,n]*x[index]/fi[n]

    cdef jacobian(self, double [:] x, double [:, :] l):
        cdef:
            Py_ssize_t i, m, n, M=self.M, dim=self.dim
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            double [:] rate
            double [:] fi=self.fi
        # infection terms
        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                J[S_index, m, S_index, m] -= rate[m]*l[i, m]
                if product_index>-1:
                    J[product_index, m, S_index, m] += rate[m]*l[i, m]
                for n in range(M):
                    J[S_index, m, infective_index, n] -= x[S_index*M+m]*rate[m]*CM[m, n]/fi[n]
                    if product_index>-1:
                        J[product_index, m, infective_index, n] += x[S_index*M+m]*rate[m]*CM[m, n]/fi[n]
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                J[reagent_index, m, reagent_index, m] -= rate[m]
                if product_index>-1:
                    J[product_index, m, reagent_index, m] += rate[m]
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] x, double [:, :] l):
        cdef:
            Py_ssize_t i, m, n, M=self.M, nClass=self.nClass, class_index
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] B=self.B
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] constant_terms=self.constant_terms
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            double [:] s, reagent, rate
            double N=self.N
        s = x[S_index*M:(S_index+1)*M]

        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                rate = parameters[rate_index]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]/N
                    B[nClass-1, m, nClass-1, m] += rate[m]/N

        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                B[S_index, m, S_index, m] += rate[m]*l[i, m]*s[m]
                if product_index>-1:
                    B[S_index, m, product_index, m] -=  rate[m]*l[i, m]*s[m]
                    B[product_index, m, product_index, m] += rate[m]*l[i, m]*s[m]
                    B[product_index, m, S_index, m] -= rate[m]*l[i, m]*s[m]

        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            reagent = x[reagent_index*M:(reagent_index+1)*M]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M): # only fill in the upper triangular form
                B[reagent_index, m, reagent_index, m] += rate[m]*reagent[m]
                if product_index>-1:
                    B[product_index, m, product_index, m] += rate[m]*reagent[m]
                    B[reagent_index, m, product_index, m] += -rate[m]*reagent[m]
                    B[product_index, m, reagent_index, m] += -rate[m]*reagent[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SppQ(SIR_type):
    """User-defined epidemic model with quarantine stage.

    To initialise the SppQ model,

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
    N: int
        Total population.
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal is 4, as required by the cubic spline fit used for interpolation.
        For robustness, set steps to be large, det_method='LSODA', lyapunov_method='LSODA'.
        For speed, set steps to be 4, det_method='RK2', lyapunov_method='euler'.
        For a combination of the two, choose something in between.
    det_method: str, optional
        The integration method used for deterministic integration.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.
    lyapunov_method: str, optional
        The integration method used for the integration of the Lyapunov equation for the covariance.
        Choose one of 'LSODA', 'RK45', 'RK2' and 'euler'. Default is 'LSODA'.


    See `SIR_type` for a table of all the methods

    Examples
    --------
    An example of model_spec and parameters for SIR class with a constant influx,
    random testing (without false positives/negatives), and quarantine

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
        readonly np.ndarray constant_terms, linear_terms, infection_terms, test_pos, test_freq
        readonly np.ndarray parameters
        readonly list param_keys
        readonly dict class_index_dict
        readonly Py_ssize_t nClassU, nClassUwoN
        readonly pyross.deterministic.SppQ det_model
        readonly object testRate


    def __init__(self, model_spec, parameters, testRate, M, fi, N, steps, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.test_pos = res[5]
        self.test_freq = res[6]
        super().__init__(parameters, self.nClass, M, fi, N, steps, det_method, lyapunov_method)
        self.det_model = pyross.deterministic.SppQ(model_spec, parameters, M, fi*N)
        self.testRate =  testRate
        self.det_model.set_testRate(testRate)

        if self.constant_terms.size > 0:
            self.nClassU = self.nClass // 2 # number of unquarantined classes with constant terms
            self.nClassUwoN = self.nClassU - 1
        else:
            self.nClassU = (self.nClass - 1) // 2 # number of unquarantined classes w/o constant terms
            self.nClassUwoN = self.nClassU



    def set_params(self, parameters):
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
        except KeyError:
            raise Exception('The parameters passed does not contain certain keys. The keys are {}'.format(self.param_keys))

    def set_testRate(self, testRate):
        self.testRate=testRate
        self.det_model.set_testRate(testRate)

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, method=None, maxNumSteps=100000):
        model.set_testRate(self.testRate)
        return super().integrate(x0, t1, t2, steps, model, contactMatrix, method, maxNumSteps)

    def make_det_model(self, parameters):
        # small hack to make this class work with SIR_type
        self.det_model.update_model_parameters(parameters)
        self.det_model.set_testRate(self.testRate)
        return self.det_model


    def make_params_dict(self, params=None):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    cdef calculate_test_r(self, double [:] x, double [:] r, double TR):
        cdef:
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, nClassUwoN=self.nClassUwoN, M=self.M
            int [:] test_freq=self.test_freq
            double [:] fi=self.fi
            double N = self.N
            double ntestpop=0, tau0=0
            double [:, :] parameters=self.parameters
            Py_ssize_t m, i

         # Compute non-quarantined recovered
        for m in range(M):
            r[m] = fi[m] - x[(nClass-1)*M+m] # subtract total quarantined
            for i in range(nClassUwoN):
                r[m] -= x[i*M+m] # subtract non-quarantined class
        # Compute normalisation of testing rates
        for m in range(M):
            for i in range(nClassUwoN):
                ntestpop += parameters[test_freq[i], m] * x[i*M+m]
            ntestpop += parameters[test_freq[nClassUwoN], m] * r[m]
        tau0 = TR / (N * ntestpop)
        return ntestpop, tau0

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, fi=self.fi
            double TR
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double ntestpop, tau0
        x = spline(t)
        cdef double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
        cdef double [:] r=np.zeros(self.M, dtype=DTYPE)
        if self.constant_terms.size > 0:
            fi = x[(nClassU-1)*M:]
        TR = self.testRate(t)
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.fill_lambdas(x, l)
        ntestpop, tau0 = self.calculate_test_r(x, r, TR)
        self.jacobian(x, l, r, ntestpop, tau0)
        self.noise_correlation(x, l, r, tau0)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fi=self.fi
            double TR
            double ntestpop, tau0
            double [:] r=np.zeros(self.M, dtype=DTYPE)
        self.CM = contactMatrix(t)
        TR = self.testRate(t)
        if self.constant_terms.size > 0:
            fi = x[(nClassU-1)*M:]
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.fill_lambdas(x, l)
        ntestpop, tau0 = self.calculate_test_r(x, r, TR)
        self.noise_correlation(x, l, r, tau0)
        if jacobian:
            self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
            self.jacobian(x, l, r, ntestpop, tau0)

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
                    l[i, m] += CM[m,n]*x[index]/fi[n]

    cdef jacobian(self, double [:] x, double [:, :] l, double [:] r, double ntestpop, double tau0):
        cdef:
            Py_ssize_t i, m, n, M=self.M, dim=self.dim
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, nClassUwoN=self.nClassUwoN
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            int [:] test_pos=self.test_pos
            int [:] test_freq=self.test_freq
            double [:] rate
            double term, term2, term3
            double [:] fi=self.fi

        # infection terms (no infection terms in Q classes, perfect quarantine)
        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                J[S_index, m, S_index, m] -= rate[m]*l[i, m]
                if product_index>-1:
                    J[product_index, m, S_index, m] += rate[m]*l[i, m]
                for n in range(M):
                    J[S_index, m, infective_index, n] -= x[S_index*M+m]*rate[m]*CM[m, n]/fi[n]
                    if product_index>-1:
                        J[product_index, m, infective_index, n] += x[S_index*M+m]*rate[m]*CM[m, n]/fi[n]
        # linear terms
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                J[reagent_index, m, reagent_index, m] -= rate[m]
                J[reagent_index + nClassU, m, reagent_index + nClassU, m] -= rate[m]
                if product_index>-1:
                    J[product_index, m, reagent_index, m] += rate[m]
                    J[product_index + nClassU, m, reagent_index + nClassU, m] += rate[m]
        # quarantining terms
        for m in range(M):
            for i in range(nClassUwoN):
                term = tau0 * parameters[test_freq[i], m] * parameters[test_pos[i], m]
                term2 = term * x[i*M+m] / ntestpop
                J[i, m, i, m] -= term
                J[i+nClassU, m, i, m] += term
                J[nClass-1,  m, i, m] += term
                for n in range(M):
                    for j in range(nClassUwoN):
                        term3 = term2 * (parameters[test_freq[j],n] - parameters[test_freq[nClassUwoN],n])
                        J[i, m, j, n] += term3
                        J[i+nClassU, m, j, n] -= term3
                        J[nClass-1,  m, j, n] -= term3
                    term3 = term2 * parameters[test_freq[nClassUwoN],n]
                    if self.constant_terms.size > 0:
                        J[i, m, nClassUwoN, n] += term3
                        J[i+nClassU, m, nClassUwoN, n] -= term3
                        J[nClass-1,  m, nClassUwoN, n] -= term3
                    J[i, m, nClass-1, n] -= term3
                    J[i+nClassU, m, nClass-1, n] += term3
                    J[nClass-1,  m, nClass-1, n] += term3
            term = tau0 * parameters[test_freq[nClassUwoN], m] * parameters[test_pos[nClassUwoN], m]
            term2 = term * r[m] / ntestpop
            for j in range(nClassUwoN):
                J[nClass-1, m, j, m] -= term
            if self.constant_terms.size > 0:
                J[nClass-1, m, nClassUwoN, m] += term
            J[nClass-1, m, nClass-1, m] -= term
            for n in range(M):
                for j in range(nClassUwoN):
                    term3 = term2 * (parameters[test_freq[j],n] - parameters[test_freq[nClassUwoN],n])
                    J[nClass-1,  m, j, n] -= term3
                term3 = term2 * parameters[test_freq[nClassUwoN],n]
                if self.constant_terms.size > 0:
                    J[nClass-1,  m, nClassUwoN, n] -= term3
                J[nClass-1,  m, nClass-1, n] += term3



        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] x, double [:, :] l, double [:] r, double tau0):
        cdef:
            Py_ssize_t i, m, n, M=self.M, class_index
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, nClassUwoN=self.nClassUwoN
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] B=self.B
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] constant_terms=self.constant_terms
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            int [:] test_pos=self.test_pos
            int [:] test_freq=self.test_freq
            double [:] s, reagent, rate
            double term
            double N=self.N
        s = x[S_index*M:(S_index+1)*M]

        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                rate = parameters[rate_index]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]/N
                    B[nClass-1, m, nClass-1, m] += rate[m]/N

        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M):
                B[S_index, m, S_index, m] += rate[m]*l[i, m]*s[m]
                if product_index>-1:
                    B[S_index, m, product_index, m] -=  rate[m]*l[i, m]*s[m]
                    B[product_index, m, product_index, m] += rate[m]*l[i, m]*s[m]
                    B[product_index, m, S_index, m] -= rate[m]*l[i, m]*s[m]

        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            reagent = x[reagent_index*M:(reagent_index+1)*M]
            rate_index = linear_terms[i, 0]
            rate = parameters[rate_index]
            for m in range(M): # only fill in the upper triangular form
                B[reagent_index, m, reagent_index, m] += rate[m]*reagent[m]
                if product_index>-1:
                    B[product_index, m, product_index, m] += rate[m]*reagent[m]
                    B[reagent_index, m, product_index, m] += -rate[m]*reagent[m]
                    B[product_index, m, reagent_index, m] += -rate[m]*reagent[m]
            # same transitions in Q classes
            reagent = x[(reagent_index+nClassU)*M:((reagent_index+nClassU)+1)*M]
            for m in range(M): # only fill in the upper triangular form
                B[reagent_index+nClassU, m, reagent_index+nClassU, m] += rate[m]*reagent[m]
                if product_index>-1:
                    B[product_index+nClassU, m, product_index+nClassU, m] += rate[m]*reagent[m]
                    B[reagent_index+nClassU, m, product_index+nClassU, m] += -rate[m]*reagent[m]
                    B[product_index+nClassU, m, reagent_index+nClassU, m] += -rate[m]*reagent[m]

        for m in range(M):
            for i in range(nClassUwoN): # only fill in the upper triangular form
                term = tau0 * parameters[test_freq[i], m] * parameters[test_pos[i], m] * x[m+M*i]
                B[i, m, i, m] += term
                B[i+nClassU, m, i+nClassU, m] += term
                B[nClass-1, m, nClassU-1, m] += term
                B[i, m, i+nClassU, m] -= term
                B[i, m, nClass-1, m] -= term
                B[i+nClassU, m, nClass-1, m] += term
            term = tau0 * parameters[test_freq[nClassUwoN], m] * parameters[test_pos[nClassUwoN], m] * r[m]
            B[nClass-1, m, nClassU-1, m] += term


        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]
