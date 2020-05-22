from itertools import compress
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, approx_fprime
from scipy.stats import lognorm
import numpy as np
from scipy.interpolate import make_interp_spline
cimport numpy as np
cimport cython
import time

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
        readonly Py_ssize_t nClass, N, M, steps, dim, vec_size
        readonly beta, gIa, gIs, fsa
        readonly np.ndarray alpha, fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols


    def __init__(self, parameters, nClass, M, fi, N, steps):
        self.N = N
        self.M = M
        self.fi = fi
        if steps < 4:
            raise Exception('Steps must be at least 4 for internal spline interpolation.')
        self.steps = steps
        self.set_params(parameters)

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


    def _inference_to_minimize(self, params, grad=0, bounds=None, eps=None, beta_rescale=None, x=None, Tf=None, Nf=None,
                               contactMatrix=None, s=None, scale=None):
        """Objective function for minimization call in inference."""
        if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
            return INFINITY

        local_params = params.copy()
        local_params[1] /= beta_rescale
        parameters = self.make_params_dict(local_params)
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(local_params, s, scale=scale))
        return minus_logp


    def inference(self, guess, stds, x, Tf, Nf, contactMatrix, beta_rescale=1, bounds=None, verbose=False,
                  ftol=1e-6, eps=1e-5, global_max_iter=100, local_max_iter=100, global_ftol_factor=10.,
                  enable_global=True, enable_local=True, cma_processes=0, cma_population=16, cma_stds=None):
        """Compute the maximum a-posteriori (MAP) estimate of the parameters of the SIR type model

        Deprecated. Use `infer_parameters` instead.
        """

        # make bounds if it does not exist and rescale
        if bounds is None:
            bounds = np.array([[eps, g*5] for g in guess])
            bounds[0][1] = min(bounds[0][1], 1-2*eps)
        bounds = np.array(bounds)
        guess[1] *= beta_rescale
        bounds[1] *= beta_rescale
        stds[1] *= beta_rescale

        s, scale = pyross.utils.make_log_norm_dist(guess, stds)

        if cma_stds is None:
            # Use prior standard deviations here
            cma_stds = stds

        minimize_args={'bounds':bounds, 'eps':eps, 'beta_rescale':beta_rescale, 'x':x, 'Tf':Tf, 'Nf':Nf,
                         'contactMatrix':contactMatrix, 's':s, 'scale':scale}
        res = minimization(self._inference_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)
        estimates = res[0]
        estimates[1] /= beta_rescale
        return estimates

    def _infer_parameters_to_minimize(self, params, grad=0, keys=None, is_scale_parameter=None, scaled_guesses=None,
                               flat_guess_range=None, eps=None, x=None, Tf=None, Nf=None,
                               contactMatrix=None, s=None, scale=None, tangent=None):
        """Objective function for minimization call in infer_parameters."""
        # Restore parameters from flattened parameters
        orig_params = []
        k=0
        for j in range(len(flat_guess_range)):
            if is_scale_parameter[j]:
                orig_params.append(np.array([params[flat_guess_range[j]]*val for val in scaled_guesses[k]]))
                k += 1
            else:
                orig_params.append(params[flat_guess_range[j]])

        parameters = self.fill_params_dict(keys, orig_params)
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        if tangent:
            minus_logp = self.obtain_log_p_for_traj_tangent_space(x, Tf, Nf, model, contactMatrix)
        else:
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def infer_parameters(self, keys, guess, stds, bounds, np.ndarray x,
                        double Tf, Py_ssize_t Nf, contactMatrix,
                        tangent=False,
                        infer_scale_parameter=False, verbose=False,
                        ftol=1e-6, eps=1e-5, global_max_iter=100, local_max_iter=100, global_ftol_factor=10.,
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
        guess: numpy.array
            Prior expectation (and initial guess) for the parameter values. For parameters that support it, age-dependent
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
        global_ftol_factor: float
            The relative tolerance for global optimisation.
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

        s, scale = pyross.utils.make_log_norm_dist(flat_guess, flat_stds)

        if cma_stds is None:
            # Use prior standard deviations here
            flat_cma_stds = flat_stds
        else:
            flat_cma_stds = np.zeros(flat_guess_size)
            for i in range(len(guess)):
                flat_cma_stds[flat_guess_range[i]] = cma_stds[i]

        minimize_args={'keys':keys, 'is_scale_parameter':is_scale_parameter, 'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                       'eps':eps, 'x':x, 'Tf':Tf, 'Nf':Nf, 'contactMatrix':contactMatrix, 's':s, 'scale':scale, 'tangent':tangent}
        res = minimization(self._infer_parameters_to_minimize, flat_guess, flat_bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=flat_cma_stds, verbose=verbose, args_dict=minimize_args)
        params = res[0]
        # Restore parameters from flattened parameters
        orig_params = []
        k=0
        for j in range(len(flat_guess_range)):
            if is_scale_parameter[j]:
                orig_params.append(np.array([params[flat_guess_range[j]]*val for val in scaled_guesses[k]]))
                k += 1
            else:
                orig_params.append(params[flat_guess_range[j]])

        return np.array(orig_params)


    def _infer_control_to_minimize(self, params, grad=0, bounds=None, eps=None, x=None, Tf=None, Nf=None, generator=None,
                                   s=None, scale=None):
        """Objective function for minimization call in infer_control."""
        if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
            return INFINITY

        parameters = self.make_params_dict()
        model =self.make_det_model(parameters)
        times = [Tf+1]
        interventions = [params]
        contactMatrix = generator.interventions_temporal(times, interventions)
        minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp


    def infer_control(self, guess, stds, x, Tf, Nf, generator, bounds, verbose=False, ftol=1e-6, eps=1e-5,
                      global_max_iter=100, local_max_iter=100, global_ftol_factor=10., enable_global=True,
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
        global_ftol_factor: float
            The relative tolerance for global optimisation.
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

        minimize_args = {'bounds':bounds, 'eps':eps, 'x':x, 'Tf':Tf, 'Nf':Nf, 'generator':generator, 's':s, 'scale':scale}
        res = minimization(self._infer_control_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        return res[0]

    def hessian(self, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, beta_rescale=1, eps=1.e-3):
        '''
        DEPRECATED. Use compute_hessian instead.
        '''
        maps[1] *= beta_rescale
        cdef:
            Py_ssize_t k=maps.shape[0], i, j
            double xx0
            np.ndarray g1, g2, s, scale, hess = np.empty((k, k))
        s, scale = pyross.utils.make_log_norm_dist(prior_mean, prior_stds)
        def minuslogP(y):
            y[1] /= beta_rescale
            parameters = self.make_params_dict(y)
            minuslogp = self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, s, scale=scale))
            y[1] *= beta_rescale
            return minuslogp
        g1 = approx_fprime(maps, minuslogP, eps)
        for j in range(k):
            xx0 = maps[j]
            maps[j] += eps
            g2 = approx_fprime(maps, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            maps[j] = xx0
        maps[1] /= beta_rescale
        hess[1, :] *= beta_rescale
        hess[:, 1] *= beta_rescale
        return hess

    def compute_hessian(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3):
        '''
        Computes the Hessian of the MAP estimatesself.

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
        eps: float, optional
            The step size of the Hessian calculation, default=1e-3

        Returns
        -------
        hess: 2d numpy.array
            The Hessian
        '''
        cdef:
            Py_ssize_t k=maps.shape[0], i, j
            double xx0
            np.ndarray g1, g2, s, scale, hess = np.empty((k, k))
        s, scale = pyross.utils.make_log_norm_dist(prior_mean, prior_stds)
        def minuslogP(y):
            parameters = self.fill_params_dict(keys, y)
            minuslogp = self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, s, scale=scale))
            return minuslogp
        g1 = approx_fprime(maps, minuslogP, eps)
        for j in range(k):
            xx0 = maps[j]
            maps[j] += eps
            g2 = approx_fprime(maps, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            maps[j] = xx0
        return hess

    def error_bars(self, keys, maps, prior_mean, prior_stds,
                        x, Tf, Nf, contactMatrix, eps=1.e-3):
        hessian = self.compute_hessian(keys, maps, prior_mean, prior_stds,
                                x,Tf,Nf,contactMatrix,eps)
        return np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    def log_G_evidence(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3):
        # M variate process, M=3 for SIIR model
        cdef double logP_MAPs
        cdef Py_ssize_t k
        s, scale = pyross.utils.make_log_norm_dist(prior_mean, prior_stds)
        parameters = self.fill_params_dict(keys, maps)
        logP_MAPs = -self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
        logP_MAPs += np.sum(lognorm.logpdf(maps, s, scale=scale))
        k = maps.shape[0]
        A = self.hessian(keys, maps, prior_mean, prior_stds, x,Tf,Nf,contactMatrix,eps)
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

    def _latent_inference_to_minimize(self, params, grad = 0, bounds=None, eps=None, param_dim=None, rescale_factor=None,
                beta_rescale=None, obs=None, fltr=None, Tf=None, Nf=None, contactMatrix=None, s=None, scale=None):
        """Objective function for minimization call in laten_inference."""
        if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
            return INFINITY
        x0 =  params[param_dim:]/rescale_factor

        local_params = params.copy()
        local_params[1] /= beta_rescale
        parameters = self.make_params_dict(local_params[:param_dim])
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj_red(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(local_params, s, scale=scale))
        return minus_logp


    def latent_inference(self, np.ndarray guess, np.ndarray stds, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, contactMatrix, np.ndarray bounds,
                            beta_rescale=1, verbose=False, double ftol=1e-5, double eps=1e-4,
                            global_max_iter=100, local_max_iter=100, global_ftol_factor=10.,
                            enable_global=True, enable_local=True, cma_processes=0,
                            cma_population=16, cma_stds=None):
        """
        DEPRECATED. Use `latent_infer_parameters` instead.
        """
        cdef:
            double eps_for_params=eps, eps_for_init_cond = 0.5/self.N
            double rescale_factor = eps_for_params/eps_for_init_cond
            Py_ssize_t param_dim = guess.shape[0] - self.dim
        guess[param_dim:] *= rescale_factor
        guess[1] *= beta_rescale
        bounds[param_dim:, :] *= rescale_factor
        bounds[1, :] *= beta_rescale
        stds[param_dim:] *= rescale_factor
        stds[1] *= beta_rescale
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)

        if cma_stds is None:
            # Use prior standard deviations here
            cma_stds = stds

        minimize_args = {'bounds':bounds, 'eps':eps, 'param_dim':param_dim, 'rescale_factor':rescale_factor, 'beta_rescale':beta_rescale,
                         'obs':obs, 'fltr':fltr, 'Tf':Tf, 'Nf':Nf, 'contactMatrix':contactMatrix, 's':s, 'scale':scale}
        res = minimization(self._latent_inference_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        params = res[0]
        params[param_dim:] /= rescale_factor
        params[1] /= beta_rescale
        return params

    def _latent_infer_parameters_to_minimize(self, params, grad = 0, param_keys=None, init_fltr=None, bounds=None, param_dim=None,
                obs=None, fltr=None, Tf=None, Nf=None, contactMatrix=None, s=None, scale=None, obs0=None, fltr0=None, tangent=None):
        """Objective function for minimization call in laten_inference."""
        inits =  np.copy(params[param_dim:])
        parameters = self.fill_params_dict(param_keys, params[:param_dim])
        self.set_params(parameters)
        model = self.make_det_model(parameters)

        x0 = self.fill_initial_conditions(inits, obs0, init_fltr, fltr0)
        penalty = self._penalty_from_negative_values(x0)
        x0[x0<0] = 0.1/self.N # set to be small and positive

        minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix, tangent)
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
                            tangent=False,
                            verbose=False, double ftol=1e-5,
                            global_max_iter=100, local_max_iter=100, global_ftol_factor=10.,
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
        guess: numpy.array
            Prior expectation for the parameter values listed, and prior for initial conditions.
            Expect of length len(param_keys)+ (total no. of variables - total no. of observed)
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
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimizer.
        ftol: float, optional
            Relative tolerance
        global_max_iter: int, optional
            Number of global optimisations performed.
        local_max_iter: int, optional
            Number of local optimisation performed.
        global_ftol_factor: float
            The relative tolerance for global optimisation.
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
            Observed initial condition, if more detailed than obs[0,:]
        fltr0: 2d numpy.array, optional
            Matrix filter for obs0

        Returns
        -------
        params: numpy.array
            MAP estimate of paramters and initial values of the classes.
        """
        cdef:
            Py_ssize_t param_dim = len(param_keys)

        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless specified otherwise
            obs0=obs[0,:]
            fltr0=fltr

        assert int(np.sum(init_fltr)) == self.dim - fltr0.shape[0]
        assert guess.shape[0] == param_dim + int(np.sum(init_fltr)), 'len(guess) must equal to total number of params + inits to be inferred'
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)

        if cma_stds is None:
            # Use prior standard deviations here
            cma_stds = stds

        minimize_args = {'param_keys':param_keys, 'init_fltr':init_fltr, 'bounds':bounds, 'param_dim':param_dim,
                         'obs':obs, 'fltr':fltr, 'Tf':Tf, 'Nf':Nf, 'contactMatrix':contactMatrix,
                         's':s, 'scale':scale, 'obs0':obs0, 'fltr0':fltr0, 'tangent':tangent}

        res = minimization(self._latent_infer_parameters_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        params = res[0]
        return params


    def _latent_infer_control_to_minimize(self, params, grad = 0, bounds=None, eps=None, generator=None, x0=None,
                                          obs=None, fltr=None, Tf=None, Nf=None, s=None, scale=None):
        """Objective function for minimization call in latent_infer_control."""
        if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
            return INFINITY

        parameters = self.make_params_dict()
        model = self.make_det_model(parameters)
        times = [Tf+1]
        interventions = [params]
        contactMatrix = generator.interventions_temporal(times, interventions)
        minus_logp = self.obtain_log_p_for_traj_red(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def latent_infer_control(self, np.ndarray guess, np.ndarray stds, np.ndarray x0, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, generator, np.ndarray bounds,
                            verbose=False, double ftol=1e-5, double eps=1e-4, global_max_iter=100,
                            local_max_iter=100, global_ftol_factor=10., enable_global=True, enable_local=True,
                            cma_processes=0, cma_population=16, cma_stds=None):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown with partially observed classes. The unobserved classes are treated as latent variables. The lockdown
        is modelled by scaling the contact matrices for contact at work, school, and other (but not home) uniformly in
        all age groups. This function infers the scaling parameters.

        Parameters
        ----------
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
        global_ftol_factor: float
            The relative tolerance for global optimisation.
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

        minimize_args = {'bounds':bounds, 'eps':eps, 'generator':generator, 'x0':x0, 'obs':obs, 'fltr':fltr, 'Tf':Tf,
                         'Nf':Nf, 's':s, 'scale':scale}
        res = minimization(self._latent_infer_control_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_ftol_factor=global_ftol_factor,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        return res[0]


    def compute_hessian_latent(self, param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1.e-3, obs0=None, fltr0=None):
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
        eps: float, optional
            Step size in the calculation of the Hessian
        obs0: numpy.array, optional
            Observed initial condition, if more detailed than obs[0,:]
        fltr0: 2d numpy.array, optional
            Matrix filter for obs0

        Returns
        -------
        hess_params: numpy.array
            The Hessian over parameters
        hess_init: numpy.array
            The Hessian over initial conditions
        '''

        s, scale = pyross.utils.make_log_norm_dist(prior_mean, prior_stds)
        dim = maps.shape[0]
        param_dim = dim - self.dim
        map_params = maps[:param_dim]
        map_x0 = maps[param_dim:]
        s_params = s[:param_dim]
        a_x0 = s[param_dim:]
        scale_params = scale[:param_dim]
        scale_x0 = scale[param_dim:]
        hess_params = self.latent_hess_selected_params(param_keys, map_params, map_x0, s_params, scale_params,
                                                obs, fltr, Tf, Nf, contactMatrix,
                                                beta_rescale=beta_rescale, eps=eps)

        if obs0 is None or fltr0 is None:
            # Use the same filter and observation for initial condition as for the rest of the trajectory, unless specified otherwise
            obs0=obs[0,:]
            fltr0=fltr
        hess_init = self.latent_hess_selected_init(init_fltr, map_x0, map_params, a_x0, scale_x0,
                                                obs, fltr, Tf, Nf, contactMatrix, obs0, fltr0,
                                                eps=0.5/self.N)
        return hess_params, hess_init



    def hessian_latent(self, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1.e-3):
        '''
        DEPRECATED. Use `compute_hessian_latent` instead.
        '''
        s, scale = pyross.utils.make_log_norm_dist(prior_mean, prior_stds)
        dim = maps.shape[0]
        param_dim = dim - self.dim
        map_params = maps[:param_dim]
        map_x0 = maps[param_dim:]
        s_params = s[:param_dim]
        a_x0 = s[param_dim:]
        scale_params = scale[:param_dim]
        scale_x0 = scale[param_dim:]
        hess_params = self.latent_hess_params(map_params, map_x0, s_params, scale_params,
                                                obs, fltr, Tf, Nf, contactMatrix,
                                                beta_rescale=beta_rescale, eps=eps)
        hess_init = self.latent_hess_init(map_x0, map_params, a_x0, scale_x0,
                                                obs, fltr, Tf, Nf, contactMatrix,
                                                eps=0.5/self.N)
        return hess_params, hess_init

    def latent_hess_params(self, map_params, x0, s_params, scale_params,
                                    obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1e-3):
        cdef Py_ssize_t j
        dim = map_params.shape[0]
        hess = np.empty((dim, dim))
        map_params[1] *= beta_rescale
        def minuslogP(y):
            y[1] /= beta_rescale
            parameters = self.make_params_dict(y)
            minuslogp = self.minus_logp_red(parameters, x0, obs, fltr, Tf, Nf, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, s_params, scale=scale_params))
            y[1] *= beta_rescale
            return minuslogp
        g1 = approx_fprime(map_params, minuslogP, eps)
        for j in range(dim):
            temp = map_params[j]
            map_params[j] += eps
            g2 = approx_fprime(map_params, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            map_params[j] = temp
        map_params[1] /= beta_rescale
        hess[1, :] *= beta_rescale
        hess[:, 1] *= beta_rescale
        return hess

    def latent_hess_selected_params(self, keys, map_params, x0, s_params, scale_params,
                                    obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1e-3):
        cdef Py_ssize_t j
        dim = map_params.shape[0]
        hess = np.empty((dim, dim))
        map_params[1] *= beta_rescale
        def minuslogP(y):
            y[1] /= beta_rescale
            parameters = self.fill_params_dict(keys, y)
            minuslogp = self.minus_logp_red(parameters, x0, obs[1:], fltr, Tf, Nf, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, s_params, scale=scale_params))
            y[1] *= beta_rescale
            return minuslogp
        g1 = approx_fprime(map_params, minuslogP, eps)
        for j in range(dim):
            temp = map_params[j]
            map_params[j] += eps
            g2 = approx_fprime(map_params, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            map_params[j] = temp
        map_params[1] /= beta_rescale
        hess[1, :] *= beta_rescale
        hess[:, 1] *= beta_rescale
        return hess



    def latent_hess_init(self, map_x0, params, a_x0, scale_x0,
                            obs, fltr, Tf, Nf, contactMatrix,
                                    eps=1e-6):
        cdef Py_ssize_t j
        dim = map_x0.shape[0]
        hess = np.empty((dim, dim))
        parameters = self.make_params_dict(params)
        model = self.make_det_model(parameters)
        def minuslogP(y):
            minuslogp = self.obtain_log_p_for_traj_red(y, obs, fltr, Tf, Nf, model, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, a_x0, scale=scale_x0))
            return minuslogp
        g1 = approx_fprime(map_x0, minuslogP, eps)
        for j in range(dim):
            temp = map_x0[j]
            map_x0[j] += eps
            g2 = approx_fprime(map_x0, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            map_x0[j] = temp
        return hess

    def latent_hess_selected_init(self, init_fltr, map_x0, params, a_x0, scale_x0,
                            obs, fltr, Tf, Nf, contactMatrix, obs0, fltr0,
                                    eps=1e-6):
        cdef Py_ssize_t j
        dim = map_x0.shape[0]
        hess = np.empty((dim, dim))
        parameters = self.make_params_dict(params)
        model = self.make_det_model(parameters)
        def minuslogP(y):
            x0 = self.fill_initial_conditions(y, obs0, init_fltr, fltr0)
            minuslogp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
            minuslogp -= np.sum(lognorm.logpdf(y, a_x0, scale=scale_x0))
            return minuslogp
        g1 = approx_fprime(map_x0, minuslogP, eps)
        for j in range(dim):
            temp = map_x0[j]
            map_x0[j] += eps
            g2 = approx_fprime(map_x0, minuslogP, eps)
            hess[:,j] = (g2 - g1)/eps
            map_x0[j] = temp
        return hess



    def minus_logp_red(self, parameters, double [:] x0, double [:, :] obs,
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

        Returns
        -------
        minus_logp: float
            -log(p) for the observed trajectory with the given parameters and initial conditions
        '''

        cdef double minus_log_p
        cdef Py_ssize_t nClass=int(self.dim/self.M)
        if np.any(np.sum(np.reshape(x0, (nClass, self.M)), axis=0) > self.fi):
            raise Exception("the sum over x0 is larger than fi")
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        if fltr.ndim == 1:
            print('Vector filter is deprecated. Use matrix filter instead. Also does not support tangent.')
            minus_logp = self.obtain_log_p_for_traj_red(x0, obs, fltr, Tf, Nf, model, contactMatrix)
        else:
            assert fltr.ndim == 2
            minus_logp = self.obtain_log_p_for_traj_matrix_fltr(x0, obs, fltr, Tf, Nf, model, contactMatrix, tangent)
        return minus_logp

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

    def fill_initial_conditions(self, double [:] partial_inits, double [:] obs_inits,
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
            double [:] z, unknown_inits
        z = np.subtract(obs_inits, np.dot(fltr[:, init_fltr], partial_inits))
        unknown_inits = np.linalg.solve(fltr[:, np.invert(init_fltr)], z)
        x0[init_fltr] = partial_inits
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

        self.alpha = np.zeros( self.M, dtype = DTYPE)
        if np.size(parameters['alpha'])==1:
            self.alpha = parameters['alpha']*np.ones(self.M)
        elif np.size(parameters['alpha'])==self.M:
            self.alpha = parameters['alpha']
        else:
            raise Exception('alpha can be a number or an array of size M')

        self.beta = parameters['beta']
        self.gIa = parameters['gIa']
        self.gIs = parameters['gIs']
        self.fsa = parameters['fsa']

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

    cdef double obtain_log_p_for_traj_red(self, double [:] x0, double [:, :] obs, np.ndarray fltr,
                                            double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t reduced_dim=(Nf-1)*int(np.sum(fltr))
            double [:, :] xm
            double [:] xm_red, dev, obs_flattened
            np.ndarray[BOOL_t, ndim=1, cast=True] full_fltr

        xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, model, contactMatrix)
        full_fltr = np.tile(fltr, (Nf-1))
        cov_red = full_cov[full_fltr][:, full_fltr]
        obs_flattened = np.ravel(obs)
        xm_red = np.ravel(np.compress(fltr, xm, axis=1))
        dev=np.subtract(obs_flattened, xm_red)
        cov_red_inv=np.linalg.inv(cov_red)
        log_p= - (dev@cov_red_inv@dev)*(self.N/2)
        sign,ldet=np.linalg.slogdet(cov_red)
        if sign <0:
            raise ValueError('Cov has negative determinant')
        log_p -= (ldet-reduced_dim*log(self.N))/2 + (reduced_dim/2)*log(2*PI)
        return -log_p

    cdef double obtain_log_p_for_traj_matrix_fltr(self, double [:] x0, double [:, :] obs, np.ndarray fltr,
                                            double Tf, Py_ssize_t Nf, model, contactMatrix, tangent=False):
        cdef:
            Py_ssize_t reduced_dim=(Nf-1)*fltr.shape[0]
            double [:, :] xm
            double [:] xm_red, dev, obs_flattened
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf, model, contactMatrix)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf, model, contactMatrix)
        full_fltr = sparse.block_diag([fltr,]*(Nf-1))
        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        obs_flattened = np.ravel(obs)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        cov_red_inv=np.linalg.inv(cov_red)
        log_p= - (dev@cov_red_inv@dev)*(self.N/2)
        sign,ldet=np.linalg.slogdet(cov_red)
        if sign <0:
            raise ValueError('Cov has negative determinant')
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
            double [:, :] invcov
            double log_cond_p
            double det
        invcov = np.linalg.inv(cov)
        sign, ldet = np.linalg.slogdet(cov)
        if sign < 0:
            raise ValueError('Cov has negative determinant')
        log_cond_p = - np.dot(x, np.dot(invcov, x))*(self.N/2) - (self.dim/2)*log(2*PI)
        log_cond_p -= (ldet - self.dim*log(self.N))/2
        return log_cond_p

    cdef estimate_cond_mean_cov(self, double [:] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] cov
            double [:, :] x
            double [:] time_points = np.linspace(t1, t2, self.steps)
            np.ndarray sigma0 = np.zeros((self.vec_size), dtype=DTYPE)
            double first_step

        x = self.integrate(x0, t1, t2, self.steps, model, contactMatrix, maxNumSteps=self.steps*2)
        spline = make_interp_spline(time_points, x)

        def rhs(t, sig):
            self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
            self.lyapunov_fun(t, sig, spline)
            return self.dsigmadt
        first_step = (t2-t1)/self.steps
        res = solve_ivp(rhs, (t1, t2), sigma0, method='RK23', t_eval=np.array([t2]), first_step=first_step, max_step=self.steps, rtol=1e-1)
        cov = res.y
        return x[self.steps-1], self.convert_vec_to_mat(cov[0])

    cdef estimate_dx_and_cov(self, double [:] xt, double t, double dt, model, contactMatrix):
        cdef:
            double [:] dx_det
            double [:, :] cov
        model.set_contactMatrix(t, contactMatrix)
        model.rhs(xt, t)
        dx_det = np.multiply(dt, model.dxdt)
        self.compute_tangent_space_variables(xt, t, contactMatrix)
        cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
        return dx_det, cov

    cpdef obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf
            double [:, :] cov
            np.ndarray[DTYPE_t, ndim=2] invcov, temp
            double ti, tf
        xm[0]=x0
        full_cov_inv=[[None]*(Nf-1) for i in range(Nf-1)]
        for i in range(Nf-1):
            ti = time_points[i]
            tf = time_points[i+1]
            xi = xm[i]
            xf, cov = self.estimate_cond_mean_cov(xi, ti, tf, model, contactMatrix)
            self.obtain_time_evol_op(xi, xf, ti, tf, model, contactMatrix)
            invcov=np.linalg.inv(cov)
            full_cov_inv[i][i]=invcov
            if i>0:
                temp = invcov@self.U
                full_cov_inv[i-1][i-1] += np.transpose(self.U)@temp
                full_cov_inv[i-1][i]=-np.transpose(self.U)@invcov
                full_cov_inv[i][i-1]=-temp
            xm[i+1]=xf
        full_cov_inv=sparse.bmat(full_cov_inv, format='csc').todense()
        full_cov=np.linalg.inv(full_cov_inv)
        return xm[1:], full_cov # returns mean and cov for all but first (fixed!) time point

    cdef obtain_full_mean_cov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xt
            double [:, :] cov, U, J_dt
            np.ndarray[DTYPE_t, ndim=2] invcov, temp
            double t, dt=time_points[1]
        xm = self.integrate(x0, 0, Tf, Nf, model, contactMatrix)
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
        full_cov=np.linalg.inv(full_cov_inv)
        return xm[1:], full_cov # returns mean and cov for all but first (fixed!) time point

    cdef obtain_time_evol_op(self, double [:] x0, double [:] xf, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] U=self.U
            double epsilon=1./self.N
            Py_ssize_t i, j
        for i in range(self.dim):
            x0[i] += epsilon
            pos = self.integrate(x0, t1, t2, 2, model, contactMatrix)[1]
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
            cov_mat[i, i] =cov[count]
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

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, maxNumSteps=100000):
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

        Returns
        -------
        sol: np.array
            The state of the system evaulated at the time point specified.
        """
        cdef:
            double [:, :] sol
            double [:] time_points = np.linspace(t1, t2, steps)
        def rhs0(t, xt):
            model.set_contactMatrix(t, contactMatrix)
            model.rhs(xt, t)
            return model.dxdt
        sol = solve_ivp(rhs0, [t1,t2], x0, method='RK23', t_eval=time_points, max_step=maxNumSteps).y.T
        return sol

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
    """

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 3, M, fi, N, steps)

    def make_det_model(self, parameters):
        return pyross.deterministic.SIR(parameters, self.M, self.fi)

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
        self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        self.noise_correlation(s, Ia, Is, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double fsa=self.fsa, beta=self.beta
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+fsa*Is[n])

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double gIa=self.gIa, gIs=self.gIs, fsa=self.fsa, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = alpha[m]*l[m]
            J[2, m, 0, m] = balpha[m]*l[m]
            for n in range(M):
                J[0, m, 1, n] = -s[m]*beta*CM[m, n]
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 1, n] = alpha[m]*s[m]*beta*CM[m, n]
                J[1, m, 2, n] = alpha[m]*s[m]*beta*CM[m, n]*fsa
                J[2, m, 1, n] = balpha[m]*s[m]*beta*CM[m, n]
                J[2, m, 2, n] = balpha[m]*s[m]*beta*CM[m, n]*fsa
            J[1, m, 1, m] -= gIa
            J[2, m, 2, m] -= gIs
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double gIa=self.gIa, gIs=self.gIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - alpha[m]*l[m]*s[m]
            B[1, m, 1, m] = alpha[m]*l[m]*s[m] + gIa*Ia[m]
            B[0, m, 2, m] = - balpha[m]*l[m]*s[m]
            B[2, m, 2, m] = balpha[m]*l[m]*s[m] + gIs*Is[m]
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
    """

    cdef:
        readonly double gE

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 4, M, fi, N, steps)

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE = parameters['gE']

    def make_det_model(self, parameters):
        return pyross.deterministic.SEIR(parameters, self.M, self.fi)


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
        self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(Ia, Is, l)
        self.noise_correlation(s, e, Ia, Is, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double fsa=self.fsa, beta=self.beta
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+fsa*Is[n])

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double gIa=self.gIa, gIs=self.gIs, gE=self.gE, fsa=self.fsa, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE
            J[2, m, 1, m] = alpha[m]*gE
            J[2, m, 2, m] = - gIa
            J[3, m, 1, m] = balpha[m]*gE
            J[3, m, 3, m] = - gIs
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]
                J[0, m, 3, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 2, n] = s[m]*beta*CM[m, n]
                J[2, m, 3, n] = s[m]*beta*CM[m, n]*fsa
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double gIa=self.gIa, gIs=self.gIs, gE=self.gE
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + gE*e[m]
            B[1, m, 2, m] = -alpha[m]*gE*e[m]
            B[1, m, 3, m] = -balpha[m]*gE*e[m]
            B[2, m, 2, m] = alpha[m]*gE*e[m]+gIa*Ia[m]
            B[3, m, 3, m] = balpha[m]*gE*e[m]+gIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEAI5R(SIR_type):
    """
    Susceptible, Exposed, Activates, Infected, Removed (SEAIR). The infected class has 5 groups:

    * E: exposed
    * A: activated
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality

    To initialise the SEAI5R class,

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
            rate of removal from exposeds individuals.
        gA: float
            rate of removal from activated individuals.
        gIh: float
            rate of hospitalisation of infected individuals.
        gIc: float
            rate hospitalised individuals are moved to intensive care.
        hh: np.array (M,)
            fraction hospitalised from Is
        cc: np.array (M,)
            fraction sent to intensive care from hospitalised.
        mm: np.array (M,)
            mortality rate in intensive care
    M: int
        Number of age groups
    fi: float numpy.array
        Fraction of each age group
    N: int
        Total population
    """
    cdef:
        readonly double gE, gA, gIh, gIc, fh
        readonly np.ndarray hh, cc, mm

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 8, M, fi, N, steps)

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE    = parameters.get('gE')                       # removal rate of E class
        self.gA    = parameters.get('gA')                       # removal rate of A class
        self.gIh   = parameters.get('gIh')                      # removal rate of Is
        self.gIc   = parameters.get('gIc')                      # removal rate of Ih
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics
        self.fh    = parameters.get('fh')                       # the self-isolation parameter of hospitalizeds

        hh = parameters.get('hh')
        cc = parameters.get('cc')
        mm = parameters.get('mm')

        self.hh    = np.zeros(self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(self.M)
        elif np.size(hh)==self.M:
            self.hh= hh
        else:
            print('hh can be a number or an array of size M')

        self.cc    = np.zeros(self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(self.M)
        elif np.size(cc)==self.M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        self.mm    = np.zeros(self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(self.M)
        elif np.size(mm)==self.M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')

    def make_det_model(self, parameters):
        return pyross.deterministic.SEAI5R(parameters, self.M, self.fi)

    def make_params_dict(self, params=None):
        if params is None:
            parameters = {'alpha':self.alpha,
                          'beta':self.beta,
                          'gIa':self.gIa,
                          'gIs':self.gIs,
                          'gE': self.gE,
                          'gA': self.gA,
                          'gIh': self.gIh,
                          'gIc': self.gIc,
                          'fsa':self.fsa,
                          'fh': self.fh,
                          'sa': 0,
                          'hh': self.hh,
                          'cc': self.cc,
                          'mm': self.mm}

        else:
            parameters = {'alpha':params[0],
                      'beta':params[1],
                      'gIa':params[2],
                      'gIs':params[3],
                      'gE': params[4],
                      'gA': params[5],
                      'gIh': self.gIh,
                      'gIc': self.gIc,
                      'fsa':self.fsa,
                      'fh': self.fh,
                      'sa': 0,
                      'hh': self.hh,
                      'cc': self.cc,
                      'mm': self.mm}
        return parameters

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Ih':5, 'Ic':6, 'Im':7}


    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x, s, e, a, Ia, Is, Ih, Ic, Im
            Py_ssize_t M=self.M
        x = spline(t)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        Ih = x[5*M:6*M]
        Ic = x[6*M:7*M]
        Im = x[7*M:8*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, Ih, l)
        self.jacobian(s, l)
        self.noise_correlation(s, e, a, Ia, Is, Ih, Ic, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef:
            double [:] s, e, a, Ia, Is, Ih, Ic, Im
            Py_ssize_t M=self.M
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        Ih = x[5*M:6*M]
        Ic = x[6*M:7*M]
        Im = x[7*M:8*M]
        self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, Ih, l)
        self.noise_correlation(s, e, a, Ia, Is, Ih, Ic, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] a, double [:] Ia, double [:] Is, double [:] Ih, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double fsa=self.fsa, fh=self.fh, beta=self.beta
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+a[n]+fsa*Is[n]+fh*Ih[n])


    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double gIa=self.gIa, gIs=self.gIs, gIh=self.gIh, gIc=self.gIc
            double gE=self.gE, gA=self.gA, fsa=self.fsa, fh=self.fh, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:] hh=self.hh, cc=self.cc, mm=self.mm
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE
            J[2, m, 1, m] = gE
            J[2, m, 2, m] = - gA
            J[3, m, 2, m] = alpha[m]*gA
            J[3, m, 3, m] = - gIa
            J[4, m, 2, m] = balpha[m]*gA
            J[4, m, 4, m] = -gIs
            J[5, m, 4, m] = hh[m]*gIs
            J[5, m, 5, m] = -gIh
            J[6, m, 5, m] = cc[m]*gIh
            J[6, m, 6, m] = -gIc
            J[7, m, 6, m] = mm[m]*gIc
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]
                J[0, m, 3, n] = -s[m]*beta*CM[m, n]
                J[0, m, 4, n] = -s[m]*beta*CM[m, n]*fsa
                J[0, m, 5, n] = -s[m]*beta*CM[m, n]*fh
                J[1, m, 2, n] = s[m]*beta*CM[m, n]
                J[1, m, 3, n] = s[m]*beta*CM[m, n]
                J[1, m, 4, n] = s[m]*beta*CM[m, n]*fsa
                J[1, m, 5, n] = s[m]*beta*CM[m, n]*fh
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] Ih, double [:] Ic, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double gIa=self.gIa, gIs=self.gIs, gIh=self.gIh, gIc=self.gIc, gE=self.gE, gA=self.gA
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:] mm=self.mm, cc=self.cc, hh=self.hh
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + gE*e[m]
            B[1, m, 2, m] = -gE*e[m]
            B[2, m, 2, m] = gE*e[m]+gA*a[m]
            B[2, m, 3, m] = -alpha[m]*gA*a[m]
            B[2, m, 4, m] = -balpha[m]*gA*a[m]
            B[3, m, 3, m] = alpha[m]*gA*a[m]+gIa*Ia[m]
            B[4, m, 4, m] = balpha[m]*gA*a[m] + gIs*Is[m]
            B[4, m, 5, m] = -hh[m]*gIs*Is[m]
            B[5, m, 5, m] = hh[m]*gIs*Is[m] + gIh*Ih[m]
            B[5, m, 6, m] = -cc[m]*gIh*Ih[m]
            B[6, m, 6, m] = cc[m]*gIh*Ih[m] + gIc*Ic[m]
            B[6, m, 7, m] = -mm[m]*gIc*Ic[m]
            B[7, m, 7, m] = mm[m]*gIc*Ic[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, maxNumSteps=100000):
        cdef:
            double [:, :] sol
            double [:] time_points=np.linspace(t1, t2, steps), init
            Py_ssize_t M=self.M
        def rhs0(t, xt):
            model.set_contactMatrix(t, contactMatrix)
            model.rhs(xt, t)
            return model.dxdt
        init = np.concatenate(x0, self.fi)
        first_step = (t2-t1)/steps
        sol = solve_ivp(rhs0, [t1,t2], init, method='RK23', t_eval=time_points, first_step=first_step, max_step=maxNumSteps).y.T
        return sol[:, :8*M]

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
    """

    cdef:
        readonly double gE, gA, tE, tA, tIa, tIs

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 6, M, fi, N, steps)

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE    = parameters.get('gE')                       # removal rate of E class
        self.gA    = parameters.get('gA')                       # removal rate of A class
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics
        # testing rate, note that we do not account for false positive here (no tau_S)
        self.tE    = parameters.get('tE')                       # testing rate in E
        self.tA    = parameters.get('tA')                       # testing rate in A
        self.tIa   = parameters.get('tIa')                       # testing rate in Ia
        self.tIs   = parameters.get('tIs')                      # testing rate in Is

    def make_det_model(self, parameters):
        return pyross.deterministic.SEAIRQ(parameters, self.M, self.fi)

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
        self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(a, Ia, Is, l)
        self.noise_correlation(s, e, a, Ia, Is, q, l)
        if jacobian:
            self.jacobian(s, l)

    cdef fill_lambdas(self, double [:] a, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double fsa=self.fsa, beta=self.beta
            Py_ssize_t m, n, M=self.M
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+a[n]+fsa*Is[n])

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M, dim=self.dim
            double gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs, beta=self.beta
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE - tE
            J[2, m, 1, m] = gE
            J[2, m, 2, m] = - gA - tA
            J[3, m, 2, m] = alpha[m]*gA
            J[3, m, 3, m] = - gIa - tIa
            J[4, m, 2, m] = balpha[m]*gA
            J[4, m, 4, m] = -gIs - tIs
            J[5, m, 1, m] = tE
            J[5, m, 2, m] = tA
            J[5, m, 3, m] = tIa
            J[5, m, 4, m] = tIs
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]
                J[0, m, 3, n] = -s[m]*beta*CM[m, n]
                J[0, m, 4, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 2, n] = s[m]*beta*CM[m, n]
                J[1, m, 3, n] = s[m]*beta*CM[m, n]
                J[1, m, 4, n] = s[m]*beta*CM[m, n]*fsa
        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double beta=self.beta, gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA
            double tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + (gE+tE)*e[m]
            B[1, m, 2, m] = -gE*e[m]
            B[2, m, 2, m] = gE*e[m]+(gA+tA)*a[m]
            B[2, m, 3, m] = -alpha[m]*gA*a[m]
            B[2, m, 4, m] = -balpha[m]*gA*a[m]
            B[3, m, 3, m] = alpha[m]*gA*a[m]+(gIa+tIa)*Ia[m]
            B[4, m, 4, m] = balpha[m]*gA*a[m] + (gIs+tIs)*Is[m]
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
cdef class SEAIRQ_testing(SIR_type):
    """
    Susceptible, Exposed, Asymptomatic and infected, Infected, Removed, Quarantined (SEAIRQ)
    Ia: asymptomatic
    Is: symptomatic
    A : Asymptomatic and infectious

    Attributes
    ----------

    N : int
        Total popuation.
    M : int
        Number of compartments of individual for each class.
    steps : int
        Number of internal integration points used for interpolation.
    dim : int
        6 * M.
    fi : np.array(M)
        Age group size as a fraction of total population
    alpha : float or np.array(M)
        Fraction of infected who are asymptomatic.
    beta : float
        Rate of spread of infection.
    gIa : float
        Rate of removal from asymptomatic individuals.
    gIs : float
        Rate of removal from symptomatic individuals.
    gE : float
        rate of removal from exposed individuals.
    gA : float
        rate of removal from activated individuals.
    fsa : float
        fraction by which symptomatic individuals self isolate.
    ars : float
        fraction of population admissible for random and symptomatic tests
    kapE : float
        fraction of positive tests for exposed individuals
    testRate: python function
        number of tests per day and age group

    Methods
    -------
    All methods of the superclass SIR_type
    make_det_model : returns deterministic model
    make_params_dict : returns a dictionary of the input parameters
    set_tesRate : update testRate function
    integrate : returns numerical integration of the chosen model
    """
    cdef:
        readonly double gE, gA, tE, tA, tIa, tIs, ars, kapE
        readonly object testRate

    def __init__(self, parameters, testRate, M, fi, N, steps):
        super().__init__(parameters, 6, M, fi, N, steps)
        self.testRate=testRate

    def get_init_keys_dict(self):
        return {'S':0, 'E':1, 'A':2, 'Ia':3, 'Is':4, 'Q':5}


    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE    = parameters.get('gE')                       # removal rate of E class
        self.gA    = parameters.get('gA')                       # removal rate of A class
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics
        self.ars    = parameters.get('ars')                     # fraction of population admissible for testing
        self.kapE    = parameters.get('kapE')                   # fraction of positive tests for exposed

    def set_testRate(self,testRate):
        self.testRate=testRate

    def make_det_model(self, parameters):
        # note: unline the other classes in inference, SEAIRQ_testing works with extensive variables
        return pyross.deterministic.SEAIRQ_testing(parameters, self.M, self.fi*self.N)


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
            double [:, :] CM=self.CM
            double beta=self.beta, fsa=self.fsa
            Py_ssize_t m, n, M=self.M
        x = spline(t)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        TR=self.testRate(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+a[n]+fsa*Is[n])
        self.jacobian(s, e, a, Ia, Is, q, l, TR)
        self.noise_correlation(s, e, a, Ia, Is, q, l, TR)
        self.flatten_lyaponuv()
        self.compute_dsigdt(sig)

    cdef jacobian(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l, double [:] TR):
        cdef:
            Py_ssize_t m, n, M=self.M, N=self.N
            double gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double ars=self.ars, kapE=self.kapE, beta=self.beta
            double t0, tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            t0 = 1./(ars*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars*kapE*t0/N
            tA= TR[m]*ars*t0/N
            tIa = TR[m]*ars*t0/N
            tIs = TR[m]*t0/N

            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]
                J[0, m, 3, n] = -s[m]*beta*CM[m, n]
                J[0, m, 4, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 2, n] = s[m]*beta*CM[m, n]
                J[1, m, 3, n] = s[m]*beta*CM[m, n]
                J[1, m, 4, n] = s[m]*beta*CM[m, n]*fsa
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE - tE
            J[1, m, 4, m] += (1-ars)*tE*t0*e[m]
            J[1, m, 5, m] = -ars*tE*t0*e[m]
            J[2, m, 1, m] = gE
            J[2, m, 2, m] = - gA - tA
            J[2, m, 4, m] = (1-ars)*tA*t0*a[m]
            J[2, m, 5, m] = - ars*tA*t0*a[m]
            J[3, m, 2, m] = alpha[m]*gA
            J[3, m, 3, m] = - gIa - tIa
            J[3, m, 4, m] = (1-ars)*tIa*t0*Ia[m]
            J[3, m, 5, m] = - ars*tIa*t0*Ia[m]
            J[4, m, 2, m] = balpha[m]*gA
            J[4, m, 4, m] = - gIs - tIs + (1-ars)*tIs*t0*Is[m]
            J[4, m, 5, m] = - ars*tIs*t0*Is[m]
            J[5, m, 1, m] = tE
            J[5, m, 2, m] = tA
            J[5, m, 3, m] = tIa
            J[5, m, 4, m] = tIs - (1-ars)*t0*(tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m])
            J[5, m, 5, m] = ars*t0*(tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m])


    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l, double [:] TR):
        cdef:
            Py_ssize_t m, M=self.M
            double beta=self.beta, gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA,  N=self.N
            double ars=self.ars, kapE=self.kapE
            double tE, tA, tIa, tIs
            double [:] alpha=self.alpha, balpha=1-self.alpha
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            t0 = 1./(ars*(self.fi[m]-q[m]-Is[m])+Is[m])
            tE = TR[m]*ars*kapE*t0/N
            tA= TR[m]*ars*t0/N
            tIa = TR[m]*ars*t0/N
            tIs = TR[m]*t0/N

            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + (gE+tE)*e[m]
            B[1, m, 2, m] = -gE*e[m]
            B[2, m, 2, m] = gE*e[m]+(gA+tA)*a[m]
            B[2, m, 3, m] = -alpha[m]*gA*a[m]
            B[2, m, 4, m] = -balpha[m]*gA*a[m]
            B[3, m, 3, m] = alpha[m]*gA*a[m]+(gIa+tIa)*Ia[m]
            B[4, m, 4, m] = balpha[m]*gA*a[m] + (gIs+tIs)*Is[m]
            B[1, m, 5, m] = -tE*e[m]
            B[2, m, 5, m] = -tA*a[m]
            B[3, m, 5, m] = -tIa*Ia[m]
            B[4, m, 5, m] = -tIs*Is[m]
            B[5, m, 5, m] = tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix, maxNumSteps=100000):
        cdef:
            np.ndarray S, E, A, Ia, Is, Q
            double N=self.N
            double [:, :] sol
            Py_ssize_t M=self.M
        # need to switch to extensive variables to calculate test rate correctly in deterministic model
        S = np.asarray(x0[0:M])*N
        E = np.asarray(x0[M:2*M])*N
        A = np.asarray(x0[2*M:3*M])*N
        Ia = np.asarray(x0[3*M:4*M])*N
        Is = np.asarray(x0[4*M:5*M])*N
        Q = np.asarray(x0[5*M:])*N
        data = model.simulate(S, E, A, Ia, Is, Q, contactMatrix, self.testRate, t2, steps, Ti=t1, maxNumSteps=maxNumSteps)
        sol = data['X']/N
        return sol

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Spp(SIR_type):
    """
    User-defined epidemic model.

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
        Number of internal steps used.

    See `SIR_type` for a table of all the methods

    Examples
    --------
    An example of model_spec and parameters for SIR class.

    >>> model_spec = {
            "classes" : ["S", "I"],
            "S" : {
                "linear"    : [],
                "infection" : [ ["I", "-beta"] ]
            },
            "I" : {
                "linear"    : [ ["I", "-gamma"] ],
                "infection" : [ ["I", "beta"] ]
            }
        }
    >>> parameters = {
            'beta': 0.1,
            'gamma': 0.1
        }
    """

    cdef:
        readonly np.ndarray linear_terms, infection_terms
        readonly np.ndarray parameters
        readonly list param_keys
        readonly dict class_index_dict
        readonly pyross.deterministic.Spp det_model


    def __init__(self, model_spec, parameters, M, fi, N, steps):
        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.linear_terms = res[2]
        self.infection_terms = res[3]
        super().__init__(parameters, self.nClass, M, fi, N, steps)
        self.det_model = pyross.deterministic.Spp(model_spec, parameters, M, fi)


    def set_params(self, parameters):
        nParams = len(parameters)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        for (i, param) in enumerate(parameters.values()):
            if type(param) == list:
                param = np.array(param)

            if type(param) == np.ndarray:
                if param.size != self.M:
                    raise Exception("Parameter array size must be equal to M.")
            else:
                param = np.full(self.M, param)
            self.parameters[i] = param

    def make_det_model(self, parameters):
        # small hack to make this class work with SIR_type
        self.det_model.update_model_parameters(parameters)
        return self.det_model


    def make_params_dict(self, params=None):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    cdef lyapunov_fun(self, double t, double [:] sig, spline):
        cdef:
            double [:] x
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
        x = spline(t)
        cdef double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
        self.B = np.zeros((self.nClass, self.M, self.nClass, self.M), dtype=DTYPE)
        self.J = np.zeros((self.nClass, self.M, self.nClass, self.M), dtype=DTYPE)
        self.fill_lambdas(x, l)
        self.jacobian(x, l)
        self.noise_correlation(x, l)
        self.compute_dsigdt(sig)

    cdef compute_tangent_space_variables(self, double [:] x, double t, contactMatrix, jacobian=False):
        cdef Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
        self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
        cdef double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
        self.B = np.zeros((self.nClass, self.M, self.nClass, self.M), dtype=DTYPE)
        self.fill_lambdas(x, l)
        self.noise_correlation(x, l)
        if jacobian:
            self.jacobian(x, l)

    cdef fill_lambdas(self, double [:] x, double [:, :] l):
        cdef:
            double [:, :] CM=self.CM
            int [:, :] infection_terms=self.infection_terms
            double infection_rate
            Py_ssize_t m, n, i, infective_index, index, M=self.M, num_of_infection_terms=infection_terms.shape[0]
        for i in range(num_of_infection_terms):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                for n in range(M):
                    index = n + M*infective_index
                    l[i, m] += CM[m,n]*x[index]

    cdef jacobian(self, double [:] x, double [:, :] l):
        cdef:
            Py_ssize_t i, m, n, M=self.M, dim=self.dim
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            double [:] rate
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
                    J[S_index, m, infective_index, n] -= x[S_index*M+m]*rate[m]*CM[m, n]
                    if product_index>-1:
                        J[product_index, m, infective_index, n] += x[S_index*M+m]*rate[m]*CM[m, n]
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
            Py_ssize_t i, m, n, M=self.M
            Py_ssize_t rate_index, infective_index, product_index, reagent_index, S_index=self.class_index_dict['S']
            double [:, :, :, :] B=self.B
            double [:, :] CM=self.CM
            double [:, :] parameters=self.parameters
            int [:, :] linear_terms=self.linear_terms, infection_terms=self.infection_terms
            double [:] s, reagent, rate
        s = x[S_index*M:(S_index+1)*M]
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
