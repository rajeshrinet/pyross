from itertools import compress
from scipy import sparse
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, approx_fprime
from scipy.stats import lognorm
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import eig
cimport numpy as np
cimport cython
import time, sympy
from sympy import MutableDenseNDimArray as Array
from sympy import Inverse

try:
    # Optional support for nested sampling.
    import nestle
except ImportError:
    nestle = None


import pyross.deterministic
cimport pyross.deterministic
import pyross.contactMatrix
from pyross.utils_python import minimization, nested_sampling
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
        readonly double Omega
        readonly np.ndarray beta, gIa, gIs, fsa
        readonly np.ndarray alpha, fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols
        readonly str det_method, lyapunov_method
        readonly dict class_index_dict
        readonly list param_keys
        readonly object contactMatrix


    def __init__(self, parameters, nClass, M, fi, Omega, steps, det_method, lyapunov_method):
        self.Omega = Omega
        self.M = M
        self.fi = fi
        if steps < 4:
            raise Exception('Steps must be at least 4 for internal spline interpolation.')
        self.steps = steps
        self.set_params(parameters)
        self.det_method=det_method
        self.lyapunov_method=lyapunov_method

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

    def _infer_params_minus_logp(self, params, grad=0, keys=None,
                               is_scale_parameter=None, scaled_guesses=None,
                               flat_guess_range=None, x=None, Tf=None,
                               s=None, scale=None, tangent=None):
        """Objective function for minimization call in infer_parameters."""
        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params, flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
        parameters = self.fill_params_dict(keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)
        if tangent:
            minus_logp = self._obtain_logp_for_traj_tangent(x, Tf)
        else:
            minus_logp = self._obtain_logp_for_traj(x, Tf)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def symbolic_test(self):
        x  =  sympy.symbols('x')
        expr = sympy.sin(x)
        diff = sympy.diff(expr, x)
        return diff, float(x.subs(x, 2))

    def infer_parameters(self, x, Tf, contactMatrix, prior_dict,
                        tangent=False, verbose=False,
                        enable_global=True, global_max_iter=100, global_atol=1,
                        enable_local=True, local_max_iter=200, ftol=1e-6,
                        cma_processes=0, cma_population=16):
        """Infers the MAP estimates for epidemiological parameters

        Parameters
        ----------
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
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimization algorithm.
            Default is False.
        enable_global: bool, optional
            Set to True to perform global optimization. Default is True.
        enable_local: bool, optional
            Set to True to perform local optimization. Default is True.
        global_max_iter: int, optional
            The maximum number of iterations for the global algorithm.
        global_atol: float, optional
            Absolute tolerance of global optimization. Default is 1.
        cma_processes: int, optional
            Number of parallel processes used in the CMA algorithm.
            Default is to use all cores on the computer.
        cma_population: int, optional
            he number of samples used in each step of the CMA algorithm.
            Should ideally be factor of `cma_processes`.
        local_max_iter: int, optional
            The maximum number of iterations for the local algorithm.
        ftol: float, optional
            The relative tolerance in -logp value for the local optimization.

        Returns
        -------
        output: dict
            Contains the following keys for users:

            map_dict: dict
                A dictionary for MAPs. Keys are the names of the parameters and
                the corresponding values are its MAP estimates.
            -logp: float
                The value of -logp at MAP.

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
                    'scale_factor_bounds': [0.1, 10]
                },
                'beta':{
                    'mean': 0.02,
                    'std': 0.1,
                    'bounds': [1e-4, 1]
                }
            }
        """
        # Read in the priors
        keys, guess, stds, bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = pyross.utils.parse_param_prior_dict(prior_dict, self.M)
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        self.contactMatrix = contactMatrix
        cma_stds = np.minimum(stds, (bounds[:, 1] - bounds[:, 0])/3)

        minimize_args={'keys':keys, 'is_scale_parameter':is_scale_parameter,
                       'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                       'x':x, 'Tf':Tf, 's':s, 'scale':scale, 'tangent':tangent}
        res = minimization(self._infer_params_minus_logp, guess, bounds,
                           ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local,
                           cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds,
                           verbose=verbose, args_dict=minimize_args)
        params = res[0]
        # Get the parameters (in their original structure) from the flattened parameter vector.
        orig_params = pyross.utils.unflatten_parameters(params, flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
        output_dict = {
            'map_dict':self.fill_params_dict(keys, orig_params),
            'flat_map':params, '-logp':res[1], 'keys': keys,
            'is_scale_parameter':is_scale_parameter,
            'flat_guess_range':flat_guess_range,
            'scaled_guesses':scaled_guesses,
            's':s, 'scale':scale
        }
        return output_dict

    def _nested_sampling_prior_transform(self, x, s=None, scale=None, ppf_bounds=None):
        # Tranform a sample x ~ Unif([0,1]^d) to a sample of the prior using inverse tranform sampling.
        y = ppf_bounds[:,0] + x * ppf_bounds[:,1]
        return lognorm.ppf(y, s, scale=scale)

    def _nested_sampling_loglike(self, params, keys=None,
                               is_scale_parameter=None, scaled_guesses=None,
                               flat_guess_range=None, x=None, Tf=None,
                               s=None, scale=None, tangent=None):
        orig_params = pyross.utils.unflatten_parameters(params, flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
        parameters = self.fill_params_dict(keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)
        if tangent:
            minus_logp = self._obtain_logp_for_traj_tangent(x, Tf)
        else:
            minus_logp = self._obtain_logp_for_traj(x, Tf)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return -minus_logp

    def nested_sampling_inference(self, x, Tf, contactMatrix, prior_dict, tangent=False, verbose=False,
                                  queue_size=1, max_workers=None, npoints=100, method='single', max_iter=1000,
                                  dlogz=None, decline_factor=None):
        '''Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        using nested sampling as implemented in the `nestle` Python package. This function assumes that full data on
        all classes is available.

        This function provides a computational alterantive to `log_G_evidence` and `infer_parameters`. It does not use
        the Laplace approximation to compute the evidence and, in addition,  returns a set of representative samples that can
        be used to compute a posterior mean estimate (insted of the MAP estimate). This approach approach is much more resource
        intensive and typically only viable for small models or tangent space inference.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        prior_dict: dict
            A dictionary for the priors for the parameters.
            See `infer_parameters` for examples.
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        queue_size: int
            Size of the internal queue of samples of the nested sampling algorithm. The log-likelihood of these samples
            is computed in parallel (if queue_size > 1).
        max_workers: int
            The maximal number of processes used to compute samples.
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

        keys, guess, stds, bounds, flat_guess_range, is_scale_parameter, scaled_guesses \
            = pyross.utils.parse_param_prior_dict(prior_dict, self.M)
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        self.contactMatrix = contactMatrix

        k = len(guess)
        # We sample the prior by inverse transform sampling from the unite cube. To implement bounds (if supplied)
        # we shrink the unit cube according the the provided bounds in each dimension.
        ppf_bounds = np.zeros((k, 2))
        ppf_bounds[:,0] = lognorm.cdf(bounds[:,0], s, scale=scale)
        ppf_bounds[:,1] = lognorm.cdf(bounds[:,1], s, scale=scale)
        ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]

        prior_transform_args = {'s':s, 'scale':scale, 'ppf_bounds':ppf_bounds}
        loglike_args = {'keys':keys, 'is_scale_parameter':is_scale_parameter,
                       'scaled_guesses':scaled_guesses, 'flat_guess_range':flat_guess_range,
                       'x':x, 'Tf':Tf, 's':s, 'scale':scale, 'tangent':tangent}

        result = nested_sampling(self._nested_sampling_loglike, self._nested_sampling_prior_transform, k, queue_size,
                                 max_workers, verbose, method, npoints, max_iter, dlogz, decline_factor, loglike_args,
                                 prior_transform_args)

        log_evidence = result.logz

        output_samples = []
        for i in range(len(result.samples)):
            sample = result.samples[i]
            weight = result.weights[i]

            orig_sample = pyross.utils.unflatten_parameters(sample, flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
            output_dict = {
                'map_dict':self.fill_params_dict(keys, orig_sample),
                'flat_map':sample, 'weight':weight, 'keys': keys,
                'is_scale_parameter':is_scale_parameter,
                'flat_guess_range':flat_guess_range,
                'scaled_guesses':scaled_guesses,
                's':s, 'scale':scale
            }
            output_samples.append(output_dict)


        return log_evidence, output_samples


    def _infer_control_to_minimize(self, params, grad=0, keys=None,
                                   x=None, Tf=None, generator=None,
                                   intervention_fun=None, tangent=None,
                                   is_scale_parameter=None, flat_guess_range=None,
                                   scaled_guesses=None, s=None, scale=None):
        """Objective function for minimization call in infer_control."""
        orig_params = pyross.utils.unflatten_parameters(params, flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
        kwargs = {k:orig_params[i] for (i, k) in enumerate(keys)}
        if intervention_fun is None:
            self.contactMatrix=generator.constant_contactMatrix(**kwargs)
        else:
            self.contactMatrix=generator.intervention_custom_temporal(intervention_fun, **kwargs)
        if tangent:
            minus_logp = self._obtain_logp_for_traj_tangent(x, Tf)
        else:
            minus_logp = self._obtain_logp_for_traj(x, Tf)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp


    def infer_control(self, x, Tf, generator, prior_dict,
                      intervention_fun=None, tangent=False,
                      verbose=False, ftol=1e-6,
                      global_max_iter=100, local_max_iter=100, global_atol=1.,
                      enable_global=True, enable_local=True,
                      cma_processes=0, cma_population=16):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown. The lockdown is modelled by scaling the contact matrices for contact at work, school, and other
        (but not home). This function infers the scaling parameters (can be age dependent) assuming that full data
        on all classes is available (with latent variables, use `latent_infer_control`).

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        generator: pyross.contactMatrix
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
        prior_dict: dict
            Priors for intervention parameters.
            Same format as the prior_dict for epidemiological parameters in
            `infer_parameters` function.
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

        Returns
        -------
        output_dict: dict
            Dictionary of MAP estimates, containing the following keys for users:

            map_dict: dict
                Dictionary for MAP estimates of the control parameters.
            -logp: float
                Value of -logp at MAP.
        """
        keys, guess, stds, bounds, \
        flat_guess_range, is_scale_parameter, scaled_guesses  \
                = pyross.utils.parse_param_prior_dict(prior_dict, self.M)
        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        cma_stds = np.minimum(stds, (bounds[:, 1] - bounds[:, 0])/3)
        minimize_args = {'keys':keys, 'x':x, 'Tf':Tf,
                         'flat_guess_range':flat_guess_range,
                         'is_scale_parameter':is_scale_parameter,
                         'scaled_guesses': scaled_guesses,
                         'generator':generator, 's':s, 'scale':scale,
                          'intervention_fun': intervention_fun, 'tangent': tangent}
        res = minimization(self._infer_control_to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local, cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds, verbose=verbose, args_dict=minimize_args)

        orig_params = pyross.utils.unflatten_parameters(res[0], flat_guess_range,
                                             is_scale_parameter, scaled_guesses)
        map_dict = {k:orig_params[i] for (i, k) in enumerate(keys)}
        output_dict = {
            'map_dict': map_dict,
            'flat_map': res[0], '-logp':res[1], 'keys': keys,
            'is_scale_parameter':is_scale_parameter,
            'flat_guess_range':flat_guess_range,
            'scaled_guesses':scaled_guesses,
            's':s, 'scale':scale
        }
        return output_dict


    def compute_hessian(self, x, Tf, contactMatrix, map_dict, tangent=False,
                        eps=1.e-3, fd_method="central"):
        '''
        Computes the Hessian of the MAP estimate.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        contactMatrix: callable
            A function that takes time (t) as an argument and returns the contactMatrix
        map_dict: dict
            Dictionary returned by infer_parameters.
        eps: float or numpy.array, optional
            The step size of the Hessian calculation, default=1e-3
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        hess: 2d numpy.array
            The Hessian
        '''
        self.contactMatrix = contactMatrix
        flat_maps = map_dict['flat_map']
        kwargs = {}
        for key in ['flat_guess_range', 'is_scale_parameter', 'scaled_guesses', \
                    'keys', 's', 'scale']:
            kwargs[key] = map_dict[key]
        def minuslogp(y):
            return self._infer_params_minus_logp(y, x=x, Tf=Tf, tangent=tangent, **kwargs)
        hess = pyross.utils.hessian_finite_difference(flat_maps, minuslogp, eps, method=fd_method)
        return hess

    def robustness(self, FIM, FIM_det, map_dict, param_pos_1, param_pos_2,
                   range_1, range_2, resolution_1, resolution_2=None):
        '''
        Robustness analysis in a two-dimensional slice of the parameter space, revealing neutral spaces as in https://doi.org/10.1073/pnas.1015814108.

        Parameters
        ----------
        FIM: 2d numpy.array
            Fisher Information matrix of a stochastic model
        FIM_det: 2d numpy.array
            Fisher information matrix of the corresponding deterministic model
        map_dict: dict
            Dictionary returned by infer_parameters
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
        -------
        >>> from matplotlib import pyplot as plt
        >>> from matplotlib import cm
        >>>
        >>> # positions 0 and 1 of map_dict['flat_map'] correspond to a scale parameter for alpha, and beta, respectively.
        >>> ff, ss, Z_sto, Z_det = estimator.robustness(FIM, FIM_det, map_dict, 0, 1, 0.5, 0.01, 20)
        >>> cmap = plt.cm.PuBu_r
        >>> levels=11
        >>> colors='black'
        >>>
        >>> c = plt.contourf(ff, ss, Z_sto, cmap=cmap, levels=levels) # heat map for the stochastic coefficient
        >>> plt.contour(ff, ss, Z_sto, colors='black', levels=levels, linewidths=0.25)
        >>> plt.contour(ff, ss, Z_det, colors=colors, levels=levels) # contour plot for the deterministic model
        >>> plt.plot(map_dict['flat_map'][0], map_dict['flat_map'][1], 'o',
                    color="#A60628", markersize=6) # the MAP estimate
        >>> plt.colorbar(c)
        >>> plt.xlabel(r'$\alpha$ scale', fontsize=20, labelpad=10)
        >>> plt.ylabel(r'$\beta$', fontsize=20, labelpad=10)
        >>> plt.show()
        '''
        flat_maps = map_dict['flat_map']
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
        '''
        Computes the normalized sensitivity measure as defined in https://doi.org/10.1073/pnas.1015814108.

        Parameters
        ----------
        FIM: 2d numpy.array
            The Fisher Information Matrix

        Returns
        -------
        T_j: numpy.array
            Normalized sensitivity measure for parameters to be estimated. A larger entry translates into greater anticipated model sensitivity to changes in the parameter of interest.
        '''

        evals, evecs = eig(FIM)
        if np.any(np.real(evals)<0.):
            raise Exception('Negative eigenvalue - FIM is not positive definite. Check for appropriate step size eps in FIM computation.')
        L = np.diag(np.real(evals))
        S_ij = np.sqrt(L)@evecs
        S2_ij = S_ij**2
        S2_j = np.sum(S2_ij, axis=0)
        S2_norm = np.sum(S2_j)
        T_j = np.divide(S2_j,S2_norm)
        return T_j

    def FIM(self, obs, fltr, Tf, contactMatrix, map_dict, tangent=False,
            eps=None):
        '''
        Computes the Fisher Information Matrix (FIM) of the stochastic model.

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
           A function that takes time (t) as an argument and returns the contactMatrix
        map_dict: dict
           Dictionary returned by infer_parameters
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        eps: float or numpy.array, optional
            Step size for numerical differentiation of the process mean and its
            full covariance matrix with respect to the parameters.
            If not specified, the array of square roots of the machine epsilon of the MAP estimates is used.
            Decreasing the step size too small can result in round-off error.
        Returns
        -------
        FIM: 2d numpy.array
            The Fisher Information Matrix
        '''
        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)
        flat_maps = map_dict['flat_map']
        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs']:
            kwargs[key] = map_dict[key]

        def mean(y):
            return self._mean(y, obs=obs, fltr=fltr, Tf=Tf, obs0=obs0,
                              **kwargs)

        def covariance(y):
            return self._cov(y, obs=obs, fltr=fltr, Tf=Tf, obs0=obs0,
                             tangent=tangent, **kwargs)

        cov = covariance(flat_maps)

        if eps == None:
            eps = np.sqrt(np.spacing(flat_maps))
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_maps))

        print('eps-vector used for differentiation: ', eps)

        invcov = np.linalg.inv(cov)

        dim = len(flat_maps)
        FIM = np.zeros((dim,dim))

        rows,cols = np.triu_indices(dim)
        for i,j in zip(rows,cols):
            dmu_i = pyross.utils.partial_derivative(mean, var=i, point=flat_maps, dx=eps[i])
            dmu_j = pyross.utils.partial_derivative(mean, var=j, point=flat_maps, dx=eps[j])
            dcov_i = pyross.utils.partial_derivative(covariance, var=i, point=flat_maps, dx=eps[i])
            dcov_j = pyross.utils.partial_derivative(covariance, var=j, point=flat_maps, dx=eps[j])
            t1 = dmu_i@invcov@dmu_j
            t2 = np.multiply(0.5,np.trace(invcov@dcov_i@invcov@dcov_j))
            FIM[i,j] = t1 + t2
        i_lower = np.tril_indices(dim,-1)
        FIM[i_lower] = FIM.T[i_lower]
        return FIM

    def FIM_det(self, obs, fltr, Tf, contactMatrix, map_dict,
                eps=None, measurement_error=1e-2):
        '''
        Computes the Fisher Information Matrix (FIM) of the deterministic model.

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
           A function that takes time (t) as an argument and returns the contactMatrix
        map_dict: dict
           Dictionary returned by infer_parameters
        eps: float or numpy.array, optional
           Step size for numerical differentiation of the process mean and its full covariance matrix with respect
            to the parameters. If not specified, the array of square roots of the machine epsilon of the MAP estimates is used. Decreasing the step size too small can result in round-off error.
        measurement_error: float, optional
            Standard deviation of measurements (uniform and independent Gaussian measurement error assumed). Default is 1e-2.
        Returns
        -------
        FIM_det: 2d numpy.array
            The Fisher Information Matrix
        '''
        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)
        flat_maps = map_dict['flat_map']
        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs']:
            kwargs[key] = map_dict[key]

        def mean(y):
            return self._mean(y, obs=obs, fltr=fltr, Tf=Tf, obs0=obs0,
                              **kwargs)

        fltr_ = fltr[1:]
        sigma_sq = measurement_error*measurement_error
        cov_diag = np.repeat(sigma_sq, repeats=(int(self.dim)*(fltr_.shape[0])))
        cov = np.diag(cov_diag)
        full_fltr = sparse.block_diag(fltr_)
        cov_red = full_fltr@cov@np.transpose(full_fltr)
        invcov = np.linalg.inv(cov_red)

        if eps == None:
            eps = np.sqrt(np.spacing(flat_maps))
        elif np.isscalar(eps):
            eps = np.repeat(eps, repeats=len(flat_maps))

        print('eps-vector used for differentiation: ', eps)

        dim = len(flat_maps)
        FIM_det = np.zeros((dim,dim))

        rows,cols = np.triu_indices(dim)
        for i,j in zip(rows,cols):
            dmu_i = pyross.utils.partial_derivative(mean, var=i, point=flat_maps, dx=eps[i])
            dmu_j = pyross.utils.partial_derivative(mean, var=j, point=flat_maps, dx=eps[j])
            FIM_det[i,j] = dmu_i@invcov@dmu_j
        i_lower = np.tril_indices(dim,-1)
        FIM_det[i_lower] = FIM_det.T[i_lower]
        return FIM_det

    def _mean(self, params, grad=0, param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None, param_length=None,
                            obs=None, fltr=None, Tf=None, obs0=None,
                            init_flags=None, init_fltrs=None):
        """Objective function for differentiation call in FIM and FIM_det."""
        inits =  np.copy(params[param_length:])

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                          param_guess_range, is_scale_parameter, scaled_param_guesses)

        parameters = self.fill_params_dict(param_keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)
        fltr_ = fltr[1:]
        Nf=fltr_.shape[0]+1
        full_fltr = sparse.block_diag(fltr_)

        x0 = self._construct_inits(inits, init_flags, init_fltrs, obs0, fltr[0])
        xm = self.integrate(x0, 0, Tf, Nf, method='LSODA')
        xm_red = full_fltr@(np.ravel(xm[1:]))
        return xm_red

    def _cov(self, params, grad=0, param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None, param_length=None,
                            obs=None, fltr=None, Tf=None, obs0=None,
                            init_flags=None, init_fltrs=None, tangent=None):
        """Objective function for differentiation call in FIM."""
        inits =  np.copy(params[param_length:])

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                          param_guess_range, is_scale_parameter, scaled_param_guesses)

        parameters = self.fill_params_dict(param_keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)

        x0 = self._construct_inits(inits, init_flags, init_fltrs, obs0, fltr[0])
        fltr_ = fltr[1:]
        Nf=fltr_.shape[0]+1
        full_fltr = sparse.block_diag(fltr_)

        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf)

        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        return cov_red

    # def error_bars(self, keys, maps, prior_mean, prior_stds, x, Tf, Nf, contactMatrix, eps=1.e-3,
    #                tangent=False, infer_scale_parameter=False, fd_method="central"):
    #     hessian = self.compute_hessian(keys, maps, prior_mean, prior_stds, x,Tf,eps,
    #                                    tangent, infer_scale_parameter, fd_method=fd_method)
    #     return np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    def log_G_evidence(self, x, Tf, contactMatrix, map_dict, tangent=False, eps=1.e-3,
                       fd_method="central"):
        """Compute the evidence using a Laplace approximation at the MAP estimate.

        Parameters
        ----------
        x: 2d numpy.array
            Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
            Total time of the trajectory
        contactMatrix: callable
            A function that takes time (t) as an argument and returns the contactMatrix
        map_dict: dict
            MAP estimate returned by infer_parameters
        eps: float or numpy.array, optional
            The step size of the Hessian calculation, default=1e-3
        fd_method: str, optional
            The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        log_evidence: float
            The log-evidence computed via Laplace approximation at the MAP estimate."""
        logP_MAPs = -map_dict['-logp']
        A = self.compute_hessian(x, Tf, contactMatrix, map_dict, tangent, eps, fd_method)
        k = A.shape[0]

        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + k/2*np.log(2*np.pi)

    def obtain_minus_log_p(self, parameters, np.ndarray x, double Tf, contactMatrix, tangent=False):
        '''Computes -logp of a full trajectory
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
        '''

        cdef:
            double minus_log_p
            double [:, :] x_memview=x.astype('float')
        self.set_params(parameters)
        self.set_det_model(parameters)
        self.contactMatrix = contactMatrix
        if tangent:
            minus_logp = self._obtain_logp_for_traj_tangent(x_memview, Tf)
        else:
            minus_logp = self._obtain_logp_for_traj(x_memview, Tf)
        return minus_logp

    def _latent_minus_logp(self, params, grad=0, param_keys=None,
                            param_guess_range=None, is_scale_parameter=None,
                            scaled_param_guesses=None, param_length=None,
                            obs=None, fltr=None, Tf=None, obs0=None,
                            init_flags=None, init_fltrs=None,
                            s=None, scale=None, tangent=None):
        """Objective function for minimization call in laten_inference."""
        inits =  np.copy(params[param_length:])

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                          param_guess_range, is_scale_parameter, scaled_param_guesses)

        parameters = self.fill_params_dict(param_keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)

        x0 = self._construct_inits(inits, init_flags, init_fltrs, obs0, fltr[0])
        penalty = self._penalty_from_negative_values(x0)
        x0[x0<0] = 0.1/self.Omega # set to be small and positive

        minus_logp = self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))

        # add penalty for being negative
        minus_logp += penalty*fltr.shape[0]

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

    def latent_infer_parameters(self, np.ndarray obs, np.ndarray fltr, double Tf,
                            contactMatrix, param_priors, init_priors,
                            tangent=False, verbose=False,
                            double ftol=1e-5,
                            global_max_iter=100, local_max_iter=100, global_atol=1,
                            enable_global=True, enable_local=True, cma_processes=0,
                            cma_population=16):
        """
        Compute the maximum a-posteriori (MAP) estimate of the parameters and the initial conditions of a SIR type model
        when the classes are only partially observed. Unobserved classes are treated as latent variables.

        Parameters
        ----------
        obs:  np.array
            The partially observed trajectory.
        fltr: 2d np.array
            The filter for the observation such that
            :math:`F_{ij} x_j (t) = obs_i(t)`
        Tf: float
            The total time of the trajectory.
        contactMatrix: callable
            A function that returns the contact matrix at time t (input).
        param_priors: dict
            A dictionary that specifies priors for parameters.
            See `infer_parameters` for examples.
        init_priors: dict
            A dictionary that specifies priors for initial conditions.
            See below for examples.
        tangent: bool, optional
            Set to True to use tangent space inference. Default is False.
        verbose: bool, optional
            Set to True to see intermediate outputs from the optimization algorithm.
            Default is False.
        enable_global: bool, optional
            Set to True to perform global optimization. Default is True.
        enable_local: bool, optional
            Set to True to perform local optimization. Default is True.
        global_max_iter: int, optional
            The maximum number of iterations for the global algorithm.
        global_atol: float, optional
            Absolute tolerance of global optimization. Default is 1.
        cma_processes: int, optional
            Number of parallel processes used in the CMA algorithm.
            Default is to use all cores on the computer.
        cma_population: int, optional
            he number of samples used in each step of the CMA algorithm.
            Should ideally be factor of `cma_processes`.
        local_max_iter: int, optional
            The maximum number of iterations for the local algorithm.
        ftol: float, optional
            The relative tolerance in -logp value for the local optimization.

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

        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        keys, param_guess, param_stds, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M)

        # Read in initial conditions priors
        init_guess, init_stds, init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        cma_stds = np.minimum(stds, (bounds[:, 1]-bounds[:, 0])/3)

        minimize_args = {'param_keys':keys, 'param_guess_range':param_guess_range,
                        'is_scale_parameter':is_scale_parameter,
                        'scaled_param_guesses':scaled_param_guesses,
                        'param_length':param_length,
                        'obs':obs, 'fltr':fltr, 'Tf':Tf, 'obs0':obs0,
                        'init_flags':init_flags, 'init_fltrs': init_fltrs,
                        's':s, 'scale':scale, 'tangent':tangent}

        res = minimization(self._latent_minus_logp,
                           guess, bounds, ftol=ftol, global_max_iter=global_max_iter,
                           local_max_iter=local_max_iter, global_atol=global_atol,
                           enable_global=enable_global, enable_local=enable_local,
                           cma_processes=cma_processes,
                           cma_population=cma_population, cma_stds=cma_stds,
                           verbose=verbose, args_dict=minimize_args)

        estimates = res[0]

        # Get the parameters (in their original structure) from the flattened parameter vector.
        param_estimates = estimates[:param_length]
        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                        param_guess_range,
                                                        is_scale_parameter,
                                                        scaled_param_guesses)
        init_estimates = estimates[param_length:]
        map_params_dict = self.fill_params_dict(keys, orig_params)
        self.set_params(map_params_dict)
        self.set_det_model(map_params_dict)
        map_x0 = self._construct_inits(init_estimates, init_flags, init_fltrs,
                                      obs0, fltr[0])
        output_dict = {
            'map_params_dict':map_params_dict, 'map_x0':map_x0,
            'flat_map':estimates, '-logp':res[1],
            'param_keys': keys, 'param_guess_range': param_guess_range,
            'is_scale_parameter':is_scale_parameter, 'param_length':param_length,
            'scaled_param_guesses':scaled_param_guesses,
            'init_flags': init_flags, 'init_fltrs': init_fltrs,
            's':s, 'scale': scale
        }
        return output_dict

    def _nested_sampling_loglike_latent(self, params, param_keys=None,
                                        param_guess_range=None, is_scale_parameter=None,
                                        scaled_param_guesses=None, param_length=None,
                                        obs=None, fltr=None, Tf=None, obs0=None,
                                        init_flags=None, init_fltrs=None,
                                        s=None, scale=None, tangent=None):
        inits =  np.copy(params[param_length:])

        # Restore parameters from flattened parameters
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                          param_guess_range, is_scale_parameter, scaled_param_guesses)

        parameters = self.fill_params_dict(param_keys, orig_params)
        self.set_params(parameters)
        self.set_det_model(parameters)

        x0 = self._construct_inits(inits, init_flags, init_fltrs, obs0, fltr[0])

        minus_logp = self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))

        return -minus_logp


    def nested_sampling_latent_inference(self, np.ndarray obs, np.ndarray fltr, double Tf, contactMatrix, param_priors,
                                         init_priors,tangent=False, verbose=False, queue_size=1,
                                         max_workers=None, npoints=100, method='single', max_iter=1000, dlogz=None,
                                         decline_factor=None):
        '''Compute the log-evidence and weighted samples of the a-posteriori distribution of the parameters of a SIR type model
        with latent variables using nested sampling as implemented in the `nestle` Python package.

        This function provides a computational alterantive to `latent_infer_parameters`. It computes an estimate of the evidence and,
        in addition, returns a set of representative samples that can be used to compute a posterior mean estimate (insted of the MAP
        estimate). This approach approach is much more resource intensive and typically only viable for small models or tangent space inference.

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
            See `infer_parameters` for further explanations.
        init_priors: dict
            A dictionary for priors for the initial conditions.
            See `latent_infer_parameters` for further explanations.
        tangent: bool, optional
            Set to True to do inference in tangent space (might be less robust but a lot faster). Default is False.
        verbose: bool, optional
            Set to True to see intermediate outputs from the nested sampling procedure.
        queue_size: int
            Size of the internal queue of samples of the nested sampling algorithm. The log-likelihood of these samples
            is computed in parallel (if queue_size > 1).
        max_workers: int
            The maximal number of processes used to compute samples.
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

        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        keys, param_guess, param_stds, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M)

        # Read in initial conditions priors
        init_guess, init_stds, init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        s, scale = pyross.utils.make_log_norm_dist(guess, stds)

        k = len(guess)
        ppf_bounds = np.zeros((k, 2))
        ppf_bounds[:,0] = lognorm.cdf(bounds[:,0], s, scale=scale)
        ppf_bounds[:,1] = lognorm.cdf(bounds[:,1], s, scale=scale)
        ppf_bounds[:,1] = ppf_bounds[:,1] - ppf_bounds[:,0]

        prior_transform_args = {'s':s, 'scale':scale, 'ppf_bounds':ppf_bounds}
        loglike_args = {'param_keys':keys, 'param_guess_range':param_guess_range,
                        'is_scale_parameter':is_scale_parameter,
                        'scaled_param_guesses':scaled_param_guesses,
                        'param_length':param_length,
                        'obs':obs, 'fltr':fltr, 'Tf':Tf, 'obs0':obs0,
                        'init_flags':init_flags, 'init_fltrs': init_fltrs,
                        's':s, 'scale':scale, 'tangent':tangent}

        result = nested_sampling(self._nested_sampling_loglike_latent, self._nested_sampling_prior_transform, k, queue_size,
                                 max_workers, verbose, method, npoints, max_iter, dlogz, decline_factor, loglike_args,
                                 prior_transform_args)

        log_evidence = result.logz

        output_samples = []
        for i in range(len(result.samples)):
            sample = result.samples[i]
            weight = result.weights[i]
            param_estimates = sample[:param_length]
            orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                            param_guess_range,
                                                            is_scale_parameter,
                                                            scaled_param_guesses)

            init_estimates = sample[param_length:]
            sample_params_dict = self.fill_params_dict(keys, orig_params)
            self.set_params(sample_params_dict)
            self.set_det_model(sample_params_dict)
            map_x0 = self._construct_inits(init_estimates, init_flags, init_fltrs,
                                        obs0, fltr[0])
            output_dict = {
                'map_params_dict':sample_params_dict, 'map_x0':map_x0,
                'flat_map':sample, 'weight':weight,
                'is_scale_parameter':is_scale_parameter,
                'flat_param_guess_range':param_guess_range,
                'scaled_param_guesses':scaled_param_guesses,
                'init_flags': init_flags, 'init_fltrs': init_fltrs
            }
            output_samples.append(output_dict)

        return log_evidence, output_samples


    def _latent_infer_control_to_minimize(self, params, grad=0, generator=None,
                                            intervention_fun=None, param_keys=None,
                                            param_guess_range=None, is_scale_parameter=None,
                                            scaled_param_guesses=None, param_length=None,
                                            obs=None, fltr=None, Tf=None, obs0=None,
                                            init_flags=None, init_fltrs=None,
                                            s=None, scale=None, tangent=None):
        """Objective function for minimization call in latent_infer_control."""
        inits = params[param_length:].copy()
        orig_params = pyross.utils.unflatten_parameters(params[:param_length],
                                                        param_guess_range,
                                                        is_scale_parameter,
                                                         scaled_param_guesses)
        kwargs = {k:orig_params[i] for (i, k) in enumerate(param_keys)}
        x0 = self._construct_inits(inits, init_flags, init_fltrs,
                                    obs0, fltr[0])
        penalty = self._penalty_from_negative_values(x0)
        x0[x0<0] = 0.1/self.Omega # set to be small and positive
        if intervention_fun is None:
            self.contactMatrix = generator.constant_contactMatrix(**kwargs)
        else:
            self.contactMatrix = generator.intervention_custom_temporal(intervention_fun, **kwargs)

        minus_logp = self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent=tangent)
        minus_logp -= np.sum(lognorm.logpdf(params, s, scale=scale))
        return minus_logp

    def latent_infer_control(self, obs, fltr, Tf, generator, param_priors, init_priors,
                            intervention_fun=None, tangent=False,
                            verbose=False, ftol=1e-5, global_max_iter=100,
                            local_max_iter=100, global_atol=1., enable_global=True,
                            enable_local=True, cma_processes=0, cma_population=16):
        """
        Compute the maximum a-posteriori (MAP) estimate of the change of control parameters for a SIR type model in
        lockdown with partially observed classes. The unobserved classes are treated as latent variables. The lockdown
        is modelled by scaling the contact matrices for contact at work, school, and other (but not home) uniformly in
        all age groups. This function infers the scaling parameters.

        Parameters
        ----------
        obs:
            Observed trajectory (number of data points x (age groups * observed model classes)).
        fltr: boolean sequence or array
            True for observed and False for unobserved classes.
            e.g. if only Is is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            Total time of the trajectory
        generator: pyross.contactMatrix
            A pyross.contactMatrix object that generates a contact matrix function with specified lockdown
            parameters.
        param_priors: dict
            A dictionary for param priors. See `infer_parameters` for further explanations.
        init_priors: dict
            A dictionary for priors for initial conditions. See `latent_infer_parameters` for further explanations.
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
            The absolute tolerance for global minimisation.
        enable_global: bool, optional
            Set to True to enable global optimisation.
        enable_local: bool, optional
            Set to True to enable local optimisation.
        cma_processes: int, optional
            Number of parallel processes used for global optimisation.
        cma_population: int, optional
            The number of samples used in each step of the CMA algorithm.

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
        """
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # Read in parameter priors
        keys, param_guess, param_stds, param_bounds, param_guess_range, \
        is_scale_parameter, scaled_param_guesses \
            = pyross.utils.parse_param_prior_dict(param_priors, self.M)

        # Read in initial conditions priors
        init_guess, init_stds, init_bounds, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))

        # Concatenate the flattend parameter guess with init guess
        param_length = param_guess.shape[0]
        guess = np.concatenate([param_guess, init_guess]).astype(DTYPE)
        stds = np.concatenate([param_stds,init_stds]).astype(DTYPE)
        bounds = np.concatenate([param_bounds, init_bounds], axis=0).astype(DTYPE)

        s, scale = pyross.utils.make_log_norm_dist(guess, stds)
        cma_stds = np.minimum(stds, (bounds[:, 1]-bounds[:, 0])/3)

        minimize_args = {'generator':generator, 'intervention_fun':intervention_fun,
                       'param_keys':keys, 'param_guess_range':param_guess_range,
                       'is_scale_parameter':is_scale_parameter,
                       'scaled_param_guesses':scaled_param_guesses,
                       'param_length':param_length,
                       'obs':obs, 'fltr':fltr, 'Tf':Tf, 'obs0':obs0,
                       'init_flags':init_flags, 'init_fltrs': init_fltrs,
                       's':s, 'scale':scale, 'tangent':tangent}
        res = minimization(self._latent_infer_control_to_minimize,
                          guess, bounds, ftol=ftol,
                          global_max_iter=global_max_iter,
                          local_max_iter=local_max_iter, global_atol=global_atol,
                          enable_global=enable_global, enable_local=enable_local,
                          cma_processes=cma_processes,
                          cma_population=cma_population, cma_stds=cma_stds,
                          verbose=verbose, args_dict=minimize_args)
        estimates = res[0]

        # Get the parameters (in their original structure) from the flattened parameter vector.
        param_estimates = estimates[:param_length]
        orig_params = pyross.utils.unflatten_parameters(param_estimates,
                                                      param_guess_range,
                                                      is_scale_parameter,
                                                      scaled_param_guesses)
        init_estimates = estimates[param_length:]
        map_params_dict = {k:orig_params[i] for (i, k) in enumerate(keys)}
        map_x0 = self._construct_inits(init_estimates, init_flags, init_fltrs,
                                    obs0, fltr[0])
        output_dict = {
            'map_params_dict':map_params_dict, 'map_x0':map_x0,
            'flat_map':estimates, '-logp':res[1],
            'param_keys': keys, 'param_guess_range': param_guess_range,
            'is_scale_parameter':is_scale_parameter, 'param_length':param_length,
            'scaled_param_guesses':scaled_param_guesses,
            'init_flags': init_flags, 'init_fltrs': init_fltrs,
            's':s, 'scale': scale
        }
        return output_dict



    def compute_hessian_latent(self, obs, fltr, Tf, contactMatrix, map_dict,
                               tangent=False, eps=1.e-3, fd_method="central"):
        '''Computes the Hessian over the parameters and initial conditions.

        Parameters
        ----------
        x: 2d numpy.array
           Observed trajectory (number of data points x (age groups * model classes)).
        Tf: float
           Total time of the trajectory.
        contactMatrix: callable
           A function that takes time (t) as an argument and returns the contactMatrix.
        map_dict: dict
           Dictionary returned by `latent_infer_parameters`.
        eps: float or numpy.array, optional
           The step size of the Hessian calculation, default=1e-3.
        fd_method: str, optional
           The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        hess: numpy.array
            The Hessian over (flat) parameters and initial conditions.
        '''
        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)
        flat_maps = map_dict['flat_map']
        kwargs = {}
        for key in ['param_keys', 'param_guess_range', 'is_scale_parameter',
                    'scaled_param_guesses', 'param_length', 'init_flags',
                    'init_fltrs', 's', 'scale', ]:
            kwargs[key] = map_dict[key]
        def minuslogp(y):
            return self._latent_minus_logp(y, obs=obs, fltr=fltr, Tf=Tf, obs0=obs0,
                                           tangent=tangent, **kwargs)
        hess = pyross.utils.hessian_finite_difference(flat_maps, minuslogp, eps, method=fd_method)
        return hess

    # def error_bars_latent(self, param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf, contactMatrix,
    #                       tangent=False, infer_scale_parameter=False, eps=1.e-3, obs0=None, fltr0=None, fd_method="central"):
    #     hessian = self.compute_hessian_latent(param_keys, init_fltr, maps, prior_mean, prior_stds, obs, fltr, Tf, Nf,
    #                                           contactMatrix, tangent, infer_scale_parameter, eps, obs0, fltr0, fd_method=fd_method)
    #     return np.sqrt(np.diagonal(np.linalg.inv(hessian)))


    def log_G_evidence_latent(self, obs, fltr, Tf, contactMatrix, map_dict, tangent=False, eps=1.e-3,
                              fd_method="central"):
        """Compute the evidence using a Laplace approximation at the MAP estimate.

        Parameters
        ----------
        x: 2d numpy.array
           Observed trajectory (number of data points x (age groups * model classes))
        Tf: float
           Total time of the trajectory
        contactMatrix: callable
           A function that takes time (t) as an argument and returns the contactMatrix
        map_dict: dict
           MAP estimate returned by infer_parameters
        eps: float or numpy.array, optional
           The step size of the Hessian calculation, default=1e-3
        fd_method: str, optional
           The type of finite-difference scheme used to compute the hessian, supports "forward" and "central".

        Returns
        -------
        log_evidence: float
            The log-evidence computed via Laplace approximation at the MAP estimate."""
        logP_MAPs = -map_dict['-logp']
        A = self.compute_hessian_latent(obs, fltr, Tf, contactMatrix, map_dict, tangent, eps, fd_method)
        k = A.shape[0]

        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + k/2*np.log(2*np.pi)

    def minus_logp_red(self, parameters, np.ndarray x0, np.ndarray obs,
                            np.ndarray fltr, double Tf, contactMatrix, tangent=False):
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
        self.contactMatrix = contactMatrix
        fltr, obs, obs0 = pyross.utils.process_latent_data(fltr, obs)

        # check that x0 is consistent with obs0
        x0_obs = fltr[0].dot(x0)
        if not np.allclose(x0_obs, obs0):
            print('x0 not consistent with obs0. '
                  'Using x0 in the calculation of logp...')
        self.set_params(parameters)
        self.set_det_model(parameters)
        minus_logp = self._obtain_logp_for_lat_traj(x0, obs, fltr[1:], Tf, tangent)
        return minus_logp

    def get_mean_inits(self, init_priors, np.ndarray obs0, np.ndarray fltr0):
        '''Construct full initial conditions from the prior dict

        Parameters
        ----------
        init_priors: dict
            A dictionary for priors for initial conditions.
            Same as the `init_priors` passed to `latent_infer_parameters`.
            In this function, only takes the mean.
        obs0: numpy.array
            Observed initial conditions.
        fltr0: numpy.array
            Filter for the observed initial conditons.

        Returns
        -------
        x0: numpy.array
            Full initial conditions.
        '''
        init_mean, _, _, init_flags, init_fltrs \
            = pyross.utils.parse_init_prior_dict(init_priors, self.dim, len(obs0))
        x0 = self._construct_inits(init_mean, init_flags, init_fltrs, obs0, fltr0)
        return x0

    cpdef find_fastest_growing_lin_mode(self, double t):
        cdef:
            np.ndarray [DTYPE_t, ndim=2] J
            np.ndarray [DTYPE_t, ndim=1] x0, v, mode=np.empty((self.dim), dtype=DTYPE)
            list indices
            Py_ssize_t S_index, M=self.M, i, j, n_inf, n, index
        # assume no infected at the start and compute eig vecs for the infectious species
        x0 = np.zeros((self.dim), dtype=DTYPE)
        S_index = self.class_index_dict['S']
        x0[S_index*M:(S_index+1)*M] = self.fi
        self.compute_jacobian_and_b_matrix(x0, t,
                                           b_matrix=False, jacobian=True)
        indices = self.infection_indices()
        n_inf = len(indices)
        J = self.J[indices][:, :, indices, :].reshape((n_inf*M, n_inf*M))
        sign, eigvec = pyross.utils.largest_real_eig(J)
        if not sign: # if eigval not positive, just return the zero state
            return x0
        else:
            eigvec = np.abs(eigvec)

            # substitute in infections and recompute fastest growing linear mode
            for (j, i) in enumerate(indices):
                x0[i*M:(i+1)*M] = eigvec[j*M:(j+1)*M]
            self.compute_jacobian_and_b_matrix(x0, t,
                                               b_matrix=False, jacobian=True)
            _, v = pyross.utils.largest_real_eig(self.J_mat)
            if v[S_index*M] > 0:
                v = - v
            return v/np.linalg.norm(v, ord=1)

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

    def set_det_model(self, parameters):
        '''
        Sets the internal deterministic model with given epidemiological parameters

        Parameters
        ----------
        parameters: dict
            A dictionary of parameter values, same as the ones required for initialisation.
        '''
        raise NotImplementedError("Please Implement set_det_model in subclass")

    def set_contact_matrix(self, contactMatrix):
        '''
        Sets the internal contact matrix

        Parameters
        ----------
        contactMatrix: callable
            A function that returns the contact matrix given time, with call
            signature contactMatrix(t).
        '''
        self.contactMatrix = contactMatrix

    def make_params_dict(self):
        raise NotImplementedError("Please Implement make_params_dict in subclass")

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
            if k in self.param_keys:
                full_parameters[k] = params[i]
            else:
                raise Exception('{} is not a parameter of the model'.format(k))
        return full_parameters

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

    def _construct_inits(self, init_guess, flags, fltrs, obs0, fltr0):
        cdef:
            np.ndarray x0=np.empty(self.dim, dtype=DTYPE), x0_lin_mode, x0_ind
            np.ndarray [BOOL_t, ndim=1] mask, init_fltr
            Py_ssize_t start=0
        if flags[0]: # lin mode
            coeff = init_guess[0]
            x0_lin_mode = self._lin_mode_inits(coeff)
            mask = fltrs[0].astype('bool')
            x0[mask] = x0_lin_mode[mask]
            start += 1
        if flags[1]: # independent guesses
            x0_ind = init_guess[start:]
            mask = fltrs[1].astype('bool')
            x0[mask] = x0_ind
        init_fltr = np.logical_or(fltrs[0], fltrs[1])
        partial_inits = x0[init_fltr]
        return self._fill_initial_conditions(partial_inits, obs0, init_fltr, fltr0)

    def _lin_mode_inits(self, double coeff):
        cdef double [:] v, x0, fi=self.fi
        v = self.find_fastest_growing_lin_mode(0)
        v = np.multiply(v, coeff)
        x0 = np.zeros((self.dim), dtype=DTYPE)
        x0[:self.M] = fi
        return np.add(x0, v)

    def _fill_initial_conditions(self, np.ndarray partial_inits, double [:] obs_inits,
                                        np.ndarray init_fltr, np.ndarray fltr):
        cdef:
            np.ndarray x0=np.empty(self.dim, dtype=DTYPE)
            double [:] z, unknown_inits, partial_inits_memview=partial_inits.astype(DTYPE)
        z = np.subtract(obs_inits, np.dot(fltr[:, init_fltr], partial_inits_memview))
        unknown_inits = np.linalg.solve(fltr[:, np.invert(init_fltr)], z)
        x0[init_fltr] = partial_inits_memview
        x0[np.invert(init_fltr)] = unknown_inits
        return x0

    cdef double _obtain_logp_for_traj(self, double [:, :] x, double Tf):
        cdef:
            double log_p = 0
            double [:] xi, xf, dev, mean
            double [:, :] cov
            Py_ssize_t i, Nf=x.shape[0]
            double [:] time_points = np.linspace(0, Tf, Nf)
        for i in range(Nf-1):
            xi = x[i]
            xf = x[i+1]
            ti = time_points[i]
            tf = time_points[i+1]
            mean, cov = self._estimate_cond_mean_cov(xi, ti, tf)
            dev = np.subtract(xf, mean)
            log_p += self._log_cond_p(dev, cov)
        return -log_p

    cdef double _obtain_logp_for_lat_traj(self, double [:] x0, double [:] obs_flattened, np.ndarray fltr,
                                            double Tf, tangent=False):
        cdef:
            Py_ssize_t reduced_dim=obs_flattened.shape[0], Nf=fltr.shape[0]+1
            double [:, :] xm
            double [:] xm_red, dev
            np.ndarray[DTYPE_t, ndim=2] cov_red, full_cov
        if tangent:
            xm, full_cov = self.obtain_full_mean_cov_tangent_space(x0, Tf, Nf)
        else:
            xm, full_cov = self.obtain_full_mean_cov(x0, Tf, Nf)
        full_fltr = sparse.block_diag(fltr)
        cov_red = full_fltr@full_cov@np.transpose(full_fltr)
        xm_red = full_fltr@(np.ravel(xm))
        dev=np.subtract(obs_flattened, xm_red)
        cov_red_inv_dev, ldet = pyross.utils.solve_symmetric_close_to_singular(cov_red, dev)
        log_p = -np.dot(dev, cov_red_inv_dev)*(self.Omega/2)
        log_p -= (ldet-reduced_dim*log(self.Omega))/2 + (reduced_dim/2)*log(2*PI)
        return -log_p

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
            dx_det, cov = self._estimate_dx_and_cov(xt, t, dt)
            dev = np.subtract(dx[i], dx_det)
            logp += self._log_cond_p(dev, cov)
        return -logp

    cdef double _log_cond_p(self, double [:] x, double [:, :] cov):
        cdef:
            double [:] invcov_x
            double _log_cond_p
            double det
        invcov_x, ldet = pyross.utils.solve_symmetric_close_to_singular(cov, x)
        _log_cond_p = - np.dot(x, invcov_x)*(self.Omega/2) - (self.dim/2)*log(2*PI)
        _log_cond_p -= (ldet - self.dim*log(self.Omega))/2
        _log_cond_p -= self.dim*np.log(self.Omega)
        return _log_cond_p

    cdef _estimate_cond_mean_cov(self, double [:] x0, double t1, double t2):
        cdef:
            double [:, :] cov_array
            double [:] cov, sigma0=np.zeros((self.vec_size), dtype=DTYPE)
            double [:, :] x, cov_mat
            Py_ssize_t steps = self.steps
            double [:] time_points = np.linspace(t1, t2, steps)
        x = self.integrate(x0, t1, t2, steps)
        spline = make_interp_spline(time_points, x)

        def rhs(t, sig):
            self.CM = self.contactMatrix(t)
            self._lyapunov_fun(t, sig, spline)
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

    cdef _estimate_dx_and_cov(self, double [:] xt, double t, double dt):
        cdef:
            double [:] dx_det
            double [:, :] cov
        self.det_model.set_contactMatrix(t, self.contactMatrix)
        self.det_model.rhs(np.multiply(xt, self.Omega), t)
        dx_det = np.multiply(dt/self.Omega, self.det_model.dxdt)
        self.compute_jacobian_and_b_matrix(xt, t)
        cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
        return dx_det, cov

    cpdef obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf
            double [:, :] cond_cov, cov, temp
            double [:, :, :, :] full_cov
            double ti, tf
        xm = self.integrate(x0, 0, Tf, Nf, method='LSODA', maxNumSteps=self.steps*Nf)
        cov = np.zeros((dim, dim), dtype=DTYPE)
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        for i in range(Nf-1):
            ti = time_points[i]
            tf = time_points[i+1]
            xi = xm[i]
            xf, cond_cov = self._estimate_cond_mean_cov(xi, ti, tf)
            self._obtain_time_evol_op(xi, xf, ti, tf)
            cov = np.add(self.U@cov@self.U.T, cond_cov)
            full_cov[i, :, i, :] = cov
            if i>0:
                for j in range(0, i):
                    temp = full_cov[j, :, i-1, :]@self.U.T
                    full_cov[j, :, i, :] = temp
                    full_cov[i, :, j, :] = temp.T
        # returns mean and cov for all but first (fixed!) time point
        return xm[1:], np.reshape(full_cov, ((Nf-1)*dim, (Nf-1)*dim))

    cpdef obtain_full_mean_cov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xt
            double [:, :] cov, cond_cov, U, J_dt, temp
            double [:, :, :, :] full_cov
            double t, dt=time_points[1]
        xm = self.integrate(x0, 0, Tf, Nf,
                                method='LSODA', maxNumSteps=self.steps*Nf)
        full_cov = np.zeros((Nf-1, dim, Nf-1, dim), dtype=DTYPE)
        cov = np.zeros((dim, dim), dtype=DTYPE)
        for i in range(Nf-1):
            t = time_points[i]
            xt = xm[i]
            self.compute_jacobian_and_b_matrix(xt, t, b_matrix=True, jacobian=True)
            cond_cov = np.multiply(dt, self.convert_vec_to_mat(self.B_vec))
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

    cpdef _obtain_time_evol_op(self, double [:] x0, double [:] xf, double t1, double t2):
        cdef:
            double [:, :] U=self.U
            double epsilon=1./self.Omega
            Py_ssize_t i, j, steps=self.steps
        if t1==t2:
            U = np.eye(self.dim)
        else:
            for i in range(self.dim):
                x0[i] += epsilon
                pos = self.integrate(x0, t1, t2, steps)[steps-1]
                for j in range(self.dim):
                    U[j, i] = (pos[j]-xf[j])/(epsilon)
                x0[i] -= epsilon

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

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
        raise NotImplementedError("Please Implement _lyapunov_fun in subclass")

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                             b_matrix=True, jacobian=False):
        raise NotImplementedError("Please Implement compute_jacobian_and_b_matrix in subclass")


    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, method=None, maxNumSteps=100000):
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
        maxNumSteps:
            The maximum number of steps taken by the integrator.

        Returns
        -------
        sol: np.array
            The state of the system evaulated at the time point specified. Only used if det_method is set to 'solve_ivp'.
        """

        def rhs0(double t, double [:] xt):
            self.det_model.set_contactMatrix(t, self.contactMatrix)
            self.det_model.rhs(xt, t)
            return self.det_model.dxdt

        x0 = np.multiply(x0, self.Omega)

        if method is None:
            method = self.det_method
        if method=='LSODA':
            time_points = np.linspace(t1, t2, steps)
            sol = solve_ivp(rhs0, [t1,t2], x0, method='LSODA', t_eval=time_points, max_step=maxNumSteps, rtol=1e-4).y.T
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
        return np.divide(sol, self.Omega)

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
        Number of people in each age group divided by Omega.
    Omega: float, optional
        System size parameter, e.g. total population. Default to 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal and the default value is 4, as required by the cubic spline fit used for interpolation.
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
    cdef readonly pyross.deterministic.SIR det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = ['alpha', 'beta', 'gIa', 'gIs', 'fsa']
        super().__init__(parameters, 3, M, fi, Omega, steps, det_method, lyapunov_method)
        self.class_index_dict = {'S':0, 'Ia':1, 'Is':2}
        self.set_det_model(parameters)

    def set_det_model(self, parameters):
        self.det_model = pyross.deterministic.SIR(parameters, self.M, self.fi*self.Omega)

    def infection_indices(self):
        return [1, 2]

    def make_params_dict(self):
        parameters = {'alpha':self.alpha, 'beta':self.beta, 'gIa':self.gIa, 'gIs':self.gIs, 'fsa':self.fsa}
        return parameters

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)

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
            Fraction by which symptomatic individuals self isolate.
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
        The minimal and default is 4, as required by the cubic spline fit used for interpolation.
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
        readonly pyross.deterministic.SEIR det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = ['alpha', 'beta', 'gE', 'gIa', 'gIs', 'fsa']
        super().__init__(parameters, 4, M, fi, Omega, steps, det_method, lyapunov_method)
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


    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)

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
        Number of compartments
    fi: float numpy.array
        Number of people in each compartment divided by Omega.
    Omega: float, optional
        System size, e.g. total population. Default is 1.
    steps: int, optional
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal and default is 4, as required by the cubic spline fit used for interpolation.
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
        readonly pyross.deterministic.SEAIRQ det_model

    def __init__(self, parameters, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = ['alpha', 'beta', 'gE', 'gA', \
                           'gIa', 'gIs', 'fsa', \
                           'tE', 'tA', 'tIa', 'tIs']
        super().__init__(parameters, 6, M, fi, Omega, steps, det_method, lyapunov_method)
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

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)

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
            fraction by which symptomatic individuals self isolate.
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
    steps: int
        The number of internal integration steps performed between the observed points (not used in tangent space inference).
        The minimal and default is 4, as required by the cubic spline fit used for interpolation.
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
        readonly pyross.deterministic.SEAIRQ_testing det_model

    def __init__(self, parameters, testRate, M, fi, Omega=1, steps=4, det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = ['alpha', 'beta', 'gE', 'gA', \
                           'gIa', 'gIs', 'fsa', \
                           'ars', 'kapE']
        super().__init__(parameters, 6, M, fi, Omega, steps, det_method, lyapunov_method)
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

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, method=None, maxNumSteps=100000):
        self.det_model.set_testRate(self.testRate)
        return super().integrate(x0, t1, t2, steps, maxNumSteps=maxNumSteps, method=method)

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

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)


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
    Omega: int
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
        readonly pyross.deterministic.Spp det_model


    def __init__(self, model_spec, parameters, M, fi, Omega=1, steps=4,
                                    det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        super().__init__(parameters, self.nClass, M, fi, Omega, steps, det_method, lyapunov_method)
        self.det_model = pyross.deterministic.Spp(model_spec, parameters, M, fi*Omega)

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
                    indices.add(self.linear_terms[i, 1])
                    temp.pop(i)
            linear_terms_indices = temp
        return list(indices)

    def set_params(self, parameters):
        nParams = len(self.param_keys)
        self.parameters = np.empty((nParams, self.M), dtype=DTYPE)
        try:
            for (i, key) in enumerate(self.param_keys):
                param = parameters[key]
                self.parameters[i] = pyross.utils.age_dep_rates(param, self.M, key)
        except KeyError:
            raise Exception('The parameters passed does not contain certain keys. The keys are {}'.format(self.param_keys))

    def set_det_model(self, parameters):
        self.det_model.update_model_parameters(parameters)


    def make_params_dict(self):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    cdef np.ndarray _get_r_from_x(self, np.ndarray x):
        cdef:
            np.ndarray r
            np.ndarray xrs=x.reshape(int(self.dim/self.M), self.M)
        if self.constant_terms.size > 0:
            r = xrs[-1,:] - np.sum(xrs[:-1,:], axis=0)
        else:
            r = self.fi - np.sum(xrs, axis=0)
        return r

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                            b_matrix=True, jacobian=False):
        cdef:
            Py_ssize_t nClass=self.nClass, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fi=self.fi
        self.CM = self.contactMatrix(t)
        if self.constant_terms.size > 0:
            fi = x[(nClass-1)*M:]
        self.fill_lambdas(x, l)
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
            double Omega=self.Omega
        s = x[S_index*M:(S_index+1)*M]

        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                rate = parameters[rate_index]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]/Omega
                    B[nClass-1, m, nClass-1, m] += rate[m]/Omega

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

    def adj_RHS_mean(self, double t, double [:] lam, double [:] x0, spline):
        """RHS function for the adjoint gradient calculation of the time evolution operator."""
        cdef:
            Py_ssize_t M=self.M, nClass=self.nClass
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fsa=self.fsa, beta=self.beta, fi=self.fi
        self.CM = self.contactMatrix(t)
        x  = spline(t)
        self.fill_lambdas(x, l)
        self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.jacobian(x, l)
        return -np.dot(self.J_mat.T, lam )

    cpdef _obtain_time_evol_op_2(self, double [:] x0, double [:] xf, double t1, double t2):
        """
        xf is a redundant input here, added for consistency with the finite difference version '_obtain_time_evol_op'
        """
        cdef:
            Py_ssize_t steps=self.steps
            double [:,:] U=self.U
        ## Interpolate the dynamics
        if t1==t2:
            self.U = np.eye(self.dim)
        else:
            x = self.integrate(x0, t1, t2, steps)
            tsteps = np.linspace(t1, t2, steps)
            spline = make_interp_spline(tsteps, x)
            for k, row in enumerate(np.eye(self.dim)):
                a = solve_ivp(self.adj_RHS_mean, [t2,t1], row, t_eval=np.array([t1]), method='DOP853', args=(x0, spline,))
                self.U[k,:] = a.y.T[0]
        return self.U

    def lambdify_derivative_functions(self, keys):
        """Create python functions from sympy expressions"""
        M=self.M
        nClass=self.nClass
        parameters=self.parameters
        p = sympy.Matrix( sympy.symarray('p', (parameters.shape[0], parameters.shape[1]) )) ## epi-p only
        CM = sympy.Matrix( sympy.symarray('CM', (M, M)))
        fi = sympy.Matrix( sympy.symarray('fi', (1, M)))
        x=sympy.Matrix( sympy.symarray('x', (nClass,  M)))
        xi=sympy.Matrix( sympy.symarray('xi', (nClass,  M)))
        xf=sympy.Matrix( sympy.symarray('xf', (nClass,  M)))
        expr_var_list = [p, CM, fi, x]
        expr_var_list_ext = [p, CM, fi, xi, xf]

        global dA, dB, dJ, dinvcov ## instead of saving for now
        dA = sympy.lambdify(expr_var_list, self.dAd(p, keys=keys))
        dB = sympy.lambdify(expr_var_list, self.dBd(p, keys=keys))
        dJ = sympy.lambdify(expr_var_list, self.dJd(p, keys=keys))
        #dinvcov = sympy.lambdify(expr_var_list_ext, self.dinvcovelemd(p, keys=keys) )


    def dmudp(self, x0, t1, tf, steps, C, full_output=False):
        """
        calculates the derivatives of the mean traj x with respect to epi params and initial conditions.
        Note that although we can calculate the evolution operator T via adjoint gradient ODES it
        is comparable accuracy to finite difference anyway, and far slower.
        """

        def integrand(t, dummy, k, tf, xf, spline): ## can be optimized by outputting full res
            xt = spline(t)
            Tn=self._obtain_time_evol_op_2(xt, xf, t, tf) ## for inner product expression
            dAdp, _ = dA(param_values, CM_f, fi, xt.ravel())
            dAdp = np.array(dAdp)
            res=np.einsum('ij,kj->ki', Tn, dAdp)
            return res[:,k]

        fi=self.fi
        parameters=self.parameters
        param_values = self.parameters.ravel()

        keys = np.ones((parameters.shape[0], parameters.shape[1]), dtype=int) ## default to all params
        self.lambdify_derivative_functions(keys) ## could probably check for saved functions here
        no_inferred_params = np.sum(keys)
        CM_f=self.CM.ravel()
        xd = self.integrate(x0, t1, tf, steps)
        tsteps=np.linspace(t1,tf,steps)
        spline = make_interp_spline(tsteps, xd)
        xf = spline(tf)
        T=self._obtain_time_evol_op_2(x0, xf, t1, tf)

        dmudp = np.zeros((tsteps.size, no_inferred_params, self.dim), dtype=DTYPE)
        for k in range(self.dim):
            res = solve_ivp(integrand, [t1,tf], np.zeros(no_inferred_params), method='BDF', t_eval=tsteps, first_step=(tf-t1)/steps, max_step=steps, args=(k, tf, xf, spline,))
            dmudp[:,:,k] = res.y.T
        if full_output==False:
            dmu  = np.concatenate((dmudp[steps-1,:,:], np.transpose(T)), axis=0)
            return dmu
        else:
            return dmudp, xd

    def dfullinvcovdp(self, x0, t1, tf, steps, det_model, C):
        """ calculates the derivatives of full inv_cov. Relies on derivatives of the elements created by dinvcovelemd() """
        fi=self.fi
        parameters=self.parameters
        param_values = self.parameters.ravel()

        keys = np.ones((parameters.shape[0], parameters.shape[1]), dtype=int) ## default to all epi-params
        self.lambdify_derivative_functions(keys) ## could probably check for saved functions here
        no_inferred_params = np.sum(keys)
        CM_f = C(0).ravel()
        self.set_contact_matrix(C)
        xd = self.integrate(x0, t1, tf, steps)
        time_points=np.linspace(t1,tf,steps)
        dt = time_points[1]-time_points[0]
        Nf = steps
        full_cov_inv=[[[None]*(Nf-1) for i in range(Nf-1)] for j in range(no_inferred_params)]

        for i, ti in enumerate(time_points[:steps-1]):
            tf = time_points[i]+dt # make general
            xi = xd[i]
            xf = xd[i+1]

            (ddiagdp, doffdiagdp), _ , _ = dinvcov(param_values, CM_f, fi, xi, xf)
            for k in range(no_inferred_params): ## num params
                full_cov_inv[k][i][i]   = ddiagdp[k]
                if i<Nf-2:
                    full_cov_inv[k][i][i+1] = doffdiagdp[k]
                    full_cov_inv[k][i+1][i] = np.transpose(doffdiagdp[k])
        full_cov_inv_mat = np.empty((1, (Nf-1)*self.dim, (Nf-1)*self.dim))

        ## Make block mat into full mat. np.sparse.bmat doesn't handle slices very well hence the workaround
        for k in range(no_inferred_params):
            f=sparse.bmat(full_cov_inv[k], format='csc').todense().copy()
            f = np.array(f).reshape((1,)+f.shape)
            full_cov_inv_mat = np.concatenate((full_cov_inv_mat, f), axis=0)
        return full_cov_inv_mat[1:]


    def FIM_sym(self,x0, t1, t2, Nf, model, C, keys=None):
        """Does the fullinfo FIM based off symbolic expressions. In development,"""
        C = self.contactMatrix

        M=self.M
        nClass=self.nClass
        num_of_infection_terms=self.infection_terms.shape[0]
        parameters=self.parameters
        xd = self.integrate(x0, 0, t2-t1, Nf)
        time_points = np.linspace(0, t2-t1, Nf)
        dt = time_points[1]
        l = np.zeros((num_of_infection_terms,M), dtype=DTYPE)
        FIM=0
        if keys == None:
            keys = np.ones((parameters.shape[0], parameters.shape[1]), dtype=int) ## default to all params
        self.lambdify_derivative_functions(keys)
        ## Ready the inputs for these functions
        param_values = self.parameters.ravel()
        fi = self.fi
        indices = np.triu_indices(self.dim, 1)
        for k in range(time_points.size-1):
            #print(f"Progress :\t{k/(time_points.size-1)}")
            xi, xf = xd[k], xd[k+1]
            ti = time_points[k]
            tf = ti+dt
            CM_f = C(ti).ravel() ## for generality - redundant for a constant ContactMatrix
            self.CM=C(ti)
            ## yield the required arrays
            self.fill_lambdas(xi, l)
            self.noise_correlation(xi, l)
            Bmat = self.convert_vec_to_mat(self.B_vec)
            Binv = np.linalg.inv(Bmat)
            ## yield the derivatives by calling saved functions (faster)
            dtan, dAdx = dA(param_values, CM_f, fi, xi.ravel())
            dcov, dBdx = dB(param_values, CM_f, fi, xi.ravel())
            dtan = np.array(dtan)
            dAdx = np.array(dAdx)
            dcov = np.array(dcov)
            dBdx = np.array(dBdx)
            ## Make things true symmetric where necessary
            for i in range(np.sum(keys)): ## len params
                dcov_i = dcov[i]
                dcov_i.T[indices] = dcov_i[indices]
            for i in range(self.dim): ## len params
                dBdx_i = dBdx[i]
                dBdx_i.T[indices] = dBdx_i[indices]
            ## construct d[..]dx0 and stack
            self._obtain_time_evol_op(x0, xi, t1, ti) ## initial to current time
            U=self.U

            dtandx0 = np.einsum('ij,ik->ik', U, dAdx)
            dcovdx0 = np.einsum('ji, jkl->ikl', U, dBdx)
            dtan = np.concatenate((dtan, dtandx0), axis=0)
            dcov = np.concatenate((dcov, dcovdx0), axis=0)
            ## Sum for FIM
            dmu   =  self.dmudp(x0.flatten(), t1, tf, Nf, model, C)
            dtan +=  np.einsum('lj,il->ij',dAdx, dmu) ## missing part by product rule
            dcov += np.einsum('ijk,li->ljk', dBdx, dmu)
            FIM  += np.einsum('ia, ak, jk', dtan, Binv, dtan)
            FIM  += 0.5*np.einsum('ab,ibc,cd,jda', Binv, dcov, Binv, dcov)
        return FIM

    def construct_l(self, x):
        """constructs sympy l. x is a sympy matrix"""
        M=self.M
        infection_terms=self.infection_terms
        num_of_infection_terms=infection_terms.shape[0]
        CM = sympy.Matrix( sympy.symarray('CM', (M, M)))
        fi = sympy.Matrix( sympy.symarray('fi', (1, M)))
        l = sympy.Matrix(np.zeros((num_of_infection_terms,M)))
        for i in range(num_of_infection_terms):
            infective_index = infection_terms[i, 1]
            for m in range(M):
                for n in range(M):
                    index = n + M*infective_index
                    l[i, m] += CM[m,n]*x[index]/fi[n]
        return l

    def construct_A_spp(self, x):
        """construct Spp A. x is a sympy matrix"""
        M=self.M
        nClass=self.nClass
        constant_terms=self.constant_terms
        linear_terms=self.linear_terms
        infection_terms=self.infection_terms
        x=x.reshape(1,self.dim)
        S_index=self.class_index_dict['S']
        s = x[:,S_index*M:(S_index+1)*M]
        parameters=self.parameters

        A=sympy.Matrix(np.zeros((nClass,M) ))
        l=self.construct_l(x)
        p = sympy.Matrix( sympy.symarray('p', (parameters.shape[0], parameters.shape[1]) )) ## epi-p
        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M):
                A[S_index, m] -= rate[m]*l[i, m]*s[m]
                if product_index>-1:
                    A[product_index, m] += rate[m]*l[i, m]*s[m]
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            reagent = x[:,reagent_index*M:(reagent_index+1)*M]
            rate_index = linear_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M):
                A[reagent_index, m] -= rate[m]*reagent[m]
                if product_index >-1:
                    A[product_index, m] += rate[m]*reagent[m]
        A=A.reshape(1, self.dim)
        return A


    def construct_B_spp(self, x):
        """constructs Spp B. x is a sympy array"""
        Omega=self.Omega
        M=self.M
        nClass=self.nClass
        constant_terms=self.constant_terms
        linear_terms=self.linear_terms
        infection_terms=self.infection_terms
        parameters=self.parameters
        x = x.reshape(1, self.dim)
        S_index=self.class_index_dict['S']
        s = x[:,S_index*M:(S_index+1)*M]
        B = Array(np.zeros((nClass, M, nClass, M)))
        l = self.construct_l(x)
        p = sympy.Matrix( sympy.symarray('p', (parameters.shape[0], parameters.shape[1]) )) ## epi-p
        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                rate = p[rate_index,:]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]/Omega
                    B[nClass-1, m, nClass-1, m] += rate[m]/Omega
        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M):
                B[S_index, m, S_index, m] += rate[m]*l[i, m]*s[m]
                if product_index>-1:
                    B[S_index, m, product_index, m] -=  rate[m]*l[i, m]*s[m]
                    B[product_index, m, product_index, m] += rate[m]*l[i, m]*s[m]
                    B[product_index, m, S_index, m] -= rate[m]*l[i, m]*s[m]
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            reagent = x[:,reagent_index*M:(reagent_index+1)*M]
            rate_index = linear_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M): # only fill in the upper triangular form
                B[reagent_index, m, reagent_index, m] += rate[m]*reagent[m]
                if product_index>-1:
                    B[product_index, m, product_index, m] += rate[m]*reagent[m]
                    B[reagent_index, m, product_index, m] += -rate[m]*reagent[m]
                    B[product_index, m, reagent_index, m] += -rate[m]*reagent[m] ## make sure the symmetrising method reassigns the Lower traingular elem
        B=B.reshape(self.dim, self.dim)
        return B

    def construct_J_spp(self, x):
        """constructs Spp J. x is a sympy matrix"""
        M=self.M
        nClass=self.nClass
        constant_terms=self.constant_terms
        linear_terms=self.linear_terms
        infection_terms=self.infection_terms
        x=x.reshape(1,self.dim)
        S_index=self.class_index_dict['S']
        s = x[:,S_index*M:(S_index+1)*M]
        parameters=self.parameters
        J = Array(np.zeros((nClass, M, nClass, M)))
        CM = sympy.Matrix( sympy.symarray('CM', (M, M)))
        fi = sympy.Matrix( sympy.symarray('fi', (1, M)))
        l = self.construct_l(x)
        p = sympy.Matrix( sympy.symarray('p', (parameters.shape[0], parameters.shape[1]) )) ## epi-p
        for i in range(infection_terms.shape[0]):
            product_index = infection_terms[i, 2]
            infective_index = infection_terms[i, 1]
            rate_index = infection_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M):
                J[S_index, m, S_index, m] -= rate[m]*l[i, m]
                if product_index>-1:
                    J[product_index, m, S_index, m] += rate[m]*l[i, m]
                for n in range(M):
                    J[S_index, m, infective_index, n] -= s[m]*rate[m]*CM[m, n]/fi[n]
                    if product_index>-1:
                        J[product_index, m, infective_index, n] += s[m]*rate[m]*CM[m, n]/fi[n]
        for i in range(linear_terms.shape[0]):
            product_index = linear_terms[i, 2]
            reagent_index = linear_terms[i, 1]
            rate_index = linear_terms[i, 0]
            rate = p[rate_index,:]
            for m in range(M):
                J[reagent_index, m, reagent_index, m] -= rate[m]
                if product_index>-1:
                    J[product_index, m, reagent_index, m] += rate[m]
        J=J.reshape(self.dim, self.dim)
        return J

    def construct_fullcov_elem(self, xi, xf, dt=1):
        """Creates a sympy version of full cov. Takes symbol and symbolic matrices"""

        xi = xi.reshape(1,self.dim)
        xf = xf.reshape(1,self.dim)

        Bi = sympy.Matrix(self.construct_B_spp(xi))
        Bf = sympy.Matrix(self.construct_B_spp(xf))
        Uf = sympy.eye(self.dim) + dt * self.construct_J_spp(xf) ## tangent space approx
        elem_diag = Inverse(Bi) + Uf.T*Inverse(Bf)*Uf
        elem_offdiag = -Uf.T*Inverse(Bf)
        return elem_diag, elem_offdiag

    def dinvcovelemd(self, p, return_x0_deriv=False, keys=None):
        """ Constuction of invcov elements in symbolic form. The inverses of B are slow, but calculating full cov elements directly is faster"""
        assert (keys is not None), "Error: integer 1-0 array 'keys' was not passed"
        M=self.M
        nClass=self.nClass
        xi=sympy.Matrix( sympy.symarray('xi', (nClass,  M)))
        xf=sympy.Matrix( sympy.symarray('xf', (nClass,  M)))
        no_inferred_params = np.sum(keys)
        d_diagdp = Array(np.zeros((no_inferred_params,self.dim,self.dim)))
        d_offdiagdp = Array(np.zeros((no_inferred_params,self.dim,self.dim)))

        rows, cols = np.where(keys==1)
        elem_diag, elem_offdiag =  self.construct_fullcov_elem(xi, xf)
        for k, (r, c) in enumerate(zip(rows, cols)):
            param = p[r,c]
            d_diagdp[k,:,:] = sympy.diff(elem_diag, param)
            d_offdiagdp[k,:,:] = sympy.diff(elem_offdiag, param)

        d_diagdxi = sympy.diff(elem_diag, xi)
        d_diagdxf = sympy.diff(elem_diag, xf)
        d_offdiagdxi = sympy.diff(elem_offdiag, xi)
        d_offdiagdxf = sympy.diff(elem_offdiag, xf)

        return (d_diagdp, d_offdiagdp), (d_diagdxi, d_offdiagdxi), (d_diagdxf, d_offdiagdxf)


    def dAd(self, p, return_x0_deriv=False, keys=None):
        """
        constructs Spp B. param is a string or sympy symbol. Most likely you'll wish to use 'all' string
        keys can be passed as a integer 0,1 numpy array which selects the parameters to be used for FIM calculation
        p is a sympy array which contains epi parameters. nParams*M due to age dependence
        [dAdp]_ij = dA_j/dp_i
        """
        assert (keys is not None), "Error: integer 1-0 array 'keys' was not passed"
        M=self.M
        nClass=self.nClass
        x=sympy.Matrix( sympy.symarray('x', (nClass,  M)))
        no_inferred_params = np.sum(keys)
        A=self.construct_A_spp(x)
        dAdp = Array(np.zeros((no_inferred_params, 1, self.dim)))
        rows, cols = np.where(keys==1)
        for k, (r, c) in enumerate(zip(rows, cols)):
            param = p[r,c]
            dAdp[k,:,:] = sympy.diff(A, param)
        dAdx = sympy.diff(A, x).reshape(self.dim, 1, self.dim)
        return dAdp[:,0,:], dAdx[:,0,:]


    def dBd(self, p, return_x0_deriv=False, keys=None):
        """
        constructs Spp B. param is a string or sympy symbol. Most likely you'll wish to use 'all' string
        keys can be passed as a integer 0,1 numpy array which selects the parameters to be used for FIM calculation
        p is a sympy array which contains epi parameters. nParams*M due to age dependence
        [dBdp]_ijk = dB_jk/dp_i
        """
        assert (keys is not None), "Error: integer 1-0 'keys' was not passed"
        M=self.M
        nClass=self.nClass
        x =sympy.Matrix( sympy.symarray('x', (nClass, M)))
        no_inferred_params = np.sum(keys)
        B=self.construct_B_spp(x)
        dBdp = Array(np.zeros((no_inferred_params, self.dim, self.dim)))
        rows, cols = np.where(keys==1)
        for k, (r, c) in enumerate(zip(rows, cols)):
            param = p[r,c]
            dBdp[k,:,:] = sympy.diff(B, param)
        dBdx = sympy.diff(B, x).reshape(self.dim, self.dim, self.dim)
        return dBdp, dBdx

    def dJd(self, p, return_x0_deriv=False, keys=None):
        """
        constructs Spp J. param is a string or sympy symbol. Most likely you'll wish to use 'all' string
        keys can be passed as a integer 0,1 numpy array which selects the parameters to be used for FIM calculation
        p is a sympy array which contains epi parameters. nParams*M due to age dependence
        [dJdp]_ijk = dJ_jk/dp_i
        """
        assert (keys is not None), "Error: integer 1-0 'keys' was not passed"
        M=self.M
        nClass=self.nClass
        x =sympy.Matrix( sympy.symarray('x', (nClass, M)))
        no_inferred_params = np.sum(keys)
        J=self.construct_J_spp(x)
        dJdp = Array(np.zeros((no_inferred_params, self.dim, self.dim)))
        rows, cols = np.where(keys==1)
        for k, (r, c) in enumerate(zip(rows, cols)):
            param = p[r,c]
            dJdp[k,:,:] = sympy.diff(J, param)
        dJdx = sympy.diff(J, x).reshape(self.dim, self.dim, self.dim)
        return dJdp, dJdx

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
    Omega: int
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
        readonly Py_ssize_t nClassU, nClassUwoN
        readonly pyross.deterministic.SppQ det_model
        readonly object testRate


    def __init__(self, model_spec, parameters, testRate, M, fi, Omega=1, steps=4,
                                    det_method='LSODA', lyapunov_method='LSODA'):
        self.param_keys = list(parameters.keys())
        res = pyross.utils.parse_model_spec(model_spec, self.param_keys)
        self.nClass = res[0]
        self.class_index_dict = res[1]
        self.constant_terms = res[2]
        self.linear_terms = res[3]
        self.infection_terms = res[4]
        self.test_pos = res[5]
        self.test_freq = res[6]
        super().__init__(parameters, self.nClass, M, fi, Omega, steps, det_method, lyapunov_method)
        self.det_model = pyross.deterministic.SppQ(model_spec, parameters, M, fi*Omega)
        self.testRate =  testRate
        self.det_model.set_testRate(testRate)

        if self.constant_terms.size > 0:
            self.nClassU = self.nClass // 2 # number of unquarantined classes with constant terms
            self.nClassUwoN = self.nClassU - 1
        else:
            self.nClassU = (self.nClass - 1) // 2 # number of unquarantined classes w/o constant terms
            self.nClassUwoN = self.nClassU

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
                    indices.add(self.linear_terms[i, 1])
                    temp.pop(i)
            linear_terms_indices = temp
        return list(indices)



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

    def integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, method=None, maxNumSteps=100000):
        self.det_model.set_testRate(self.testRate)
        return super().integrate(x0, t1, t2, steps, method, maxNumSteps)

    def set_det_model(self, parameters):
        self.det_model.update_model_parameters(parameters)
        self.det_model.set_testRate(self.testRate)

    def make_params_dict(self):
        param_dict = {k:self.parameters[i] for (i, k) in enumerate(self.param_keys)}
        return param_dict

    cdef np.ndarray _get_r_from_x(self, np.ndarray x):
        cdef:
            np.ndarray r
            np.ndarray xrs=x.reshape(int(self.dim/self.M), self.M)
        r = self.fi - xrs[-1,:] - np.sum(xrs[:self.nClassUwoN,:], axis=0) # subtract total quarantined and all non-quarantined classes
        return r

    cdef np.ndarray _get_rq_from_x(self, np.ndarray x):
        cdef:
            np.ndarray r
            np.ndarray xrs=x.reshape(int(self.dim/self.M), self.M)
        r = xrs[-1,:] - np.sum(xrs[self.nClassU:-1,:], axis=0) # subtract all quarantined classes
        return r

    cdef double _penalty_from_negative_values(self, np.ndarray x0):
        cdef:
            double eps=0.1/self.Omega, dev
            np.ndarray R_init, RQ_init
        R_init = self._get_r_from_x(x0)
        RQ_init = self._get_rq_from_x(x0)
        dev = - (np.sum(R_init[R_init<0]) + np.sum(RQ_init[RQ_init<0]) + np.sum(x0[x0<0]))
        return (dev/eps)**2 + (dev/eps)**8

    cdef calculate_test_r(self, double [:] x, double [:] r, double TR):
        cdef:
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, nClassUwoN=self.nClassUwoN, M=self.M
            int [:] test_freq=self.test_freq
            double [:] fi=self.fi
            double Omega = self.Omega
            double ntestpop=0, tau0=0
            double [:, :] parameters=self.parameters
            Py_ssize_t m, i

        # Compute non-quarantined recovered
        r = self._get_r_from_x(np.array(x))
        # Compute normalisation of testing rates
        for m in range(M):
            for i in range(nClassUwoN):
                ntestpop += parameters[test_freq[i], m] * x[i*M+m]
            ntestpop += parameters[test_freq[nClassUwoN], m] * r[m]
        tau0 = TR / (Omega * ntestpop)
        return ntestpop, tau0

    cdef _lyapunov_fun(self, double t, double [:] sig, spline):
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
        self._compute_dsigdt(sig)

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                        b_matrix=True, jacobian=False):
        cdef:
            Py_ssize_t nClass=self.nClass, nClassU=self.nClassU, M=self.M
            Py_ssize_t num_of_infection_terms=self.infection_terms.shape[0]
            double [:, :] l=np.zeros((num_of_infection_terms, self.M), dtype=DTYPE)
            double [:] fi=self.fi
            double TR
            double ntestpop, tau0
            double [:] r=np.zeros(self.M, dtype=DTYPE)
        self.CM = self.contactMatrix(t)
        TR = self.testRate(t)
        if self.constant_terms.size > 0:
            fi = x[(nClassU-1)*M:]
        self.fill_lambdas(x, l)
        ntestpop, tau0 = self.calculate_test_r(x, r, TR)
        if b_matrix:
            self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
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
            double Omega=self.Omega
        s = x[S_index*M:(S_index+1)*M]

        if self.constant_terms.size > 0:
            for i in range(constant_terms.shape[0]):
                rate_index = constant_terms[i, 0]
                class_index = constant_terms[i, 1]
                rate = parameters[rate_index]
                for m in range(M):
                    B[class_index, m, class_index, m] += rate[m]/Omega
                    B[nClass-1, m, nClass-1, m] += rate[m]/Omega

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
                B[nClass-1, m, nClass-1, m] += term
                B[i, m, i+nClassU, m] -= term
                B[i+nClassU, m, i, m] -= term
                B[i, m, nClass-1, m] -= term
                B[nClass-1, m, i, m] -= term
                B[i+nClassU, m, nClass-1, m] += term
                B[nClass-1, m, i+nClassU, m] += term
            term = tau0 * parameters[test_freq[nClassUwoN], m] * parameters[test_pos[nClassUwoN], m] * r[m]
            B[nClass-1, m, nClass-1, m] += term


        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]
