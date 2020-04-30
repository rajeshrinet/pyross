from scipy import sparse
from scipy.integrate import odeint
from scipy.optimize import minimize, approx_fprime, basinhopping
from scipy.stats import gamma
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
cimport numpy as np
cimport cython

import pyross.deterministic
cimport pyross.deterministic
import pyross.contactMatrix
from pyross.utils import BoundedSteps, minimisation
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
    cdef:
        readonly Py_ssize_t nClass, N, M, steps, dim, vec_size
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols


    def __init__(self, parameters, nClass, M, fi, N, steps):
        self.N = N
        self.M = M
        self.fi = fi
        self.steps = steps
        self.set_params(parameters)

        self.dim = nClass*M
        self.vec_size = int(self.dim*(self.dim+1)/2)
        self.CM = np.empty((M, M), dtype=DTYPE)
        self.dsigmadt = np.zeros((self.vec_size), dtype=DTYPE)
        self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
        self.J_mat = np.empty((self.vec_size, self.vec_size), dtype=DTYPE)
        self.B_vec = np.empty((self.vec_size), dtype=DTYPE)
        self.U = np.empty((self.dim, self.dim), dtype=DTYPE)

        # preparing the indices
        self.rows, self.cols = np.triu_indices(self.dim)
        self.flat_indices = np.ravel_multi_index((self.rows, self.cols), (self.dim, self.dim))
        r, c = np.triu_indices(self.dim, k=1)
        self.flat_indices1 = np.ravel_multi_index((r, c), (self.dim, self.dim))
        self.flat_indices2 = np.ravel_multi_index((c, r), (self.dim, self.dim))


    def inference(self, guess, stds, x, Tf, Nf, contactMatrix, beta_rescale=1, bounds=None, niter=2, verbose=False, ftol=1e-6, eps=1e-5):
        '''
        guess: numpy.array
            initial guess for the parameter values
        Tf: float
            total time of the trajectory
        Nf: float
            number of data points along the trajectory
        contactMatrix: callable
        bounds: 2d numpy.array
            bounds for the parameters.
            Note that the upper bound must be smaller than the absolute physical upper bound minus epsilon
        niter: int
            number of iterations of basinhopping
        verbose: bool
            whether to print messages
        ftol: double
            relative tolerance of logp
        eps: double
            size of steps taken by L-BFGS-B algorithm for the calculation of Hessian
        '''
        a, scale = pyross.utils.make_gamma_dist(guess, stds)
        def to_minimize(params):
            if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
                return INFINITY
            params[1] /= beta_rescale
            parameters = self.make_params_dict(params)
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
            minus_logp -= np.sum(gamma.logpdf(params, a, scale=scale))
            params[1] *=beta_rescale
            return minus_logp
        # make bounds if it does not exist and rescale
        if bounds is None:
            bounds = np.array([[eps, g*5] for g in guess])
            bounds[0][1] = min(bounds[0][1], 1-2*eps)
        assert bounds[0][1] < 1-eps # the upper bound of alpha must be less than 1-eps
        bounds = np.array(bounds)
        guess[1] *= beta_rescale
        bounds[1] *= beta_rescale

        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
        if verbose:
            def callback(params):
                print('parameters:', params)
            minimizer_kwargs['callback']= callback
        take_step = BoundedSteps(bounds)
        res = basinhopping(to_minimize, guess, niter=niter,
                            minimizer_kwargs=minimizer_kwargs,
                            take_step=take_step, disp=verbose)
        estimates = res.x
        estimates[1] /= beta_rescale
        return estimates, res.nit

    def infer_control(self, guess, x, Tf, Nf, generator, bounds, niter=2, verbose=False, ftol=1e-6, eps=1e-5):
        '''
        guess: numpy.array
            initial guess for the control parameter values
        Tf: float
            total time of the trajectory
        Nf: float
            number of data points along the trajectory
        generator: pyross.contactMatrix
        bounds: 2d numpy.array
            bounds for the parameters.
            Note that the upper bound must be smaller than the absolute physical upper bound minus epsilon
        niter: int
            number of iterations of basinhopping
        verbose: bool
            whether to print messages
        ftol: double
            relative tolerance of logp
        eps: double
            size of steps taken by L-BFGS-B algorithm for the calculation of Hessian
        '''
        def to_minimize(params):
            if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
                return INFINITY
            parameters = self.make_params_dict()
            model =self.make_det_model(parameters)
            times = [Tf+1]
            interventions = [params]
            contactMatrix = generator.interventions_temporal(times, interventions)
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
            return minus_logp
        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
        if verbose:
            def callback(params):
                print('parameters:', params)
            minimizer_kwargs['callback'] = callback
        take_step = BoundedSteps(bounds)
        res = basinhopping(to_minimize, guess, niter=niter,
                            minimizer_kwargs=minimizer_kwargs,
                            take_step=take_step, disp=verbose)
        return res.x, res.nit

    def hessian(self, maps, x, Tf, Nf, contactMatrix, beta_rescale=1, eps=1.e-3):
        maps[1] *= beta_rescale
        cdef:
            Py_ssize_t k=maps.shape[0], i, j
            double xx0
            np.ndarray g1, g2, hess = np.empty((k, k))
        def minuslogP(y):
            y[1] /= beta_rescale
            parameters = self.make_params_dict(y)
            minuslogp = self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
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

    def error_bars(self, maps, x, Tf, Nf, contactMatrix, eps=1.e-3):
        hessian = self.hessian(maps,x,Tf,Nf,contactMatrix,eps)
        return np.sqrt(np.diagonal(np.linalg.inv(hessian)))

    def log_G_evidence(self, maps, x, Tf, Nf, contactMatrix, eps=1.e-3):
        # M variate process, M=3 for SIIR model
        cdef double logP_MAPs
        cdef Py_ssize_t k
        parameters = self.make_params_dict(maps)
        logP_MAPs = -self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
        k = maps.shape[0]
        A = self.hessian(maps,x,Tf,Nf,contactMatrix,eps)
        return logP_MAPs - 0.5*np.log(np.linalg.det(A)) + k/2*np.log(2*np.pi)

    def log_NS_evidence(self, x, Tf, Nf, contactMatrix, UB=1., LB=0.001, P=4): # this is very slow
        import nestle
        # For now universal upper and lower parameter bounds UB, LB. Easy to generalize
        # P is dimension of parameter space, P=4 for SIIR
        def logP(y):
            parameters = self.make_params_dict(y)
            return -self.obtain_minus_log_p(parameters, x, Tf, Nf, contactMatrix)
        def prior_transform(x):
            return (UB - LB)*x + LB  #Flat prior between LB and UB
        res = nestle.sample(logP, prior_transform, P)
        return res.logz

    def obtain_minus_log_p(self, parameters, double [:, :] x, double Tf, int Nf, contactMatrix):
        cdef double minus_log_p
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        return minus_logp

    def latent_inference(self, np.ndarray guess, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, contactMatrix, np.ndarray bounds,
                            beta_rescale=1, verbose=False, Py_ssize_t niter=1,
                            double ftol=1e-5, double eps=1e-4, global_max_iter=100, local_max_iter=100, 
                            global_ftol_factor=10., enable_global=True, enable_local=True, cma_processes=0, 
                            cma_population=16, cma_stds=None):
        '''
        guess: numpy.array
            initial guess, arranged in the order of parameters and initial conditions
        obs: numpy.array
            the observed trajectories with reduced number of variables
        fltr: boolean sequence or array
            True for observed and False for unobserved.
            e.g. if only Is is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            total time of the trajectory
        Nf: int
            total number of data points along the trajectory
        contactMatrix: callable
            a function that takes time as an input and outputs the contactMatrix
        bounds: 2d numpy.array
            bounds for the parameters + initial conditions.
            Better bounds makes it easier to find the true global minimum.
        verbose: bool, optional
            set True to see intermediate outputs from the optimizer
        niter: int, optional
            number of basinhopping performed by the optimizer
        ftol: float, optional
            relative tolerance
        eps: float, optional
            step size used by L-BFGS-B in calculation of Hessian
        '''
        cdef:
            double eps_for_params=eps, eps_for_init_cond = 0.5/self.N
            double rescale_factor = eps_for_params/eps_for_init_cond
            Py_ssize_t param_dim = guess.shape[0] - self.dim
        guess[param_dim:] *= rescale_factor
        guess[1] *= beta_rescale
        bounds[param_dim:, :] *= rescale_factor
        bounds[1, :] *= beta_rescale

        def to_minimize(params, grad = 0):
            if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
                return INFINITY
            x0 =  params[param_dim:]/rescale_factor
            # NLopt does not like us changing params here, so we just make a copy.
            local_params = params.copy()
            local_params[1] /= beta_rescale
            parameters = self.make_params_dict(local_params[:param_dim])
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            minus_logp = self.obtain_log_p_for_traj_red(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
            return minus_logp
            
#        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
#        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
#        if verbose:
#            def callback(params):
#                print('parameters:', params[:-self.dim])
#            minimizer_kwargs['callback'] = callback
#        take_step = BoundedSteps(bounds)
#        res = basinhopping(to_minimize, guess, niter=niter,
#                            minimizer_kwargs=minimizer_kwargs,
#                            take_step=take_step, disp=verbose)

        res = minimisation(to_minimize, guess, bounds, ftol=ftol, global_max_iter=global_max_iter, local_max_iter=local_max_iter,
                           global_ftol_factor=global_ftol_factor, enable_global=enable_global, enable_local=enable_local, 
                           cma_processes=cma_processes, cma_population=cma_population, cma_stds=cma_stds, 
                           verbose=verbose)

        params = res[0]
        params[param_dim:] /= rescale_factor
        params[1] /= beta_rescale
        return params

    def latent_infer_control(self, np.ndarray guess, np.ndarray x0, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, generator, np.ndarray bounds,
                            verbose=False, Py_ssize_t niter=1,
                            double ftol=1e-5, double eps=1e-4):
        def to_minimize(params):
            if (params>(bounds[:, 1]-eps)).all() or (params < (bounds[:,0]+eps)).all():
                return INFINITY
            parameters = self.make_params_dict()
            model = self.make_det_model(parameters)
            times = [Tf+1]
            interventions = [params]
            contactMatrix = generator.interventions_temporal(times, interventions)
            minus_logp = self.obtain_log_p_for_traj_red(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
            return minus_logp
        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
        if verbose:
            def callback(params):
                print('parameters:', params)
            minimizer_kwargs['callback'] = callback
        take_step = BoundedSteps(bounds)
        res = basinhopping(to_minimize, guess, niter=niter,
                            minimizer_kwargs=minimizer_kwargs,
                            take_step=take_step, disp=verbose)
        return res.x


    def hessian_latent(self, maps, obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1.e-3):
        '''
        compute the Hessian over the params and initial conditions
        maps: numpy.array
            maximum a posteriori
        obs: numpy.array
            the observed data without the initial datapoint
        fltr: boolean sequence or array
            True for observed and False for unobserved.
            e.g. if only Is is known for SIR with one age group, fltr = [False, False, True]
        Tf: float
            total time of the trajectory
        Nf: int
            total number of data points along the trajectory
        contactMatrix: callable
            a function that takes time as an input and outputs the contactMatrix
        eps: float, optional
            step size in the calculation of the Hessian
        '''
        dim = maps.shape[0]
        param_dim = dim - self.dim
        map_params = maps[:param_dim]
        map_x0 = maps[param_dim:]
        hess_params = self.latent_hess_params(map_params, map_x0, obs, fltr, Tf, Nf, contactMatrix,
                                                beta_rescale=beta_rescale, eps=eps)
        hess_init = self.latent_hess_init(map_x0, map_params, obs, fltr, Tf, Nf, contactMatrix,
                                                eps=0.5/self.N)
        return hess_params, hess_init

    def latent_hess_params(self, map_params, x0, obs, fltr, Tf, Nf, contactMatrix,
                                    beta_rescale=1, eps=1e-3):
        cdef Py_ssize_t j
        dim = map_params.shape[0]
        hess = np.empty((dim, dim))
        map_params[1] *= beta_rescale
        def minuslogP(y):
            y[1] /= beta_rescale
            parameters = self.make_params_dict(y)
            minuslogp = self.minus_logp_red(parameters, x0, obs, fltr, Tf, Nf, contactMatrix)
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

    def latent_hess_init(self, map_x0, params, obs, fltr, Tf, Nf, contactMatrix,
                                    eps=1e-6):
        cdef Py_ssize_t j
        dim = map_x0.shape[0]
        hess = np.empty((dim, dim))
        parameters = self.make_params_dict(params)
        model = self.make_det_model(parameters)
        def minuslogP(y):
            minuslogp = self.obtain_log_p_for_traj_red(y, obs, fltr, Tf, Nf, model, contactMatrix)
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
                            np.ndarray fltr, double Tf, int Nf, contactMatrix):
        cdef double minus_log_p
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj_red(x0, obs, fltr, Tf, Nf, model, contactMatrix)
        return minus_logp

    def make_det_model(self, parameters):
        pass # to be implemented in subclass

    def make_params_dict(self, params=None):
        pass # to be implemented in subclass

    def set_params(self, parameters):
        self.alpha = parameters['alpha']
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

    def estimate_cond_mean_cov(self, double [:] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] cov
            double [:, :] x
            double [:, :] cheb_coef
            double [:] time_points = np.linspace(t1, t2, self.steps)
            np.ndarray sigma0 = np.zeros((self.vec_size), dtype=DTYPE)
        x = self.integrate(x0, t1, t2, self.steps, model, contactMatrix)
        cheb_coef, _ = chebfit(time_points, x, 16, full=True) # even number seems to behave better
        def rhs(sig, t):
            self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
            self.lyapunov_fun(t, sig, cheb_coef)
            return self.dsigmadt
        def jac(sig, t):
            self.CM = np.einsum('ij,j->ij', contactMatrix(t), 1/self.fi)
            self.lyapunov_fun(t, sig, cheb_coef)
            return self.J_mat
        cov = odeint(rhs, sigma0, np.array([t1, t2]), Dfun=jac)
        return x[self.steps-1], self.convert_vec_to_mat(cov[1])

    def obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf, model, contactMatrix):
        cdef:
            Py_ssize_t dim=self.dim, i
            double [:, :] xm=np.empty((Nf, self.dim), dtype=DTYPE)
            double [:] time_points=np.linspace(0, Tf, Nf)
            double [:] xi, xf, dev, mean
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
            double [:] dsigdt=self.dsigmadt, B_vec=self.B_vec
            double [:, :] J_mat=self.J_mat
        for i in range(self.vec_size):
            dsigdt[i] = B_vec[i]
            for j in range(self.vec_size):
                dsigdt[i] += J_mat[i, j]*sig[j]


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

    cdef lyapunov_fun(self, double t, double [:] sig, double [:, :] cheb_coef):
        pass # to be implemented in subclasses

    cdef flatten_lyaponuv(self):
        cdef:
            double [:, :] I
            double [:, :] J_reshaped
        J_reshaped = np.reshape(self.J, (self.dim, self.dim))
        I = np.eye(self.dim)
        self.J_mat = (np.kron(I,J_reshaped) + np.kron(J_reshaped,I))
        self.J_mat[:, self.flat_indices1] += self.J_mat[:, self.flat_indices2]
        self.J_mat = self.J_mat[self.flat_indices][:, self.flat_indices]

    cpdef integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix):
        pass # to be implemented

cdef class SIR(SIR_type):

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

    cdef lyapunov_fun(self, double t, double [:] sig, double [:, :] cheb_coef):
        cdef:
            double [:] x, s, Ia, Is
            double [:, :] CM=self.CM
            double fsa=self.fsa, beta=self.beta
            Py_ssize_t m, n, M=self.M
        x = chebval(t, cheb_coef)
        s = x[0:M]
        Ia = x[M:2*M]
        Is = x[2*M:3*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+fsa*Is[n])
        self.jacobian(s, l)
        self.noise_correlation(s, Ia, Is, l)
        self.flatten_lyaponuv()
        self.compute_dsigdt(sig)


    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa, beta=self.beta
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = alpha*l[m]
            J[2, m, 0, m] = balpha*l[m]
            for n in range(M):
                J[0, m, 1, n] = -s[m]*beta*CM[m, n]
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 1, n] = alpha*s[m]*beta*CM[m, n]
                J[1, m, 2, n] = alpha*s[m]*beta*CM[m, n]*fsa
                J[2, m, 1, n] = balpha*s[m]*beta*CM[m, n]
                J[2, m, 2, n] = balpha*s[m]*beta*CM[m, n]*fsa
            J[1, m, 1, m] -= gIa
            J[2, m, 2, m] -= gIs

    cdef noise_correlation(self, double [:] s, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - alpha*l[m]*s[m]
            B[1, m, 1, m] = alpha*l[m]*s[m] + gIa*Ia[m]
            B[0, m, 2, m] = - balpha*l[m]*s[m]
            B[2, m, 2, m] = balpha*l[m]*s[m] + gIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    cpdef integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix):
        cdef:
            double [:] S0, Ia0, Is0
            double [:, :] sol
        S0 = x0[0:self.M]
        Ia0 = x0[self.M:2*self.M]
        Is0 = x0[2*self.M:3*self.M]
        data = model.simulate(S0, Ia0, Is0, contactMatrix, t2, steps, Ti=t1)
        sol = data['X']
        return sol

cdef class SEIR(SIR_type):
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


    cdef lyapunov_fun(self, double t, double [:] sig, double [:, :] cheb_coef):
        cdef:
            double [:] x, s, e, Ia, Is
            double [:, :] CM=self.CM
            double fsa=self.fsa, beta=self.beta
            Py_ssize_t m, n, M=self.M
        x = chebval(t, cheb_coef)
        s = x[0:M]
        e = x[M:2*M]
        Ia = x[2*M:3*M]
        Is = x[3*M:4*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+fsa*Is[n])
        self.jacobian(s, l)
        self.noise_correlation(s, e, Ia, Is, l)
        self.flatten_lyaponuv()
        self.compute_dsigdt(sig)

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs,
            double gE=self.gE, fsa=self.fsa, beta=self.beta
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE
            J[2, m, 1, m] = alpha*gE
            J[2, m, 2, m] = - gIa
            J[3, m, 1, m] = balpha*gE
            J[3, m, 3, m] = - gIs
            for n in range(M):
                J[0, m, 2, n] = -s[m]*beta*CM[m, n]
                J[0, m, 3, n] = -s[m]*beta*CM[m, n]*fsa
                J[1, m, 2, n] = s[m]*beta*CM[m, n]
                J[2, m, 3, n] = s[m]*beta*CM[m, n]*fsa

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] Ia, double [:] Is, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs, gE=self.gE
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + gE*e[m]
            B[1, m, 2, m] = -alpha*gE*e[m]
            B[1, m, 3, m] = -balpha*gE*e[m]
            B[2, m, 2, m] = alpha*gE*e[m]+gIa*Ia[m]
            B[3, m, 3, m] = balpha*gE*e[m]+gIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    cpdef integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix):
        cdef:
            double [:] s, e, Ia, Is
            double [:, :] sol
            Py_ssize_t M=self.M
        s = x0[0:M]
        e = x0[M:2*M]
        Ia = x0[2*M:3*M]
        Is = x0[3*M:4*M]
        data = model.simulate(s, e, Ia, Is, contactMatrix, t2, steps, Ti=t1)
        sol = data['X']
        return sol


cdef class SEAI5R(SIR_type):
    cdef:
        readonly double gE, gA, gIh, gIc, fh
        readonly np.ndarray hh, cc, mm

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 8, M, fi, N, steps)

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE    = parameters.get('gE')                       # recovery rate of E class
        self.gA    = parameters.get('gA')                       # recovery rate of A class
        self.gIh   = parameters.get('gIh')                      # recovery rate of Is
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ih
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


    cdef lyapunov_fun(self, double t, double [:] sig, double [:, :] cheb_coef):
        cdef:
            double [:] x, s, e, a, Ia, Is, Ih, Ic, Im
            double [:, :] CM=self.CM
            double fsa=self.fsa, fh=self.fh, beta=self.beta
            Py_ssize_t m, n, M=self.M
        x = chebval(t, cheb_coef)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        Ih = x[5*M:6*M]
        Ic = x[6*M:7*M]
        Im = x[7*M:8*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+a[n]+fsa*Is[n]+fh*Ih[n])
        self.jacobian(s, l)
        self.noise_correlation(s, e, a, Ia, Is, Ih, Ic, l)
        self.flatten_lyaponuv()
        self.compute_dsigdt(sig)


    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs, gIh=self.gIh, gIc=self.gIc
            double gE=self.gE, gA=self.gA, fsa=self.fsa, fh=self.fh, beta=self.beta
            double [:] hh=self.hh, cc=self.cc, mm=self.mm
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE
            J[2, m, 1, m] = gE
            J[2, m, 2, m] = - gA
            J[3, m, 2, m] = alpha*gA
            J[3, m, 3, m] = - gIa
            J[4, m, 2, m] = balpha*gA
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


    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] Ih, double [:] Ic, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, gIa=self.gIa, gIs=self.gIs, gIh=self.gIh, gIc=self.gIc, gE=self.gE, gA=self.gA
            double [:] mm=self.mm, cc=self.cc, hh=self.hh
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + gE*e[m]
            B[1, m, 2, m] = -gE*e[m]
            B[2, m, 2, m] = gE*e[m]+gA*a[m]
            B[2, m, 3, m] = -alpha*gA*a[m]
            B[2, m, 4, m] = -balpha*gA*a[m]
            B[3, m, 3, m] = alpha*gA*a[m]+gIa*Ia[m]
            B[4, m, 4, m] = balpha*gA*a[m] + gIs*Is[m]
            B[4, m, 5, m] = -hh[m]*gIs*Is[m]
            B[5, m, 5, m] = hh[m]*gIs*Is[m] + gIh*Ih[m]
            B[5, m, 6, m] = -cc[m]*gIh*Ih[m]
            B[6, m, 6, m] = cc[m]*gIh*Ih[m] + gIc*Ic[m]
            B[6, m, 7, m] = -mm[m]*gIc*Ic[m]
            B[7, m, 7, m] = mm[m]*gIc*Ic[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    cpdef integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix):
        cdef:
            double [:] s, e, a, Ia, Is, Ih, Ic, Im
            double [:, :] sol
            Py_ssize_t M=self.M
        s = x0[0:M]
        e = x0[M:2*M]
        a = x0[2*M:3*M]
        Ia = x0[3*M:4*M]
        Is = x0[4*M:5*M]
        Ih = x0[5*M:6*M]
        Ic = x0[6*M:7*M]
        Im = x0[7*M:8*M]
        data = model.simulate(s, e, a, Ia, Is, Ih, Ic, Im, contactMatrix, t2, steps, Ti=t1)
        sol = data['X'][:, :8*M]
        return sol


cdef class SEAIRQ(SIR_type):
    cdef:
        readonly double gE, gA, tE, tA, tIa, tIs

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 6, M, fi, N, steps)

    def infer_control(self, guess, x, Tf, Nf, generator, bounds, niter=2, verbose=False, ftol=1e-6, eps=1e-5):
        '''
        guess: numpy.array
            initial guess for the control parameter values
        Tf: float
            total time of the trajectory
        Nf: float
            number of data points along the trajectory
        generator: pyross.contactMatrix
        bounds: 2d numpy.array
            bounds for the parameters.
            Note that the upper bound must be smaller than the absolute physical upper bound minus epsilon
        niter: int
            number of iterations of basinhopping
        verbose: bool
            whether to print messages
        ftol: double
            relative tolerance of logp
        eps: double
            size of steps taken by L-BFGS-B algorithm for the calculation of Hessian
        '''
        def to_minimize(params):
            tau_control = params
            parameters = self.make_params_dict()
            parameters['tE'] = tau_control[0]
            parameters['tA'] = tau_control[1]
            parameters['tIa'] = tau_control[2]
            parameters['tIs'] = tau_control[3]
            model =self.make_det_model(parameters)
            contactMatrix = generator.constant_contactMatrix()
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
            return minus_logp

        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
        if verbose:
            def callback(params):
                print('parameters:', params)
            minimizer_kwargs['callback'] = callback
        take_step = BoundedSteps(bounds)
        res = basinhopping(to_minimize, guess, niter=niter,
                            minimizer_kwargs=minimizer_kwargs,
                            take_step=take_step, disp=verbose)
        return res.x, res.nit

    def latent_infer_control(self, np.ndarray guess, np.ndarray x0, np.ndarray obs, np.ndarray fltr,
                            double Tf, Py_ssize_t Nf, generator, np.ndarray bounds,
                            verbose=False, Py_ssize_t niter=1,
                            double ftol=1e-5, double eps=1e-4):
        def to_minimize(params):
            tau_control = params
            parameters = self.make_params_dict()
            parameters['tE'] = tau_control[0]
            parameters['tA'] = tau_control[1]
            parameters['tIa'] = tau_control[2]
            parameters['tIs'] = tau_control[3]
            model = self.make_det_model(parameters)
            contactMatrix = generator.constant_contactMatrix()
            minus_logp = self.obtain_log_p_for_traj_red(x0, obs[1:], fltr, Tf, Nf, model, contactMatrix)
            return minus_logp
        options={'eps': eps, 'ftol': ftol, 'disp': verbose}
        minimizer_kwargs = {'method':'L-BFGS-B', 'bounds': bounds, 'options': options}
        if verbose:
            def callback(params):
                print('parameters:', params)
            minimizer_kwargs['callback'] = callback
        take_step = BoundedSteps(bounds)
        res = basinhopping(to_minimize, guess, niter=niter,
                            minimizer_kwargs=minimizer_kwargs,
                            take_step=take_step, disp=verbose)
        return res.x

    def set_params(self, parameters):
        super().set_params(parameters)
        self.gE    = parameters.get('gE')                       # recovery rate of E class
        self.gA    = parameters.get('gA')                       # recovery rate of A class
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

    cdef lyapunov_fun(self, double t, double [:] sig, double [:, :] cheb_coef):
        cdef:
            double [:] x, s, e, a, Ia, Is, Q
            double [:, :] CM=self.CM
            double beta=self.beta, fsa=self.fsa
            Py_ssize_t m, n, M=self.M
        x = chebval(t, cheb_coef)
        s = x[0:M]
        e = x[M:2*M]
        a = x[2*M:3*M]
        Ia = x[3*M:4*M]
        Is = x[4*M:5*M]
        q = x[5*M:6*M]
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        for m in range(M):
            for n in range(M):
                l[m] += beta*CM[m,n]*(Ia[n]+a[n]+fsa*Is[n])
        self.jacobian(s, l)
        self.noise_correlation(s, e, a, Ia, Is, q, l)
        self.flatten_lyaponuv()
        self.compute_dsigdt(sig)

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, beta=self.beta
            double gE=self.gE, gA=self.gA, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa
            double tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - gE - tE
            J[2, m, 1, m] = gE
            J[2, m, 2, m] = - gA - tE
            J[3, m, 2, m] = alpha*gA
            J[3, m, 3, m] = - gIa - tIa
            J[4, m, 2, m] = balpha*gA
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

    cdef noise_correlation(self, double [:] s, double [:] e, double [:] a, double [:] Ia, double [:] Is, double [:] q, double [:] l):
        cdef:
            Py_ssize_t m, M=self.M
            double alpha=self.alpha, balpha=1-self.alpha, beta=self.beta
            double gIa=self.gIa, gIs=self.gIs, gE=self.gE, gA=self.gA
            double tE=self.tE, tA=self.tE, tIa=self.tIa, tIs=self.tIs
            double [:, :, :, :] B = self.B
        for m in range(M): # only fill in the upper triangular form
            B[0, m, 0, m] = l[m]*s[m]
            B[0, m, 1, m] =  - l[m]*s[m]
            B[1, m, 1, m] = l[m]*s[m] + (gE+tE)*e[m]
            B[1, m, 2, m] = -gE*e[m]
            B[2, m, 2, m] = gE*e[m]+(gA+tA)*a[m]
            B[2, m, 3, m] = -alpha*gA*a[m]
            B[2, m, 4, m] = -balpha*gA*a[m]
            B[3, m, 3, m] = alpha*gA*a[m]+(gIa+tIa)*Ia[m]
            B[4, m, 4, m] = balpha*gA*a[m] + (gIs+tIs)*Is[m]
            B[1, m, 5, m] = -tE*e[m]
            B[2, m, 5, m] = -tA*a[m]
            B[3, m, 5, m] = -tIa*Ia[m]
            B[4, m, 5, m] = -tIs*Is[m]
            B[5, m, 5, m] = tE*e[m]+tA*a[m]+tIa*Ia[m]+tIs*Is[m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]

    cpdef integrate(self, double [:] x0, double t1, double t2, Py_ssize_t steps, model, contactMatrix):
        cdef:
            double [:] s, e, a, Ia, Is, q
            double [:, :] sol
            Py_ssize_t M=self.M
        s = x0[0:M]
        e = x0[M:2*M]
        a = x0[2*M:3*M]
        Ia = x0[3*M:4*M]
        Is = x0[4*M:5*M]
        q = x0[5*M:]
        data = model.simulate(s, e, a, Ia, Is, q, contactMatrix, t2, steps, Ti=t1)
        sol = data['X']
        return sol
