from scipy import linalg
from scipy.integrate import odeint
from scipy.optimize import minimize
import  numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
cimport numpy as np
cimport cython

import pyross.deterministic
from libc.math cimport sqrt, log, INFINITY
cdef double PI = 3.14159265359


DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR_type:
    cdef:
        readonly Py_ssize_t nClass, N, M, steps, dim, vec_size
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray fi, CM, dsigmadt, J, B, J_mat, B_vec,
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

        # preparing the indices
        self.rows, self.cols = np.triu_indices(self.dim)
        self.flat_indices = np.ravel_multi_index((self.rows, self.cols), (self.dim, self.dim))
        r, c = np.triu_indices(self.dim, k=1)
        self.flat_indices1 = np.ravel_multi_index((r, c), (self.dim, self.dim))
        self.flat_indices2 = np.ravel_multi_index((c, r), (self.dim, self.dim))



    def inference(self, guess, x, Tf, Nf, contactMatrix, method='L-BFGS-B', fatol=0.01, eps=1e-5):

        def to_minimize(params):
            parameters = {'alpha':params[0], 'beta':params[1], 'gIa':params[2], 'gIs':params[3],'fsa':self.fsa}
            self.set_params(parameters)
            model = self.make_det_model(parameters)
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
            return minus_logp

        if method == 'Nelder-Mead':
            options={'fatol': fatol, 'adaptive': True}
            res = minimize(to_minimize, guess, method='Nelder-Mead', options=options)
        elif method == 'L-BFGS-B':
            bounds = [(eps, INFINITY), (eps, INFINITY), (eps, INFINITY), (eps, INFINITY)]
            options={'eps': eps}
            res = minimize(to_minimize, guess, bounds=bounds, method='L-BFGS-B', options=options)
        else:
            print('optimisation method not implemented')
            return
        return res.x, res.nit

    def obtain_minus_log_p(self, parameters, double [:, :] x, double Tf, int Nf, contactMatrix):
        cdef double minus_log_p
        self.set_params(parameters)
        model = self.make_det_model(parameters)
        minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        return minus_logp

    def make_det_model(self, parameters):
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

    cdef double log_cond_p(self, double [:] x, double [:, :] cov):
        cdef:
            double [:, :] invcov
            double log_cond_p
            double det
        invcov = linalg.inv(cov)
        det = linalg.det(cov)
        log_cond_p = - np.dot(x, np.dot(invcov, x))*(self.N/2) - (self.dim/2)*log(2*PI)
        log_cond_p -= (log(det) - log(self.N))/2
        return log_cond_p

    def estimate_cond_mean_cov(self, double [:] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :] cov
            double [:, :] x
            double [:, :] cheb_coef
            double [:] time_points = np.linspace(t1, t2, self.steps)
            np.ndarray sigma0 = np.zeros((self.vec_size), dtype=DTYPE)
        x = self.integrate(x0, t1, t2, model, contactMatrix)
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

    cpdef integrate(self, double [:] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:] S0, Ia0, Is0
            double [:, :] sol
        x_reshaped = np.reshape(x0, (3, self.M))
        S0 = x0[0:self.M]
        Ia0 = x0[self.M:2*self.M]
        Is0 = x0[2*self.M:3*self.M]
        data = model.simulate(S0, Ia0, Is0, contactMatrix, t2, self.steps, Ti=t1)
        sol = data['X']
        return sol

cdef class SIR(SIR_type):

    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 3, M, fi, N, steps)

    def make_det_model(self, parameters):
        return pyross.deterministic.SIR(parameters, self.M, self.fi)

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
        cdef:
            double [:] dsigdt=self.dsigmadt, B_vec=self.B_vec
            double [:, :] J_mat=self.J_mat
        for i in range(self.vec_size):
            dsigdt[i] = B_vec[i]
            for j in range(self.vec_size):
                dsigdt[i] += J_mat[i, j]*sig[j]


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

cdef class SEIR(SIR_type):
    def __init__(self, parameters, M, fi, N, steps):
        super().__init__(parameters, 4, M, fi, N, steps)

    def make_det_model(self, parameters):
        return pyross.deterministic.SEIR(parameters, self.M, self.fi)

    # more to come 
