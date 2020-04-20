from scipy import linalg
from scipy.integrate import simps, trapz
from scipy.optimize import minimize

import  numpy as np
cimport numpy as np
cimport cython

from pyross.deterministic import SIR as detSIR
from libc.math cimport sqrt, log, INFINITY
cdef double PI = 3.14159265359


DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR:
    cdef:
        readonly Py_ssize_t N, M, steps
        readonly double alpha, beta, gIa, gIs, fsa
        readonly np.ndarray fi, time_evol_op, B


    def __init__(self, parameters, M, fi, N, steps):
        self.N = N
        self.M = M
        self.fi = fi
        self.steps = steps
        self.set_params(parameters)

        self.time_evol_op = np.empty((steps, 3, M, 3, M), dtype=DTYPE)
        self.B = np.zeros((steps, M, 3, 3), dtype=DTYPE)

    def inference(self, guess, x, Tf, Nf, contactMatrix, method='Nelson-Mead', fatol=None):

        def to_minimize(params):
            parameters = {'alpha':params[0], 'beta':params[1], 'gIa':params[2], 'gIs':params[3],'fsa':self.fsa}
            self.set_params(parameters)
            model = detSIR(parameters, self.M, self.fi)
            minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
            return minus_logp

        if method == 'Nelson-Mead':
            options={'fatol': fatol, 'adaptive': True}
            res = minimize(to_minimize, guess, method='Nelder-Mead', options=options)
        elif method == 'L-BFGS-B':
            bounds = [(0, INFINITY), (0, INFINITY), (0, INFINITY), (0, INFINITY)]
            options={'eps': 1e-5}
            res = minimize(to_minimize, guess, bounds=bounds, method='L-BFGS-B', options=options)
        else:
            print('optimisation method not implemented')
            return
        return res.x, res.nit

    def obtain_minus_log_p(self, parameters, double [:, :, :] x, double Tf, int Nf, contactMatrix):
        cdef double minus_log_p
        self.set_params(parameters)
        model = detSIR(parameters, self.M, self.fi)
        minus_logp = self.obtain_log_p_for_traj(x, Tf, Nf, model, contactMatrix)
        return minus_logp

    def set_params(self, parameters):
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.gIa = parameters['gIa']
        self.gIs = parameters['gIs']
        self.fsa = parameters['fsa']

    cdef double obtain_log_p_for_traj(self, double [:, :, :] x, double Tf, int Nf, model, contactMatrix):
        cdef:
            double log_p = 0
            double [:] time_points = np.linspace(0, Tf, Nf)
            double [:, :] xi, xf, dev, mean
            double [:, :, :, :] cov
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

    cdef double log_cond_p(self, double [:, :] x, double [:, :, :, :] cov):
        cdef:
            Py_ssize_t dim = 3*self.M
            double [:, :] cov_mat, invcov
            double [:] x_vec
            double log_cond_p
        cov_mat = np.reshape(cov, (dim, dim))
        invcov = linalg.inv(cov_mat)
        x_vec = np.reshape(x, dim)
        log_cond_p = - np.dot(x_vec, np.dot(invcov, x_vec))*(self.N/2) - (dim/2)*log(2*PI)
        log_cond_p -= (log(linalg.det(cov_mat)) - log(self.N))/2
        return log_cond_p

    cpdef estimate_cond_mean_cov(self, double [:, :] x0, double t1, double t2, model, contactMatrix):
        cdef:
            double [:, :, :, :, :] integrand
            double [:, :, :, :] cov
            double [:, :, :] x
            double [:, :] T1, T2, invT1
            double [:, :, :, :] T_between_ij, temp
            int dim = 3*self.M
            Py_ssize_t j, steps=self.steps
        x = self.integrate(x0, t1, t2, model, contactMatrix)
        self.obtain_time_evol_op(x0, t1, t2, model, contactMatrix)
        self.get_noise_correlation(x, t1, t2, contactMatrix)
        integrand = np.empty((steps, 3, self.M, 3, self.M), dtype=DTYPE)
        T2 = np.reshape(self.time_evol_op[steps-1], (dim, dim))
        for j in range(steps):
            T1 = np.reshape(self.time_evol_op[j], (dim, dim))
            invT1 = np.linalg.inv(T1)
            T_between_ij = np.reshape(np.matmul(invT1, T2), (3, self.M, 3, self.M))
            temp = np.einsum('aibj,jbc,dkcj->aidk', T_between_ij, self.B[j], T_between_ij)
            integrand[j] = temp
        cov = simps(integrand, axis=0, dx=(t2-t1)/steps)
        return x[steps-1], cov


    cpdef integrate(self, double [:, :] x0, double t1, double t2, model, contactMatrix):
        cdef double [:] S0, Ia0, Is0
        cdef double [:, :, :] sol
        S0 = x0[0]
        Ia0 = x0[1]
        Is0 = x0[2]
        data = model.simulate(S0, Ia0, Is0, contactMatrix, t2, self.steps, Ti=t1)
        sol = np.reshape(data['X'], (self.steps, 3, self.M))
        return sol

    cdef obtain_time_evol_op(self, double [:, :] x0, double t1, double t2, model, contactMatrix):
        cdef:
            Py_ssize_t a, m, i, j, k, M=self.M, steps=self.steps
            double epsilon = 1e-3
            double sqrtN = sqrt(self.N)
            double [:, :, :, :, :] time_evol_op = self.time_evol_op
            double [:, :, :] pos, neg
        pos = np.empty((steps, 3, self.M))
        neg = np.empty((steps, 3, self.M))
        for a in range(3): # three classes
            for m in range(M): # age groups
                x0[a, m] += epsilon/sqrtN
                pos = self.integrate(x0, t1, t2, model, contactMatrix)
                x0[a, m] -= 2*epsilon/sqrtN
                neg = self.integrate(x0, t1, t2, model, contactMatrix)
                for i in range(steps):
                    for j in range(3):
                        for k in range(M):
                            time_evol_op[i, j, k, a, m] = (pos[i, j, k] - neg[i, j, k])/(2*epsilon/sqrtN)
                x0[a, m] += epsilon/sqrtN
        return

    cdef get_noise_correlation(self, double [:, :, :] x, double t1, double t2, contactMatrix):
        cdef:
            Py_ssize_t i, m, n, M=self.M, steps=self.steps
            double [:, :, :, :] B
            double [:] time_points = np.linspace(t1, t2, steps)
            double [:] fi = self.fi
            double [:, :] CM
            double t, l, s, Ia, Is, alpha=self.alpha, beta=self.beta, gIa=self.gIa, gIs=self.gIs, fsa=self.fsa

        B = self.B
        for i in range(steps):
            t = time_points[i]
            CM = contactMatrix(t)
            for m in range(M):
                l = 0
                for n in range(M):
                    Ia = x[i, 1, n]
                    Is = x[i, 2, n]
                    l += beta*CM[m, n]*(Ia + fsa*Is)/fi[n]
                s = x[i, 0, m]
                Ia = x[i, 1, m]
                Is = x[i, 2, m]
                B[i, m, 0, 0] = l*s
                B[i, m, 0, 1] = - alpha*l*s
                B[i, m, 1, 0] = - alpha*l*s
                B[i, m, 1, 1] = alpha*l*s + gIa*Ia
                B[i, m, 0, 2] = - (1-alpha)*l*s
                B[i, m, 2, 0] = - (1-alpha)*l*s
                B[i, m, 2, 2] = (1-alpha)*l*s + gIs*Is
        return
