import pyross
cimport pyross.inference
cimport pyross.tsi.deterministic
import numpy as np
cimport numpy as np
cimport cython

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.uint8_t BOOL_t

cdef class SIR(pyross.inference.SIR_type):
    cdef:
        readonly pyross.tsi.deterministic.SIR det_model
        readonly np.ndarray gI
        readonly int kI
        readonly double tsi_max, dt

    def __init__(self, parameters, M, fi, kI, tsi_max, Omega=1, steps=2):
        self.param_keys = ['beta', 'gI']
        self.kI = kI
        self.tsi_max = tsi_max
        self.dt = tsi_max/float(kI)
        super().__init__(parameters, (kI+1), M, fi, Omega, steps, 'euler', 'euler', 1e-4, 1e-4)
        self.class_index_dict = {'S':0, 'I':1} # only the S index matters
        self.set_det_model(parameters)

    def set_det_model(self, parameters):
        self.det_model = pyross.tsi.deterministic.SIR(parameters, self.M, self.fi*self.Omega, self.kI, self.tsi_max)

    def make_params_dict(self):
        parameters = {'beta':self.beta, 'gI':self.gI}
        return parameters

    def set_params(self, parameters):
        self.set_det_model(parameters)
        self.beta = parameters['beta']
        self.gI = parameters['gI']

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                                b_matrix=True, jacobian=False):
        cdef:
            double [:] S, I
            Py_ssize_t M=self.M, nClass=self.nClass
        S = x[0:M]
        I = x[M:]
        self.CM = self.contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(I, l)
        if b_matrix:
            self.B = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
            self.noise_correlation(S, I, l)
        if jacobian:
            self.J = np.zeros((nClass, M, nClass, M), dtype=DTYPE)
            self.jacobian(S, l)

    cdef fill_lambdas(self, double [:] I, double [:] l):
        cdef:
            double [:, :] CM=self.CM
            double [:] beta=self.beta, fi=self.fi
            Py_ssize_t j, k, n, M=self.M, kI=self.kI
        for j in range(M):
            for k in range(kI-1):
                for n in range(M):
                    l[j] += CM[j,n]*0.5*(beta[k]*I[k*M+n] + beta[k+1]*I[(k+1)*M+n])/fi[n]

    cdef jacobian(self, double [:] s, double [:] l):
        cdef:
            Py_ssize_t m, n, k, M=self.M, kI=self.kI, dim=self.dim
            double [:] beta=self.beta, gI=self.gI, fi=self.fi
            double [:, :, :, :] J = self.J
            double [:, :] CM=self.CM
            double dt=self.dt
        for m in range(M):
            J[0, m, 0, m] = -l[m]
            J[1, m, 0, m] = l[m]
            J[1, m, 1, m] = - 1/dt
            J[kI, m, kI-1, m] = - gI[kI-2] + 1/dt
            for n in range(M):
                J[0, m, 1, n]  = -s[m]*beta[0]*CM[m, n]/(fi[n]*2)
                J[1, m, 1, n] +=  s[m]*beta[0]*CM[m, n]/(fi[n]*2)
                J[0, m, kI, n] = -s[m]*beta[kI-1]*CM[m, n]/(fi[n]*2)
                J[1, m, kI, n] = s[m]*beta[kI-1]*CM[m, n]/(fi[n]*2)
            for k in range(1, kI-1):
                J[k+1, m, k+1, m] = -1/dt
                J[k+1, m, k, m] = -gI[k-1] + 1/dt
                for n in range(M):
                    J[0, m, 1+k, n] = -s[m]*beta[k]*CM[m, n]/fi[n]
                    J[1, m, 1+k, n] = s[m]*beta[k]*CM[m, n]/fi[n]

        self.J_mat = self.J.reshape((dim, dim))

    cdef noise_correlation(self, double [:] S, double [:] I, double [:] l):
        cdef:
            Py_ssize_t m, n, k, M=self.M, kI=self.kI, dim=self.dim
            double [:] beta=self.beta, gI=self.gI
            double [:, :, :, :] B = self.B
        for m in range(M):
            B[0, m, 0, m] = S[m]*l[m]
            B[1, m, 1, m] = S[m]*l[m]
            B[0, m, 1, m] = - S[m]*l[m]
            B[1, m, 0, m] = - S[m]*l[m]
            for k in range(1, kI):
                B[k+1, m, k+1, m] = gI[k-1]*I[(k-1)*M+m]
        self.B_vec = self.B.reshape((self.dim, self.dim))[(self.rows, self.cols)]
