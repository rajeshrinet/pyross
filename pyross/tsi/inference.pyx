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
        readonly int kI
        readonly double tsi_max

    def __init__(self, parameters, M, fi, kI, tsi_max, Omega=1, steps=4):
        self.param_keys = ['beta', 'gI']
        self.kI = kI
        self.tsi_max = tsi_max
        super().__init__(parameters, (kI+1), M, fi, Omega, steps, 'Euler', 'Euler', 1e-4, 1e-4)
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
            Py_ssize_t M=self.M
        S = x[0:M]
        I = x[M:]
        self.CM = self.contactMatrix(t)
        cdef double [:] l=np.zeros((M), dtype=DTYPE)
        self.fill_lambdas(I, l)
        if b_matrix:
            self.noise_correlation(S, I, l)
        if jacobian:
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
