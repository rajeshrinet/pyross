import  numpy as np
cimport numpy as np
cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR_type:

    cdef:
        readonly Py_ssize_t nClass, M, steps, dim, vec_size
        readonly double Omega, rtol_det, rtol_lyapunov
        readonly np.ndarray beta, gIa, gIs, fsa
        readonly np.ndarray alpha, fi, CM, dsigmadt, J, B, J_mat, B_vec, U
        readonly np.ndarray flat_indices1, flat_indices2, flat_indices, rows, cols
        readonly str det_method, lyapunov_method
        readonly dict class_index_dict
        readonly list param_keys, _interp
        readonly object contactMatrix
        readonly bint param_mapping_enabled

    cdef np.ndarray _get_r_from_x(self, np.ndarray x)

    cdef double _penalty_from_negative_values(self, np.ndarray x0)

    cpdef find_fastest_growing_lin_mode(self, double t)

    cdef double _obtain_logp_for_traj(self, double [:, :] x, double Tf,
                                     Py_ssize_t inter_steps=*)

    cdef double _obtain_logp_for_lat_traj(self, double [:] x0, double [:] obs_flattened,
                                          np.ndarray fltr, double Tf, tangent=*,
                                          Py_ssize_t inter_steps=*)

    cdef double _obtain_logp_for_traj_tangent(self, double [:, :] x, double Tf)

    cdef double _log_cond_p(self, double [:] x, double [:, :] cov)

    cdef _estimate_cond_cov(self, object sol, double t1, double t2)

    cpdef obtain_full_mean_cov(self, double [:] x0, double Tf, Py_ssize_t Nf, Py_ssize_t inter_steps=*)

    cpdef interpolate_euler(self, t)

    cpdef obtain_full_mean_cov_tangent_space(self, double [:] x0, double Tf, Py_ssize_t Nf, Py_ssize_t inter_steps=*)

    cpdef _rhs0(self, t, xt)

    cdef _compute_dsigdt(self, double [:] sig)

    cpdef convert_vec_to_mat(self, double [:] cov)

    cdef compute_jacobian_and_b_matrix(self, double [:] x, double t,
                                             b_matrix=*, jacobian=*)
