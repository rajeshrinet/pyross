import  numpy as np
import scipy as sp
import scipy.linalg
import copy
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log, sin, cos, atan2, sqrt
from cython.parallel import prange
cdef double PI = 3.1415926535
from scipy.sparse import spdiags
from scipy.sparse.linalg.eigen.arpack import eigs, ArpackNoConvergence
from scipy.misc import derivative



def parse_model_spec(model_spec, param_keys):

    # First, extract the classes
    class_list = model_spec['classes']
    nClass = len(class_list) # total number of classes

    # Make dictionaries for class and parameter index look-up
    class_index_dict = {class_name:i for (i, class_name) in enumerate(class_list)}
    params_index_dict = {param: i for (i, param) in enumerate(param_keys)}

    try:
        # dictionaries of all constant, linear and all infection terms
        constant_dict = {}
        linear_dict = {}
        infection_dict = {}
        finres_dict = {}

        for class_name in class_list:
            reaction_dict = model_spec[class_name]
            if 'constant' in reaction_dict.keys():
                constant_dict[class_name] = reaction_dict['constant']
            if 'linear' in reaction_dict.keys():
                linear_dict[class_name] = reaction_dict['linear']
            if 'infection' in reaction_dict.keys():
                infection_dict[class_name] = reaction_dict['infection']
            if 'finite-resource' in reaction_dict.keys():
                finres_dict[class_name] = reaction_dict['finite-resource']

        # parse the constant term
        constant_term_set = set() # used to check for duplicates
        constant_term_list = []
        for (k, val) in constant_dict.items():
            for vv in val:
                if len(vv)==1:
                    (rate,) = vv
                    if (k, rate) in constant_term_set:
                        raise Exception('Duplicate constant term: {}, {}'.format(k, rate))
                    else:
                        constant_term_set.add((k, rate))
                        overdispersion_index = -1
                elif len(vv)==2:
                    (rate, overdispersion) = vv
                    if (k, rate) in constant_term_set:
                        raise Exception('Duplicate constant term: {}, {}, {}'.format(k, rate, overdispersion))
                    else:
                        constant_term_set.add((k, rate, overdispersion))
                        overdispersion_index = params_index_dict[overdispersion]
                sign = 1
                class_index = class_index_dict[k]
                if rate.startswith('-'):
                    rate = rate[1:]
                    sign = -1
                rate_index = params_index_dict[rate]
                constant_term_list.append([rate_index, class_index, sign, overdispersion_index])


        if len(constant_term_list) > 0: # add one class for Ni
            class_index_dict['Ni'] = nClass
            nClass += 1 # add one class for Ni

        # parse the linear terms into a list of [rate_index, reagent_index] and a dictionary for the product
        linear_terms_set = set() # used to check for duplicates
        linear_terms_list = [] # collect all linear terms
        linear_terms_destination_dict = {} # a dictionary for the product
        for (k, val) in linear_dict.items():
            for vv in val:
                if len(vv) == 2:
                    (reagent, rate) = vv
                    if (reagent,rate) in linear_terms_set:
                        raise Exception('Duplicate linear terms: {}, {}'.format(reagent, rate))
                    else:
                        linear_terms_set.add((reagent, rate))
                        overdispersion_index = -1
                elif len(vv) == 3:
                    (reagent, rate, overdispersion) = vv
                    if (reagent, rate, overdispersion) in linear_terms_set:
                        raise Exception('Duplicate linear terms: {}, {}, {}'.format(reagent, rate, overdispersion))
                    else:
                        linear_terms_set.add((reagent, rate, overdispersion))
                        overdispersion_index = params_index_dict[overdispersion]
                else:
                    raise Exception('Wrong number of values in linear term: {}, {} (2 or 3 allowed)'.format(k, vv))
                reagent_index = class_index_dict[reagent]
                if rate.startswith('-'):
                    rate = rate[1:]
                    rate_index = params_index_dict[rate]
                    linear_terms_list.append([rate_index, reagent_index, -1, overdispersion_index])
                else:
                    rate_index = params_index_dict[rate]
                    linear_terms_destination_dict[(rate_index, reagent_index, overdispersion_index)] = class_index_dict[k]


        # parse the infection terms into a list of [rate_index, reagent_index, susceptible_index] and a dictionary for the product
        infection_terms_set = set() # used to check to duplicates
        infection_terms_list = [] # collect all infection terms
        infection_terms_destination_dict = {} # a dictionary for the product
        for (k, val) in infection_dict.items():
            for vv in val:
                if len(vv) == 3:  # no overdispersion specified
                    (reagent, susceptible, rate) = vv
                    if (reagent, susceptible, rate) in infection_terms_set:
                        raise Exception('Duplicate infection terms: {}, {}, {}'.format(reagent, susceptible, rate))
                    else:
                        infection_terms_set.add((reagent, susceptible, rate))
                        overdispersion_index = -1
                elif len(vv) == 4:  # overdispersion specified
                    (reagent, susceptible, rate, overdispersion) = vv
                    if (reagent, susceptible, rate, overdispersion) in infection_terms_set:
                        raise Exception('Duplicate infection terms: {}, {}, {}, {}'.format(reagent, susceptible, rate, overdispersion))
                    else:
                        infection_terms_set.add((reagent, susceptible, rate, overdispersion))
                        overdispersion_index = params_index_dict[overdispersion]
                else:
                    raise Exception('Wrong number of values in infection term: {}, {} (3 or 4 allowed)'.format(k, vv))
                reagent_index = class_index_dict[reagent]
                susceptible_index = class_index_dict[susceptible]
                if rate.startswith('-'):
                    rate = rate[1:]
                    rate_index = params_index_dict[rate]
                    infection_terms_list.append([rate_index, reagent_index, susceptible_index, -1, overdispersion_index])
                else:
                    rate_index = params_index_dict[rate]
                    infection_terms_destination_dict[(rate_index, reagent_index, susceptible_index, overdispersion_index)] = class_index_dict[k]


        # parse the finite resource-terms into
        #  1) a list resource_list = [rate_index, (class_index0, priority_index0), (class_index1, priority_index1), ...]
        #  2) a list finres_terms_list = [resource_index, priority_index, positivity_index, class_index, origin_class_index, destination_class_index]

        finres_terms_set = set() # used to check to duplicates
        finres_terms_dict = {} # collect all finite-resource terms
        finres_terms_list = [] # collect all finite-resource terms
        resource_list = []
        resource_dict = {}
        resource_count = 0
        for (k, val) in finres_dict.items():
            for vv in val:
                if len(vv) == 4:  # no overdispersion specified
                    (reagent, rate, priority, prob) = vv
                    if (reagent, rate, priority, prob) in finres_terms_set:
                        raise Exception('Duplicate finite-resource terms: {}, {}, {}, {}'.format(reagent, rate, priority, prob))
                    else:
                        finres_terms_set.add((reagent, rate, priority, prob))
                        overdispersion_index = -1
                elif len(vv) == 5:  # overdispersion specified 
                    (reagent, rate, priority, prob, overdispersion) = vv
                    if (reagent, rate, priority, prob, overdispersion) in finres_terms_set:
                        raise Exception('Duplicate finite-resource terms: {}, {}, {}, {}, {}'.format(reagent, rate, priority, prob, overdispersion))
                    else:
                        finres_terms_set.add((reagent, rate, priority, prob, overdispersion))
                        overdispersion_index = params_index_dict[overdispersion]
                else:
                    raise Exception('Wrong number of values in finite resource term: {}, {} (4 or 5 allowed)'.format(k, vv))
                class_index = class_index_dict[reagent]
                neg_rate = rate.startswith('-')
                if neg_rate:
                    rate = rate[1:]
                rate_index = params_index_dict[rate]
                priority_index = params_index_dict[priority]
                prob_index = params_index_dict[prob]
                if rate not in resource_dict.keys():
                    resource_dict[rate] = resource_count
                    resource_list.append([rate_index])
                    resource_count += 1
                resource_index = resource_dict[rate]
                if (class_index, priority_index) not in resource_list[resource_index]:
                    resource_list[resource_index].append((class_index, priority_index))
                dict_key = (resource_index, priority_index, prob_index, class_index, overdispersion_index)
                if dict_key not in finres_terms_dict.keys():
                    finres_terms_dict[dict_key] = [-1,-1]
                if neg_rate:
                    finres_terms_dict[dict_key][0] = class_index_dict[k]
                else:
                    finres_terms_dict[dict_key][1] = class_index_dict[k]
        for (k,val) in finres_terms_dict.items():
            (resource_index, priority_index, prob_index, class_index, overdispersion_index) = k
            finres_terms_list.append([resource_index, priority_index, prob_index, class_index, *val, overdispersion_index])

        # parse parameters for testing (for SppQ only, otherwise ignore empty parameters lists)
        test_pos_list = []
        test_freq_list = []
        if "test_pos" in model_spec and "test_freq" in model_spec:
            test_pos_strings = model_spec['test_pos']
            test_freq_strings = model_spec['test_freq']

            if len(test_pos_strings) != len(class_list)+1 or len(test_freq_strings) != len(class_list)+1:
                raise Exception('Test parameters must be specified for every class (including R). {} parameters expected'.format(len(class_list)+1))

            for rate in test_pos_strings:
                rate_index = params_index_dict[rate]
                test_pos_list.append(rate_index)
            for rate in test_freq_strings:
                rate_index = params_index_dict[rate]
                test_freq_list.append(rate_index)

            for class_name in class_list: # add quarantine classes
                class_index_dict[class_name+'Q'] = nClass
                nClass += 1
            class_index_dict['NiQ'] = nClass
            nClass += 1




    except KeyError:
        raise Exception('No reactions for some classes. Please check model_spec again')

    # set the product index
    set_destination(linear_terms_list, linear_terms_destination_dict)
    set_destination(infection_terms_list, infection_terms_destination_dict)






    res = (nClass, class_index_dict, np.array(constant_term_list, dtype=np.intc, ndmin=2),
                                     np.array(linear_terms_list, dtype=np.intc, ndmin=2),
                                     np.array(infection_terms_list, dtype=np.intc, ndmin=2),
                                     np.array(finres_terms_list, dtype=np.intc, ndmin=2),
                                     np.array(resource_list, dtype=object),
                                     np.array(test_pos_list, dtype=np.intc, ndmin=1),
                                     np.array(test_freq_list, dtype=np.intc, ndmin=1))
    return res


def set_destination(term_list, destination_dict):
    '''
    A function used by parse_model_spec that sets the product_index
    '''
    for term in term_list:
        indices = tuple(term[:-2]+term[-1:])
        if indices in destination_dict.keys():
            product_index = destination_dict[indices]
            term[-2] = product_index
            

def age_dep_rates(rate, int M, str name, bint check_length=True):
    if np.size(rate)==1:
        return rate*np.ones(M)
    elif np.size(rate)==M or check_length==False:
        return rate
    else:
        raise Exception('{} can be a number or an array of size M'.format(name))


DTYPE = np.float
ctypedef np.float_t DTYPE_t

cpdef forward_euler_integration(f, double [:] x, double t1, double t2, Py_ssize_t steps, Py_ssize_t inter_steps=1):
    cdef:
        double dt=(t2-t1)/(inter_steps*(steps-1)),t=t1
        double [:] fx
        Py_ssize_t i, j, size=x.shape[0]
        double [:, :] sol=np.empty((inter_steps*steps, size), dtype=DTYPE)
    for j in range(size):
        sol[0, j] = x[j]
    for i in range(1, inter_steps*steps):
        fx = f(t, sol[i-1])
        for j in range(size):
            sol[i, j] = sol[i-1, j] + fx[j]*dt
        t += dt
    return sol

cpdef RK2_integration(f, double [:] x, double t1, double t2, Py_ssize_t steps):
    cdef:
        double dt=(t2-t1)/(steps-1),t=t1
        double [:] fx
        Py_ssize_t i, j, size=x.shape[0]
        double [:] k1=np.empty((size), dtype=DTYPE), temp=np.empty((size), dtype=DTYPE)
        double [:, :] sol=np.empty((steps, size), dtype=DTYPE)

    for j in range(size):
        sol[0, j] = x[j]
    for i in range(1, steps):
        fx = f(t, sol[i-1])
        for j in range(size):
            k1[j] = dt*fx[j]
            temp[j] = sol[i-1, j] + k1[j]
        t += dt
        fx = f(t, temp)
        for j in range(size):
            sol[i, j] = sol[i-1, j] + 0.5*(k1[j] + dt*fx[j])
    return sol

cpdef RK4_integration(f, double [:] x, double t1, double t2, Py_ssize_t steps):
    cdef:
        double dt=(t2-t1)/(steps-1),t=t1
        double [:] fx
        Py_ssize_t i, j, size=x.shape[0]
        double [:] k1=np.empty((size), dtype=DTYPE), 
        double [:] k2=np.empty((size), dtype=DTYPE), 
        double [:] k3=np.empty((size), dtype=DTYPE), 
        double [:] temp=np.empty((size), dtype=DTYPE)
        double [:, :] sol=np.empty((steps, size), dtype=DTYPE)

    for j in range(size):
        sol[0, j] = x[j]
    for i in range(1, steps):
        fx = f(t, sol[i-1])
        for j in range(size):
            k1[j] = dt*fx[j]
            temp[j] = sol[i-1, j] + 0.5*k1[j]
        t += 0.5*dt
        fx = f(t, temp)
        for j in range(size):
            k2[j] = dt*fx[j]
            temp[j] = sol[i-1, j] + 0.5*k2[j]
        fx = f(t, temp)
        for j in range(size):
            k3[j] = dt*fx[j]
            temp[j] = sol[i-1, j] + k3[j]
        t += 0.5*dt
        fx = f(t, temp)
        for j in range(size):
            sol[i, j] = sol[i-1, j] + (k1[j] + 2*k2[j] + 2*k3[j] + dt*fx[j])/6.
    return sol

cpdef nearest_positive_definite(double [:, :] A):
    """Find the nearest positive-definite matrix to input

    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd

    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    cdef:
        np.ndarray[DTYPE_t, ndim=2] B, V, H, A2, A3, I
        np.ndarray[DTYPE_t, ndim=1] s
        double spacing, mineig
        int k

    B = np.add(A, np.transpose(A))/2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = np.add(B, H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3
    else:
        spacing = np.spacing(np.linalg.norm(B))
        I = np.eye(B.shape[0])
        k = 1
        while not is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (np.abs(mineig) * k**2 + spacing)
            k += 1
        return A3

cpdef is_positive_definite(double [:, :] B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        sign, _ = np.linalg.slogdet(B)
        if sign > 0:
            return True
        else:
            return False
    except np.linalg.LinAlgError:
        return False


cpdef solve_symmetric_close_to_singular(double [:, :] A, double [:] b, double eps=1e-13):
    """ Solve the linear system Ax=b and return x and the log-determinant of A for a
    symmetric matrix A that should be positive definite but might be close to singular
    or slightly non-positive.

    If A is symmteric positive, this function simply solves the linear system and computes
    the determinant using a Cholesky decomposition. Otherwise, it computes the pseudo-inverse
    and pseudo-determinant of A accounting only for positive and large-enough eigenvalues.

    Parameters
    ----------
    A: numpy.array(dims=2)
        The symmetric matrix.
    b: numpy.array(dims=1)
        The right-hand side of the linear equation.
    eps: float, optional
        Any eigenvalue smaller than eps*max_eigenval is ignored for the computation of the
        pseudo-inverse.

    Returns
    -------
    x: numpy.array(dims=1)
        The solution of Ax=b.
    log_det:
        The (pseudo-)log-determinant of A.
    """
    cdef:
        double log_det
        np.ndarray[DTYPE_t, ndim=1] x
    try:
        chol_fact = sp.linalg.cho_factor(A)
        log_det = 2*np.sum(np.log(np.diag(chol_fact[0])))
        x = sp.linalg.cho_solve(chol_fact, b)
        return x, log_det

    except sp.linalg.LinAlgError:
        # Not positive definite:
        eigvals, eigvecs = np.linalg.eigh(A)

        # Set all small or negative eigenvalues to 0 and compute pseudo-inverse and
        # pseudo-determinant.
        eigvals[eigvals < eps] = 0
        log_det = np.sum(np.log(eigvals[eigvals > 0]))

        eigvals[eigvals != 0] = 1/eigvals[eigvals != 0]

        x = eigvecs @ (np.diag(eigvals) @ (eigvecs.T @ b))
        return x, log_det

def largest_real_eig(np.ndarray A):
    w, v = np.linalg.eig(A)
    max_index = np.argmax(np.real(w))
    eigval = np.real(w[max_index])
    eigvec = np.real(v[:, max_index])
    eigval_sign = (eigval > 0)
    return eigval_sign, eigvec


def partial_derivative(func, var, point, dx, *func_args):
    args = point[:]
    def wraps(x, *wraps_args):
        args[var] = x
        return func(args, *wraps_args)
    return derivative(wraps, point[var], dx=dx, args=func_args)

cpdef make_fltr(fltr_list, n_list):
    fltr = [f for (i, f) in enumerate(fltr_list) for n in range(n_list[i])]
    return np.array(fltr)

def process_fltr(np.ndarray fltr, Py_ssize_t Nf):
    if fltr.ndim == 2:
        return np.array([fltr]*Nf)
    elif fltr.shape[0] == Nf:
        return fltr
    else:
        raise Exception("fltr must be a 2D array or an array of 2D arrays")

def process_obs(np.ndarray obs, Py_ssize_t Nf):
    if obs.shape[0] != Nf:
        raise Exception("Wrong length of obs")
    if obs.ndim == 2:
        return np.ravel(obs)
    elif obs.ndim == 1:
        return np.concatenate(obs)
    else:
        raise Exception("Obs must be a 2D array or an array of 1D arrays")

def process_latent_data(np.ndarray fltr, np.ndarray obs):
    cdef Py_ssize_t Nf=obs.shape[0]
    fltr = process_fltr(fltr, Nf)
    obs0 = obs[0]
    obs = process_obs(obs[1:], Nf-1)
    return fltr, obs, obs0

def _parse_prior_name(sub_dict, dim):
    if 'prior_fun' in sub_dict:
        prior = sub_dict['prior_fun']
        if np.size(prior) == 1:
            return [prior]*dim
        elif np.size(prior) == dim:
            return prior
        else:
            raise Exception('prior_fun must be of length 1 or {}'.format(dim))
    else:
        return ['lognorm']*dim

def mode_of_lognormal(mean,std):
    mode = mean**4/(mean**2+std**2)**(3/2)
    logvar = np.sqrt(np.log(1+std**2/mean**2))*mode  # std on logscale
    return mode, logvar

def parse_param_prior_dict(prior_dict, M, check_length=True):
    flat_stds = []
    flat_mean = []
    flat_bounds = []
    flat_guess_range = []
    key_list = []
    scaled_guesses = []
    is_scale_parameter = []
    names = []

    count = 0
    for key in prior_dict:
        key_list.append(key)
        sub_dict = prior_dict[key]
        try:
            mean = sub_dict['mean']
        except KeyError:
            raise Exception('Sub-dict under {} must have "mean" as a key'.format(key))
        if np.size(mean) == 1:
            flat_mean.append(mean)
            try:
                flat_stds.append(sub_dict['std'])
                flat_bounds.append(sub_dict['bounds'])
            except KeyError:
                raise Exception('Sub-dict under {} must have "std" and "bounds"'
                                ' as keys'.format(key))
            flat_guess_range.append(count)
            is_scale_parameter.append(False)
            names += _parse_prior_name(sub_dict, 1)
            count += 1
        else:
            infer_scale = False
            if 'infer_scale' in sub_dict.keys():
                infer_scale = sub_dict['infer_scale']
            if infer_scale:
                flat_mean.append(1.0)
                try:
                    flat_stds.append(sub_dict['scale_factor_std'])
                    flat_bounds.append(sub_dict['scale_factor_bounds'])
                except KeyError:
                    raise Exception('Sub-dict under {} must have "scale_factor_std"'
                                    'and "scale_factor_bounds" as keys because'
                                    'infer_scale" is True'.format(key))
                scaled_guesses.append(mean)
                flat_guess_range.append(count)
                is_scale_parameter.append(True)
                names += _parse_prior_name(sub_dict, 1)
                count += 1
            else:
                if check_length:
                    assert len(mean) == M, 'length of mean must be either 1 or M'
                else:
                    M = len(mean)
                flat_mean += list(mean)
                try:
                    assert len(sub_dict['std']) == M
                    assert len(sub_dict['bounds']) == M
                    flat_stds += list(sub_dict['std'])
                    flat_bounds += list(sub_dict['bounds'])
                except KeyError:
                    raise Exception('Sub-dict under {} must have "std" and "bounds"'
                                    ' as keys'.format(key))
                is_scale_parameter += [False]*M
                flat_guess_range.append(list(range(count, count+M)))
                names += _parse_prior_name(sub_dict, M)
                count += M
    
    # set guess to mode for all lognormal distributions
    flat_guess = flat_mean.copy()
    flat_guess_std = flat_stds.copy()
    for i in range(len(names)):
        if names[i] == 'lognorm':
            flat_guess[i], flat_guess_std[i] = mode_of_lognormal(flat_mean[i], flat_stds[i])
    
    return names, key_list, np.array(flat_mean), np.array(flat_stds), np.array(flat_guess), np.array(flat_guess_std), \
          np.array(flat_bounds), flat_guess_range, is_scale_parameter, scaled_guesses

def unflatten_parameters(params, flat_guess_range, is_scale_parameter, scaled_guesses):
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

def parse_init_prior_dict(prior_dict, dim, obs_dim):
    mean = []
    stds = []
    bounds = []
    names = []
    flags = [False, False]
    fltrs = [None, None]
    count = 0
    if 'lin_mode_coeff' in prior_dict.keys():
        sub_dict = prior_dict['lin_mode_coeff']
        try:
            mean.append(sub_dict['mean'])
            stds.append(sub_dict['std'])
            bounds.append(sub_dict['bounds'])
            fltrs[0] = _process_init_fltr(sub_dict['fltr'], dim)
        except KeyError:
            raise Exception('Sub dict of "lin_mode_coeff" must have'
                            ' "mean", "std", "bounds" and "fltr" as keys')
        flags[0] = True
        names += _parse_prior_name(sub_dict, 1)
        count += fltrs[0].shape[0]

    if 'independent' in prior_dict.keys():
        sub_dict = prior_dict['independent']
        try:
            fltrs[1] = _process_init_fltr(sub_dict['fltr'], dim)
            mean.extend(sub_dict['mean'])
            stds.extend(sub_dict['std'])
            bounds.extend(sub_dict['bounds'])
        except KeyError:
            raise Exception('Sub dict of "independent" must have'
                            ' "mean", "std", "bounds" and "fltr" as keys')
        assert fltrs[1].shape[0] == len(sub_dict['mean'])
        assert len(sub_dict['std']) == len(sub_dict['mean'])
        assert len(sub_dict['bounds']) == len(sub_dict['mean'])
        flags[1] = True
        names += _parse_prior_name(sub_dict, len(sub_dict['mean']))
        count += fltrs[1].shape[0]

    # make sure that there are some priors
    if np.sum(flags) == 0:
        raise Exception('Prior for inits must have at least one of "independent"'
                        ' and "coeff" as keys')
    # check that the total number of means is correct
    assert count == dim - obs_dim, 'Total No. of guessed values must be dim - obs_dim'

    # set guess to mode for all lognormal distributions
    guess = mean.copy()
    guess_std = stds.copy()
    for i in range(len(names)):
        if names[i] == 'lognorm':
            (guess[i],guess_std[i]) = mode_of_lognormal(mean[i], stds[i])
    
    return names, np.array(mean), np.array(stds), np.array(guess), np.array(guess_std), np.array(bounds), \
           flags, fltrs

def _process_init_fltr(fltr, dim):
    fltr = np.array(fltr).astype('bool')
    if fltr.ndim == 1:
        assert len(fltr) == dim, 'init_fltr has wrong shape'
        l = np.sum(fltr)
        fltr_mat = np.zeros((l, dim), dtype=DTYPE)
        row = 0
        for i in range(dim):
            if fltr[i]:
                fltr_mat[row, i] = 1
                row += 1
        return fltr_mat
    elif fltr.ndim == 2:
        assert fltr.shape[1] == dim, 'init_fltr has wrong shape'
        return fltr.astype('float')
    else:
        raise Exception('init_fltr must either be 1D or 2D')

cpdef double distance_on_Earth(double [:] coord1, double [:] coord2):
    cdef:
        double lat1=coord1[0], lon1=coord1[1], lat2=coord2[0], lon2=coord2[1]
        double Earth_radius_km=6371.0, degree_to_radius=PI/180.0
        double d_lat, d_lon, a, c
    d_lat = (lat2 - lat1) * degree_to_radius
    d_lon = (lon2 - lon2) * degree_to_radius

    lat1 *= degree_to_radius
    lat2 *= degree_to_radius

    a = sin(d_lat/2) * sin(d_lat/2) + sin(d_lon/2) * sin(d_lon/2) * cos(lat1) * cos(lat2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return Earth_radius_km * c


def plotSIR(data, showPlot=True):
    t = data['t']
    X = data['X']
    M = data['M']
    Ni = data['Ni']
    N = Ni.sum()

    S = X[:, 0: M]
    Is = X[:, 2*M:3*M]
    R = Ni - X[:, 0:M] - X[:, M:2*M] - X[:, 2*M:3*M]


    sumS = S.sum(axis=1)
    sumI = Is.sum(axis=1)
    sumR = R.sum(axis=1)

    try:
        import matplotlib.pyplot as plt
        plt.fill_between(t, 0, sumS/N, color="#348ABD", alpha=0.3)
        plt.plot(t, sumS/N, '-', color="#348ABD", label='$S$', lw=4)

        plt.fill_between(t, 0, sumI/N, color='#A60628', alpha=0.3)
        plt.plot(t, sumI/N, '-', color='#A60628', label='$I$', lw=4)

        plt.fill_between(t, 0, sumR/N, color="dimgrey", alpha=0.3)
        plt.plot(t, sumR/N, '-', color="dimgrey", label='$R$', lw=4)

        plt.legend(fontsize=26); plt.grid()
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.ylabel('Fraction of compartment value')
        plt.xlabel('Days')
        if True != showPlot:
            pass
        else:
            plt.show()

    except ImportError:
        print('Please install matplotlib to use this method')




def getPopulation(country='India', M=16):
    """
    Takes coutry name and number, M, of age-groups as argument
    Returns population structured in M age-groups

    Parameters
    ----------
    country: string
        Default is 'India'
    M: int
        Deafault is 16 age-groups
    """
    u0 = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/'

    u1 = u0 + 'age_structures/India-2019.csv'
    u2 = u0 + 'age_structures/UK.csv'
    u3 = u0 + 'age_structures/Germany-2019.csv'
    u4 = u0 + 'age_structures/Italy-2019.csv'
    u5 = u0 + 'age_structures/Denmark-2019.csv'
    u6 = u0 + 'age_structures/UK.csv'
    u7 = u0 + 'age_structures/US.csv'
    u8 = u0 + 'age_structures/China-2019.csv'
    u9 = u0 + 'age_structures/France.txt'

    if country=='India':
        data = np.asanyarray(np.genfromtxt(u1, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='UK':
        data = np.asanyarray(np.genfromtxt(u2, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='Germany':
        data = np.asanyarray(np.genfromtxt(u3, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='Italy':
        data = np.asanyarray(np.genfromtxt(u4, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='Denmark':
        data = np.asanyarray(np.genfromtxt(u5, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='UK':
        data = np.asanyarray(np.genfromtxt(u6, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='USA':
        data = np.asanyarray(np.genfromtxt(u7, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='China':
        data = np.asanyarray(np.genfromtxt(u8, delimiter=','))
        N_m  = np.array((data[:, 1]))[1:M+1]
        N_f  = np.array((data[:, 2 ]))[1:M+1]
        Ni   = N_m + N_f 
        Ni   = Ni[0:M];  Ni=Ni.astype('double')

    elif country=='France':
        Ni = np.genfromtxt(u9)[:, 3]

    else:
        print('Direct extraction of Ni is not implemnted , please do it locally')

    return Ni


def get_summed_CM(CH0, CW0, CS0, CO0, M, M0, Ni, Ni0):
    CH = np.zeros((M, M))
    CW = np.zeros((M, M))
    CS = np.zeros((M, M))
    CO = np.zeros((M, M))

    for i in range(16):
        CH0[i,:] = Ni0[i]*CH0[i,:]
        CS0[i,:] = Ni0[i]*CS0[i,:]
        CW0[i,:] = Ni0[i]*CW0[i,:]
        CO0[i,:] = Ni0[i]*CO0[i,:]

    for i in range(M):
        for j in range(M):
            i1, j1 = i*M, j*M
            CH[i,j] = np.sum( CH0[i1:i1+M,j1:j1+M] )/Ni[i]
            CW[i,j] = np.sum( CW0[i1:i1+M,j1:j1+M] )/Ni[i]
            CS[i,j] = np.sum( CS0[i1:i1+M,j1:j1+M] )/Ni[i]
            CO[i,j] = np.sum( CO0[i1:i1+M,j1:j1+M] )/Ni[i]
    return CH, CW, CS, CO


class GPR:
    def __init__(self, nS, nT, iP, nP, xS, xT, yT):
        self.nS   =  nS           # # of test data points
        self.nT   =  nT           # # of training data points
        self.iP   =  iP           # # inverse of sigma
        self.nP   =  nP           # # number of priors
        self.xS   =  xS           # test input
        self.xT   =  xT           # training input
        self.yT   =  yT           # training output

        self.yS   =  0            # test output
        self.yP   =  0            # prior output
        self.K    =  0            # kernel
        self.Ks   =  0            # kernel
        self.Kss  =  0            # kernel
        self.mu   =  0            # mean
        self.sd   =  0            # stanndard deviation


    def calcDistM(self, r, s):
        '''Calculate distance matrix between 2 1D arrays'''
        return r[..., np.newaxis] - s[np.newaxis, ...]


    def calcKernels(self):
        '''Calculate the kernel'''
        cc = self.iP*0.5
        self.K   = np.exp(-cc*self.calcDistM(self.xT, self.xT)**2)
        self.Ks  = np.exp(-cc*self.calcDistM(self.xT, self.xS)**2)
        self.Kss = np.exp(-cc*self.calcDistM(self.xS, self.xS)**2)
        return


    def calcPrior(self):
        '''Calculate the prior'''
        L  = np.linalg.cholesky(self.Kss + 1e-6*np.eye(self.nS))
        G  = np.random.normal(size=(self.nS, self.nP))
        yP = np.dot(L, G)
        return


    def calcMuSigma(self):
        '''Calculate the mean'''
        self.mu =  np.dot(self.Ks.T, np.linalg.solve(self.K, self.yT))

        vv = self.Kss - np.dot(self.Ks.T, np.linalg.solve(self.K, self.Ks))
        self.sd = np.sqrt(np.abs(np.diag(vv)))

        # Posterior
        L       = np.linalg.cholesky(vv + 1e-6*np.eye(self.nS))
        self.yS = self.mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(self.nS, self.nP)))
        return


    def plotResults(self):
        try:
            import matplotlib.pyplot as plt
            plt.plot(self.xT, self.yT, 'o', ms=10, mfc='#348ABD', mec='none', label='training set' )
            plt.plot(self.xS, self.yS, '#dddddd', lw=1.5, label='posterior')
            plt.plot(self.xS, self.mu, '#A60628', lw=2, label='mean')

            # fill 95% confidence interval (2*sd about the mean)
            plt.fill_between(self.xS.flat, self.mu-2*self.sd, self.mu+2*self.sd, color="#348ABD", alpha=0.4, label='2 sigma')
            plt.axis('tight'); plt.legend(fontsize=15); plt.rcParams.update({'font.size':18})
        except ImportError:
            print('Please install matplotlib to use this method')


    def runGPR(self):
        self.calcKernels()
        self.calcPrior()
        self.calcMuSigma()
        self.plotResults()


def resample(weighted_samples, N):
    """
    Given a set of weighted samples, produce a set of unweighted samples approximating the same
    distribution. We implement residual resampling here, see https://doi.org/10.1109/ISPA.2005.195385
    for some context. This function can (and for large N will) duplicate likely samples.

    Parameters
    ----------
    weighted_samples: list of dict
        The set of weighed samples (e.g. given by nested sampling).
    N: int
        The number of samples that should be drawn from the given samples.

    Returns
    -------
    sample_list: list of dict
        A set of uniformly weighted samples.
    """

    weights = np.array([w['weight'] for w in weighted_samples])
    weights /= np.sum(weights)

    # Deterministic part
    selected_samples = np.array([int(w*N) for w in weights])
    # Random part
    selected_samples += np.random.multinomial(N - sum(selected_samples), weights, size=1)[0,:]

    sample_list = []
    for i in range(len(selected_samples)):
        for j in range(selected_samples[i]):
            sample_list.append(weighted_samples[i])

    return sample_list


def posterior_mean(weighted_samples):
    """
    Compute the posterior mean of a set of (potentially) weighted samples of the posterior
    (e.g. computed by nested sampling or by MCMC).

    Parameters
    ----------
    weighted_samples: list of dict
        The set of (weighted) samples.

    Returns
    -------
    sample: dict
        An approximation of the posterior mean.
    """
    weights = np.array([w['weight'] for w in weighted_samples])
    weights /= np.sum(weights)

    sample = weighted_samples[0].copy()
    sample['weight'] = 1.0

    # Set the parameter to averages.
    potential_list_keys = {'params_dict', 'control_params_dict',  # Current keys
                           'map_params_dict', 'map_dict' }        # Keys from deprecated interface (to be deleted)
    potential_keys = {'flat_params', 'x0',   # Current keys
                      'flat_map', 'map_x0'}  # Keys from deprecated interface (to be deleted)

    # Select the keys that are actually in the result dictionaries.
    list_keys = list(potential_list_keys & set(sample.keys()))
    keys = list(potential_keys & set(sample.keys()))

    # Set the average parameters
    for list_key in list_keys:
        for key in sample[list_key].keys():
            vals = [w[list_key][key] for w in weighted_samples]
            avg = sum([weights[i] * vals[i] for i in range(len(vals))])
            sample[list_key][key] = avg

    for key in keys:
        vals = [w[key] for w in weighted_samples]
        avg = sum([weights[i] * vals[i] for i in range(len(vals))])
        sample[key] = avg

    # Reset the value of log-posterior (we cannot compute this here)
    sample['log_posterior']  = None
    sample['log_prior']      = None
    sample['log_likelihood'] = None
    return sample


def build_SppQ_model_spec(input_model_spec, testRate_name = "tau"):
    """
    Adds a quarantined version of every state and automatically specifies finite-resource 
    transtions for testing. If not yet specified, an explicit R-class is added. 

    Parameters
    input_model_spec: dict
        Model specification of the model without quarantine (Spp syntax)
  
    Returns
    -------
    model_spec: dict
        Full model specification with quarantine.
    """
    
    model_spec = copy.deepcopy(input_model_spec)
    
    # collect all terms
    terms = list()
    for name in model_spec['classes']:
        for key, listval in model_spec[name].items():
            for val in listval:
                terms.append((key,val))
            
    
    # Add R-class, if not yet specified
    if "R" not in model_spec['classes']:
        model_spec['classes'].append("R")
        model_spec['R'] = dict()
        # check if transition is balanced, otherwise add to R
        for term in terms:
            term2 = copy.deepcopy(term)
            if term[1][1].startswith('-'):
                term2[1][1] = term[1][1][1:]
                if term2 not in terms:
                    if term2[0] not in model_spec['R'].keys():
                        model_spec['R'][term2[0]] = list()
                    model_spec['R'][term2[0]].append(term2[1])
    n_class = len(model_spec['classes'])
                    
    # add Q-classes
    for class_name in model_spec['classes'].copy():
        classQ = class_name + 'Q'
        model_spec['classes'].append(classQ)
        model_spec[classQ] = copy.deepcopy(model_spec[class_name])
        for terms in model_spec[classQ].values():
            for term in terms:
                term[0] += 'Q'
        if 'infection' in model_spec[classQ]:
            del model_spec[classQ]['infection'] # no infection in quarantine
    
    # add testing transitions
    test_pos = model_spec.pop('test_pos')
    assert len(test_pos) == n_class, "number of elements of test_pos must match number of classes (incl. R)"
    test_freq = model_spec.pop('test_freq')
    assert len(test_freq) == n_class, "number of elements of test_freq must match number of classes (incl. R)"
    
    for i in range(n_class):
        class_name = model_spec['classes'][i]
        class_nameQ = model_spec['classes'][i+n_class]
        model_spec[class_name]['finite-resource'] = [ [class_name, "-"+testRate_name, test_freq[i], test_pos[i]] ]
        model_spec[class_nameQ]['finite-resource'] = [ [class_name, testRate_name, test_freq[i], test_pos[i]] ]
        
                
    return model_spec


def Spp2Xpp(model_spec):
    """
    Converts Spp syntax to Xpp syntax by adding susceptible class `S` to every infection term.

    Parameters
    model_spec: dict
        Model specification in Spp syntax
  
    Returns
    -------
    Xpp_model_spec: dict
        Model specification in Xpp syntax
    """
    Xpp_model_spec=copy.deepcopy(model_spec)
    for key in Xpp_model_spec.keys():
        if "infection" in Xpp_model_spec[key]:
            for term in Xpp_model_spec[key]["infection"]:
                term.insert(1,'S')
    return Xpp_model_spec