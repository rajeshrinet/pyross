import  numpy as np
cimport numpy as np
import scipy.linalg as spl
import pyross.utils
from libc.math cimport exp, pow, sqrt
cimport cython
import warnings
from types import ModuleType
import os


DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class ContactMatrixFunction:
    """Generates a time dependent contact matrix

    Parameters
    ----------
    CH: 2d np.array
        Contact matrix at home
    CW: 2d np.array
        Contact matrix at work
    CS: 2d np.array
        Contact matrix at school
    CO: 2d np.array
        Contact matrix at other locations
    """

    cdef:
        np.ndarray CH, CW, CS, CO
        double aW, aS, aO
        Protocol protocol

    def __init__(self, CH, CW, CS, CO):
        self.CH, self.CW, self.CS, self.CO = CH, CW, CS, CO

    def contactMatrix(self, t, **kwargs):
        cdef np.ndarray[DTYPE_t, ndim=2] C
        aW, aS, aO = self.protocol(t, **kwargs)
        C = self.CH + aW*self.CW + aS*self.CS + aO*self.CO
        return C

    cpdef get_individual_contactMatrices(self):
        return self.CH, self.CW, self.CS, self.CO

    def constant_contactMatrix(self, aW=1, aS=1, aO=1):
        '''Constant contact matrix

        Parameters
        ----------
        aW: float, optional
            Scale factor for work contacts.
        aS: float, optional
            Scale factor for school contacts.
        aO: float, optional
            Scale factor for other contacts.

        Returns
        -------
        contactMatrix: callable
            A function that takes t as an argument and outputs the contact matrix
        '''

        self.protocol=ConstantProtocol(aW, aS, aO)
        return self.contactMatrix

    def interventions_temporal(self,times,interventions):
        '''Temporal interventions

        Parameters
        ----------
        time: np.array
            Ordered array with temporal boundaries between the different interventions.
        interventions: np.array
            Ordered matrix with prefactors of CW, CS, CO matrices
            during the different time intervals.
            Note that len(interventions) = len(times) + 1

        Returns
        -------
        contactMatrix: callable
            A function that takes t as an argument and outputs the contact matrix
        '''

        self.protocol = TemporalProtocol(np.array(times), np.array(interventions))
        return self.contactMatrix

    def interventions_threshold(self,thresholds,interventions):
        '''Temporal interventions

        Parameters
        ----------
        threshold: np.array
            Ordered array with temporal boundaries between the different interventions.
        interventions: np.array
            Array of shape [K+1,3] with prefactors during different phases of intervention
            The current state of the intervention is defined by
            the largest integer "index" such that state[j] >= thresholds[index,j] for all j.

        Returns
        -------
        contactMatrix: callable
            A function that takes t as an argument and outputs the contact matrix
        '''

        self.protocol = ThresholdProtocol(np.array(thresholds), np.array(interventions))
        return self.contactMatrix

    def intervention_custom_temporal(self, intervention_func, **kwargs):
        '''Custom temporal interventions

        Parameters
        ----------
        intervention_func: callable
            The calling signature is `intervention_func(t, **kwargs)`,
            where t is time and kwargs are other keyword arguments for the function.
            The function must return (aW, aS, aO).
        kwargs: dict
            Keyword arguments for the function.

        Returns
        -------
        contactMatrix: callable
            A function that takes t as an argument and outputs the contact matrix.

        Examples
        --------
        An example for an custom temporal intervetion that
        allows for some anticipation and reaction time

        >>> def fun(t, width=1, loc=0) # using keyword arguments for parameters of the intervention
                a = (1-np.tanh((t-loc)/width))/2
                return a, a, a
        >>> contactMatrix = generator.intervention_custom_temporal(fun, width=5, loc=10)
        '''

        self.protocol = CustomTemporalProtocol(intervention_func, **kwargs)
        return self.contactMatrix




cdef class Protocol:

    def __init__(self):
        pass

    def __call__(self, t):
        pass

cdef class ConstantProtocol(Protocol):
    cdef:
        double aW, aS, aO

    def __init__(self, aW=1, aS=1, aO=1):
        self.aW = aW
        self.aS = aS
        self.aO = aO

    def __call__(self, double t):
        return self.aW, self.aS, self.aO

cdef class TemporalProtocol(Protocol):
    cdef:
        np.ndarray times, interventions

    def __init__(self, np.ndarray times, np.ndarray interventions):
        self.times = times
        self.interventions = interventions

    def __call__(self, double t):
        cdef:
            Py_ssize_t index
            np.ndarray t_arr=self.times, prefac_arr=self.interventions
        index = np.argmin( t_arr < t)
        if index == 0:
            if t >= t_arr[len(t_arr)-1]:
                index = -1
        return prefac_arr[index,0], prefac_arr[index,1], prefac_arr[index,2]




cdef class ThresholdProtocol(Protocol):
    cdef:
        np.ndarray thresholds, interventions

    def __init__(self, np.ndarray thresholds, np.ndarray interventions):
        self.thresholds = thresholds
        self.interventions = interventions

    def __call__(self, double t, S=None, Ia=None, Is=None):
        cdef:
            np.ndarray[DTYPE_t, ndim=1] state
            np.ndarray thresholds=self.thresholds, prefac_arr=self.interventions
            Py_ssize_t index
        state = np.concatenate((S, Ia, Is))
        index = np.argmin((thresholds <= state ).all(axis=1))
        if index == 0:
            N = len(thresholds)
            if (thresholds[N-1] <= state ).all():
                index = N
        return prefac_arr[index,0], prefac_arr[index,1], prefac_arr[index,2]




cdef class CustomTemporalProtocol(Protocol):
    cdef:
        object intervention_func
        dict kwargs

    def __init__(self, intervention_func, **kwargs):
        self.intervention_func = intervention_func
        self.kwargs = kwargs

    def __call__(self, double t):
        return self.intervention_func(t, **self.kwargs)




cdef class SpatialContactMatrix:
    '''A class for generating a spatial compartmental model with commute data

    Let :math:`\\mu, \\nu` denote spatial index and i, j denote age group index.

    .. math::
        C^{\\mu\\nu}_{ij} = \\frac{1}{N^\\mu_i} \widetilde{C}^{\\mu \\nu}_{ij}

    Parameters
    ----------
    b: float
        Parameter b in the above equation
    populations: np.array(n_loc, M)
        Populations of regions by age groups. Here n_loc is the number of regions
        and M is the number of age groups.
    areas: np.array(n_loc)
        Areas of the geographical regions.
    commutes: np.array(n_loc, n_loc, M)
        Each entry commute[mu, nu, i] needs to be the number of people of age
        group i commuting from :math:`\\mu` to :math:`\\nu`.
        Entries with :math:`\\mu = \\nu` are ignored.
    '''
    cdef:
        readonly np.ndarray Ni, pops, commute_fraction,
        readonly np.ndarray density_factor, norm_factor, local_contacts
        readonly Py_ssize_t n_loc, M
        readonly double work_ratio

    def __init__(self, double b, double work_ratio, np.ndarray populations,
                    np.ndarray areas, np.ndarray commutes):
        cdef:
            double [:, :] densities
        self.n_loc = populations.shape[0]
        if self.n_loc != commutes.shape[0] or self.n_loc !=commutes.shape[1]:
            raise Exception('The first two dimensions of `commutes` must be \
                             the number of regions')
        if self.n_loc != areas.shape[0] :
            raise Exception('The first dimension of populations \
                             areas must be equal')

        self.M = populations.shape[1]
        if self.M != commutes.shape[2]:
            raise Exception('The last dimension of `commutes` must be \
                             the number of age groups')

        self.Ni = np.sum(populations, axis=0)
        self.work_ratio = work_ratio

        # for pops, etc, pops[0] gives the values without commutors
        # and pops[1] gives the values with commutors
        self.pops = np.empty((2, self.n_loc, self.M), dtype=DTYPE)
        self.commute_fraction = np.empty((self.n_loc, self.n_loc, self.M), dtype=DTYPE)
        self.density_factor = np.empty((2, self.n_loc, self.M, self.M), dtype=DTYPE)
        self.norm_factor = np.zeros((2, self.M, self.M), dtype=DTYPE)
        self.local_contacts = np.empty((2, self.n_loc, self.M, self.M), dtype=DTYPE)

        self.pops[0] = populations
        self._process_commute(populations.astype('float'), commutes.astype('float'))
        rho = np.sum(populations)/np.sum(areas)
        self._compute_density_factor(b, rho, areas.astype('float'))

    def spatial_contact_matrix(self, np.ndarray CM):
        self._compute_local_contacts(CM.astype('float'))
        cdef:
            Py_ssize_t mu, nu, i, j, M=self.M, n_loc=self.n_loc
            double [:, :, :] f=self.commute_fraction,
            double [:, :, :] C=self.local_contacts[0], CC=self.local_contacts[1]
            double [:, :] pop=self.pops[0], commute_time_pop=self.pops[1]
            np.ndarray spatial_CM
            double p, cc, work_ratio=self.work_ratio
        spatial_CM = np.zeros((n_loc, M, n_loc, M))
        for i in range(M):
            for j in range(M):
                for mu in range(n_loc):
                    p = pop[mu, i]
                    spatial_CM[mu, i, mu, j] = C[mu, i, j]*(1-work_ratio)
                    for nu in range(n_loc):
                        for eta in range(n_loc):
                            cc = CC[eta, i, j]
                            f1 = f[mu, eta, i]
                            f2 = f[nu, eta, j]
                            spatial_CM[mu, i, nu, j] += f1*f2*cc*work_ratio
                        spatial_CM[mu, i, nu, j] /= p
        return spatial_CM

    cdef _process_commute(self, double [:, :] pop, double [:, :, :] commutes):
        cdef:
            Py_ssize_t i, mu, nu
            double influx, outflux, p, f
            double [:, :, :] pops=self.pops
            double [:, :, :] commute_fraction=self.commute_fraction
        for i in range(self.M):
            for mu in range(self.n_loc):
                influx = np.sum(commutes[:, mu, i])
                outflux = np.sum(commutes[mu, :, i])
                p = pops[0, mu, i] + influx - outflux
                pops[1, mu, i] = p
                commute_fraction[mu, mu, i] = 1
                for nu in range(self.n_loc):
                    if nu != mu:
                        f = commutes[nu, mu, i]/p
                        commute_fraction[nu, mu, i] = f
                        commute_fraction[mu, mu, i] -= f


    cdef _compute_density_factor(self, double b, double rho, double [:] areas):
        cdef:
            Py_ssize_t mu, i, j, a
            double rhoi, rhoj
            double [:, :, :] pops=self.pops
            double [:, :, :, :] density_factor=self.density_factor
            double [:, :, :] norm_factor=self.norm_factor
        for mu in range(self.n_loc):
            for i in range(self.M):
                for j in range(self.M):
                    for a in range(2):
                        rhoi = pops[a, mu, i]/areas[mu]
                        rhoj = pops[a, mu, j]/areas[mu]
                        density_factor[a, mu, i, j] = pow(rhoi*rhoj/rho**2, b)
                        norm_factor[a, i, j] += density_factor[a, mu, i, j]


    cdef _compute_local_contacts(self, double [:, :] CM):
        cdef:
            Py_ssize_t mu, i, j, M=self.M, n_loc=self.n_loc
            double [:] Ni = self.Ni
            double [:, :, :, :] local_contacts=self.local_contacts
            double [:, :, :] norm=self.norm_factor
            double [:, :, :, :] density_factor=self.density_factor
            double c, d
        for mu in range(n_loc):
            for i in range(M):
                for j in range(M):
                    c = CM[i, j] * Ni[i]
                    for a in range(2):
                        d = density_factor[a, mu, i, j]
                        local_contacts[a, mu, i, j] = c * d / norm[a, i, j]




cdef class MinimalSpatialContactMatrix:
    '''A class for generating a minimal spatial compartmental model

    Let :math:`\\mu, \\nu` denote spatial index and i, j denote age group index,

    .. math::
        C^{\\mu\\nu}_{ij} = \\frac{1}{a_{ij}} k^{\\mu\\nu} g^{\\mu\\nu}_{ij} f^{\\mu\\nu}_{ij} C_{ij}

    where
    .. math::
        a_{ij} = \sum_{\mu\nu} k^{\mu\nu}_{ij} g^{\mu\nu}_{ij} \sqrt{ \frac{ N_i^\mu N_j^\nu} {N_i N_j}}
        k^{\mu \nu} = \exp ( - b_{\mu\nu} | r^\mu - r^\nu |) \\
        b_{\mu, \nv} = (\rho^\mu \rho^\nu)^{-c} \\
        g^{\mu\nu}_{ij} = (\rho^\mu_i \rho^\nu_j)^b \\
        f^{\mu\nu}_{ij} = \sqrt{ \frac{N_i N_j^\nu}{N_i^\mu N_j}}

    where :math:`C_{ij}` is the non-spatial age structured contact matrix,
    :math:`\rho^\mu_i` is the population density of age group i
    in spatial location :math:`\mu`,
    :math:`\rho_i` is the total population density of age group i,
    :math:`r^\mu` is the spatial position of the :math:`\mu`th location,
    :math:`N_i^\mu` is the population of the age group i at :math:`\mu`,
    :math:`N_i` is the total population of the age group i,
    and b, c are free parameters

    Parameters
    ----------
    b: float
        Parameter b in the above equation.
    c: float
        Paramter c in the above equation.
    populations: np.array(n_loc, M)
        The population of each age group (total number of age group = M)
        in each location (total number of location = n_loc ).
    areas: np.array(n_loc)
        The area of each geographical region.
    coordinates: np.array(n_loc, 2)
        The GPS coordinates of each geographical region.
    '''

    cdef:
        Py_ssize_t n_loc, M
        readonly np.ndarray Ni, spatial_kernel, density_factor, rescale_factor, normalisation_factor

    def __init__(self, double b, double c, np.ndarray populations,
                        np.ndarray areas, np.ndarray coordinates):
        cdef:
            double [:, :] densities
        self.n_loc = populations.shape[0]
        if self.n_loc != areas.shape[0] or self.n_loc != coordinates.shape[0]:
            raise Exception('The first dimension of populations \
                             areas and coordinates must be equal')
        self.M = populations.shape[1]
        self.Ni = np.sum(populations, axis=0)
        self.spatial_kernel = np.empty((self.n_loc, self.n_loc))
        self.density_factor = np.empty((self.n_loc, self.M, self.n_loc, self.M))
        self.rescale_factor = np.empty((self.n_loc, self.M, self.n_loc, self.M))
        self.normalisation_factor = np.zeros((self.M, self.M))

        densities = populations/areas[:, np.newaxis]
        self._compute_spatial_kernel(densities, coordinates, c)
        self._compute_density_factor(densities, b)
        self._compute_rescale_factor(populations)
        self._compute_normalisation_factor(populations)

    def spatial_contact_matrix(self, double [:, :] CM):
        cdef:
            Py_ssize_t n_loc=self.n_loc, M=self.M, i, j, m, n
            double [:] Ni=self.Ni
            np.ndarray spatial_CM=np.empty((n_loc, M, n_loc, M),
                                                        dtype=DTYPE)
            double [:, :] a=self.normalisation_factor, k=self.spatial_kernel
            double [:, :, :, :] g=self.density_factor, f=self.rescale_factor

        for i in range(n_loc):
            for j in range(n_loc):
                for m in range(M):
                    for n in range(M):
                        spatial_CM[i, m, j, n]=k[i, j]*g[i, m, j, n]*f[i, m, j, n]/a[m, n]*CM[m, n]
        return spatial_CM

    cdef _compute_spatial_kernel(self, double [:, :] densities,
                                        double [:, :] coordinates, double c):
        cdef:
            Py_ssize_t n_loc=self.n_loc, i, j
            double [:] densities_sum_over_age
            double [:, :] spatial_kernel=self.spatial_kernel
            double d, k, rhoi, rhoj
        densities_sum_over_age = np.sum(densities, axis=1)
        for i in range(n_loc):
            spatial_kernel[i, i] = 1
            rho_i = densities_sum_over_age[i]
            for j in range(i+1, n_loc):
                rho_j = densities_sum_over_age[j]
                d = pyross.utils.distance_on_Earth(coordinates[i], coordinates[j])
                k = exp(-d*pow(rho_i*rho_j, -c))
                spatial_kernel[i, j] = k
                spatial_kernel[j, i] = k

    cdef _compute_density_factor(self, double [:, :] densities, double b):
        cdef:
            Py_ssize_t n_loc=self.n_loc, M=self.M, i, j, m, n
            double [:, :, :, :] density_factor=self.density_factor
            double rho_im, rho_jn, f
        for i in range(n_loc):
            for m in range(M):
                rho_im = densities[i, m]
                for j in range(n_loc):
                    for n in range(M):
                        rho_jn = densities[j, n]
                        f = pow(rho_im*rho_jn, b)
                        density_factor[i, m, j, n] = f

    cdef _compute_rescale_factor(self, double [:, :] populations):
        cdef:
            Py_ssize_t n_loc=self.n_loc, M=self.M, i, j, m, n
            double [:] Ni=self.Ni
            double [:, :, :, :] rescale_factor=self.rescale_factor
        for i in range(n_loc):
            for j in range(n_loc):
                for m in range(M):
                    for n in range(M):
                        rescale_factor[i, m, j, n] = sqrt(Ni[m]/Ni[n]*populations[j, n]/populations[i, m])

    cdef _compute_normalisation_factor(self, double [:, :] populations):
        cdef:
            Py_ssize_t n_loc=self.n_loc, M=self.M, i, j, m, n
            double [:] Ni=self.Ni
            double [:, :] norm_fac=self.normalisation_factor, k=self.spatial_kernel
            double [:, :, :, :] g=self.density_factor, f=self.rescale_factor
            double a
        for m in range(M):
            for n in range(M):
                for i in range(n_loc):
                    for j in range(n_loc):
                        a = sqrt(populations[i, m]*populations[j, n]/(Ni[m]*Ni[n]))
                        norm_fac[m, n] += k[i, j]*g[i, m, j, n]*a

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR(ContactMatrixFunction):


    def basicReproductiveRatio(self, data, state='constant'):
        C = self.CH + self.CW + self.CS + self.CO
        # matrix for linearised dynamics
        alpha = data['alpha']                         # infection rate
        beta  = data['beta']                         # infection rate
        gIa   = data['gIa']                          # removal rate of Ia
        gIs   = data['gIs']                          # removal rate of Is
        fsa   = data['fsa']                          # the self-isolation parameters
        M     = data['M']
        Ni    = data['Ni']

        L0 = np.zeros((M, M))
        L  = np.zeros((2*M, 2*M))

        if state=='constant':
            for i in range(M):
                for j in range(M):
                    L0[i,j]=C[i,j]*Ni[i]/Ni[j]

            L[0:M, 0:M]     = alpha*beta/gIa*L0
            L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
            L[M:2*M, 0:M]   = ((1-alpha)*beta/gIa)*L0
            L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0
            r0 = np.max(np.linalg.eigvals(L))

        else:
            t = data.get('t');
            Nt = t.size
            r0 = np.zeros((Nt))

            for tt in range(Nt):
                S = np.array((data['X'][tt,0:M]))
                for i in range(M):
                    for j in range(M):
                        L0[i,j]=C[i,j]*S[i]/Ni[j]

                L[0:M, 0:M]     = alpha*beta/gIa*L0
                L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
                L[M:2*M, 0:M]   = ((1-alpha)*beta/gIa)*L0
                L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0
                r0[tt] = np.real(np.max(np.linalg.eigvals(L)))

        return r0





"""
---
KreissPy
https://gitlab.com/AustenBolitho/kreisspy
sublibrary dedicated to calculating the transient effects of non-normal
contact matricies
---
"""
def _epsilon_eval(z, A, ord=2):
    """
    Finds the value of \epsilon for a given complex number and matrix.
    Uses the first definition of the pseudospectrum in Trfethen & Embree
    ord="svd" uses fourth definition (may be faster)


    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    order: order of the matrix norm given from associated vector norm
    default is regular L2 norm -> returns maximum singular value.
    accepted inputs are any in spl.norm or "svd"
    """
    z=np.array(z)
    A=np.array(A)
    zc = complex(z[0], z[1])
    try :
        ep = 1/spl.norm(spl.inv(zc*np.eye(*A.shape)-A),ord=ord)
        # ep = spl.norm(zc*np.eye(*A.shape)-A,ord=ord)
    except TypeError:
        if ord=="svd":
            ep = np.min(spl.svdvals(zc*np.eye(*A.shape)-A))
        else: raise Exception("invalid method")
    return ep


def _inv_epsilon_eval(z, A, ord=2):
    """
    Finds the value of 1/\epsilon for a given complex number and matrix.
    Uses the first definition of the pseudospectrum in Trfethen & Embree
    ord="svd" uses fourth definition (may be faster)


    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    order: order of the matrix norm given from associated vector norm
    default is regular L2 norm -> returns maximum singular value.
    accepted inputs are any in spl.norm or "svd"
    """
    z=np.array(z)
    A=np.array(A)
    zc = complex(z[0], z[1])
    try :
        iep = spl.norm(spl.inv(zc*np.eye(*A.shape)-A),ord=ord)
    except TypeError:
        if ord=="svd":
            iep = 1/np.min(spl.svdvals(zc*np.eye(*A.shape)-A))
        else: raise Exception("invalid method")
    return iep


def _kreiss_eval(z, A, theta=0, ord=2):
    """
    Kreiss constant guess for a matrix and pseudo-eigenvalue.

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm
    """
    z=np.array(z)
    A=np.array(A)
    kg = (z[0]-theta)*_inv_epsilon_eval(z, A, ord=ord)
    return kg


def _inv_kreiss_eval(z, A, theta=0, ord=2):
    """
    1/Kreiss constant guess for a matrix and pseudo-eigenvalue.
    for minimizer

    inputs:
    z: length 2 array representing a complex number
    A: an MxM matrix
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm
    """
    z=np.array(z)
    A=np.array(A)
    ikg = _epsilon_eval(z, A, ord=ord)/np.real(z[0]-theta) if z[0]-theta > 0 else np.inf
    # print(z[0]-theta)
    return ikg


def _transient_properties(guess, A, theta=0, ord=2):
    """
    returns the maximal eigenvalue (spectral abcissa),
    initial groth rate (numerical abcissa),
    the Kreiss constant (minimum bound of transient)
    and time of transient growth

    inputs:
    A: an MxM matrix
    guess: initial guess for the minimizer
    theta: normalizing factor found in Townley et al 2007, default 0
    ord: default 2, order of matrix norm

    returns: [spectral abcissa, numerical abcissa, Kreiss constant ,
              duration of transient, henrici's departure from normalcy']
    """
    from scipy.optimize import minimize
    A = np.array(A)
    if np.array_equal(A@A.T, A.T@A):
        warnings.warn("The input matrix is normal")
        # print("The input matrix is normal")
    evals = spl.eigvals(A)
    sa = evals[
        np.where(np.real(evals) == np.amax(np.real(evals)))[0]
    ]
    na = np.real(np.max(spl.eigvals((A+A.T)/2)))
    m = minimize(_inv_kreiss_eval, guess, args=(A, theta, ord),
                 bounds=((0, None), (None, None)))
    K = 1/m.fun
    tau = np.log(1/m.fun)/m.x[0]
    evals2 = np.dot(evals,np.conj(evals))
    frobNorm = spl.norm(A,ord='fro')
    henrici = np.sqrt(frobNorm**2-evals2)#/frobNorm
    return np.array([sa, na, K, tau, henrici],dtype=np.complex64)


def _first_estimate( A, tol=0.001):
    """
    Takes the eigenvalue with the largest real part

    returns a first guess of the
    maximal pseudoeigenvalue in the complex plane
    """
    evals = spl.eigvals(A)
    revals = np.real(evals)
    idxs = np.where(revals == np.amax(revals))[0]
    mevals = evals[idxs]
    iguesses = []
    for evl in mevals:
        guess = []
        a, b = np.real(evl), np.imag(evl)
        if a > 0:
            guess = [a+tol, b]
        else:
            guess = [tol, b]
        iguesses.append(guess)
    return iguesses


def characterise_transient(A, tol=0.001, theta=0, ord=2):
    """
    The maximal eigenvalue (spectral abcissa),
    initial groth rate (numerical abcissa),
    the Kreiss constant (minimum bound of transient)
    and time of transient growth

    Parameters 
    -----------
    A    : an MxM matrix
    tol  : Used to find a first estimate of the pseudospectrum
    theta: normalizing factor found in Townley et al 2007, default 0
    ord  : default 2, order of matrix norm

    Returns 
    ---------
    [spectral abcissa, numerical abcissa, Kreiss constant, 
    duration of transient, henrici's departure from normalcy]

    """
    guesses = _first_estimate(A, tol)
    transient_properties = [1, 0, 0]
    for guess in guesses:
        tp = _transient_properties(guess, A, theta, ord)
        if tp[2] > transient_properties[2]:
            transient_properties = tp
        else:
            pass
    return transient_properties









'''
---
Contact structure: Projecting social contact matrices in 152 countries using
contact surveys and demographic data,
Kiesha Prem, Alex R. Cook, Mark Jit, PLOS Computational Biology, (2017)

---
Below we provide the contact matrix for some of the countries using
data from above
'''


def getCM(country='India', sheet=1):
    """
    Method to compute contact matrices of a given country
    
    The data is read from sheets at: 

    https://github.com/rajeshrinet/pyross/tree/master/examples/data/contact_matrices_152_countries

    Parameters
    ----------
    country: string 
        Default is India
    sheet: int 
        Default is 1
        sheet takes value 1 and 2

    Returns
    ----------
    four np.arrays: CH, CW, CS, CO of the given country 

    CH - home, 

    CW - work, 

    CS - school, 

    CO - other locations 

    """

    u1 ='https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples'
    u2 ='/data/contact_matrices_152_countries/MUestimates_'

    if sheet==1:
        uH = u1 + u2 + 'home_1.xlsx'
        uW = u1 + u2 + 'work_1.xlsx'
        uS = u1 + u2 + 'school_1.xlsx'
        uO = u1 + u2 + 'other_locations_1.xlsx'
    elif sheet==2:
        uH = u1 + u2 + 'home_2.xlsx'
        uW = u1 + u2 + 'work_2.xlsx'
        uS = u1 + u2 + 'school_2.xlsx'
        uO = u1 + u2 + 'other_locations_2.xlsx'
    else:
        raise Exception('There are only two sheets, choose 1 or 2')

    import pandas as pd
    CH = np.array(pd.read_excel(uH,  sheet_name=country))
    CW = np.array(pd.read_excel(uW,  sheet_name=country))
    CS = np.array(pd.read_excel(uS,  sheet_name=country))
    CO = np.array(pd.read_excel(uO,  sheet_name=country))
    return CH, CW, CS, CO


def Denmark():
    uH = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_home_1.xlsx'
    uW = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_work_1.xlsx'
    uS = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_school_1.xlsx'
    uO = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'

    import pandas as pd
    CH = np.array(pd.read_excel(uH,  sheet_name='Denmark'))
    CW = np.array(pd.read_excel(uW,  sheet_name='Denmark'))
    CS = np.array(pd.read_excel(uS,  sheet_name='Denmark'))
    CO = np.array(pd.read_excel(uO,  sheet_name='Denmark'))
    return CH, CW, CS, CO



def France():
    uH = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_home_1.xlsx'
    uW = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_work_1.xlsx'
    uS = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_school_1.xlsx'
    uO = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'

    import pandas as pd
    CH = np.array(pd.read_excel(uH,  sheet_name='France'))
    CW = np.array(pd.read_excel(uW,  sheet_name='France'))
    CS = np.array(pd.read_excel(uS,  sheet_name='France'))
    CO = np.array(pd.read_excel(uO,  sheet_name='France'))
    return CH, CW, CS, CO


def Germany():
    uH = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_home_1.xlsx'
    uW = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_work_1.xlsx'
    uS = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_school_1.xlsx'
    uO = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'

    import pandas as pd
    CH = np.array(pd.read_excel(uH,  sheet_name='Germany'))
    CW = np.array(pd.read_excel(uW,  sheet_name='Germany'))
    CS = np.array(pd.read_excel(uS,  sheet_name='Germany'))
    CO = np.array(pd.read_excel(uO,  sheet_name='Germany'))
    return CH, CW, CS, CO


def India():
    uH = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_home_1.xlsx'
    uW = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_work_1.xlsx'
    uS = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_school_1.xlsx'
    uO = 'https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx'

    import pandas as pd
    CH = np.array(pd.read_excel(uH,  sheet_name='India'))
    CW = np.array(pd.read_excel(uW,  sheet_name='India'))
    CS = np.array(pd.read_excel(uS,  sheet_name='India'))
    CO = np.array(pd.read_excel(uO,  sheet_name='India'))
    return CH, CW, CS, CO


def UK():
    CH=np.array([[4.78812800e-01, 5.51854140e-01, 3.34323605e-01, 1.32361228e-01,
        1.38531588e-01, 2.81604887e-01, 4.06440259e-01, 4.93947983e-01,
        1.13301081e-01, 7.46826414e-02, 4.19640343e-02, 1.79831987e-02,
        5.53694265e-03, 1.42187285e-03, 0.00000000e+00, 5.05582194e-04],
       [2.63264243e-01, 9.18274813e-01, 5.24179768e-01, 1.16285331e-01,
        2.50556852e-02, 1.69858750e-01, 4.48042262e-01, 5.77627602e-01,
        3.24781960e-01, 7.22768451e-02, 2.65332306e-02, 5.85664492e-03,
        0.00000000e+00, 2.42040180e-03, 2.85809310e-04, 0.00000000e+00],
       [1.68812075e-01, 5.35714614e-01, 1.08021932e+00, 3.88591513e-01,
        3.93145217e-02, 1.10571286e-02, 2.16808970e-01, 5.91908717e-01,
        4.85585642e-01, 1.33959469e-01, 4.43333867e-02, 1.75951639e-02,
        4.67677888e-03, 7.51152994e-03, 1.19829213e-03, 2.39287711e-04],
       [9.39012807e-02, 1.53999088e-01, 4.17215826e-01, 9.79076097e-01,
        1.28063144e-01, 3.34981190e-02, 6.10027034e-02, 2.53606638e-01,
        4.21886057e-01, 2.06528142e-01, 7.49213694e-02, 4.80225161e-02,
        7.19471109e-03, 2.45565827e-02, 2.39744368e-03, 0.00000000e+00],
       [1.67946317e-01, 8.08360897e-02, 7.28418698e-02, 3.56303405e-01,
        8.04795340e-01, 2.05901995e-01, 7.48746418e-02, 4.13841822e-02,
        1.64106139e-01, 2.83182758e-01, 8.09986834e-02, 8.53108114e-02,
        1.64317496e-02, 7.56076605e-03, 1.05155415e-03, 1.57920160e-03],
       [4.89661848e-01, 2.96565218e-01, 3.98977026e-02, 6.61533092e-02,
        1.25000620e-01, 6.59542030e-01, 2.10168617e-01, 2.53383507e-02,
        7.44119119e-03, 4.68015533e-02, 1.67113552e-01, 1.10846330e-01,
        3.36293191e-02, 8.39367044e-04, 0.00000000e+00, 2.94523467e-06],
       [3.19984866e-01, 4.72630659e-01, 2.69616873e-01, 7.55885940e-02,
        4.66457340e-02, 8.84928988e-02, 6.42245625e-01, 1.48784529e-01,
        3.25759003e-02, 4.94466873e-03, 1.23221822e-02, 3.97889759e-02,
        1.91884087e-02, 9.65680966e-04, 0.00000000e+00, 7.67613500e-13],
       [3.78242810e-01, 7.00783540e-01, 5.64193426e-01, 1.96423246e-01,
        2.36900269e-02, 8.98937822e-03, 8.70751437e-02, 5.90987777e-01,
        1.53469321e-01, 3.24455468e-03, 2.35796486e-02, 8.02859504e-03,
        1.17312985e-02, 6.12173390e-03, 0.00000000e+00, 0.00000000e+00],
       [1.66028735e-01, 5.16161356e-01, 7.30664425e-01, 4.15001753e-01,
        6.74170325e-02, 4.11946127e-03, 9.01634890e-02, 1.91874369e-01,
        4.57516113e-01, 1.04913169e-01, 1.71962553e-02, 1.05488179e-03,
        2.59360053e-02, 1.20795828e-02, 8.40324698e-03, 1.24138793e-03],
       [1.27256603e-01, 1.46586243e-01, 3.34933702e-01, 7.06502619e-01,
        3.52848641e-01, 8.31144216e-02, 1.08163950e-02, 7.63594681e-02,
        7.19518913e-02, 5.33831753e-01, 1.12210304e-01, 2.84184279e-02,
        1.22674409e-02, 3.04004664e-06, 1.47155837e-03, 7.00229693e-03],
       [1.21649534e-01, 9.95989954e-02, 1.75445499e-01, 2.57908401e-01,
        3.09365024e-01, 7.01661197e-02, 7.04766396e-02, 3.64505998e-02,
        7.93133496e-02, 5.51750268e-02, 3.71421901e-01, 1.27825451e-01,
        1.49161500e-02, 6.01013855e-06, 4.00038647e-03, 2.43485013e-03],
       [1.86135805e-02, 5.51357874e-03, 1.09080090e-01, 2.78496861e-01,
        2.29404069e-01, 1.83248470e-01, 6.35861333e-02, 1.56945824e-02,
        6.86912630e-03, 7.77296348e-02, 9.45600495e-02, 3.89562252e-01,
        9.76702124e-02, 3.08734001e-03, 0.00000000e+00, 0.00000000e+00],
       [2.12793806e-02, 0.00000000e+00, 4.77735588e-02, 3.46141167e-02,
        8.50215545e-02, 1.00750300e-01, 1.10507830e-01, 6.99822752e-02,
        8.30907984e-02, 1.72938762e-02, 4.47522374e-02, 1.13862629e-01,
        4.78747390e-01, 5.72918912e-02, 1.45300042e-02, 0.00000000e+00],
       [4.59608954e-02, 7.01063209e-02, 1.14559880e-01, 1.50158357e-01,
        1.53288467e-02, 7.94963407e-03, 2.16081408e-02, 9.60965711e-02,
        2.16056745e-01, 2.71571261e-02, 1.75702661e-02, 4.79514960e-02,
        8.21058370e-02, 5.10463324e-01, 5.48425643e-02, 0.00000000e+00],
       [0.00000000e+00, 2.23559403e-02, 1.63429174e-01, 2.45762052e-01,
        8.40593873e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.96636658e-01, 8.44905492e-02, 3.44102225e-02, 0.00000000e+00,
        5.82160069e-02, 1.42592716e-01, 1.56981294e-01, 8.97547179e-02],
       [2.06073830e-02, 0.00000000e+00, 3.61661933e-02, 0.00000000e+00,
        1.85817325e-02, 1.01978827e-06, 4.13712762e-02, 0.00000000e+00,
        7.08730070e-02, 5.42524362e-02, 7.66046555e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.10469936e-01, 2.70621588e-01]])

    CW=np.array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 8.20604524e-092, 1.20585150e-005,
        3.16436834e-125],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 4.99018010e-109, 3.44591393e-132,
        1.26863435e-134],
       [0.00000000e+000, 0.00000000e+000, 1.70706524e-002,
        3.39774261e-058, 5.21109185e-055, 1.77192380e-062,
        1.03290434e-001, 1.96122421e-041, 3.90287364e-002,
        1.97799359e-044, 1.59467901e-041, 3.90846629e-047,
        1.67090789e-052, 3.05322327e-067, 4.57147850e-105,
        1.71364811e-089],
       [0.00000000e+000, 0.00000000e+000, 1.11260263e-002,
        1.08280649e+000, 8.66967626e-001, 2.77814213e-001,
        1.07287816e-001, 2.81420074e-001, 1.51004085e-001,
        4.82846088e-001, 3.22855848e-001, 6.17464937e-002,
        1.70700125e-002, 9.94370020e-006, 3.67138547e-006,
        2.32484156e-060],
       [0.00000000e+000, 0.00000000e+000, 2.01011380e-002,
        4.37985950e-001, 6.40587498e-001, 7.09850066e-001,
        5.57489171e-001, 7.12702933e-001, 3.50716589e-001,
        3.71190606e-001, 1.73785224e-001, 1.62704445e-001,
        1.88254228e-002, 1.03983436e-005, 3.77157952e-025,
        6.28788788e-006],
       [0.00000000e+000, 0.00000000e+000, 8.74447054e-003,
        4.87143168e-001, 6.13381662e-001, 8.36808081e-001,
        7.66883222e-001, 6.23447246e-001, 9.47190892e-001,
        5.39282641e-001, 5.49818808e-001, 9.05871444e-002,
        2.23459466e-002, 3.10746661e-005, 2.16644121e-005,
        3.05975144e-046],
       [0.00000000e+000, 0.00000000e+000, 4.84246852e-002,
        9.84231314e-002, 5.92950691e-001, 6.26772329e-001,
        6.46893403e-001, 1.04522455e+000, 5.81682804e-001,
        6.87889559e-001, 2.70550441e-001, 2.14812973e-001,
        1.80432671e-002, 1.16738785e-005, 1.78221099e-006,
        2.15436357e-017],
       [0.00000000e+000, 0.00000000e+000, 1.87403399e-002,
        3.27483491e-001, 3.66842833e-001, 5.91763122e-001,
        5.36313765e-001, 7.02885834e-001, 6.24741019e-001,
        6.60862381e-001, 5.23040167e-001, 1.93609278e-001,
        7.90879435e-003, 5.88071946e-006, 9.26733449e-006,
        9.30126148e-006],
       [0.00000000e+000, 0.00000000e+000, 5.58367043e-003,
        2.29767122e-001, 3.92219793e-001, 5.33086777e-001,
        5.51139706e-001, 8.30081835e-001, 8.77503953e-001,
        7.68856044e-001, 3.49715761e-001, 2.49994838e-001,
        2.45716729e-002, 1.36056185e-005, 4.58282153e-006,
        4.59779025e-006],
       [0.00000000e+000, 0.00000000e+000, 2.63051547e-001,
        2.80066640e-001, 3.93355530e-001, 7.57348378e-001,
        7.56714423e-001, 6.79089053e-001, 8.49060790e-001,
        1.04442896e+000, 4.24243495e-001, 1.94960632e-001,
        2.98059949e-002, 1.32082459e-005, 1.71616271e-005,
        5.85947581e-006],
       [0.00000000e+000, 0.00000000e+000, 1.74585026e-018,
        2.05215851e-031, 1.21657066e-001, 4.25861003e-001,
        4.72693450e-001, 4.75417257e-001, 8.81593841e-001,
        5.17705523e-001, 4.15224535e-001, 1.56205347e-001,
        5.37668689e-002, 2.68951406e-005, 2.64826387e-005,
        3.43401106e-005],
       [0.00000000e+000, 0.00000000e+000, 5.37184378e-003,
        8.58035375e-002, 3.29750962e-001, 3.57451427e-001,
        3.12520128e-001, 2.49804092e-001, 2.92846051e-001,
        3.27971964e-001, 2.95271255e-001, 1.40112625e-001,
        1.34526516e-002, 2.68206223e-006, 1.79208323e-005,
        1.11046250e-019],
       [0.00000000e+000, 0.00000000e+000, 1.31286159e-063,
        6.70806763e-030, 3.17184608e-002, 6.74302678e-002,
        3.18002325e-002, 8.84916618e-002, 7.48812045e-002,
        5.21685926e-002, 5.92572568e-002, 4.19842467e-002,
        2.03529414e-003, 6.73992976e-018, 4.66686893e-006,
        4.52861395e-006],
       [2.58407181e-111, 7.39195225e-130, 5.33140539e-087,
        3.78777949e-005, 6.22554272e-005, 8.71080864e-005,
        6.24939435e-005, 3.79228316e-005, 8.74563518e-005,
        3.81468121e-005, 6.26557947e-005, 8.79220665e-005,
        6.20688567e-005, 1.36475462e-005, 3.77998670e-005,
        1.98536122e-047],
       [2.50639944e-126, 2.99924183e-135, 1.21230989e-089,
        9.79382617e-070, 5.30156997e-066, 1.31128123e-065,
        8.71796045e-063, 2.46221075e-070, 4.22265747e-070,
        1.58532169e-082, 1.40541939e-086, 1.17749352e-085,
        2.02031502e-067, 5.50642578e-066, 2.02545953e-087,
        1.10077206e-089],
       [4.27086792e-122, 8.44726024e-119, 7.93527188e-138,
        2.41833942e-104, 1.45250090e-126, 1.89014730e-117,
        9.02173256e-119, 4.45637903e-114, 2.00246187e-100,
        5.01682767e-115, 1.73033587e-131, 2.38801926e-101,
        2.90450714e-115, 5.81582549e-106, 1.23483720e-102,
        6.02725794e-149]])

    CS=np.array([[9.74577996e-001, 1.51369805e-001, 8.74880926e-003,
        2.62790908e-002, 1.11281607e-002, 8.91043051e-002,
        1.25477587e-001, 8.83182775e-002, 3.71824197e-002,
        2.94092695e-002, 5.10911446e-038, 1.13982464e-032,
        7.58428706e-003, 1.51636768e-003, 1.23262014e-050,
        5.97486362e-064],
       [2.40133743e-001, 2.34515888e+000, 6.39939810e-002,
        3.88802754e-003, 1.80337209e-002, 3.66886638e-002,
        6.36078932e-002, 6.07560447e-002, 6.21345771e-002,
        4.73177817e-002, 4.20258909e-002, 4.36578029e-003,
        3.43822581e-003, 5.95795228e-035, 4.93943288e-004,
        2.07656805e-118],
       [2.82088459e-003, 1.12010442e+000, 2.56184136e+000,
        1.18940767e-001, 1.47157881e-002, 5.39857062e-002,
        4.13607860e-002, 1.22566825e-001, 8.55246510e-002,
        7.33569952e-002, 3.21595148e-002, 2.48458132e-002,
        1.20293062e-002, 2.65862310e-039, 6.67045155e-080,
        1.18947185e-080],
       [4.10378936e-002, 8.15005829e-002, 1.16880401e+000,
        4.14022281e+000, 6.25698482e-002, 1.16615502e-001,
        7.52334095e-002, 7.66148284e-002, 7.30893629e-002,
        5.46524014e-002, 4.43356626e-002, 2.65571157e-002,
        4.12575831e-003, 1.38647056e-079, 1.26698624e-064,
        5.27476571e-157],
       [4.19673041e-010, 1.26450247e-001, 2.15657834e-012,
        2.73508417e-001, 2.31775202e-001, 1.37233733e-002,
        2.25585392e-002, 3.11109807e-002, 1.42312028e-002,
        5.30426306e-003, 1.32960881e-002, 4.26520062e-003,
        2.41842796e-038, 2.49126607e-003, 1.81465300e-104,
        9.56826578e-094],
       [1.05598140e-001, 6.67002690e-002, 1.49726793e-002,
        9.32703539e-018, 4.45742804e-016, 5.54165061e-002,
        3.89811348e-002, 5.54955408e-002, 4.26568613e-002,
        2.53100174e-002, 5.22686376e-074, 7.65053681e-050,
        4.31059746e-063, 3.20010844e-058, 1.10434594e-112,
        4.70398310e-138],
       [9.08419300e-013, 1.01860702e-001, 1.03416290e-002,
        1.17480169e-002, 5.53566940e-003, 2.27561024e-016,
        3.77912488e-002, 1.02625777e-021, 1.73005609e-002,
        2.19613255e-048, 5.26097369e-054, 4.26686266e-073,
        4.40714004e-003, 1.31284672e-038, 9.04985044e-075,
        5.97459017e-166],
       [3.59540282e-002, 1.70694149e-001, 7.34853682e-002,
        3.00452237e-002, 2.91989884e-019, 2.42879579e-002,
        5.25451809e-002, 7.21974967e-002, 6.63406664e-002,
        4.59161426e-002, 2.12417476e-039, 1.28484230e-002,
        3.84942036e-003, 1.17170192e-002, 4.88904328e-097,
        2.56517709e-088],
       [7.00269779e-002, 1.01401068e-001, 3.43308925e-002,
        1.13430645e-028, 4.33430214e-002, 5.87478614e-002,
        3.24249496e-002, 5.84762944e-002, 2.08536379e-002,
        7.77340682e-003, 6.90530538e-003, 6.17465018e-003,
        4.38008111e-046, 3.90101960e-068, 4.69082078e-106,
        2.76209236e-130],
       [1.53340511e-002, 2.79941643e-063, 1.74657731e-002,
        2.07384215e-001, 4.23521169e-050, 3.34922994e-036,
        4.51382885e-002, 9.70364652e-003, 6.70494015e-002,
        4.85106354e-002, 6.26475837e-002, 3.90012986e-002,
        7.40698582e-003, 9.91522890e-119, 5.15067801e-112,
        5.20851281e-116],
       [1.66846886e-064, 6.53087623e-032, 1.33938898e-024,
        4.12769147e-065, 5.74369752e-078, 1.74432966e-090,
        2.13522353e-093, 2.49943297e-019, 1.86659948e-002,
        3.93009404e-028, 3.02015724e-031, 4.08034054e-002,
        1.37379009e-046, 5.46886479e-068, 4.10421908e-139,
        1.54549575e-131],
       [4.72436907e-002, 1.39679917e-001, 5.02310347e-002,
        4.67713263e-029, 2.06363138e-039, 4.03823645e-105,
        3.15124853e-002, 1.27297693e-035, 1.50283699e-001,
        3.34822053e-002, 2.15611077e-042, 7.65831683e-002,
        1.43175261e-028, 1.46897214e-106, 9.60833222e-084,
        1.31966960e-132],
       [1.20464428e-055, 1.28480957e-001, 3.64533950e-028,
        3.62669643e-061, 1.92138523e-002, 7.52981787e-003,
        4.62798825e-002, 1.12103189e-001, 2.05379654e-002,
        3.47636881e-002, 7.30372614e-003, 6.66334027e-003,
        1.06836353e-056, 4.73494581e-003, 6.15572896e-085,
        5.05308567e-099],
       [6.77119009e-075, 6.40703844e-002, 3.86996137e-002,
        1.38552114e-041, 9.87472775e-058, 3.59625833e-062,
        3.50098778e-002, 3.45924571e-002, 4.58618693e-066,
        3.70697105e-037, 2.54826485e-055, 3.42366512e-002,
        6.92907148e-079, 3.38216408e-097, 2.81376672e-076,
        4.14574730e-142],
       [1.07774809e-128, 1.83355530e-067, 3.22341977e-058,
        2.54051489e-102, 1.86466231e-074, 8.74732370e-179,
        2.11147203e-055, 3.80703957e-077, 3.33820454e-103,
        5.82123795e-122, 2.74223987e-101, 5.36542607e-088,
        5.87831910e-114, 2.81586216e-098, 3.11490409e-135,
        2.02469770e-131],
       [5.16754842e-145, 3.94484225e-099, 2.86062514e-060,
        5.86630493e-002, 8.06704383e-060, 4.70909855e-107,
        1.70979948e-133, 2.32792123e-102, 7.42127342e-124,
        4.68662514e-136, 1.26541787e-131, 1.97909543e-091,
        1.02301794e-133, 8.92335236e-124, 6.57361457e-126,
        9.04895761e-118]])
    CO=np.array([[2.57847576e-01, 1.00135168e-01, 4.58036774e-02, 1.27084549e-01,
        1.87303683e-01, 2.57979215e-01, 1.93228849e-01, 3.36594917e-01,
        3.09223290e-01, 7.05385230e-02, 1.52218422e-01, 1.13554852e-01,
        6.15771478e-02, 4.04298741e-02, 3.73564987e-02, 6.69781558e-03],
       [1.78249137e-01, 7.70818807e-01, 1.26353007e-01, 8.93938321e-02,
        5.75063231e-02, 2.19815579e-01, 2.27399212e-01, 1.77127045e-01,
        2.63450303e-01, 1.70585267e-01, 1.58471291e-01, 6.28050238e-02,
        8.89132534e-02, 5.56176145e-02, 3.87559151e-02, 1.34886967e-02],
       [1.45914285e-01, 3.52969647e-01, 8.82339649e-01, 3.10008842e-01,
        8.78033161e-02, 2.11294343e-01, 9.11142295e-02, 1.65645507e-01,
        2.55105780e-01, 1.33087803e-01, 1.09222892e-01, 5.18379440e-02,
        6.16156923e-02, 2.48153626e-02, 3.92820997e-02, 5.22958037e-02],
       [4.15053747e-02, 2.60686913e-01, 6.81929499e-01, 1.67100807e+00,
        2.66344449e-01, 2.04639942e-01, 1.57765096e-01, 3.72781663e-01,
        2.54490869e-01, 2.08705381e-01, 1.58881713e-01, 8.28919449e-02,
        5.87426148e-02, 3.62337333e-02, 1.29249951e-02, 6.26015302e-04],
       [1.52449194e-01, 4.03527309e-02, 1.77345560e-01, 9.56584354e-01,
        7.47473605e-01, 3.65269979e-01, 2.42523037e-01, 2.97250794e-01,
        1.62561984e-01, 3.18278622e-01, 9.37373497e-02, 1.35322516e-01,
        8.49893699e-02, 2.76649176e-02, 2.14095967e-02, 4.97165066e-02],
       [2.46624662e-01, 9.32518220e-02, 7.90416058e-02, 2.04867788e-01,
        9.92753107e-01, 8.53320503e-01, 4.57489561e-01, 2.65745555e-01,
        2.87944489e-01, 3.21539788e-01, 1.99559888e-01, 9.54298936e-02,
        9.31612406e-02, 5.45181631e-02, 1.70750122e-02, 3.48407388e-06],
       [1.63303370e-01, 1.69284990e-01, 9.07730944e-02, 1.80129515e-01,
        3.17586122e-01, 4.39556514e-01, 6.68057369e-01, 4.44733217e-01,
        2.31203718e-01, 2.28177542e-01, 2.20039677e-01, 2.46935289e-01,
        1.01243142e-01, 5.72768339e-02, 9.74062159e-03, 2.48212962e-02],
       [1.28967710e-01, 2.19607305e-01, 9.58601352e-02, 9.96702012e-02,
        2.39546405e-01, 2.98953187e-01, 3.06342303e-01, 5.21367017e-01,
        5.10712253e-01, 2.51296074e-01, 1.47519291e-01, 1.96720545e-01,
        2.07242065e-01, 1.00568610e-01, 1.11409484e-01, 3.83885126e-02],
       [3.75239432e-11, 2.52235248e-01, 1.50369063e-01, 1.45922200e-01,
        2.66348797e-01, 1.77376873e-01, 3.63119379e-01, 3.48387652e-01,
        3.93890700e-01, 3.89781423e-01, 2.38353809e-01, 1.97666911e-01,
        9.66813603e-02, 2.34037924e-02, 5.32265244e-02, 7.30241759e-05],
       [1.38622045e-04, 7.32972090e-03, 2.82757900e-02, 1.29584707e-01,
        1.86858340e-01, 1.31553248e-01, 1.69202815e-01, 3.93752639e-01,
        4.64534939e-01, 5.56651247e-01, 3.18427668e-01, 1.22253182e-01,
        1.62523947e-01, 9.90464287e-02, 6.13700738e-02, 5.96663544e-02],
       [2.83551289e-02, 3.04613089e-02, 1.01989787e-01, 3.86949156e-01,
        2.98423992e-01, 4.77952248e-01, 2.36001646e-01, 4.64053305e-01,
        4.06314308e-01, 6.03299664e-01, 2.31973686e-01, 2.91053827e-01,
        1.97333552e-01, 1.43771820e-01, 1.24694792e-01, 9.82611919e-03],
       [1.06247959e-01, 6.21835454e-02, 5.85213722e-02, 1.08802565e-01,
        3.56437338e-01, 5.74277189e-01, 5.46699177e-01, 4.62661146e-01,
        4.86305996e-01, 2.20824496e-01, 4.62143721e-01, 6.39590054e-01,
        3.97758991e-01, 2.04284068e-01, 1.23780529e-01, 1.09684868e-01],
       [6.06193015e-08, 9.46852632e-02, 6.52705519e-02, 2.53132268e-02,
        2.86419963e-01, 2.16646288e-01, 2.83354109e-01, 3.12674071e-01,
        3.22305670e-01, 2.70592125e-01, 2.77529477e-01, 3.98381793e-01,
        2.58761860e-01, 9.47893694e-02, 1.13889277e-01, 8.17784031e-02],
       [1.89542956e-02, 1.10966265e-01, 6.29986791e-02, 6.30557072e-02,
        1.07816765e-01, 4.93773233e-01, 2.88114222e-01, 4.62696678e-01,
        4.17082281e-01, 4.30372098e-01, 3.87893725e-01, 5.16293634e-01,
        2.89415248e-01, 2.15854635e-01, 1.36359067e-01, 5.64939475e-02],
       [7.28168981e-02, 2.60738708e-02, 9.09315193e-02, 2.42115267e-01,
        5.06491395e-01, 4.39531838e-01, 2.42898620e-01, 1.38245804e-01,
        4.88467243e-01, 3.56689765e-01, 3.37426065e-01, 4.67690646e-01,
        4.82265927e-01, 5.51656694e-01, 2.41565408e-01, 2.49247170e-01],
       [1.35424505e-09, 6.88316177e-04, 4.13998603e-02, 1.11795121e-01,
        6.84467694e-02, 3.86038638e-02, 1.29207583e-01, 2.15447768e-01,
        1.98027155e-01, 4.06778809e-01, 3.52954451e-01, 1.98892238e-04,
        4.66582938e-01, 3.38156572e-04, 1.92071526e-01, 4.67710373e-01]])
    return CH, CW, CS, CO

