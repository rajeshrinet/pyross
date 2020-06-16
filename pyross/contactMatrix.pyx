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


def China():
    CH=np.array([[0.50245185, 0.39319654, 0.22049273, 0.11464803, 0.21323029,
        0.38346006, 0.46250082, 0.42127211, 0.20640593, 0.07465477,
        0.08125642, 0.05577225, 0.04468528, 0.01588087, 0.00711776,
        0.01088331],
       [0.23860502, 0.71275983, 0.3728312 , 0.15123651, 0.05357911,
        0.20940918, 0.44600141, 0.46210248, 0.35268315, 0.12581834,
        0.05977214, 0.05840022, 0.03339426, 0.01772328, 0.0086909 ,
        0.01061312],
       [0.13880461, 0.37675977, 1.13783701, 0.34670914, 0.07161731,
        0.06003415, 0.16794907, 0.3692376 , 0.43620455, 0.19019275,
        0.07638444, 0.02679322, 0.01575774, 0.01803953, 0.01501266,
        0.01019597],
       [0.08769989, 0.16275172, 0.4216981 , 1.10489232, 0.23226524,
        0.0743461 , 0.04494658, 0.23121244, 0.40176347, 0.39999886,
        0.20217199, 0.05792504, 0.02441132, 0.01508249, 0.00803504,
        0.00847246],
       [0.22623601, 0.12253121, 0.14267313, 0.51242707, 1.37170599,
        0.36852607, 0.08272318, 0.03856995, 0.18893472, 0.45515718,
        0.31115952, 0.14034998, 0.03435313, 0.00846109, 0.00866136,
        0.01294392],
       [0.43971289, 0.18424735, 0.07126216, 0.12811871, 0.32408357,
        0.95074676, 0.22547206, 0.04470752, 0.02590678, 0.11685738,
        0.23820279, 0.14379627, 0.06558941, 0.01899097, 0.00356559,
        0.01183111],
       [0.48617709, 0.5700758 , 0.30169896, 0.06351614, 0.09206835,
        0.25391134, 0.78818618, 0.23316616, 0.07701034, 0.02676613,
        0.05849728, 0.08919717, 0.0817643 , 0.01816635, 0.01016936,
        0.00709265],
       [0.55304458, 0.86147279, 0.78184935, 0.35416204, 0.05558047,
        0.06866975, 0.2280478 , 0.97395323, 0.2088356 , 0.05595726,
        0.03464154, 0.03790391, 0.06938065, 0.04499021, 0.02156145,
        0.00640075],
       [0.37049783, 0.71361405, 0.92469148, 0.71133231, 0.211384  ,
        0.06928312, 0.13626539, 0.25078984, 0.89942926, 0.20992613,
        0.0698799 , 0.01754183, 0.04997377, 0.05157699, 0.02124004,
        0.01603826],
       [0.20886991, 0.42516528, 0.61322755, 0.83199025, 0.49790148,
        0.1782792 , 0.0579498 , 0.12865486, 0.21668584, 0.78878795,
        0.20475256, 0.04636657, 0.02422325, 0.01991155, 0.01872582,
        0.03635468],
       [0.21908936, 0.19064061, 0.30634307, 0.35992597, 0.36091717,
        0.25482451, 0.11134677, 0.04902627, 0.0867571 , 0.15608597,
        0.45574892, 0.12136358, 0.02640483, 0.00989299, 0.00993613,
        0.02738699],
       [0.51398229, 0.52429881, 0.33340351, 0.41696555, 0.37293697,
        0.47616806, 0.39059245, 0.13550662, 0.07280143, 0.1618376 ,
        0.25019114, 0.62079057, 0.15650671, 0.0562186 , 0.01172889,
        0.02977188],
       [0.48933121, 0.45411679, 0.29426714, 0.26763078, 0.19159465,
        0.27321849, 0.36069834, 0.27382629, 0.12918232, 0.06608195,
        0.10735181, 0.19662248, 0.46673642, 0.11934614, 0.02294591,
        0.00714077],
       [0.30001849, 0.4499232 , 0.39991344, 0.20060218, 0.14502266,
        0.17618424, 0.29453414, 0.38477539, 0.28527436, 0.0939626 ,
        0.08423378, 0.1096708 , 0.14464806, 0.41537419, 0.06512316,
        0.01425282],
       [0.14526383, 0.44325346, 0.39095056, 0.28735645, 0.06432907,
        0.15515522, 0.12754867, 0.3119905 , 0.28000306, 0.2246746 ,
        0.11920272, 0.0656158 , 0.09475726, 0.10928283, 0.26681268,
        0.06641991],
       [0.25457252, 0.34463553, 0.53890775, 0.43392491, 0.11871411,
        0.10655955, 0.12608649, 0.2690276 , 0.30844247, 0.30876886,
        0.31492335, 0.14027073, 0.05369849, 0.09421629, 0.06909398,
        0.17735295]])

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
        0.00000000e+000, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [0.00000000e+000, 0.00000000e+000, 2.11173918e-002,
        7.62776977e-003, 1.06695575e-002, 3.25812680e-003,
        2.58512440e-002, 7.34320923e-003, 2.58406210e-002,
        1.50218485e-002, 6.38697537e-003, 1.62940058e-008,
        1.94799699e-017, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [0.00000000e+000, 0.00000000e+000, 1.35953741e-002,
        3.88689155e-001, 4.08017318e-001, 2.47651484e-001,
        2.24134258e-001, 2.07666212e-001, 2.31940702e-001,
        1.82988043e-001, 1.15384878e-001, 5.67136025e-002,
        1.30854460e-002, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [0.00000000e+000, 0.00000000e+000, 2.33086881e-002,
        2.75000161e-001, 6.97312457e-001, 7.05311149e-001,
        5.71488663e-001, 6.27606090e-001, 4.75070177e-001,
        3.80754541e-001, 3.12166265e-001, 1.48558734e-001,
        4.72091100e-002, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [0.00000000e+000, 0.00000000e+000, 3.16370108e-002,
        2.55106239e-001, 7.00370968e-001, 1.24474906e+000,
        8.64716869e-001, 8.41492886e-001, 7.79395842e-001,
        5.46826767e-001, 4.82650058e-001, 2.30971561e-001,
        6.64575141e-002, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [0.00000000e+000, 0.00000000e+000, 3.48004415e-002,
        1.33839789e-001, 4.71133006e-001, 8.19827452e-001,
        1.04157009e+000, 9.10454740e-001, 8.18015473e-001,
        6.68328658e-001, 4.32866558e-001, 2.61606563e-001,
        6.04122985e-002, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [0.00000000e+000, 0.00000000e+000, 2.13307841e-002,
        2.66081463e-001, 3.89539819e-001, 7.67511713e-001,
        7.97888867e-001, 1.09893976e+000, 1.07185311e+000,
        7.42267142e-001, 5.80325957e-001, 2.39182664e-001,
        4.26744461e-002, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [0.00000000e+000, 0.00000000e+000, 2.41659874e-002,
        1.66607023e-001, 4.59676455e-001, 7.61033278e-001,
        8.71439853e-001, 9.29535349e-001, 1.14172749e+000,
        9.18027938e-001, 7.08490856e-001, 2.54601905e-001,
        6.27344600e-002, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [0.00000000e+000, 0.00000000e+000, 3.22656159e-002,
        2.08403227e-001, 3.13515624e-001, 5.72495562e-001,
        7.12610202e-001, 7.80892927e-001, 8.09947753e-001,
        7.91053589e-001, 5.47131673e-001, 2.88371980e-001,
        5.00035106e-002, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [0.00000000e+000, 0.00000000e+000, 3.67589160e-002,
        1.64245967e-001, 2.74964360e-001, 6.07745584e-001,
        6.87760573e-001, 7.13846209e-001, 9.69106136e-001,
        9.35437991e-001, 7.63088434e-001, 3.83459339e-001,
        6.29666466e-002, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [0.00000000e+000, 0.00000000e+000, 6.04304597e-002,
        1.17760250e-001, 1.97669003e-001, 3.83729200e-001,
        5.17965264e-001, 4.86786574e-001, 6.23902503e-001,
        4.92515202e-001, 4.90299003e-001, 3.17781210e-001,
        6.33744043e-002, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [0.00000000e+000, 0.00000000e+000, 2.07765111e-002,
        1.62626260e-002, 8.00162168e-002, 1.50520379e-001,
        1.60973010e-001, 1.89618207e-001, 2.05553479e-001,
        2.00856121e-001, 1.64785405e-001, 1.31752561e-001,
        2.21392798e-002, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

    CS=np.array([[3.21886487e-001, 4.31838793e-002, 7.87813457e-003,
        8.10212428e-003, 5.34544349e-003, 2.18036508e-002,
        4.01447341e-002, 2.99589878e-002, 1.40901776e-002,
        1.66873506e-002, 9.48109665e-003, 7.42117505e-003,
        1.28313946e-003, 7.79274003e-004, 8.23610731e-066,
        6.37927908e-120],
       [5.40357094e-002, 4.49544742e+000, 2.31911004e-001,
        2.98796511e-002, 2.38196210e-002, 6.83021911e-002,
        9.06742014e-002, 9.77616359e-002, 1.03627127e-001,
        6.96350873e-002, 4.21970508e-002, 1.67056619e-002,
        6.29322763e-003, 1.54094275e-003, 3.48249318e-004,
        8.10468990e-039],
       [4.56197949e-004, 9.00348813e-001, 4.84950716e+000,
        1.74748367e-001, 1.43525174e-002, 4.99152969e-002,
        5.17116761e-002, 9.37433059e-002, 1.06808876e-001,
        7.90777709e-002, 4.02470348e-002, 2.57559823e-002,
        6.39241986e-003, 7.80018912e-004, 4.94640390e-025,
        1.82425735e-004],
       [2.59826163e-003, 4.49284060e-002, 1.75119107e+000,
        7.00065237e+000, 6.66362834e-002, 6.41143377e-002,
        7.28389736e-002, 1.17085535e-001, 1.07937607e-001,
        1.19734711e-001, 4.70041131e-002, 3.57218809e-002,
        7.75480384e-003, 1.20800064e-003, 6.22849724e-033,
        1.71079556e-070],
       [7.18494994e-003, 1.90456035e-002, 7.11285512e-003,
        6.81510790e-001, 3.09313466e-001, 3.67088609e-002,
        2.49957467e-002, 3.36072662e-002, 2.30794471e-002,
        2.73609182e-002, 1.18261720e-002, 9.88352922e-003,
        7.46851390e-004, 1.26957295e-003, 1.77236275e-004,
        1.22001800e-047],
       [7.03585259e-003, 9.51897617e-002, 2.80091830e-002,
        1.63955799e-001, 2.03330974e-001, 1.25288550e-001,
        2.22053144e-002, 3.24086916e-002, 4.16550881e-002,
        3.52298852e-002, 7.38769725e-003, 1.42939798e-002,
        4.71360888e-003, 2.46304801e-003, 4.63369124e-004,
        1.28657489e-003],
       [1.41420736e-002, 3.23792522e-001, 2.00821638e-001,
        1.44937649e-001, 3.76608865e-002, 6.95028072e-002,
        6.40384939e-002, 4.89542483e-002, 5.76898427e-002,
        2.90238938e-002, 1.90733105e-002, 3.44191189e-003,
        6.08792703e-003, 4.59321795e-004, 1.65542910e-048,
        3.11351133e-055],
       [2.41117438e-002, 2.18059588e-001, 1.49439881e-001,
        8.64197646e-002, 1.62193635e-002, 4.93001010e-002,
        6.97018124e-002, 5.76676277e-002, 6.97705209e-002,
        3.22215970e-002, 3.77851285e-003, 1.04503681e-002,
        6.82947373e-004, 2.21545239e-003, 1.84622561e-123,
        9.70360629e-067],
       [7.82544035e-003, 1.37251237e-001, 1.01756371e-001,
        4.67973539e-001, 8.40278679e-003, 2.67121948e-002,
        2.65835609e-002, 3.99551556e-002, 9.35346534e-002,
        3.21395696e-002, 2.80089918e-002, 9.31318169e-003,
        7.20730916e-003, 5.21243798e-004, 4.81876334e-068,
        2.40487229e-092],
       [6.56388616e-002, 2.92068424e-001, 1.76863201e-001,
        7.18119988e-001, 5.87723732e-003, 3.21101668e-002,
        6.28059070e-002, 6.02134218e-002, 6.12425924e-002,
        3.55161817e-002, 3.44055391e-002, 1.87608794e-002,
        4.26471008e-003, 1.83259892e-003, 6.21757597e-134,
        3.27638131e-072],
       [1.72826706e-002, 3.45863937e-001, 4.08190750e-001,
        4.86725338e-001, 4.79873348e-003, 1.31118474e-002,
        3.71201551e-002, 3.98996911e-002, 5.34836050e-002,
        7.10680700e-002, 3.02702926e-002, 1.87965544e-002,
        5.20108172e-003, 9.20539920e-024, 1.23874350e-117,
        5.64540093e-078],
       [6.15255001e-002, 4.12075699e-001, 3.35840049e-001,
        4.27871217e-001, 6.50142252e-003, 5.96517196e-002,
        2.34897735e-002, 4.30134488e-002, 5.83665228e-002,
        4.10375410e-002, 2.95751073e-002, 3.79164500e-002,
        1.11383707e-002, 1.21296823e-031, 7.82718666e-004,
        7.63029675e-004],
       [3.64017022e-002, 7.87072237e-002, 3.80986281e-002,
        1.97501943e-001, 1.35485027e-002, 1.86683162e-003,
        1.73959006e-002, 5.17606541e-002, 1.20597774e-002,
        1.86965053e-002, 1.27376809e-002, 7.23776318e-003,
        2.37743850e-002, 1.15917987e-002, 4.42529176e-067,
        2.12162613e-037],
       [1.67970136e-003, 3.00909263e-002, 1.13558325e-002,
        8.52722239e-032, 1.99202048e-003, 1.96529324e-003,
        1.27604064e-002, 5.72956376e-003, 5.90308814e-003,
        9.42327139e-003, 2.01849092e-003, 1.59633979e-002,
        8.47170013e-003, 1.76301994e-002, 1.11255457e-002,
        3.45732208e-126],
       [1.28088730e-028, 5.12236581e-026, 1.93369392e-040,
        7.62526866e-003, 2.63694184e-022, 1.69841476e-024,
        1.25953049e-026, 7.62912659e-003, 7.86061664e-003,
        2.11802390e-002, 3.52336553e-002, 2.14568223e-002,
        7.74353528e-003, 8.01446712e-003, 7.91286559e-003,
        2.13826122e-002],
       [2.81656250e-094, 2.11695749e-002, 8.47825559e-042,
        2.13052758e-002, 4.89979496e-036, 7.59588107e-003,
        9.77754732e-069, 2.23293606e-060, 1.43869010e-048,
        8.56906259e-060, 4.69700127e-042, 1.59939884e-046,
        2.21076487e-083, 8.85961526e-107, 1.02042962e-080,
        6.61414633e-113]])

    CO=np.array([[0.58532239, 0.25010009, 0.12864745, 0.11996247, 0.26376152,
        0.33189934, 0.37388734, 0.38380709, 0.29222448, 0.19850351,
        0.16119892, 0.18468299, 0.10981365, 0.08488349, 0.05152729,
        0.02319481],
       [0.27232076, 1.01244942, 0.37297503, 0.13060205, 0.14385994,
        0.23524093, 0.27995777, 0.37202393, 0.35385488, 0.16304678,
        0.08772002, 0.12473224, 0.10376296, 0.0695574 , 0.03192514,
        0.02255547],
       [0.08819483, 0.4628153 , 1.64558134, 0.32469192, 0.27931711,
        0.19463275, 0.21900695, 0.31327683, 0.40677363, 0.24782665,
        0.1093901 , 0.09334526, 0.0586752 , 0.05452218, 0.03746198,
        0.03084418],
       [0.06078714, 0.18887618, 0.8834097 , 3.34437168, 1.0530404 ,
        0.40582407, 0.25052814, 0.35882935, 0.41319925, 0.35242427,
        0.10690029, 0.07284691, 0.0478246 , 0.03911358, 0.02257566,
        0.01434166],
       [0.11223842, 0.14110774, 0.21147037, 1.71904563, 3.61567468,
        1.17764477, 0.68024681, 0.54053088, 0.47266504, 0.52864144,
        0.21570717, 0.19603786, 0.07064235, 0.04876693, 0.051688  ,
        0.03896279],
       [0.15985468, 0.0820242 , 0.07422764, 0.36630707, 1.33098828,
        1.39947365, 0.77568763, 0.62135959, 0.46715861, 0.40563311,
        0.23037947, 0.16951392, 0.05981635, 0.04005329, 0.02429178,
        0.0117171 ],
       [0.15740429, 0.11042966, 0.16861333, 0.16966841, 0.5842739 ,
        0.70777311, 0.86527832, 0.73903438, 0.52293257, 0.38317283,
        0.26113507, 0.25863305, 0.10602544, 0.06953108, 0.03392475,
        0.03016622],
       [0.17087745, 0.19881315, 0.1505326 , 0.1406533 , 0.43578369,
        0.63021968, 0.76882401, 1.12052283, 0.87851519, 0.54074937,
        0.24767632, 0.26937532, 0.20718835, 0.13036818, 0.07442076,
        0.02818315],
       [0.13152065, 0.16846405, 0.26530249, 0.21506319, 0.50210503,
        0.49676924, 0.66836956, 0.86396813, 1.0536867 , 0.61539327,
        0.28371534, 0.18314164, 0.15536985, 0.0992677 , 0.07318432,
        0.02957073],
       [0.03810813, 0.06183575, 0.08096854, 0.1948775 , 0.34841355,
        0.35636341, 0.42600574, 0.56928321, 0.61413421, 0.60368108,
        0.31563193, 0.19830773, 0.12631215, 0.07527828, 0.06501655,
        0.04804548],
       [0.04072554, 0.07545125, 0.08609694, 0.16151649, 0.38366039,
        0.40128701, 0.28666418, 0.33264328, 0.42324745, 0.45343969,
        0.23933167, 0.26023223, 0.1416327 , 0.07042123, 0.0449786 ,
        0.02901949],
       [0.07608311, 0.06682999, 0.07226716, 0.10823789, 0.35255639,
        0.48966271, 0.49462904, 0.49095585, 0.53954755, 0.36468633,
        0.35958943, 0.46075163, 0.26716929, 0.1357402 , 0.07290507,
        0.03788561],
       [0.04579709, 0.04943731, 0.04276711, 0.08574742, 0.23506523,
        0.29806454, 0.29986279, 0.42994531, 0.4168921 , 0.32377115,
        0.1968695 , 0.33484764, 0.26080864, 0.17309502, 0.1176473 ,
        0.04724742],
       [0.03962516, 0.05171708, 0.03303299, 0.04388546, 0.16662918,
        0.21444327, 0.26461243, 0.28369042, 0.30363724, 0.2315486 ,
        0.17193873, 0.2555321 , 0.21702392, 0.17209192, 0.088359  ,
        0.05045597],
       [0.01438688, 0.03011111, 0.04360325, 0.10269269, 0.12385505,
        0.14881748, 0.14940992, 0.24878008, 0.32753649, 0.23676843,
        0.13010263, 0.17832656, 0.26282365, 0.2155843 , 0.19256528,
        0.06718525],
       [0.02124353, 0.02133733, 0.03167565, 0.0220843 , 0.06286517,
        0.06916845, 0.12801801, 0.1229236 , 0.13185378, 0.15682915,
        0.07684999, 0.1017857 , 0.08626694, 0.10467985, 0.08835442,
        0.05736747]])
    return CH, CW, CS, CO






def Denmark():
    CH = np.array([[6.43135403e-01, 4.37109727e-01, 1.65972900e-01, 6.46734926e-02,
        1.08276297e-01, 3.09772839e-01, 6.28274486e-01, 5.63719487e-01,
        1.91182516e-01, 6.22685805e-02, 5.33630354e-02, 3.12695833e-02,
        2.18438345e-02, 5.52208437e-03, 2.32422223e-03, 4.30924402e-03],
       [3.11742776e-01, 9.54070432e-01, 3.58466807e-01, 1.05591984e-01,
        3.26567003e-02, 1.16733765e-01, 4.89803309e-01, 6.76861628e-01,
        4.68554353e-01, 1.44906742e-01, 3.58000969e-02, 1.93663478e-02,
        1.13694585e-02, 5.41708469e-03, 2.27846427e-03, 3.26172556e-03],
       [1.17763249e-01, 3.61105894e-01, 1.56996233e+00, 4.11622888e-01,
        5.45149643e-02, 2.33192888e-02, 1.08434832e-01, 4.14273679e-01,
        6.32046007e-01, 2.94517778e-01, 6.74336397e-02, 1.31731720e-02,
        1.08235048e-02, 7.14073481e-03, 6.23734685e-03, 3.49362269e-03],
       [4.62644354e-02, 9.27347411e-02, 4.36962772e-01, 1.32593470e+00,
        2.09633242e-01, 3.15031354e-02, 1.47440057e-02, 1.47309585e-01,
        3.97077130e-01, 5.53258199e-01, 2.21155389e-01, 4.40654592e-02,
        1.35402291e-02, 8.87867442e-03, 4.71599799e-03, 2.61953041e-03],
       [9.36274731e-02, 4.51734157e-02, 6.67163961e-02, 3.82634233e-01,
        1.16842508e+00, 2.03534965e-01, 3.54761119e-02, 1.63877063e-02,
        1.21596739e-01, 4.97823220e-01, 2.94505597e-01, 1.08518732e-01,
        1.51941990e-02, 2.11333867e-03, 3.38856084e-03, 2.58003600e-03],
       [3.17470425e-01, 7.77646600e-02, 2.17949374e-02, 4.65763025e-02,
        1.94533898e-01, 9.31094660e-01, 1.76402402e-01, 2.81464933e-02,
        1.40081580e-02, 4.89123680e-02, 1.93717296e-01, 8.93085771e-02,
        3.33044747e-02, 7.20095434e-03, 8.44808678e-04, 4.85635824e-03],
       [5.40208435e-01, 4.34015645e-01, 1.44827315e-01, 1.99272305e-02,
        4.08146802e-02, 1.80344863e-01, 8.94217257e-01, 1.94574577e-01,
        5.18718972e-02, 1.58647439e-02, 2.88446183e-02, 5.14018535e-02,
        5.05389728e-02, 7.39397381e-03, 3.12573201e-03, 2.84777919e-03],
       [4.84213453e-01, 7.13372289e-01, 5.07985340e-01, 1.62765159e-01,
        2.07670807e-02, 3.01858815e-02, 1.53662131e-01, 1.01863510e+00,
        1.60255217e-01, 3.76018904e-02, 1.91258539e-02, 1.31603423e-02,
        2.52457095e-02, 1.14302007e-02, 6.17989939e-03, 2.36321369e-03],
       [1.80264326e-01, 4.38927703e-01, 6.73120678e-01, 4.11767023e-01,
        8.63594755e-02, 2.25986792e-02, 5.64891481e-02, 1.56985988e-01,
        8.59512635e-01, 1.53712674e-01, 2.74766544e-02, 4.26464870e-03,
        1.45950876e-02, 1.43247176e-02, 5.51412433e-03, 5.99417369e-03],
       [9.15973775e-02, 2.20071418e-01, 4.27066792e-01, 6.76078796e-01,
        3.81900850e-01, 5.52980558e-02, 2.33888701e-02, 6.93042288e-02,
        1.65947945e-01, 8.62163971e-01, 1.43140625e-01, 2.50396418e-02,
        7.81835657e-03, 7.46603580e-03, 5.81944671e-03, 9.88874911e-03],
       [1.29794608e-01, 1.04645589e-01, 2.07881927e-01, 3.44446378e-01,
        3.86639416e-01, 2.20757042e-01, 6.18983151e-02, 3.34571395e-02,
        5.69807304e-02, 1.73792832e-01, 7.48544248e-01, 1.26238161e-01,
        2.08418708e-02, 6.04119928e-03, 5.50658067e-03, 2.10188671e-02],
       [2.19318531e-01, 2.07500542e-01, 1.39641939e-01, 2.21079752e-01,
        2.48704473e-01, 2.68530306e-01, 2.04582643e-01, 5.55753938e-02,
        2.59226486e-02, 1.04967789e-01, 2.09778742e-01, 8.37487924e-01,
        1.29943327e-01, 3.62080155e-02, 4.52477334e-03, 2.07519325e-02],
       [2.61264775e-01, 2.28991138e-01, 1.67818444e-01, 1.40737444e-01,
        1.05716165e-01, 1.64123659e-01, 2.54420950e-01, 1.48506948e-01,
        6.92267737e-02, 3.16771679e-02, 8.50917484e-02, 1.98220414e-01,
        8.95641356e-01, 1.10790197e-01, 2.27419074e-02, 5.43557675e-03],
       [2.08574479e-01, 3.15744274e-01, 2.91055366e-01, 1.58011622e-01,
        9.57754619e-02, 1.23546304e-01, 2.15585235e-01, 2.16899567e-01,
        1.84570100e-01, 8.06113715e-02, 7.26392130e-02, 1.24183968e-01,
        1.91056386e-01, 1.05595421e+00, 1.37294982e-01, 1.88711376e-02],
       [8.74823588e-02, 2.61896247e-01, 2.48404200e-01, 2.20097755e-01,
        3.55022074e-02, 7.21201171e-02, 6.71751513e-02, 1.75506851e-01,
        1.50364150e-01, 1.38937017e-01, 7.53405002e-02, 3.96370431e-02,
        1.24551994e-01, 1.99755636e-01, 6.06796130e-01, 1.24262851e-01],
       [1.62028482e-01, 2.08477596e-01, 3.19002969e-01, 2.54787827e-01,
        6.10183705e-02, 6.30528510e-02, 7.70854142e-02, 1.61189359e-01,
        2.18260476e-01, 1.68968387e-01, 2.89429691e-01, 1.27433303e-01,
        4.53693140e-02, 9.00501873e-02, 9.49768097e-02, 3.81399543e-01]])

    CW = np.array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 8.20604524e-092, 1.20585150e-005,
        3.16436834e-125],
       [0.00000000e+000, 1.02576076e-004, 3.75363957e-072,
        1.23809837e-059, 8.57393963e-007, 3.15186206e-003,
        8.19034282e-003, 1.84487662e-003, 7.09499869e-003,
        2.03795201e-003, 5.88144072e-004, 3.73574408e-004,
        1.25324604e-057, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [0.00000000e+000, 9.72031306e-004, 1.84372915e-001,
        2.90132625e-002, 3.33397177e-002, 1.03031168e-002,
        8.46303790e-002, 2.43694668e-002, 8.45619724e-002,
        5.02315016e-002, 2.13574837e-002, 3.81254221e-008,
        1.56757676e-017, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [0.00000000e+000, 2.99881195e-003, 5.17118593e-002,
        6.44085556e-001, 5.55438848e-001, 3.41180123e-001,
        3.19665103e-001, 3.00239696e-001, 3.30667316e-001,
        2.66573910e-001, 1.68091551e-001, 5.78117580e-002,
        4.58744859e-003, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [0.00000000e+000, 2.68565847e-003, 7.28338621e-002,
        3.74361004e-001, 7.79832978e-001, 7.98252069e-001,
        6.69593336e-001, 7.45428404e-001, 5.56401881e-001,
        4.55676403e-001, 3.73593641e-001, 1.24406665e-001,
        1.35964338e-002, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [0.00000000e+000, 3.28964701e-003, 1.00045160e-001,
        3.51450258e-001, 7.92660906e-001, 1.42569373e+000,
        1.02532741e+000, 1.01147317e+000, 9.23791660e-001,
        6.62287173e-001, 5.84562566e-001, 1.95744289e-001,
        1.93699465e-002, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [0.00000000e+000, 5.24888416e-003, 1.13927769e-001,
        1.90885187e-001, 5.52010113e-001, 9.72100332e-001,
        1.27856066e+000, 1.13293888e+000, 1.00374101e+000,
        8.37974615e-001, 5.42746301e-001, 2.29521527e-001,
        1.82286241e-002, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [0.00000000e+000, 3.63910228e-003, 7.07891903e-002,
        3.84695307e-001, 4.62669261e-001, 9.22547914e-001,
        9.92865737e-001, 1.38623637e+000, 1.33324728e+000,
        9.43444373e-001, 7.37615642e-001, 2.12725608e-001,
        1.30530411e-002, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [0.00000000e+000, 2.02740497e-003, 7.90818284e-002,
        2.37524060e-001, 5.38372764e-001, 9.02027131e-001,
        1.06929508e+000, 1.15622231e+000, 1.40039298e+000,
        1.15059900e+000, 8.87982495e-001, 2.23287163e-001,
        1.89217810e-002, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [0.00000000e+000, 1.18127988e-003, 1.07892869e-001,
        3.03598323e-001, 3.75206745e-001, 6.93375837e-001,
        8.93496414e-001, 9.92538935e-001, 1.01513804e+000,
        1.01310539e+000, 7.00716974e-001, 2.58425755e-001,
        1.54112182e-002, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [0.00000000e+000, 1.74435089e-004, 1.22918581e-001,
        2.39271903e-001, 3.29071229e-001, 7.36072258e-001,
        8.62343141e-001, 9.07324795e-001, 1.21462299e+000,
        1.19802474e+000, 9.77299578e-001, 3.43640318e-001,
        1.94065840e-002, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [0.00000000e+000, 1.48233931e-003, 1.41397813e-001,
        1.20040463e-001, 1.65532788e-001, 3.25203670e-001,
        4.54438822e-001, 4.32940949e-001, 5.47165662e-001,
        4.41369556e-001, 4.39385583e-001, 1.99271394e-001,
        1.36673518e-002, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [0.00000000e+000, 6.21131870e-005, 1.67191100e-002,
        5.70129291e-003, 2.30450265e-002, 4.38712119e-002,
        4.85715087e-002, 5.79994466e-002, 6.19984282e-002,
        6.19044036e-002, 5.07875516e-002, 2.84138151e-002,
        1.64205505e-003, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

    CS = np.array([[2.06141581e+000, 2.90316108e-001, 5.42489211e-002,
        6.84260312e-002, 1.61192955e-002, 6.73889239e-002,
        1.35421936e-001, 1.11604162e-001, 5.43498964e-002,
        6.77522139e-002, 3.43831914e-002, 2.49807166e-002,
        2.97000411e-003, 9.25336863e-004, 8.27198936e-066,
        6.39934407e-120],
       [3.63270671e-001, 2.66074115e+000, 1.54053522e-001,
        1.98401099e-002, 1.72605055e-002, 4.57824591e-002,
        6.68121228e-002, 7.12218435e-002, 7.20588320e-002,
        5.39293964e-002, 4.37207863e-002, 1.50032848e-002,
        5.50142499e-003, 1.24785346e-003, 3.47932243e-004,
        8.09878584e-039],
       [3.14138409e-003, 5.98082469e-001, 3.61278836e+000,
        1.30449697e-001, 1.14417919e-002, 3.69021128e-002,
        4.20500165e-002, 7.55968830e-002, 8.23580249e-002,
        6.78633725e-002, 4.57412198e-002, 2.54051680e-002,
        6.04772240e-003, 6.55072888e-004, 4.94517190e-025,
        1.82381314e-004],
       [2.19434713e-002, 2.98324941e-002, 1.30726454e+000,
        5.24501430e+000, 5.28026285e-002, 4.69596145e-002,
        5.90389348e-002, 9.42114582e-002, 8.29334744e-002,
        1.02895013e-001, 5.42551706e-002, 3.54820738e-002,
        7.34915025e-003, 9.84639125e-004, 6.22693312e-033,
        1.71034870e-070],
       [2.16663653e-002, 1.38010905e-002, 5.67035079e-003,
        5.40029534e-001, 2.69464785e-001, 3.03365672e-002,
        2.19635980e-002, 2.92397859e-002, 1.94095971e-002,
        2.48192076e-002, 1.29686004e-002, 9.88391752e-003,
        7.28632301e-004, 1.18595768e-003, 1.77226947e-004,
        1.21994575e-047],
       [2.17458324e-002, 6.38050010e-002, 2.07070396e-002,
        1.20087041e-001, 1.68034736e-001, 9.78269174e-002,
        1.84607673e-002, 2.65913293e-002, 3.29462497e-002,
        3.01385204e-002, 7.76171323e-003, 1.36409350e-002,
        4.43835783e-003, 2.26573805e-003, 4.63262978e-004,
        1.28633085e-003],
       [4.77060572e-002, 2.38582369e-001, 1.63300706e-001,
        1.17477828e-001, 3.30923729e-002, 5.77823457e-002,
        5.69127033e-002, 4.31367404e-002, 4.90838367e-002,
        2.67635936e-002, 2.14764280e-002, 3.50928414e-003,
        6.00976258e-003, 4.26817390e-004, 1.65544944e-048,
        3.11346410e-055],
       [8.98218250e-002, 1.58861968e-001, 1.20511956e-001,
        6.95366171e-002, 1.41115529e-002, 4.04507295e-002,
        6.14187550e-002, 5.03840337e-002, 5.87617177e-002,
        2.95443730e-002, 4.28505188e-003, 1.06587487e-002,
        6.72912892e-004, 2.03846095e-003, 1.84624439e-123,
        9.70340911e-067],
       [3.01849902e-002, 9.54399112e-002, 7.84621472e-002,
        3.59565796e-001, 7.06666436e-003, 2.11274704e-002,
        2.26179012e-002, 3.36507961e-002, 7.58076174e-002,
        2.84239468e-002, 3.09720886e-002, 9.22983171e-003,
        6.94088504e-003, 4.73648584e-004, 4.81821615e-068,
        2.40460145e-092],
       [2.66499956e-001, 2.26194501e-001, 1.51781381e-001,
        6.17122344e-001, 5.33126747e-003, 2.74696586e-002,
        5.79147573e-002, 5.52104165e-002, 5.41623989e-002,
        3.42769607e-002, 4.10620758e-002, 2.00634886e-002,
        4.35010769e-003, 1.69526397e-003, 6.21887475e-134,
        3.27678207e-072],
       [6.26755946e-002, 3.58353084e-001, 4.63913500e-001,
        5.61809692e-001, 5.26229932e-003, 1.37756592e-002,
        4.17970619e-002, 4.52485550e-002, 5.91416844e-002,
        8.48178099e-002, 4.19790388e-002, 2.37297546e-002,
        6.00203963e-003, 9.00193774e-024, 1.23971095e-117,
        5.64848975e-078],
       [2.07103466e-001, 3.70083454e-001, 3.31265675e-001,
        4.24998844e-001, 6.50167794e-003, 5.69264296e-002,
        2.39495641e-002, 4.38711382e-002, 5.78441612e-002,
        4.38868678e-002, 3.73371644e-002, 4.36045831e-002,
        1.20158130e-002, 1.16474176e-031, 7.83065367e-004,
        7.63259645e-004],
       [8.42567850e-002, 6.88044216e-002, 3.60442417e-002,
        1.87170622e-001, 1.32179933e-002, 1.75781804e-003,
        1.71725502e-002, 5.10001397e-002, 1.16139777e-002,
        1.90708888e-002, 1.46992625e-002, 7.80792914e-003,
        2.46030329e-002, 1.12661574e-002, 4.42610632e-067,
        2.12188467e-037],
       [1.99453541e-003, 2.43675934e-002, 9.53681745e-003,
        6.95052344e-032, 1.86082414e-003, 1.80785744e-003,
        1.18574024e-002, 5.27183163e-003, 5.36407216e-003,
        8.71709148e-003, 1.97387741e-003, 1.53287083e-002,
        8.23370985e-003, 1.73824369e-002, 1.11245464e-002,
        3.45708738e-126],
       [1.28646771e-028, 5.11770199e-026, 1.93321230e-040,
        7.62335377e-003, 2.63680307e-022, 1.69802570e-024,
        1.25954597e-026, 7.62920421e-003, 7.85972403e-003,
        2.11846634e-002, 3.52611724e-002, 2.14663265e-002,
        7.74496063e-003, 8.01374728e-003, 7.91287134e-003,
        2.13826227e-002],
       [2.82542155e-094, 2.11541534e-002, 8.47619112e-042,
        2.12997110e-002, 4.89950480e-036, 7.59444032e-003,
        9.77739902e-069, 2.23289069e-060, 1.43852807e-048,
        8.57011074e-060, 4.69957119e-042, 1.59988088e-046,
        2.21103428e-083, 8.85901381e-107, 1.02043012e-080,
        6.61414849e-113]])

    CO = np.array([[0.5862964 , 0.27482419, 0.14202343, 0.10966706, 0.17463361,
        0.24291186, 0.30887191, 0.30576237, 0.21792972, 0.17194928,
        0.19781965, 0.18652441, 0.14721075, 0.16829656, 0.08647878,
        0.03826193],
       [0.29924152, 1.22048785, 0.45170778, 0.13097844, 0.10449013,
        0.1888749 , 0.25371678, 0.32513292, 0.289497  , 0.15493995,
        0.11809325, 0.13819951, 0.15259649, 0.15129144, 0.05877925,
        0.04081756],
       [0.09736479, 0.56051279, 2.0022342 , 0.32714402, 0.2038218 ,
        0.15699831, 0.19940327, 0.27506544, 0.33434079, 0.23660117,
        0.14795241, 0.10390531, 0.08669109, 0.11914121, 0.06929465,
        0.05607717],
       [0.05557027, 0.18942051, 0.89008129, 2.79032014, 0.63631201,
        0.27107461, 0.18888735, 0.26089617, 0.28123418, 0.2786164 ,
        0.11972777, 0.06714731, 0.05851177, 0.07077637, 0.03457971,
        0.02159157],
       [0.07431183, 0.10249112, 0.15431304, 1.03875348, 1.5823408 ,
        0.56970619, 0.37144838, 0.28463339, 0.23299556, 0.30268274,
        0.17497104, 0.13087088, 0.06259543, 0.06391043, 0.05733981,
        0.04248352],
       [0.1169951 , 0.06585721, 0.05987489, 0.2446788 , 0.64388878,
        0.74838884, 0.4682144 , 0.36168816, 0.25455666, 0.25673543,
        0.20657193, 0.1250934 , 0.05858997, 0.05802437, 0.02978872,
        0.01412266],
       [0.13003319, 0.10007887, 0.15352047, 0.12792262, 0.31904242,
        0.42722038, 0.58953426, 0.48556934, 0.32163338, 0.27374264,
        0.26429447, 0.21543109, 0.11722192, 0.11369636, 0.04695745,
        0.0410405 ],
       [0.13613061, 0.17375414, 0.13217165, 0.10226562, 0.22947549,
        0.36684554, 0.50514208, 0.70997116, 0.52107242, 0.37254363,
        0.24173567, 0.21637922, 0.22090089, 0.20557608, 0.09933798,
        0.03697554],
       [0.09808301, 0.1378244 , 0.21806095, 0.14637761, 0.24750771,
        0.27069162, 0.41108544, 0.51244415, 0.58504481, 0.39688317,
        0.25921963, 0.13771259, 0.15506997, 0.14653379, 0.09144671,
        0.0363175 ],
       [0.03301033, 0.05876122, 0.07730101, 0.1540645 , 0.19949017,
        0.22555139, 0.30434291, 0.39220172, 0.39607117, 0.45222004,
        0.33496413, 0.17320423, 0.14643281, 0.12907193, 0.09436398,
        0.06853915],
       [0.04997746, 0.10157639, 0.11644792, 0.18089763, 0.31120643,
        0.35981781, 0.29013245, 0.32466466, 0.38670468, 0.48121251,
        0.35982612, 0.32199942, 0.23261191, 0.17105699, 0.09248333,
        0.05864771],
       [0.07684171, 0.07404558, 0.08044267, 0.09976927, 0.23535947,
        0.36134833, 0.4120064 , 0.39436665, 0.40571053, 0.3185212 ,
        0.44493945, 0.4692047 , 0.36112404, 0.27136086, 0.12337203,
        0.06301401],
       [0.06139331, 0.07270379, 0.06318731, 0.10490904, 0.20828878,
        0.29195354, 0.33152885, 0.45840078, 0.41608745, 0.37534567,
        0.32333062, 0.45260267, 0.46791424, 0.45930133, 0.26425019,
        0.10430745],
       [0.07856391, 0.11248771, 0.07218329, 0.07941113, 0.21837222,
        0.31065954, 0.43269098, 0.44734815, 0.4482134 , 0.39701258,
        0.41764854, 0.51083916, 0.57586506, 0.67537108, 0.29353034,
        0.1647474 ],
       [0.02414565, 0.05543934, 0.08065436, 0.15729702, 0.13739796,
        0.1824931 , 0.20680797, 0.33207548, 0.40926982, 0.34364191,
        0.26751219, 0.30176925, 0.59033401, 0.71617531, 0.54150246,
        0.18569502],
       [0.03504313, 0.03861315, 0.05758884, 0.03324823, 0.06854574,
        0.08336897, 0.17416575, 0.1612725 , 0.16193714, 0.22372421,
        0.15531204, 0.16929712, 0.19045027, 0.34179769, 0.24420501,
        0.15584606]])
    return CH, CW, CS, CO





def France():
    CH = np.array([[6.88060587e-01, 4.77680362e-01, 1.99420164e-01, 8.04619856e-02,
        1.34592088e-01, 3.79075398e-01, 7.02364688e-01, 5.71826992e-01,
        1.98564546e-01, 6.62109962e-02, 5.26756718e-02, 3.68089446e-02,
        2.23902294e-02, 6.49430329e-03, 2.99304586e-03, 5.21950981e-03],
       [3.25931872e-01, 1.00963395e+00, 3.85943905e-01, 1.26837578e-01,
        3.75846591e-02, 1.45180962e-01, 5.73297142e-01, 7.14297626e-01,
        4.68638472e-01, 1.29077697e-01, 3.71703284e-02, 2.20074543e-02,
        1.19993333e-02, 6.39165808e-03, 2.60271248e-03, 3.31203274e-03],
       [1.28181133e-01, 3.69161263e-01, 1.58850747e+00, 4.21810846e-01,
        5.99276240e-02, 2.60512161e-02, 1.26748270e-01, 4.57233231e-01,
        6.37619645e-01, 2.72502343e-01, 6.39786966e-02, 1.41021480e-02,
        8.75908548e-03, 8.30009913e-03, 6.66129368e-03, 3.33535900e-03],
       [4.71730629e-02, 9.61422092e-02, 3.92455487e-01, 1.25411521e+00,
        2.06093080e-01, 3.24571801e-02, 1.70906768e-02, 1.68148986e-01,
        4.01699129e-01, 4.95991383e-01, 1.89243533e-01, 4.20548363e-02,
        9.50560312e-03, 7.75421572e-03, 4.99003206e-03, 2.22055484e-03],
       [9.64047581e-02, 4.75082114e-02, 7.20785657e-02, 3.90330520e-01,
        1.19140212e+00, 1.97241893e-01, 4.02695612e-02, 1.85899235e-02,
        1.17882495e-01, 4.87592593e-01, 2.71308105e-01, 1.12326340e-01,
        1.45866406e-02, 2.32177438e-03, 2.40011911e-03, 2.52892806e-03],
       [3.66725662e-01, 9.67866107e-02, 2.55643695e-02, 5.51078246e-02,
        2.07188836e-01, 1.01466577e+00, 1.95743059e-01, 2.88185796e-02,
        1.39560956e-02, 5.15897290e-02, 2.09116618e-01, 9.84030072e-02,
        3.70972939e-02, 7.79300363e-03, 8.56044838e-04, 4.34208939e-03],
       [5.19927950e-01, 4.75618511e-01, 1.63414315e-01, 2.35194851e-02,
        4.66565134e-02, 1.90639682e-01, 9.32505578e-01, 1.99321909e-01,
        5.33710990e-02, 1.55237482e-02, 3.02134925e-02, 5.12411346e-02,
        5.01432647e-02, 7.60085236e-03, 3.31291058e-03, 3.02359605e-03],
       [4.47115031e-01, 7.11948111e-01, 5.52991698e-01, 1.93905635e-01,
        2.31569277e-02, 3.09937987e-02, 1.63765972e-01, 1.04259109e+00,
        1.64946862e-01, 3.81400708e-02, 1.90079054e-02, 1.44560392e-02,
        2.75448795e-02, 1.24621037e-02, 6.89150950e-03, 2.37544923e-03],
       [1.69284893e-01, 4.11680040e-01, 6.61423969e-01, 4.41584147e-01,
        8.89382493e-02, 2.21940972e-02, 5.94875278e-02, 1.59125325e-01,
        8.67284832e-01, 1.44630558e-01, 2.83355489e-02, 4.29891891e-03,
        1.58378845e-02, 1.47179550e-02, 5.35116261e-03, 5.97482861e-03],
       [8.25101673e-02, 1.80596237e-01, 3.61976196e-01, 6.18390689e-01,
        3.61229601e-01, 5.18066634e-02, 2.15104530e-02, 6.46520113e-02,
        1.44426592e-01, 7.96166672e-01, 1.24661677e-01, 2.33003088e-02,
        7.60124809e-03, 7.21958411e-03, 5.59567654e-03, 8.88558826e-03],
       [1.25564042e-01, 1.03706129e-01, 2.01282171e-01, 3.38546311e-01,
        3.90068310e-01, 2.30417129e-01, 6.53289949e-02, 3.36418400e-02,
        6.02868185e-02, 1.66583251e-01, 7.47469716e-01, 1.32954385e-01,
        2.13980306e-02, 5.77012682e-03, 4.74480463e-03, 2.17550943e-02],
       [2.33811319e-01, 2.17834131e-01, 1.46831692e-01, 2.33025964e-01,
        2.70982777e-01, 2.90320743e-01, 2.13432500e-01, 6.05392468e-02,
        2.71714155e-02, 1.07779105e-01, 2.21375937e-01, 8.66687667e-01,
        1.25525236e-01, 3.42652074e-02, 4.82072516e-03, 1.81900885e-02],
       [2.61256717e-01, 2.33839803e-01, 1.63821059e-01, 1.35336211e-01,
        1.07687663e-01, 1.69805897e-01, 2.61426630e-01, 1.58203936e-01,
        7.53781402e-02, 3.34134864e-02, 8.66443527e-02, 1.93533392e-01,
        9.08575250e-01, 1.09922865e-01, 2.25611067e-02, 5.50004531e-03],
       [1.39879237e-01, 2.14594537e-01, 2.00491074e-01, 1.07826896e-01,
        6.57754134e-02, 8.21886116e-02, 1.44264815e-01, 1.49012663e-01,
        1.27642912e-01, 5.55982615e-02, 4.71190886e-02, 7.95385223e-02,
        1.26782105e-01, 6.91952303e-01, 9.02159470e-02, 1.19539490e-02],
       [8.15733379e-02, 2.42322382e-01, 2.34411929e-01, 2.13315115e-01,
        3.15537693e-02, 6.53068302e-02, 6.06260817e-02, 1.64654499e-01,
        1.41831746e-01, 1.30553160e-01, 6.29743148e-02, 3.69819096e-02,
        1.15141423e-01, 1.85317083e-01, 5.26009905e-01, 1.08658991e-01],
       [1.99006076e-01, 2.48851264e-01, 3.84293387e-01, 3.01760950e-01,
        7.38378322e-02, 7.44493293e-02, 9.49187219e-02, 1.90460562e-01,
        2.64034209e-01, 2.04618294e-01, 3.57400685e-01, 1.40653650e-01,
        5.51488148e-02, 1.03624533e-01, 1.10187565e-01, 4.29969780e-01]])

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
        0.00000000e+000, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [0.00000000e+000, 0.00000000e+000, 1.45656012e-002,
        6.93386603e-003, 1.00194792e-002, 2.89918063e-003,
        2.39203103e-002, 6.81920292e-003, 2.40093313e-002,
        1.39559489e-002, 5.31171349e-003, 5.65563543e-009,
        1.58340614e-018, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [0.00000000e+000, 0.00000000e+000, 1.23585931e-002,
        4.65661172e-001, 5.04971737e-001, 2.90427650e-001,
        2.73327506e-001, 2.54157709e-001, 2.84016661e-001,
        2.24051800e-001, 1.26467237e-001, 2.59436161e-002,
        1.40178790e-003, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [0.00000000e+000, 0.00000000e+000, 2.18885288e-002,
        3.40346605e-001, 8.91531656e-001, 8.54473590e-001,
        7.19952073e-001, 7.93497320e-001, 6.00959892e-001,
        4.81605778e-001, 3.53456500e-001, 7.02040773e-002,
        5.22444858e-003, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [0.00000000e+000, 0.00000000e+000, 2.81515774e-002,
        2.99170044e-001, 8.48488637e-001, 1.42892111e+000,
        1.03223469e+000, 1.00813180e+000, 9.34231177e-001,
        6.55397699e-001, 5.17834425e-001, 1.03426352e-001,
        6.96894905e-003, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [0.00000000e+000, 0.00000000e+000, 3.22010563e-002,
        1.63215102e-001, 5.93525658e-001, 9.78649041e-001,
        1.29291833e+000, 1.13423567e+000, 1.01961417e+000,
        8.32958383e-001, 4.82937202e-001, 1.21814617e-001,
        6.58759101e-003, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [0.00000000e+000, 0.00000000e+000, 1.98086342e-002,
        3.25650736e-001, 4.92504467e-001, 9.19500307e-001,
        9.94002194e-001, 1.37398240e+000, 1.34082458e+000,
        9.28444190e-001, 6.49786929e-001, 1.11774520e-001,
        4.67015731e-003, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [0.00000000e+000, 0.00000000e+000, 2.24533766e-002,
        2.04014086e-001, 5.81486960e-001, 9.12220693e-001,
        1.08620491e+000, 1.16279352e+000, 1.42898791e+000,
        1.14889640e+000, 7.93711421e-001, 1.19043084e-001,
        6.86908968e-003, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [0.00000000e+000, 0.00000000e+000, 2.99761568e-002,
        2.55170324e-001, 3.96557150e-001, 6.86162962e-001,
        8.88147820e-001, 9.76758179e-001, 1.01363589e+000,
        9.89896419e-001, 6.12885169e-001, 1.34820041e-001,
        5.47460000e-003, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [0.00000000e+000, 0.00000000e+000, 3.05704686e-002,
        1.80021281e-001, 3.11333899e-001, 6.52049201e-001,
        7.67315379e-001, 7.99288625e-001, 1.08567472e+000,
        1.04785758e+000, 7.65183391e-001, 1.60481185e-001,
        6.17114670e-003, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [0.00000000e+000, 0.00000000e+000, 2.09753607e-002,
        5.38693822e-002, 9.34120097e-002, 1.71829428e-001,
        2.41185616e-001, 2.27484445e-001, 2.91715328e-001,
        2.30261345e-001, 2.05194546e-001, 5.55069238e-002,
        2.59228852e-003, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [0.00000000e+000, 0.00000000e+000, 1.68879395e-003,
        1.74214561e-003, 8.85508348e-003, 1.57840519e-002,
        1.75531205e-002, 2.07512209e-002, 2.25070126e-002,
        2.19905944e-002, 1.61500567e-002, 5.38925225e-003,
        2.12072051e-004, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

    CS = np.array([[2.42331707e+000, 3.16794620e-001, 5.66157261e-002,
        6.18728423e-002, 1.65377420e-002, 8.64183659e-002,
        1.59975442e-001, 1.31679834e-001, 6.23852833e-002,
        7.38770969e-002, 3.94131905e-002, 2.68636549e-002,
        2.37904441e-003, 8.23077871e-004, 8.26941397e-066,
        6.40343617e-120],
       [3.96403062e-001, 2.69340407e+000, 1.49117638e-001,
        1.66058016e-002, 1.66903575e-002, 5.52418607e-002,
        7.41191853e-002, 7.87303110e-002, 7.74396882e-002,
        5.50063627e-002, 4.69898006e-002, 1.51585532e-002,
        4.19739257e-003, 1.09609022e-003, 3.47689685e-004,
        8.10159666e-039],
       [3.27843830e-003, 5.78919868e-001, 3.34391704e+000,
        1.04310475e-001, 1.06801603e-002, 4.28958515e-002,
        4.49075633e-002, 8.03409900e-002, 8.50675948e-002,
        6.65064879e-002, 4.72894263e-002, 2.47266017e-002,
        4.49042680e-003, 5.72045373e-004, 4.94061247e-025,
        1.82411083e-004],
       [1.98419361e-002, 2.49692405e-002, 1.04531776e+000,
        3.60597590e+000, 4.36912195e-002, 4.85541519e-002,
        5.57614223e-002, 8.81695987e-002, 7.52955880e-002,
        8.83944883e-002, 4.94279480e-002, 3.04906018e-002,
        4.86925697e-003, 8.25136101e-004, 6.21502934e-033,
        1.70951004e-070],
       [2.22288101e-002, 1.33452137e-002, 5.29289957e-003,
        4.46844212e-001, 2.57036089e-001, 3.32548007e-002,
        2.28048564e-002, 3.02915702e-002, 1.97236577e-002,
        2.43247679e-002, 1.31837540e-002, 9.66006038e-003,
        6.22289200e-004, 1.12809769e-003, 1.77168530e-004,
        1.21999611e-047],
       [2.78864714e-002, 7.69881533e-002, 2.40703317e-002,
        1.24164657e-001, 1.84198879e-001, 1.26951662e-001,
        2.26790438e-002, 3.29152308e-002, 4.00189226e-002,
        3.52837678e-002, 9.37089086e-003, 1.55961744e-002,
        4.08056434e-003, 2.16587613e-003, 4.63332571e-004,
        1.28697912e-003],
       [5.63556970e-002, 2.64675482e-001, 1.74397952e-001,
        1.10956114e-001, 3.43598901e-002, 7.09855839e-002,
        6.59250545e-002, 5.01375263e-002, 5.58997370e-002,
        2.93147349e-002, 2.43740875e-002, 3.78284679e-003,
        5.29421586e-003, 4.04359065e-004, 1.65534157e-048,
        3.11455745e-055],
       [1.05979229e-001, 1.75609778e-001, 1.28074723e-001,
        6.50771758e-002, 1.46191596e-002, 5.00706484e-002,
        7.13865819e-002, 5.87135219e-002, 6.70069836e-002,
        3.23271905e-002, 4.87124942e-003, 1.14827287e-002,
        5.83929323e-004, 1.91608697e-003, 1.84606921e-123,
        9.70714150e-067],
       [3.46477048e-002, 1.02566705e-001, 8.10435432e-002,
        3.26451029e-001, 7.18100781e-003, 2.56629695e-002,
        2.57586777e-002, 3.83725737e-002, 8.45450666e-002,
        3.03941118e-002, 3.44634202e-002, 9.74029566e-003,
        5.92529539e-003, 4.43446730e-004, 4.81733969e-068,
        2.40538618e-092],
       [2.90591878e-001, 2.30711589e-001, 1.48746610e-001,
        5.30154107e-001, 5.22505978e-003, 3.21592780e-002,
        6.34352690e-002, 6.04107473e-002, 5.79165877e-002,
        3.51046826e-002, 4.38713690e-002, 2.03655107e-002,
        3.60818550e-003, 1.57703946e-003, 6.21672191e-134,
        3.27746145e-072],
       [7.18445568e-002, 3.85147234e-001, 4.79615616e-001,
        5.11824033e-001, 5.34960270e-003, 1.66316630e-002,
        4.74364379e-002, 5.14385831e-002, 6.58084362e-002,
        9.06206850e-002, 4.65978803e-002, 2.50210817e-002,
        5.17176972e-003, 8.47242727e-024, 1.23950943e-117,
        5.65020106e-078],
       [2.22714029e-001, 3.73913433e-001, 3.22417644e-001,
        3.65211757e-001, 6.35442388e-003, 6.50860459e-002,
        2.58165279e-002, 4.72626186e-002, 6.10432833e-002,
        4.45475109e-002, 3.93689802e-002, 4.39290584e-002,
        1.02325598e-002, 1.10036552e-031, 7.82841479e-004,
        7.63368480e-004],
       [6.74917023e-002, 5.24953388e-002, 2.67628072e-002,
        1.24011868e-001, 1.12888414e-002, 1.61611341e-003,
        1.51279167e-002, 4.42560655e-002, 9.91462161e-003,
        1.58182991e-002, 1.26658945e-002, 6.64916322e-003,
        2.07072557e-002, 1.08519765e-002, 4.42375018e-067,
        2.12131051e-037],
       [1.77411927e-003, 2.14040203e-002, 8.32806912e-003,
        5.82459874e-032, 1.77003907e-003, 1.72817650e-003,
        1.12334882e-002, 4.95535020e-003, 5.02203604e-003,
        8.10917799e-003, 1.85777033e-003, 1.44814779e-002,
        7.93101161e-003, 1.72779281e-002, 1.11232494e-002,
        3.45682243e-126],
       [1.28606719e-028, 5.11413421e-026, 1.93142988e-040,
        7.60878051e-003, 2.63593394e-022, 1.69828078e-024,
        1.25946390e-026, 7.62848033e-003, 7.85829430e-003,
        2.11773297e-002, 3.52554408e-002, 2.14601890e-002,
        7.74083776e-003, 8.01281292e-003, 7.91286335e-003,
        2.13826186e-002],
       [2.82722828e-094, 2.11614954e-002, 8.47757464e-042,
        2.12892667e-002, 4.89970706e-036, 7.59826765e-003,
        9.78083252e-069, 2.23374957e-060, 1.43899752e-048,
        8.57188759e-060, 4.70099501e-042, 1.60010901e-046,
        2.21043599e-083, 8.85833486e-107, 1.02042993e-080,
        6.61415091e-113]])

    CO = np.array([[0.69354314, 0.31197129, 0.15587858, 0.11303449, 0.19062315,
        0.28771628, 0.34670863, 0.33848915, 0.23463304, 0.17037128,
        0.21496553, 0.20853794, 0.16156827, 0.12072782, 0.08479718,
        0.04864456],
       [0.33968903, 1.32952509, 0.47575945, 0.12955018, 0.10945271,
        0.21468087, 0.2732995 , 0.34540217, 0.2991026 , 0.14732038,
        0.12314814, 0.14827206, 0.160718  , 0.10414775, 0.05530945,
        0.04979868],
       [0.10686325, 0.5903579 , 2.03897308, 0.31285561, 0.20642802,
        0.17253645, 0.20767713, 0.28253153, 0.33398907, 0.21751193,
        0.14917348, 0.10778476, 0.08827977, 0.07929837, 0.06304371,
        0.06614906],
       [0.05727661, 0.18735496, 0.85120592, 2.50592081, 0.60519659,
        0.27975836, 0.18474285, 0.2516558 , 0.26382705, 0.24053663,
        0.11336339, 0.06541187, 0.05595493, 0.04423835, 0.02954417,
        0.02391829],
       [0.08111586, 0.10735876, 0.1562862 , 0.98795882, 1.59382055,
        0.62267039, 0.38474791, 0.29076232, 0.23147917, 0.27674211,
        0.17545153, 0.13501562, 0.06339438, 0.04230536, 0.05188236,
        0.04984016],
       [0.13857452, 0.07485528, 0.06580071, 0.25251697, 0.70374956,
        0.88756968, 0.52624799, 0.40091698, 0.27442054, 0.25470723,
        0.22476576, 0.14003714, 0.06438716, 0.04167753, 0.02924712,
        0.01797808],
       [0.14596222, 0.1078033 , 0.15989052, 0.12511579, 0.33046559,
        0.4801729 , 0.62795029, 0.51008429, 0.32859716, 0.25737623,
        0.27253197, 0.22855343, 0.12208305, 0.07739424, 0.04369245,
        0.04951194],
       [0.15070113, 0.18458622, 0.13575918, 0.0986436 , 0.23441672,
        0.40663374, 0.5306452 , 0.73553928, 0.52501928, 0.34544397,
        0.24583549, 0.22639634, 0.22689162, 0.13800955, 0.09115734,
        0.04399329],
       [0.10560063, 0.14239745, 0.21783155, 0.1373175 , 0.24589687,
        0.29181456, 0.41998597, 0.51632565, 0.57329518, 0.35791109,
        0.25637976, 0.1401327 , 0.15490331, 0.09567233, 0.08161244,
        0.04202423],
       [0.03270739, 0.05587149, 0.07106428, 0.1330078 , 0.18239338,
        0.22376955, 0.28614699, 0.36367209, 0.35717882, 0.37530607,
        0.30488598, 0.16219881, 0.13461539, 0.07755391, 0.07750288,
        0.07298705],
       [0.05430922, 0.1059243 , 0.11740898, 0.17128163, 0.31206102,
        0.39150877, 0.29917526, 0.33017095, 0.38246815, 0.43800197,
        0.35919908, 0.33071022, 0.23452634, 0.11272392, 0.08330641,
        0.06849542],
       [0.08591054, 0.07944233, 0.0834461 , 0.09719071, 0.24281339,
        0.40451526, 0.43710255, 0.41262357, 0.41284034, 0.29828231,
        0.45697606, 0.49579815, 0.37459855, 0.18398079, 0.11433566,
        0.07571774],
       [0.06738102, 0.07657324, 0.06434526, 0.10032473, 0.2109473 ,
        0.3208409 , 0.34527716, 0.47083241, 0.41564026, 0.34505451,
        0.32599168, 0.46949049, 0.47647826, 0.30569625, 0.24040719,
        0.12303905],
       [0.05635795, 0.07743559, 0.04804398, 0.04963546, 0.14455099,
        0.2231394 , 0.29453705, 0.30031859, 0.29263981, 0.2385482 ,
        0.27522395, 0.34634541, 0.38327733, 0.29379908, 0.17454231,
        0.12701699],
       [0.02367613, 0.0521667 , 0.07337868, 0.13439121, 0.12432078,
        0.17917514, 0.19242838, 0.30472854, 0.36525657, 0.28223947,
        0.24096755, 0.2796662 , 0.53706883, 0.42586021, 0.44013702,
        0.19569658],
       [0.04455231, 0.04710924, 0.06793224, 0.03683108, 0.08041544,
        0.10612831, 0.21011648, 0.19188108, 0.18738305, 0.23824298,
        0.18139096, 0.20342769, 0.22465147, 0.26351926, 0.25735793,
        0.21294837]])
    return CH, CW, CS, CO


def Germany():
    """
    Returns CH, CW, CS, CO of the country supplied
    """
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
    CH = np.array([[0.33616363, 0.79154253, 0.40965338, 0.28552183, 0.64323047,
        0.81023917, 0.53329121, 0.32656827, 0.15871547, 0.10282808,
        0.16544223, 0.22014068, 0.11302775, 0.05668324, 0.01602993,
        0.01054193],
       [0.51911959, 0.54389291, 0.71924201, 0.32253968, 0.17492295,
        0.62741274, 0.69935446, 0.48418804, 0.256076  , 0.11226015,
        0.09849883, 0.1700854 , 0.10687316, 0.05675574, 0.02044932,
        0.01076183],
       [0.2705121 , 0.74216994, 0.71039605, 0.56833221, 0.15917324,
        0.18666648, 0.3781164 , 0.49201769, 0.31046558, 0.13833673,
        0.07472379, 0.0626091 , 0.04364539, 0.0474856 , 0.02390485,
        0.00912792],
       [0.18645694, 0.29230237, 0.61409485, 0.70474978, 0.43102035,
        0.18118549, 0.11636666, 0.3720618 , 0.35841148, 0.25282504,
        0.17381598, 0.09702503, 0.04840177, 0.03350817, 0.01440578,
        0.00758424],
       [0.41010187, 0.19230461, 0.16903529, 0.61991946, 0.55330572,
        0.54877756, 0.13561749, 0.0798248 , 0.1775226 , 0.23476352,
        0.19711166, 0.15725248, 0.04288627, 0.01737283, 0.00938088,
        0.006357  ],
       [0.75440343, 0.37414162, 0.14063975, 0.23140621, 0.57539694,
        0.42687489, 0.34331532, 0.10111066, 0.04609492, 0.10495471,
        0.21318755, 0.16224761, 0.08184572, 0.03164476, 0.00511587,
        0.00658889],
       [0.55690224, 0.86439075, 0.57345255, 0.14044153, 0.2022419 ,
        0.488235  , 0.23328058, 0.31915981, 0.12154288, 0.04421105,
        0.09216743, 0.13386112, 0.11355351, 0.03344366, 0.01533681,
        0.0060369 ],
       [0.37068582, 0.74224317, 0.81003902, 0.45687268, 0.11037846,
        0.13252834, 0.27385295, 0.1984528 , 0.19124468, 0.06779348,
        0.0480126 , 0.0659401 , 0.09116515, 0.05979406, 0.02366743,
        0.00656628],
       [0.21917998, 0.43739173, 0.64156927, 0.63987319, 0.25235635,
        0.09441197, 0.16518783, 0.25143314, 0.11420688, 0.15497062,
        0.0649435 , 0.02509708, 0.05618279, 0.06331009, 0.02163332,
        0.0115158 ],
       [0.14284026, 0.21801673, 0.33890704, 0.59338747, 0.4338401 ,
        0.17114002, 0.0609345 , 0.14314534, 0.16615767, 0.09874375,
        0.11357915, 0.04927389, 0.02027794, 0.01945624, 0.01560716,
        0.01635286],
       [0.35958128, 0.2053999 , 0.27097746, 0.44451328, 0.58056188,
        0.44550247, 0.20917645, 0.08377493, 0.118336  , 0.12885521,
        0.08840295, 0.22013839, 0.04402996, 0.01407709, 0.01334366,
        0.01658442],
       [0.62905695, 0.47865528, 0.25075861, 0.40758738, 0.4969681 ,
        0.65108193, 0.47264335, 0.17834082, 0.0825824 , 0.14423322,
        0.28018666, 0.08842445, 0.12474771, 0.04961502, 0.00882651,
        0.01220513],
       [0.55484292, 0.43224578, 0.24103243, 0.24415045, 0.23534217,
        0.36723661, 0.43835683, 0.32922762, 0.13771196, 0.04899618,
        0.1118228 , 0.16302094, 0.04376124, 0.08427011, 0.02021926,
        0.00281394],
       [0.35166331, 0.45306846, 0.36428776, 0.21631126, 0.21186784,
        0.2565869 , 0.43950865, 0.48326209, 0.3363031 , 0.08745247,
        0.07559162, 0.11325964, 0.11206124, 0.036252  , 0.04275999,
        0.0092892 ],
       [0.1354029 , 0.34064077, 0.26933119, 0.25344292, 0.06664782,
        0.17620119, 0.15016419, 0.32662886, 0.26665802, 0.16784271,
        0.09687113, 0.05381806, 0.08594313, 0.06381126, 0.01707631,
        0.03629825],
       [0.22013997, 0.28037159, 0.41153242, 0.41495329, 0.13706238,
        0.12615806, 0.17539846, 0.3821743 , 0.34732236, 0.26841461,
        0.24477724, 0.10866985, 0.03683059, 0.08899639, 0.06074283,
        0.01474225]])

    CW = np.array([[1.95422817e-055, 7.41487361e-080, 3.51348387e-058,
        2.07681587e-106, 1.91129849e-059, 4.90845732e-003,
        1.81194115e-007, 5.55435897e-003, 5.24421256e-003,
        9.30492679e-003, 6.66142588e-003, 5.38874516e-019,
        1.32622812e-003, 8.20604524e-092, 1.20585150e-005,
        3.16436834e-125],
       [6.24464834e-116, 3.14139232e-004, 2.95159517e-072,
        1.55767813e-059, 1.18514314e-006, 4.19297487e-003,
        1.12032517e-002, 2.55994508e-003, 1.00479115e-002,
        2.78164015e-003, 7.77241302e-004, 6.90074947e-004,
        9.90358102e-057, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [1.49703242e-023, 7.64336280e-004, 3.72245511e-002,
        9.37232338e-003, 1.18326066e-002, 3.51926366e-003,
        2.97232588e-002, 8.68235366e-003, 3.07486966e-002,
        1.76039967e-002, 7.24686809e-003, 1.80826439e-008,
        3.18062786e-017, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [2.30018036e-003, 3.77286970e-003, 1.67047835e-002,
        3.32899091e-001, 3.15408297e-001, 1.86459817e-001,
        1.79632004e-001, 1.71150396e-001, 1.92380580e-001,
        1.49475877e-001, 9.12566468e-002, 4.38714306e-002,
        1.48927056e-002, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [5.82579635e-004, 3.71228379e-003, 2.58494823e-002,
        2.12582478e-001, 4.86526242e-001, 4.79302172e-001,
        4.13397097e-001, 4.66856646e-001, 3.55652904e-001,
        2.80722886e-001, 2.22836285e-001, 1.03723481e-001,
        4.84948049e-002, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [1.77043854e-003, 4.37627250e-003, 3.41726977e-002,
        1.92072593e-001, 4.75945016e-001, 8.23872974e-001,
        6.09233166e-001, 6.09672559e-001, 5.68298460e-001,
        3.92674135e-001, 3.35569211e-001, 1.57067816e-001,
        6.64910664e-002, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [1.18422202e-003, 7.17974469e-003, 4.00128726e-002,
        1.07265662e-001, 3.40802942e-001, 5.77606488e-001,
        7.81140695e-001, 7.02159403e-001, 6.34908393e-001,
        5.10862201e-001, 3.20357528e-001, 1.89368825e-001,
        6.43392096e-002, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [1.46039727e-003, 5.04960704e-003, 2.52207728e-002,
        2.19293968e-001, 2.89766553e-001, 5.56072236e-001,
        6.15346535e-001, 8.71541020e-001, 8.55502756e-001,
        5.83459229e-001, 4.41661444e-001, 1.78043507e-001,
        4.67363559e-002, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [1.84533197e-003, 2.87120359e-003, 2.87559890e-002,
        1.38190302e-001, 3.44128666e-001, 5.54909350e-001,
        6.76374097e-001, 7.41911413e-001, 9.17108659e-001,
        7.26237000e-001, 5.42655160e-001, 1.90734952e-001,
        6.91457216e-002, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [1.73558070e-003, 1.61235177e-003, 3.78118442e-002,
        1.70236561e-001, 2.31148946e-001, 4.11106795e-001,
        5.44710467e-001, 6.13821036e-001, 6.40736520e-001,
        6.16300493e-001, 4.12710653e-001, 2.12757967e-001,
        5.42779930e-002, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [7.40697289e-009, 2.30518613e-004, 4.17078506e-002,
        1.29900351e-001, 1.96280135e-001, 4.22543627e-001,
        5.09000460e-001, 5.43278038e-001, 7.42268500e-001,
        7.05616661e-001, 5.57308809e-001, 2.73917201e-001,
        6.61760839e-002, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [8.01109891e-004, 2.73821012e-003, 6.70640785e-002,
        9.10947360e-002, 1.38012195e-001, 2.60947741e-001,
        3.74938886e-001, 3.62355646e-001, 4.67396402e-001,
        3.63372798e-001, 3.50236171e-001, 2.22027695e-001,
        6.51453375e-002, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [1.11663128e-004, 4.90839755e-004, 3.39232300e-002,
        1.85086928e-002, 8.21953819e-002, 1.50596372e-001,
        1.71436553e-001, 2.07666762e-001, 2.26560389e-001,
        2.18026035e-001, 1.73184589e-001, 1.35434252e-001,
        3.34829296e-002, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

    CS = np.array([[3.21888548e-001, 4.34609554e-002, 7.89876562e-003,
        8.11032636e-003, 5.35680688e-003, 2.18490676e-002,
        4.02081492e-002, 2.99769390e-002, 1.40860630e-002,
        1.66788696e-002, 9.48667149e-003, 7.41610671e-003,
        1.28262715e-003, 7.79394394e-004, 8.23608342e-066,
        6.37926455e-120],
       [5.43824128e-002, 6.89315695e+000, 2.92214854e-001,
        3.31489243e-002, 3.53872899e-002, 9.79485694e-002,
        1.21411727e-001, 1.11298475e-001, 1.02531145e-001,
        6.61569520e-002, 4.97775866e-002, 1.45551725e-002,
        5.62013699e-003, 1.71181436e-003, 3.47399313e-004,
        8.08903385e-039],
       [4.57392628e-004, 1.13446664e+000, 5.04874334e+000,
        1.60151586e-001, 1.78667549e-002, 5.99114054e-002,
        5.81648465e-002, 9.00957839e-002, 8.96432444e-002,
        6.40388951e-002, 4.06312821e-002, 1.96296726e-002,
        5.12310056e-003, 8.01300681e-004, 4.93286865e-025,
        1.82040297e-004],
       [2.60089195e-003, 4.98442343e-002, 1.60491357e+000,
        5.62986572e+000, 7.46788631e-002, 6.90013358e-002,
        7.33952716e-002, 1.00225369e-001, 8.03384924e-002,
        8.60260793e-002, 4.27973477e-002, 2.43102950e-002,
        5.62715340e-003, 1.18550367e-003, 6.20477573e-033,
        1.70581454e-070],
       [7.20022378e-003, 2.82948370e-002, 8.85444942e-003,
        7.63764850e-001, 4.15355536e-001, 4.84327009e-002,
        3.13089497e-002, 3.75518201e-002, 2.33182612e-002,
        2.68499928e-002, 1.33342408e-002, 9.19873622e-003,
        7.09979653e-004, 1.33114715e-003, 1.77110499e-004,
        1.21933000e-047],
       [7.05050823e-003, 1.36506616e-001, 3.36183419e-002,
        1.76453030e-001, 2.68269514e-001, 1.61946671e-001,
        2.72118285e-002, 3.52293776e-002, 4.07595448e-002,
        3.34779318e-002, 8.16686264e-003, 1.29218452e-002,
        4.38281447e-003, 2.57232771e-003, 4.62967617e-004,
        1.28569645e-003],
       [1.41644133e-002, 4.33554512e-001, 2.25882443e-001,
        1.46044591e-001, 4.71729377e-002, 8.51732355e-002,
        7.45285064e-002, 5.06084571e-002, 5.37693994e-002,
        2.63330370e-002, 2.02187069e-002, 3.00471113e-003,
        5.51878238e-003, 4.73000510e-004, 1.65387017e-048,
        3.11122532e-055],
       [2.41261913e-002, 2.48253822e-001, 1.43625223e-001,
        7.39754297e-002, 1.81230635e-002, 5.35909284e-002,
        7.20571003e-002, 5.27480158e-002, 5.74038642e-002,
        2.59067652e-002, 3.62695850e-003, 8.27247173e-003,
        5.75451200e-004, 2.21240662e-003, 1.84385311e-123,
        9.69405571e-067],
       [7.82315519e-003, 1.35799638e-001, 8.54027455e-002,
        3.48315010e-001, 8.48973445e-003, 2.61379090e-002,
        2.47770151e-002, 3.28732006e-002, 6.91446331e-002,
        2.33000414e-002, 2.46539708e-002, 6.78178676e-003,
        5.70418367e-003, 5.06429666e-004, 4.81115830e-068,
        2.40199342e-092],
       [6.56055020e-002, 2.77480182e-001, 1.43227658e-001,
        5.15949356e-001, 5.76748845e-003, 3.05133545e-002,
        5.69830596e-002, 4.84127145e-002, 4.43986946e-002,
        2.53119567e-002, 2.97297420e-002, 1.35550521e-002,
        3.36403692e-003, 1.77106423e-003, 6.20816302e-134,
        3.27262761e-072],
       [1.72928327e-002, 4.07997046e-001, 4.12087837e-001,
        4.43164485e-001, 5.41066607e-003, 1.44947273e-002,
        3.93493062e-002, 3.82993335e-002, 4.70771403e-002,
        6.14097450e-002, 3.00598746e-002, 1.60820884e-002,
        4.66829371e-003, 9.25930485e-024, 1.23781523e-117,
        5.64214619e-078],
       [6.14834808e-002, 3.59029945e-001, 2.55957242e-001,
        2.91184989e-001, 6.05096312e-003, 5.39255198e-002,
        2.05060403e-002, 3.40492828e-002, 4.25020497e-002,
        2.96503162e-002, 2.53040787e-002, 2.85102055e-002,
        9.14427219e-003, 1.17200939e-031, 7.81904745e-004,
        7.62432253e-004],
       [3.63871686e-002, 7.02891117e-002, 3.05335236e-002,
        1.43314229e-001, 1.28796189e-002, 1.73582001e-003,
        1.57696026e-002, 4.36135077e-002, 9.54464191e-003,
        1.47479508e-002, 1.14328593e-002, 5.94198905e-003,
        2.08739457e-002, 1.13502740e-002, 4.42254995e-067,
        2.12063633e-037],
       [1.67996086e-003, 3.34276401e-002, 1.16656611e-002,
        8.36841728e-032, 2.08863335e-003, 2.05248872e-003,
        1.31404144e-002, 5.72168684e-003, 5.73531805e-003,
        9.10685840e-003, 2.03031094e-003, 1.54243547e-002,
        8.29518524e-003, 1.76790777e-002, 1.11241379e-002,
        3.45698345e-126],
       [1.28088359e-028, 5.10986316e-026, 1.92840260e-040,
        7.59622748e-003, 2.63507054e-022, 1.69694309e-024,
        1.25834438e-026, 7.61932277e-003, 7.84821090e-003,
        2.11481737e-002, 3.52072524e-002, 2.14345101e-002,
        7.73873755e-003, 8.01345303e-003, 7.91284789e-003,
        2.13825765e-002],
       [2.81655609e-094, 2.11286811e-002, 8.46034233e-042,
        2.12432451e-002, 4.89703184e-036, 7.59069485e-003,
        9.77036844e-069, 2.23073834e-060, 1.43696784e-048,
        8.55924513e-060, 4.69429331e-042, 1.59814657e-046,
        2.20973349e-083, 8.85874749e-107, 1.02042792e-080,
        6.61413811e-113]])

    CO = np.array([[1.69499683, 0.75693289, 0.3989942 , 0.29686159, 0.46864971,
        0.6195553 , 0.66127614, 0.52277579, 0.32669025, 0.2097554 ,
        0.22951757, 0.18040128, 0.11033282, 0.09386833, 0.05533687,
        0.02343726],
       [0.82418418, 3.20248155, 1.20896831, 0.33777568, 0.26714469,
        0.45894046, 0.51749289, 0.52959408, 0.41344196, 0.18006402,
        0.13053361, 0.12733893, 0.10895835, 0.08039128, 0.03583273,
        0.02381978],
       [0.27353223, 1.50017828, 5.46608505, 0.8605413 , 0.53152793,
        0.38911793, 0.41485051, 0.4570065 , 0.48703931, 0.28046883,
        0.16681054, 0.09765548, 0.06313855, 0.06457446, 0.04308833,
        0.03337959],
       [0.1504251 , 0.48848986, 2.34132877, 7.0722562 , 1.59888339,
        0.64736071, 0.37864608, 0.4176623 , 0.39474257, 0.31823328,
        0.13006713, 0.06080772, 0.04106148, 0.03696224, 0.02071822,
        0.01238369],
       [0.19942448, 0.26203393, 0.40241862, 2.61011212, 3.94175311,
        1.34881115, 0.73819614, 0.45173749, 0.32421733, 0.34274356,
        0.18844367, 0.11749404, 0.04354885, 0.03308906, 0.03405886,
        0.02415624],
       [0.29840015, 0.16002412, 0.14839901, 0.58432414, 1.52444259,
        1.68398546, 0.88435959, 0.54556362, 0.33665407, 0.27629834,
        0.21144505, 0.10673774, 0.03874077, 0.02855183, 0.01681653,
        0.00763196],
       [0.27839322, 0.20412566, 0.31939319, 0.25643538, 0.63404742,
        0.80693042, 0.93468714, 0.61480226, 0.35705377, 0.24729082,
        0.22708444, 0.1542997 , 0.06506191, 0.04696163, 0.02225166,
        0.01861681],
       [0.23274867, 0.28302014, 0.21959612, 0.16371453, 0.36419719,
        0.55334292, 0.63958423, 0.71788006, 0.46195277, 0.26876313,
        0.16586968, 0.1237654 , 0.09791347, 0.06781038, 0.03759241,
        0.01339474],
       [0.14703256, 0.1968324 , 0.3176527 , 0.2054568 , 0.34441124,
        0.35799274, 0.45635688, 0.45430344, 0.45475441, 0.25104035,
        0.15594895, 0.06906303, 0.06026442, 0.0423789 , 0.03034179,
        0.01153517],
       [0.04026824, 0.06828957, 0.09163321, 0.17597116, 0.22589319,
        0.24273812, 0.27493419, 0.28294501, 0.25052673, 0.23276823,
        0.16398568, 0.0706844 , 0.04630893, 0.03037643, 0.02547841,
        0.01771494],
       [0.05798567, 0.11227681, 0.13129047, 0.19651945, 0.33516908,
        0.36830605, 0.24928468, 0.22277235, 0.23264515, 0.23558332,
        0.16754599, 0.12498384, 0.0699668 , 0.03828949, 0.02374998,
        0.01441736],
       [0.07431919, 0.06822662, 0.0756041 , 0.09034974, 0.21130242,
        0.30832567, 0.2950942 , 0.22557132, 0.20346433, 0.12998804,
        0.17270293, 0.15181636, 0.09054696, 0.05063413, 0.02641034,
        0.01291306],
       [0.0460136 , 0.05191263, 0.04602036, 0.07362144, 0.14491053,
        0.19304503, 0.18400911, 0.20318438, 0.16170293, 0.11870192,
        0.09725387, 0.11348399, 0.09091698, 0.06641331, 0.0438363 ,
        0.01656413],
       [0.04381945, 0.05977226, 0.0391233 , 0.04147166, 0.11306029,
        0.15286506, 0.17872054, 0.14756021, 0.12962738, 0.09343492,
        0.09348667, 0.09531918, 0.083268  , 0.07267391, 0.03623683,
        0.01946932],
       [0.01545055, 0.03379666, 0.05015194, 0.09424355, 0.08161203,
        0.10302228, 0.09799977, 0.1256671 , 0.13579469, 0.09278382,
        0.0686979 , 0.06459996, 0.09793014, 0.08841309, 0.07669354,
        0.02517639],
       [0.02146559, 0.02253336, 0.0342794 , 0.01906929, 0.03897528,
        0.04505303, 0.07900516, 0.05842248, 0.05143449, 0.05782478,
        0.03818035, 0.03469299, 0.03024371, 0.04039254, 0.03310913,
        0.02022665]])
    return CH, CW, CS, CO



def Italy():
    CH=np.array([[4.66033163e-01, 2.50960098e-01, 1.29031039e-01, 1.41158019e-02,
        1.06827138e-02, 3.63583287e-02, 3.30153195e-01, 8.22901305e-01,
        5.29118018e-01, 5.14060903e-02, 2.11806910e-02, 2.22499885e-02,
        3.83338246e-02, 3.53988620e-02, 8.29230497e-03, 2.83669910e-02],
       [1.50190758e-01, 4.44959342e-01, 2.97744104e-01, 6.21550592e-02,
        4.05234626e-02, 3.82795609e-02, 1.79978604e-01, 4.67262228e-01,
        9.90498040e-01, 2.64746580e-01, 3.40510612e-02, 4.75941202e-02,
        1.84742850e-02, 2.88978417e-02, 1.59226084e-02, 5.22199790e-02],
       [7.63071459e-02, 4.38942411e-01, 2.24161448e+00, 1.89485695e-01,
        8.36691977e-02, 1.96204598e-02, 4.41398673e-02, 2.24320316e-01,
        6.44425154e-01, 3.37626885e-01, 1.72215656e-01, 2.29146135e-02,
        6.59991304e-03, 1.88993533e-02, 5.06867884e-02, 4.17800531e-02],
       [2.73927219e-02, 1.01393362e-01, 4.03337048e-01, 2.26713600e+00,
        2.01574507e-01, 4.68332490e-02, 2.07438380e-02, 6.89083891e-02,
        2.88205079e-01, 6.81675572e-01, 4.83802922e-01, 6.82159278e-02,
        2.86502149e-03, 3.09125201e-03, 1.36615961e-02, 1.56929004e-02],
       [8.19691796e-03, 7.41811509e-02, 6.14821505e-02, 2.93811840e-01,
        1.14418050e+00, 1.18105001e-01, 1.81231874e-02, 3.61931442e-07,
        4.16562752e-02, 5.45925034e-01, 4.29287352e-01, 2.57848421e-01,
        3.92014431e-02, 5.80674502e-03, 9.54423105e-03, 5.90690026e-02],
       [5.89036389e-04, 2.70025608e-02, 4.09921905e-02, 7.38982731e-02,
        3.14661795e-01, 1.33529662e+00, 1.31067643e-01, 2.26269385e-02,
        1.69789112e-02, 1.27615167e-01, 3.50399341e-01, 3.20658929e-01,
        2.50266722e-01, 3.95225649e-02, 8.00129301e-03, 2.01947173e-02],
       [3.82123593e-01, 2.94277387e-01, 5.16187819e-02, 1.71030548e-02,
        1.27201332e-02, 2.39853271e-01, 1.33014167e+00, 1.79008231e-01,
        6.35672589e-02, 1.58587259e-02, 5.79148772e-02, 1.53532075e-01,
        2.38024128e-01, 7.26942274e-02, 3.24264825e-02, 3.28194814e-02],
       [5.61730682e-01, 7.01980255e-01, 2.50860297e-01, 6.53725035e-02,
        2.12739780e-03, 1.88886028e-03, 1.38594157e-01, 1.69919470e+00,
        2.75392860e-01, 3.07913990e-02, 5.80452618e-03, 4.48689393e-03,
        1.19385410e-01, 9.51387233e-02, 2.79531216e-02, 3.09305790e-03],
       [3.65338243e-01, 4.48230379e-01, 4.91043826e-01, 2.43606159e-01,
        3.43604867e-02, 5.31243722e-03, 2.86323898e-02, 1.54632211e-01,
        1.33161289e+00, 1.15490671e-01, 3.07358713e-02, 1.44366761e-02,
        1.55172385e-02, 2.93886017e-02, 1.45892316e-02, 3.15968005e-02],
       [3.80116011e-02, 2.87201262e-01, 2.14400967e-01, 3.93924135e-01,
        2.22184338e-01, 2.48552734e-02, 1.07926207e-03, 4.19549121e-02,
        1.48423905e-01, 1.23163526e+00, 1.64794705e-01, 3.81753138e-02,
        7.14416210e-03, 1.56616534e-03, 1.06996137e-02, 5.93131286e-02],
       [2.04257869e-02, 4.44377367e-02, 1.66793654e-01, 3.85810400e-01,
        3.06805281e-01, 4.61718307e-02, 1.73600851e-02, 2.33629847e-03,
        2.80614473e-02, 1.83695850e-01, 9.09048403e-01, 1.16395715e-01,
        2.57422137e-03, 2.09729041e-03, 5.20417485e-03, 1.01722866e-01],
       [2.84745291e-13, 3.64505986e-02, 4.72168245e-02, 5.19852168e-03,
        2.97666585e-01, 2.16768972e-01, 1.38115921e-01, 8.81116600e-03,
        3.03863207e-02, 9.59100293e-02, 1.21262607e-01, 1.06227155e+00,
        6.51999453e-02, 2.13863379e-02, 4.67570449e-03, 7.29324596e-02],
       [1.33525634e-01, 3.57310559e-02, 2.82134158e-02, 1.76767308e-02,
        8.35521391e-02, 1.85888669e-01, 2.69961159e-01, 1.37741403e-01,
        4.11587318e-02, 2.53798896e-02, 4.00755270e-02, 2.40280699e-01,
        8.13639422e-01, 9.45236495e-02, 9.91020252e-03, 2.46394397e-02],
       [7.92232897e-02, 4.96960676e-02, 2.91610771e-02, 1.43738468e-02,
        2.70048058e-02, 6.79654445e-02, 4.60226552e-01, 2.48304162e-01,
        7.76574379e-02, 3.49618657e-02, 3.14482786e-02, 1.02157991e-01,
        2.57273847e-01, 1.39087133e+00, 7.45852450e-02, 4.38759097e-02],
       [1.38637285e-02, 4.46194715e-02, 1.63746000e-01, 6.68866595e-02,
        1.13892402e-01, 3.83132383e-02, 9.24691371e-02, 2.65968460e-01,
        4.25657985e-01, 1.63870883e-01, 4.78472596e-02, 3.30865219e-02,
        7.47486109e-02, 8.24536791e-02, 1.17764030e+00, 1.62502162e-01],
       [1.07896604e-01, 1.29962982e-01, 3.11767405e-06, 2.57945499e-01,
        5.60270239e-06, 3.06324298e-09, 5.75112451e-04, 9.92189295e-02,
        3.01629378e-01, 9.17982889e-02, 8.91531868e-07, 6.69118259e-01,
        1.91600750e-03, 3.46183266e-02, 6.45768878e-02, 1.28569956e-04]])

    CW=np.array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 4.19864706e-125, 1.32668683e-100,
        3.06468114e-139],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 3.84564026e-089, 1.77587792e-090,
        5.37792548e-087],
       [0.00000000e+000, 0.00000000e+000, 4.19972541e-002,
        3.24579808e-073, 3.51259765e-128, 1.86366287e-083,
        8.04460865e-070, 1.43227490e-002, 6.25305870e-058,
        1.74002216e-072, 3.11899387e-002, 1.49698691e-094,
        1.44611118e-080, 1.21671118e-075, 2.47975769e-005,
        1.37511700e-074],
       [0.00000000e+000, 0.00000000e+000, 1.82360835e-045,
        1.19069860e-048, 1.00851728e-068, 8.73974663e-072,
        2.34179830e-028, 1.85524849e-043, 6.79661439e-037,
        4.21497094e-038, 5.04202272e-060, 3.30223921e-028,
        2.42939722e-073, 1.75058413e-042, 6.93476624e-051,
        1.37510290e-083],
       [0.00000000e+000, 0.00000000e+000, 2.57666229e-002,
        1.79321314e-001, 4.06641299e-001, 6.30449983e-001,
        3.19548877e-001, 6.47803568e-001, 4.41885667e-001,
        5.11269426e-001, 2.97584454e-001, 9.61672288e-002,
        8.00395516e-003, 1.53396656e-005, 3.33981659e-034,
        3.49885652e-006],
       [0.00000000e+000, 0.00000000e+000, 4.24946456e-003,
        2.72934208e-002, 3.32857434e-001, 1.26383956e+000,
        1.06059658e+000, 8.29338595e-001, 9.08408991e-001,
        5.72866247e-001, 3.05976074e-001, 1.65776103e-001,
        1.02627879e-002, 2.78723156e-006, 7.59006323e-006,
        1.26665976e-005],
       [0.00000000e+000, 0.00000000e+000, 2.35572856e-002,
        1.18869956e-001, 4.14602846e-001, 7.98641797e-001,
        1.06865347e+000, 8.63363283e-001, 7.60411801e-001,
        3.29575690e-001, 3.48858328e-001, 7.17685269e-002,
        5.18683823e-003, 3.93113271e-006, 2.25138850e-027,
        1.21524218e-018],
       [0.00000000e+000, 0.00000000e+000, 8.06233595e-015,
        2.60946464e-001, 4.07873636e-001, 6.45437062e-001,
        1.03569066e+000, 1.80495096e+000, 1.82747740e+000,
        1.04598932e+000, 5.16161626e-001, 1.65260712e-001,
        9.35002089e-003, 1.25336619e-005, 1.58235245e-005,
        1.22632445e-005],
       [0.00000000e+000, 0.00000000e+000, 3.90913452e-003,
        1.51492392e-001, 4.02600960e-001, 8.14446399e-001,
        1.17811397e+000, 9.21547630e-001, 1.17908277e+000,
        9.81851820e-001, 9.19536228e-001, 1.31904891e-001,
        1.32755741e-002, 1.50920443e-005, 1.23936100e-005,
        6.94994442e-006],
       [0.00000000e+000, 0.00000000e+000, 2.67307077e-002,
        4.97790423e-001, 1.53618832e-001, 4.23021848e-001,
        9.51891500e-001, 8.57499585e-001, 8.32065778e-001,
        6.60644121e-001, 6.71148817e-001, 1.80166009e-001,
        9.35385793e-003, 2.23889851e-005, 2.49947518e-005,
        9.31589387e-006],
       [0.00000000e+000, 0.00000000e+000, 6.08516510e-002,
        4.37593818e-001, 9.55467494e-002, 5.05018005e-001,
        3.87434044e-001, 7.97968184e-001, 9.06198072e-001,
        6.83875109e-001, 4.03494300e-001, 1.56055493e-001,
        5.55382541e-003, 7.45819314e-006, 4.55885507e-006,
        1.06192493e-005],
       [0.00000000e+000, 0.00000000e+000, 9.71939874e-002,
        4.33567854e-003, 3.99531984e-002, 1.11026363e-001,
        2.36092495e-001, 2.47942489e-001, 3.48688084e-001,
        2.69473456e-001, 2.86376317e-001, 1.11879372e-001,
        7.37024488e-003, 4.61434649e-005, 3.51692831e-006,
        3.25731123e-006],
       [0.00000000e+000, 0.00000000e+000, 5.89845417e-003,
        5.16789949e-004, 2.01993670e-003, 2.35054876e-002,
        1.82005455e-002, 4.26435113e-002, 4.82803816e-002,
        6.51349278e-002, 2.95922711e-002, 1.14462934e-002,
        5.10329513e-004, 4.14240206e-005, 8.45221086e-006,
        4.14335448e-005],
       [4.17037877e-161, 2.36006542e-098, 2.51499824e-064,
        2.51954377e-082, 8.15174976e-006, 8.53453509e-006,
        6.69583207e-005, 8.50096009e-006, 3.77167286e-005,
        6.76343349e-005, 9.72046572e-005, 2.27310855e-005,
        3.73871283e-005, 6.69130609e-005, 6.71780661e-005,
        8.17844976e-006],
       [4.78088112e-125, 1.25437259e-106, 1.90753460e-113,
        9.12145428e-006, 3.26907567e-036, 4.12974594e-005,
        2.49307372e-005, 2.49347301e-005, 9.05271665e-006,
        5.70503276e-005, 9.25751259e-006, 5.76073369e-005,
        9.25471347e-006, 9.02859738e-006, 1.65891117e-047,
        4.09649731e-005],
       [6.23261201e-127, 1.16102392e-131, 9.00286378e-075,
        9.89585026e-094, 1.05310287e-004, 1.21813524e-074,
        6.07065589e-071, 5.36309147e-078, 1.79111186e-052,
        1.07029745e-004, 4.71427326e-039, 3.67962480e-093,
        7.06391875e-068, 5.51106604e-089, 3.77477441e-092,
        9.94219022e-109]])

    CS=np.array([[5.13817987e+000, 8.86623374e-001, 5.69733613e-002,
        4.15791609e-009, 3.80635888e-003, 8.14073344e-002,
        1.69548740e-001, 1.78804125e-001, 1.34257658e-001,
        3.09503928e-001, 1.10994540e-001, 1.32611039e-001,
        3.52965072e-003, 2.02549576e-003, 8.09630295e-053,
        8.28058006e-113],
       [6.93390655e-001, 3.37659364e+000, 6.39961280e-002,
        4.01805408e-005, 9.39450181e-003, 2.75478903e-002,
        7.77798118e-002, 8.62824564e-002, 9.59254817e-002,
        7.09714425e-002, 6.81061596e-002, 2.43644362e-002,
        7.95512101e-003, 3.34467532e-003, 4.63587582e-004,
        6.17135515e-043],
       [1.44357528e-022, 7.85290495e-001, 4.13719308e+000,
        7.42256289e-002, 1.11724686e-003, 4.20468293e-002,
        4.56879065e-002, 9.62398113e-002, 1.51789174e-001,
        1.45343321e-001, 9.32725827e-002, 4.92196445e-002,
        4.10391725e-003, 2.35187672e-003, 1.19977801e-034,
        1.47547249e-057],
       [1.15276998e-014, 1.50078873e-024, 1.29019463e+000,
        4.35635283e+000, 2.42392775e-002, 1.17023686e-002,
        2.58153701e-002, 8.35976467e-002, 1.00703734e-001,
        1.53761121e-001, 8.28935357e-002, 6.51755130e-002,
        1.36966181e-002, 1.06917151e-003, 1.81235105e-089,
        1.24173209e-067],
       [1.72051819e-001, 1.16844306e-002, 6.65272283e-026,
        3.75399867e-001, 1.79592965e-001, 1.19755986e-002,
        6.79090054e-006, 1.46890538e-002, 9.52714863e-003,
        8.82660697e-003, 1.33960160e-002, 1.21505864e-002,
        2.21916621e-003, 1.83513218e-003, 1.04866752e-062,
        1.79180252e-117],
       [3.95904042e-002, 1.07914417e-024, 6.87399714e-029,
        1.49728189e-002, 4.18332198e-001, 8.05856676e-002,
        4.29662011e-002, 9.49153533e-003, 2.91524287e-002,
        1.05256748e-002, 2.55687360e-031, 2.15234330e-002,
        1.62999938e-002, 1.22757063e-002, 2.97803697e-061,
        1.15121064e-002],
       [4.49271733e-002, 4.55736474e-002, 4.17609558e-001,
        3.68189190e-001, 1.07853351e-001, 2.43759935e-001,
        7.40294048e-002, 8.42859488e-002, 7.16913041e-002,
        3.11547341e-002, 2.83790081e-002, 5.18737637e-003,
        3.67931319e-003, 2.59513936e-003, 9.75229871e-046,
        9.51306647e-064],
       [5.62262609e-002, 1.03185290e-001, 5.44060909e-002,
        1.29870097e-002, 1.63429414e-027, 2.88683069e-002,
        4.77886006e-002, 1.02287252e-001, 1.48977681e-001,
        2.28960243e-002, 7.39101234e-003, 6.33022286e-003,
        3.09970613e-031, 1.43699888e-083, 5.79404180e-119,
        1.42813717e-109],
       [2.79289043e-002, 3.71848187e-001, 4.75701737e-002,
        8.05484894e-001, 7.83469389e-024, 2.35126788e-002,
        6.30103103e-003, 7.18316337e-002, 1.51662707e-001,
        9.57042642e-002, 8.35363131e-002, 2.36668601e-002,
        3.44745488e-002, 3.21543695e-057, 2.68224727e-137,
        7.40333858e-115],
       [3.52978774e-001, 4.47632529e-001, 4.30149324e-002,
        7.77929897e-001, 1.03561816e-010, 3.05413068e-002,
        6.94831173e-002, 1.38889214e-001, 7.91540863e-002,
        1.75396299e-002, 4.63941054e-002, 3.06371865e-002,
        3.20448922e-003, 6.51583286e-036, 5.52160206e-130,
        1.39107942e-130],
       [1.34636389e-001, 6.32642797e-001, 1.17783843e+000,
        7.23346149e-001, 8.95731969e-010, 3.66644994e-003,
        4.50375314e-002, 3.19890654e-002, 6.01123537e-002,
        1.29549749e-001, 9.15162625e-002, 4.39260681e-002,
        1.20571800e-002, 1.31259872e-049, 6.25314282e-139,
        1.88369568e-127],
       [2.39277768e-001, 9.27766533e-001, 1.76006429e+000,
        1.52661787e-002, 1.01249973e-027, 2.42540356e-002,
        1.05128702e-019, 1.14279589e-002, 7.42750441e-002,
        1.10872184e-001, 6.36972828e-002, 3.96578124e-002,
        2.96160771e-002, 1.58253418e-061, 3.01580520e-077,
        6.12876122e-135],
       [4.44097792e-002, 3.22019695e-002, 8.15377967e-002,
        4.32616789e-001, 1.94524841e-002, 9.80562173e-031,
        5.36507600e-003, 5.85291723e-003, 5.70731118e-003,
        5.60123379e-003, 1.45504282e-002, 1.31926840e-002,
        1.13087213e-002, 3.54770697e-003, 6.30441298e-074,
        7.97386684e-099],
       [6.03660505e-003, 2.73262746e-002, 5.99308404e-003,
        5.36762753e-041, 7.77471876e-054, 3.23523334e-042,
        5.70882912e-003, 5.55893304e-049, 5.75011781e-003,
        5.77795628e-003, 1.21031837e-049, 1.55355903e-002,
        1.49942337e-002, 2.43233113e-002, 2.77913041e-052,
        2.17907736e-117],
       [1.76399315e-065, 6.11123337e-064, 2.37496455e-081,
        2.39386567e-102, 7.38811575e-148, 7.10565020e-094,
        1.85848813e-087, 1.53000930e-065, 4.58291768e-087,
        2.11028157e-072, 2.44544213e-105, 1.41502332e-099,
        3.87287735e-067, 1.58783852e-079, 2.23160197e-091,
        3.34427661e-147],
       [3.76940876e-110, 5.25303596e-118, 3.88905356e-106,
        1.34950539e-125, 1.95720070e-117, 2.29607899e-131,
        6.99685894e-093, 3.60683256e-140, 4.88785500e-101,
        5.86965211e-116, 5.20086792e-108, 1.41499833e-101,
        5.92426633e-085, 7.26002572e-130, 2.56129544e-136,
        4.26725486e-115]])

    CO=np.array([[6.17797481e-01, 1.57503931e-01, 7.65171296e-02, 1.52737711e-01,
        1.80331753e-01, 3.86815716e-01, 3.23799703e-01, 2.80749790e-01,
        3.21871524e-01, 3.01928837e-01, 2.84194714e-01, 2.86394782e-01,
        2.19587516e-01, 2.28965969e-01, 2.34875404e-01, 1.14160919e-01],
       [4.02868374e-01, 1.54518250e+00, 5.43599753e-01, 4.99426848e-02,
        7.33115177e-02, 1.46364225e-01, 3.22270630e-01, 5.39227470e-01,
        4.61697435e-01, 7.42027531e-02, 8.75549923e-02, 6.29190865e-02,
        1.69931438e-01, 1.68634732e-01, 9.91028466e-02, 5.23461058e-03],
       [7.71626924e-02, 6.57747277e-01, 2.98031545e+00, 3.54804763e-01,
        4.27263394e-01, 2.53990016e-01, 3.01325496e-01, 3.07720916e-01,
        5.91481323e-01, 3.70023066e-01, 2.70357781e-01, 5.71023244e-02,
        1.07411192e-01, 2.16581559e-01, 1.09391432e-01, 1.47025172e-01],
       [6.28579960e-02, 2.39171319e-01, 1.68693278e+00, 4.27696457e+00,
        8.06969431e-01, 4.77120327e-01, 2.36386389e-01, 2.20368601e-01,
        3.94060771e-01, 3.85062929e-01, 1.61347510e-01, 2.61320837e-02,
        7.88393986e-02, 1.75632778e-02, 4.76662089e-02, 3.75096945e-02],
       [5.75003471e-02, 4.69321917e-02, 1.30281148e-01, 9.32959923e-01,
        3.65082759e+00, 1.59287269e+00, 8.74168332e-01, 4.92828655e-01,
        4.03079043e-01, 4.61851958e-01, 4.72105227e-01, 2.71521114e-01,
        1.73443931e-01, 9.83403037e-02, 1.49503809e-01, 8.84769237e-02],
       [8.05657906e-02, 1.08854686e-01, 8.43032109e-02, 6.00046807e-01,
        2.03693507e+00, 2.42264839e+00, 1.61549281e+00, 7.65359479e-01,
        6.77540035e-01, 6.05414035e-01, 5.78960158e-01, 1.94012641e-01,
        1.51068472e-01, 3.86161021e-02, 8.10078907e-02, 8.44390577e-03],
       [2.49103874e-01, 7.21208627e-02, 3.47982717e-01, 6.91522617e-02,
        6.23523989e-01, 8.90720055e-01, 1.44878993e+00, 8.73604587e-01,
        6.38328128e-01, 4.45405104e-01, 5.01896947e-01, 4.13639958e-01,
        2.65175055e-01, 1.28399972e-01, 1.31233064e-01, 7.02414013e-02],
       [2.51887983e-01, 1.85610899e-01, 2.18871652e-01, 1.85665662e-01,
        4.10906467e-01, 9.87030470e-01, 1.38471083e+00, 1.31845251e+00,
        1.08949485e+00, 6.31734562e-01, 5.25799011e-01, 4.39723481e-01,
        4.32416453e-01, 4.63212719e-01, 2.18536583e-01, 1.19190834e-01],
       [3.69748995e-01, 1.84726300e-01, 2.23269101e-01, 1.25212415e-01,
        5.32316132e-01, 4.87756864e-01, 8.50261338e-01, 8.20264310e-01,
        9.44077788e-01, 6.75676388e-01, 5.66099879e-01, 2.86338429e-01,
        3.40777677e-01, 1.99418294e-01, 2.53252391e-01, 4.75468115e-02],
       [9.79169652e-02, 1.34926387e-01, 1.49378165e-01, 1.65940559e-01,
        3.04656953e-01, 2.74434947e-01, 4.80866316e-01, 5.90911131e-01,
        7.50132563e-01, 3.21619705e-01, 4.22181540e-01, 3.13484168e-01,
        2.21467380e-01, 1.67795693e-01, 1.15602626e-01, 1.29804353e-01],
       [7.58589822e-02, 1.14554794e-01, 2.89736169e-01, 1.67905547e-01,
        2.93943268e-01, 5.87524630e-01, 5.10673740e-01, 6.28595409e-01,
        5.85633317e-01, 3.41121871e-01, 3.69451256e-01, 5.72992917e-01,
        4.72181209e-01, 2.04529935e-01, 2.38718333e-01, 1.11922535e-01],
       [1.91913332e-01, 1.70338642e-01, 1.66245287e-01, 1.41470278e-01,
        1.86890007e-01, 7.23937621e-01, 6.39979700e-01, 7.05339663e-01,
        7.19717751e-01, 5.68807771e-01, 7.68166371e-01, 7.63236567e-01,
        7.53113833e-01, 3.30379056e-01, 3.40313444e-01, 1.21083901e-01],
       [9.81038373e-02, 1.11193693e-01, 1.18017457e-01, 2.26915099e-01,
        4.22191358e-01, 5.96292861e-01, 5.81398297e-01, 7.00289291e-01,
        7.80492319e-01, 6.58635855e-01, 5.26596160e-01, 7.76273665e-01,
        6.74896082e-01, 6.80042075e-01, 5.07545548e-01, 3.03431334e-01],
       [5.91139071e-02, 9.05637121e-02, 7.89382100e-26, 1.36842608e-02,
        2.26952676e-01, 4.66793226e-01, 7.17926705e-01, 7.47960251e-01,
        6.81914604e-01, 5.96594417e-01, 6.69648581e-01, 5.11980519e-01,
        6.38809759e-01, 6.75590624e-01, 3.06108962e-01, 2.19115879e-01],
       [2.70733674e-02, 4.16375044e-02, 4.13833461e-02, 1.09237108e-01,
        2.70048100e-04, 3.22986781e-01, 3.09987534e-01, 2.51045265e-01,
        4.69964942e-01, 3.89594477e-01, 3.44137736e-01, 3.95151933e-01,
        1.08096951e+00, 1.27995266e+00, 2.50479137e-01, 8.85388147e-02],
       [1.10382885e-11, 7.38377732e-06, 4.64739605e-24, 1.16787701e-01,
        8.16851623e-02, 1.65666752e-04, 1.45687342e-12, 1.48885936e-01,
        2.87652379e-01, 9.43051008e-01, 5.04464875e-01, 5.06491558e-01,
        1.49337045e-01, 1.60799248e+00, 4.90741543e-01, 4.51948493e-07]])

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



def RepublicOfKorea():
    CH = np.array([[0.321884356339694,	0.0429593200835726,	0.00786188991649658,
    0.00807523798841333,	0.00533465134087154,	0.0217715658215287,
    0.04010688801543,	0.0299107836271602,	0.0140671453881273,
    0.0166644275289274,	0.00948649736945684,	0.00741543234334085,
    0.0012825627202577,	0.000779362251039537,	0.000000, 0],
       [0.2146999 , 0.80652371, 0.32294517, 0.10429544, 0.03978323,
        0.10990523, 0.2926698 , 0.40497012, 0.44149537, 0.13655079,
        0.04480706, 0.03145452, 0.01634002, 0.01051657, 0.00447817,
        0.00863129],
       [0.14939055, 0.47339215, 1.4854278 , 0.36360143, 0.07348596,
        0.04326912, 0.12760568, 0.37221514, 0.60396272, 0.27718598,
        0.12563702, 0.03239006, 0.01212187, 0.01344626, 0.01686077,
        0.01457124],
       [0.07116708, 0.14910155, 0.42647442, 1.34279173, 0.24719448,
        0.05947553, 0.03571091, 0.16789366, 0.40275512, 0.53379808,
        0.32181108, 0.07294903, 0.01791975, 0.01145263, 0.00736776,
        0.01057749],
       [0.09897732, 0.08282466, 0.09688526, 0.40712772, 1.23681987,
        0.22356532, 0.05147574, 0.01947737, 0.11798248, 0.42479297,
        0.32879236, 0.16160447, 0.03266684, 0.00616448, 0.00744444,
        0.01018096],
       [0.30304692, 0.13431638, 0.04928371, 0.10381142, 0.28074673,
        1.05735462, 0.22371133, 0.02856823, 0.02048264, 0.11650679,
        0.28840698, 0.23629232, 0.10006859, 0.02228086, 0.00417187,
        0.01518979],
       [0.54292975, 0.54205956, 0.2333011 , 0.04903098, 0.07358345,
        0.23386564, 1.09659074, 0.24396626, 0.08285029, 0.02087705,
        0.04987331, 0.11781957, 0.13323238, 0.02756597, 0.01324007,
        0.01215628],
       [0.55725481, 0.77876615, 0.58790448, 0.20555739, 0.02840511,
        0.03374218, 0.17642985, 1.0662995 , 0.21321169, 0.04374971,
        0.0227233 , 0.01877782, 0.05772289, 0.03500574, 0.0133739 ,
        0.0077663 ],
       [0.30982218, 0.66643882, 0.77093879, 0.47204492, 0.11039648,
        0.03242716, 0.08442907, 0.20994774, 0.96247278, 0.16543975,
        0.05164433, 0.01056037, 0.02797313, 0.03352994, 0.01441372,
        0.01519471],
       [0.12266882, 0.30220147, 0.46960072, 0.67316143, 0.370747  ,
        0.09563353, 0.02929383, 0.08302228, 0.17486434, 0.88849905,
        0.18042935, 0.04337915, 0.01578341, 0.01020898, 0.01054537,
        0.03436616],
       [0.17888208, 0.17946731, 0.37075359, 0.50646465, 0.48699249,
        0.26969878, 0.09740991, 0.04582824, 0.10868332, 0.23269954,
        0.91200379, 0.17789488, 0.0299    , 0.00872043, 0.01116897,
        0.05139323],
       [0.262303  , 0.28266465, 0.20226189, 0.28565407, 0.30220538,
        0.39058018, 0.28524755, 0.06857088, 0.04599524, 0.15023339,
        0.22677398, 0.7988931 , 0.14808231, 0.0483574 , 0.00708578,
        0.03698961],
       [0.25486009, 0.22916939, 0.15250561, 0.13780789, 0.12227049,
        0.19232678, 0.29592624, 0.17817411, 0.07690749, 0.04244938,
        0.07078003, 0.17431455, 0.58773223, 0.09859024, 0.02291428,
        0.01089501],
       [0.18347745, 0.26822744, 0.23578578, 0.12353499, 0.08962221,
        0.11023975, 0.23133874, 0.25639799, 0.20482069, 0.05579473,
        0.05170563, 0.10069838, 0.13310538, 0.55042837, 0.06599686,
        0.01414389],
       [0.09694548, 0.284753  , 0.28201572, 0.20294172, 0.04771416,
        0.10368858, 0.10418097, 0.20970751, 0.24666354, 0.16037086,
        0.09893085, 0.04821376, 0.11990535, 0.13094116, 0.41486002,
        0.09049405],
       [0.17892901, 0.21766337, 0.34700539, 0.27941522, 0.07342818,
        0.0690764 , 0.09554759, 0.18798061, 0.23379308, 0.2696076 ,
        0.30278513, 0.1302428 , 0.06178028, 0.07781778, 0.0702749 ,
        0.27301194]])

    CW=np.array([
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 8.20604524e-092, 1.20585150e-005,
        3.16436834e-125],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [0.00000000e+000, 0.00000000e+000, 1.45656012e-002,
        6.93386603e-003, 1.00194792e-002, 2.89918063e-003,
        2.39203103e-002, 6.81920292e-003, 2.40093313e-002,
        1.39559489e-002, 5.31171349e-003, 5.65563543e-009,
        1.58340614e-018, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [0.00000000e+000, 0.00000000e+000, 1.23585931e-002,
        4.65661172e-001, 5.04971737e-001, 2.90427650e-001,
        2.73327506e-001, 2.54157709e-001, 2.84016661e-001,
        2.24051800e-001, 1.26467237e-001, 2.59436161e-002,
        1.40178790e-003, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [0.00000000e+000, 0.00000000e+000, 2.18885288e-002,
        3.40346605e-001, 8.91531656e-001, 8.54473590e-001,
        7.19952073e-001, 7.93497320e-001, 6.00959892e-001,
        4.81605778e-001, 3.53456500e-001, 7.02040773e-002,
        5.22444858e-003, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [0.00000000e+000, 0.00000000e+000, 2.81515774e-002,
        2.99170044e-001, 8.48488637e-001, 1.42892111e+000,
        1.03223469e+000, 1.00813180e+000, 9.34231177e-001,
        6.55397699e-001, 5.17834425e-001, 1.03426352e-001,
        6.96894905e-003, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [0.00000000e+000, 0.00000000e+000, 3.22010563e-002,
        1.63215102e-001, 5.93525658e-001, 9.78649041e-001,
        1.29291833e+000, 1.13423567e+000, 1.01961417e+000,
        8.32958383e-001, 4.82937202e-001, 1.21814617e-001,
        6.58759101e-003, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [0.00000000e+000, 0.00000000e+000, 1.98086342e-002,
        3.25650736e-001, 4.92504467e-001, 9.19500307e-001,
        9.94002194e-001, 1.37398240e+000, 1.34082458e+000,
        9.28444190e-001, 6.49786929e-001, 1.11774520e-001,
        4.67015731e-003, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [0.00000000e+000, 0.00000000e+000, 2.24533766e-002,
        2.04014086e-001, 5.81486960e-001, 9.12220693e-001,
        1.08620491e+000, 1.16279352e+000, 1.42898791e+000,
        1.14889640e+000, 7.93711421e-001, 1.19043084e-001,
        6.86908968e-003, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [0.00000000e+000, 0.00000000e+000, 2.99761568e-002,
        2.55170324e-001, 3.96557150e-001, 6.86162962e-001,
        8.88147820e-001, 9.76758179e-001, 1.01363589e+000,
        9.89896419e-001, 6.12885169e-001, 1.34820041e-001,
        5.47460000e-003, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [0.00000000e+000, 0.00000000e+000, 3.05704686e-002,
        1.80021281e-001, 3.11333899e-001, 6.52049201e-001,
        7.67315379e-001, 7.99288625e-001, 1.08567472e+000,
        1.04785758e+000, 7.65183391e-001, 1.60481185e-001,
        6.17114670e-003, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [0.00000000e+000, 0.00000000e+000, 2.09753607e-002,
        5.38693822e-002, 9.34120097e-002, 1.71829428e-001,
        2.41185616e-001, 2.27484445e-001, 2.91715328e-001,
        2.30261345e-001, 2.05194546e-001, 5.55069238e-002,
        2.59228852e-003, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [0.00000000e+000, 0.00000000e+000, 1.68879395e-003,
        1.74214561e-003, 8.85508348e-003, 1.57840519e-002,
        1.75531205e-002, 2.07512209e-002, 2.25070126e-002,
        2.19905944e-002, 1.61500567e-002, 5.38925225e-003,
        2.12072051e-004, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

    CS=np.array([[0.321884356339694,	0.0429593200835726,
    	0.00786188991649658,	0.00807523798841333,
        0.00533465134087154,	0.0217715658215287,
        0.04010688801543,	0.0299107836271602,
        0.0140671453881273,	0.0166644275289274,
        0.00948649736945684,	0.00741543234334085,
        0.0012825627202577,	0.000779362251039537,
        0.0,	0.0],
        [5.37547198e-002, 2.99841914e+000, 2.09594674e-001,
        2.53671552e-002, 1.48219846e-002, 5.31960359e-002,
        8.01778969e-002, 7.68519220e-002, 8.35409054e-002,
        5.81905056e-002, 5.14694090e-002, 1.46905137e-002,
        5.67374133e-003, 1.70331756e-003, 3.48153259e-004,
        8.10213723e-039],
       [4.55257273e-004, 8.13710056e-001, 5.96391141e+000,
        2.02828057e-001, 1.12866043e-002, 5.04017742e-002,
        5.96774121e-002, 9.63778386e-002, 1.13385882e-001,
        8.68463107e-002, 6.38203025e-002, 2.90678275e-002,
        7.12655079e-003, 9.83410270e-004, 4.95472320e-025,
        1.82625475e-004],
       [2.58963950e-003, 3.81432114e-002, 2.03258369e+000,
        7.65251600e+000, 4.92281406e-002, 6.13896562e-002,
        8.00564916e-002, 1.14056648e-001, 1.08557081e-001,
        1.24785560e-001, 7.21312450e-002, 3.85146648e-002,
        8.32877192e-003, 1.52411641e-003, 6.23827897e-033,
        1.71248680e-070],
       [7.17044393e-003, 1.18513070e-002, 5.59344254e-003,
        5.03472092e-001, 2.17015034e-001, 2.83773030e-002,
        2.07634698e-002, 2.57582149e-002, 1.77760208e-002,
        2.16680445e-002, 1.20520375e-002, 8.35533057e-003,
        6.63254775e-004, 1.27928049e-003, 1.77165032e-004,
        1.21962450e-047],
       [7.02549906e-003, 7.41369771e-002, 2.82821621e-002,
        1.56988132e-001, 1.57182340e-001, 1.12203005e-001,
        2.17884713e-002, 2.91877627e-002, 3.82032481e-002,
        3.31930218e-002, 9.07125511e-003, 1.39296587e-002,
        4.63815268e-003, 2.63822158e-003, 4.63437055e-004,
        1.28665719e-003],
       [1.41287413e-002, 2.86310803e-001, 2.31756473e-001,
        1.59299330e-001, 3.12841496e-002, 6.81980852e-002,
        6.92921944e-002, 4.86186060e-002, 5.86702732e-002,
        3.02904912e-002, 2.58809381e-002, 3.65610403e-003,
        6.39369941e-003, 5.09746712e-004, 1.65629548e-048,
        3.11454864e-055],
       [2.40729478e-002, 1.71419988e-001, 1.53639693e-001,
        8.41841708e-002, 1.24312953e-002, 4.44004240e-002,
        6.92239197e-002, 5.23187904e-002, 6.45941585e-002,
        3.06814424e-002, 4.76349189e-003, 1.02854826e-002,
        6.77073635e-004, 2.40386170e-003, 1.84663259e-123,
        9.70468208e-067],
       [7.81264867e-003, 1.10647597e-001, 1.08022257e-001,
        4.70659327e-001, 6.47191037e-003, 2.44986304e-002,
        2.70353447e-002, 3.69908325e-002, 8.85622548e-002,
        3.13208750e-002, 3.65536926e-002, 9.35788263e-003,
        7.26118905e-003, 5.74856958e-004, 4.82044040e-068,
        2.40534618e-092],
       [6.55486950e-002, 2.44066748e-001, 1.94238106e-001,
        7.48412922e-001, 4.65438473e-003, 3.02536741e-002,
        6.55467452e-002, 5.73352908e-002, 5.96825535e-002,
        3.55923642e-002, 4.57923569e-002, 1.92950647e-002,
        4.37377790e-003, 2.03152148e-003, 6.22029709e-134,
        3.27724754e-072],
       [1.72925153e-002, 4.21863900e-001, 6.47273948e-001,
        7.46915585e-001, 4.89038344e-003, 1.60998629e-002,
        5.03690451e-002, 5.03007037e-002, 6.97998439e-002,
        9.45886770e-002, 5.00052865e-002, 2.43579926e-002,
        6.32730288e-003, 1.08384384e-023, 1.24031083e-117,
        5.65033817e-078],
       [6.14778899e-002, 3.62368382e-001, 3.79024201e-001,
        4.61322755e-001, 5.49616772e-003, 5.81313327e-002,
        2.49515555e-002, 4.23347842e-002, 5.86466675e-002,
        4.22060176e-002, 3.83256544e-002, 3.96328809e-002,
        1.15374100e-002, 1.31459848e-031, 7.83007729e-004,
        7.63207034e-004],
       [3.63853406e-002, 7.09595227e-002, 4.24740261e-002,
        2.12119954e-001, 1.20319909e-002, 1.83694708e-003,
        1.82696275e-002, 5.13154828e-002, 1.21499330e-002,
        1.91746591e-002, 1.54958467e-002, 7.49706070e-003,
        2.44068212e-002, 1.22115079e-002, 4.42630975e-067,
        2.12193495e-037],
       [1.67989158e-003, 3.32617178e-002, 1.43168865e-002,
        1.07586694e-031, 2.00725208e-003, 2.10506616e-003,
        1.41612596e-002, 6.21682457e-003, 6.51025741e-003,
        1.04461364e-002, 2.37657152e-003, 1.73009136e-002,
        8.92460568e-003, 1.82600786e-002, 1.11281609e-002,
        3.45788428e-126],
       [1.28088624e-028, 5.12095289e-026, 1.93694618e-040,
        7.63724399e-003, 2.63588189e-022, 1.69866375e-024,
        1.26018968e-026, 7.63080836e-003, 7.86335233e-003,
        2.11895085e-002, 3.52782350e-002, 2.14647464e-002,
        7.74531659e-003, 8.01635106e-003, 7.91287483e-003,
        2.13826287e-002],
       [2.81656046e-094, 2.11629073e-002, 8.48753850e-042,
        2.13263375e-002, 4.89821462e-036, 7.59636700e-003,
        9.78080485e-069, 2.23318362e-060, 1.43897360e-048,
        8.57132811e-060, 4.70110908e-042, 1.59977060e-046,
        2.21108667e-083, 8.86105592e-107, 1.02043041e-080,
        6.61414969e-113]])

    CO = np.array([
       [1.3230324832343,	0.56369553948105,	0.291801003171522,
       0.175606907513332,	0.244706407233471,	0.442307952115635,
       0.742980082343001,	0.752345670525179,	0.489539674047473,
       0.228863087073346,	0.252979756500437,	0.191158157599365,
       0.119824711559517,	0.096381987060268,	0.0606719735006203,
       0.0353370297215362],
       [0.19063269, 0.73806102, 0.34476798, 0.10876514, 0.08667752,
        0.17901785, 0.26679729, 0.30082774, 0.28126749, 0.13171971,
        0.1152529 , 0.11017624, 0.09059423, 0.07067949, 0.03648522,
        0.025036  ],
       [0.0782867 , 0.42781388, 1.92883358, 0.34287787, 0.21339901,
        0.18781385, 0.2646519 , 0.32122069, 0.40999199, 0.253872  ,
        0.18224658, 0.10455137, 0.06495925, 0.07025085, 0.0542879 ,
        0.04341248],
       [0.04861263, 0.15729573, 0.93288935, 3.18181614, 0.72482382,
        0.35281095, 0.27275091, 0.33147881, 0.37521024, 0.3252559 ,
        0.16045484, 0.07350911, 0.04770132, 0.04540445, 0.02947442,
        0.01818583],
       [0.06493904, 0.08501929, 0.16156392, 1.18324542, 1.80054367,
        0.74070529, 0.53580048, 0.36125611, 0.31052425, 0.35297785,
        0.23424234, 0.14311887, 0.05097662, 0.04095653, 0.04882267,
        0.03574463],
       [0.11681712, 0.06242024, 0.0716271 , 0.31845609, 0.83715402,
        1.11176421, 0.77168538, 0.52451091, 0.38763521, 0.34208693,
        0.31598132, 0.15630725, 0.05451833, 0.04248668, 0.02898064,
        0.01357681],
       [0.14404664, 0.10523849, 0.20375536, 0.18471863, 0.46020684,
        0.70412128, 1.07798986, 0.78123485, 0.54338794, 0.40467207,
        0.44852668, 0.29865048, 0.12101473, 0.0923632 , 0.05068399,
        0.04377276],
       [0.13268733, 0.16076522, 0.1543497 , 0.12993248, 0.29124982,
        0.53199001, 0.81272552, 1.00506878, 0.77458966, 0.48457684,
        0.36096548, 0.26393387, 0.20065574, 0.14694329, 0.09434227,
        0.03470009],
       [0.10038897, 0.13390647, 0.26740154, 0.19529055, 0.32986528,
        0.41220529, 0.69451395, 0.76176347, 0.9132323 , 0.54208417,
        0.40645395, 0.1763891 , 0.14791096, 0.10998501, 0.09119638,
        0.03578907],
       [0.02956337, 0.04995491, 0.08294364, 0.1798544 , 0.23263834,
        0.30053578, 0.44990827, 0.51014661, 0.54097509, 0.54046274,
        0.45957212, 0.19411921, 0.12221456, 0.08476946, 0.08234319,
        0.05909969],
       [0.05138279, 0.0991333 , 0.14343961, 0.24243247, 0.41662735,
        0.5503928 , 0.49237559, 0.48479702, 0.60634931, 0.66022548,
        0.56674471, 0.41429006, 0.22287221, 0.12896975, 0.09264549,
        0.05805464],
       [0.06453498, 0.05903106, 0.08094284, 0.10922181, 0.25738636,
        0.45151355, 0.57116135, 0.48103841, 0.51965413, 0.35698367,
        0.57246685, 0.49313569, 0.28264091, 0.16712787, 0.10095602,
        0.05095393],
       [0.03839668, 0.04316314, 0.04734743, 0.08552637, 0.16962672,
        0.27166458, 0.34225573, 0.41638922, 0.39687822, 0.31326795,
        0.30979244, 0.35423848, 0.27272154, 0.21065604, 0.16102946,
        0.06281023],
       [0.03866507, 0.05255138, 0.04256241, 0.05094382, 0.13994225,
        0.22747151, 0.35150401, 0.31975904, 0.33641902, 0.26074253,
        0.31488937, 0.31461966, 0.26411736, 0.24374838, 0.14075573,
        0.07806506],
       [0.01578877, 0.03441208, 0.06318751, 0.13407396, 0.11698914,
        0.1775426 , 0.22322022, 0.31537539, 0.40814947, 0.29986622,
        0.26798125, 0.24693944, 0.35973926, 0.34342541, 0.34500654,
        0.11691015],
       [0.02264318, 0.0236839 , 0.04458274, 0.02800384, 0.05767277,
        0.08014669, 0.1857608 , 0.15134789, 0.15958095, 0.1929121 ,
        0.15374147, 0.13689579, 0.11468238, 0.16195981, 0.15374696,
        0.09695559]])
    return CH, CW, CS, CO


'''
used pandas to read the files using
curDir='/Users/rsingh/Dropbox/repos/github/pyross/examples/'
np.array(pd.read_excel(os.path.join(curDir,'data/contact_matrices_152_countries/MUestimates_school_1.xlsx'), sheet_name='India'))
'''
