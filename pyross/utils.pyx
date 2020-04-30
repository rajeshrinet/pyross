import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange
cdef double PI = 3.1415926535
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import nlopt
import cma
import multiprocessing
# from pathos.multiprocessing import ProcessingPool as Pool


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
        plt.plot(self.xT, self.yT, 'o', ms=10, mfc='#348ABD', mec='none', label='training set' )
        plt.plot(self.xS, self.yS, '#dddddd', lw=1.5, label='posterior')
        plt.plot(self.xS, self.mu, '#A60628', lw=2, label='mean')

        # fill 95% confidence interval (2*sd about the mean)
        plt.fill_between(self.xS.flat, self.mu-2*self.sd, self.mu+2*self.sd, color="#348ABD", alpha=0.4, label='2 sigma')
        plt.axis('tight'); plt.legend(fontsize=15); plt.rcParams.update({'font.size':18})


    def runGPR(self):
        self.calcKernels()
        self.calcPrior()
        self.calcMuSigma()
        self.plotResults()

class BoundedSteps(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
    Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, bounds, stepsize=1e-1):
        self.bounds = np.array(bounds)
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.bounds[:, 0]-x, -self.stepsize)
        max_step = np.minimum(self.bounds[:, 1]-x, self.stepsize)
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        return xnew


def minimisation(objective_fct, guess, bounds, global_max_iter=100, local_max_iter=100, ftol=1e-2, global_ftol_factor=10., 
                 enable_global=True, enable_local=True, cma_processes=0, cma_population=16, cma_stds=None, 
                 verbose=True):
    """ Compute the global minimum of the objective function.
    
    This function computes the global minimum of `objective_fct` using a combination of a global minimisation step 
    (CMA-ES) and a local refinement step (Suplex) (both derivative free).
    
    objective_fct: callable 
        The objective function. It must be of the form fct(params, grad=0) for the use in NLopt. The parameters 
        should not be modified and `grad` can be ignored (since only derivative free algorithms are used).
    guess: numpy.array
        The initial guess.
    bounds: numpy.array
        The boundaries for the optimisation algorithm, given as a dimsx2 matrix.
    global_max_iter: int
        The maximum number of iterations for the global algorithm.
    local_max_iter: int
        The maximum number of iterations for the local algorithm.
    ftol: float
        Relative function value stopping criterion for the optimisation algorithms.
    global_ftol_factor: float
        For the global optimisation, `ftol` gets multiplied by this.
    enable_global: bool
        Enable (or disable) the global minimisation part.
    enable_local: bool
        Enable (or disable) the local minimisation part (run after the global minimiser).
    cma_processes: int
        Number of processes used in the CMA algorithm. By default, the number of CPU cores is used.
    cma_population: int
        The number of samples used in each step of the CMA algorithm. Should ideally be factor of `cma_threads`.
    cma_stds: numpy.array
        Initial standard deviation of the spread of the population for each parameter in the CMA algorithm. Ideally, 
        one should have the optimum within 3*sigma of the guessed initial value. If not specified, these values are
        chosen such that 3*sigma reaches one of the boundaries for each parameters.
    verbose: bool
        Enable output.
    """         
    x_result = guess
    y_result = 0
    
    # Step 1: Global optimisation
    if enable_global:
        if verbose:
            print('Starting global minimisation...')
        
        if cma_processes == 0:
            cma_processes = multiprocessing.cpu_count()
        # p = Pool(cma_processes)

        options = cma.CMAOptions()
        options['bounds'] = [bounds[:, 0], bounds[:, 1]]
        options['tolfunrel'] = ftol * global_ftol_factor
        options['popsize'] = cma_population
        

        if cma_stds is None:
            # Standard scale: 3*sigma reaches from the guess to the closest boundary for each parameter.
            cma_stds = np.amin([bounds[:, 1] - guess, guess -  bounds[:, 0]], axis=0)
            cma_stds *= 1.0/3.0
            
        options['CMA_stds'] = cma_stds

        global_opt = cma.CMAEvolutionStrategy(guess, 1.0, options)
        iteration = 0
        while not global_opt.stop() and iteration < global_max_iter:
            positions = global_opt.ask()
            # Endless parallelisation options here. Use pool for now.
            # values = p.map(objective_fct, positions)
            # No multiprocessing for now because of problems with sending objective_fct to the processes... (todo)
            values = [objective_fct(x) for x in positions]
            global_opt.tell(positions, values)
            if verbose:
                global_opt.disp()
            iteration += 1

        x_result = global_opt.best.x
        y_result = global_opt.best.f
        
        if verbose:
            print('Optimal value (global minimisation): ', y_result)
            print('Starting local minimisation...')
    
    # Step 2: Local refinement
    if enable_local:
        # Use derivative free local optimisation algorith with support for boundary conditions
        # to converge to the next minimum (which is hopefully the global one).
        local_opt = nlopt.opt(nlopt.LN_SBPLX, guess.shape[0])
        local_opt.set_min_objective(objective_fct)
        local_opt.set_lower_bounds(bounds[:,0])
        local_opt.set_upper_bounds(bounds[:,1])
        local_opt.set_ftol_rel(ftol)
        local_opt.set_maxeval(3*local_max_iter)

        x_result = local_opt.optimize(global_opt.best.x)
        y_result = local_opt.last_optimum_value()
    
        if verbose:
            print('Optimal value (local minimisation): ', y_result)
    
    return x_result, y_result
