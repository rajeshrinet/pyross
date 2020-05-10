import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log
from cython.parallel import prange
cdef double PI = 3.1415926535
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

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


def make_gamma_dist(means, stds):
    vars = stds**2
    scale = vars/means
    a = means/scale
    return a, scale 



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
