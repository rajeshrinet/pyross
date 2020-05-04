#!python
"""Unittesting for the pyross module. Run as python -m unittest pyross.test."""
import sys
#remove pwd from path that tries to import .pyx files
for i in sys.path:
    if 'pyross' in i or i == '':
        sys.path.remove(i)
import pyross
import unittest
import numpy as np
import scipy as sp


class DeterministicTest(unittest.TestCase):
    """testing deterministic.pyx."""

    def __init__(self, *args, **kwargs):
        super(DeterministicTest, self).__init__(*args, **kwargs)
        self.N = np.asarray([10000], dtype=np.float64)
        self.M = 1
        self.alpha = 0
        self.beta = 0.007
        self.gIa = 0.008
        self.gIs = 0.008
        self.gE = 0
        self.gA = 0
        self.tE = 0
        self.tIa = 0
        self.tIs = 0
        self.Tf = 100
        self.Nf = 1000
        self.fsa = 0
        self.parameters = {'N': self.N, 'M': self.M, 'alpha': self.alpha,
                              'beta': self.beta, 'gIa': self.gIa, 'gIs': self.gIs,
                              'gE': self.gE, 'gA': self.gA, 'tE': self.tE,
                              'tIa': self.tIa, 'tIs': self.tIs, 'fsa': self.fsa}

    def contactMatrix(self, t): return np.identity(self.M)

    def test_decay(self):
        """
        Exponential decay from infected to recovered. Paths agree within .1%.
        """
        SIR = pyross.deterministic.SIR(self.parameters, self.M, self.N)
        sim = SIR.simulate(np.zeros(1), np.zeros(1), self.N,
                           self.contactMatrix, self.Tf,
                           self.Nf, integrator='solve_ivp')
        time_points = np.linspace(0, self.Tf, self.Nf)
        exp_decay = sp.integrate.solve_ivp(lambda t, y: -self.gIs * y,
                                           (0, self.Tf), self.N,
                                           t_eval=time_points)
        diff = (sim['X'][:, 2] - exp_decay.y)/self.N
        self.assertTrue((diff < 0.001).all(), msg="paths differ > .1%")

    def test_integrators(self):
        """
        All integration methods produce paths which agree within .1%
        """
        integrators = ['solve_ivp', 'odeint', 'odespy',
                       'odespy-rkf45', 'odespy-rk4']
        paths = []
        model = pyross.deterministic.SIR(self.parameters, self.M, self.N)
        for integrator in integrators:
            data = model.simulate(np.zeros(1), np.zeros(1), self.N,
                                  self.contactMatrix, self.Tf,
                                  self.Nf, integrator=integrator)
            paths.append(data['X'])
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i != j:
                    diff = (paths[i]-paths[j])/self.N
                    self.assertTrue((np.asarray(diff) < 0.001).all(),
                                    msg=f"path {i} not equal to path {j}")

    def test_SIRS(self):
        """Test to make sure SIRS collapses down to SIR"""
        self.parameters['ep'] = 0
        self.parameters['sa'] = 0
        self.parameters['iaa'] = 0
        SIR = pyross.deterministic.SIR(self.parameters, self.M, self.N)
        SIRS = pyross.deterministic.SIRS(self.parameters, self.M, self.N)
        SIRdata = SIR.simulate(self.N, np.ones(1), np.zeros(1),
                               self.contactMatrix, self.Tf,
                               self.Nf)['X']
        SIRSdata = SIRS.simulate(self.N, np.ones(1), np.zeros(1),
                                 self.contactMatrix, self.Tf,
                                 self.Nf)['X']
        self.assertTrue((SIRdata-SIRSdata[:, 0:3] < 0.001).all(),
                        msg="paths differ > .1%")


if __name__ == '__main__':
    unittest.main()