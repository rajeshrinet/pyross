#!python
"""Unittesting for the pyross module. Run as python -m unittest pyross.test."""
import sys
#remove pwd from path that tries to import .pyx files
for i in sys.path:
    if 'pyross' in i or i == '':
        sys.path.remove(i)
import pyross
import unittest
import inspect
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
        self.gI = 0.008
        self.gE = 0.007
        self.gIc = 0.1
        self.gIh = 0.1
        self.gA = 0
        self.tE = 0
        self.tIa = 0
        self.tIs = 0
        self.Tf = 100
        self.Nf = 1000
        self.fsa = 0
        self.fh = 1
        self.sa = 0
        self.iaa = 0
        self.hh = 0
        self.cc = 0
        self.mm = 0
        self.tE = 0
        self.tA = 0
        self.tIa = 0
        self.tIs = 0
        self.kI = 1
        self.kE = 1
        self.k = 1
        self.ep = 0
        self.parameters = {'N': self.N, 'M': self.M, 'alpha': self.alpha,
                              'beta': self.beta, 'gIa': self.gIa, 'gIs': self.gIs,
                              'gI': self.gI, 'iaa': self.iaa,
                              'gE': self.gE, 'gA': self.gA, 'tE': self.tE,
                              'gIc': self.gIc, 'gIh': self.gIh, 'fh': self.fh,
                              'tIa': self.tIa, 'tIs': self.tIs, 'fsa': self.fsa,
                              'sa': self.sa, 'hh': self.hh, 'cc': self.cc,
                              'mm': self.mm, 'tA': self.tA, 'tE': self.tE,
                              'tIa': self.tIa, 'tIs': self.tIs, 'kI': self.kI,
                              'kE': self.kE, 'ep': self.ep, 'k': self.k}

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


    def test_init_models(self):
        """Test initialisation of deterministic models"""
        deterministic_models = dict(inspect.getmembers(pyross.deterministic,
                                                       inspect.isclass))
        for name, model in deterministic_models.items():
            if name.startswith('S'):
                m = model(self.parameters, self.M, self.N)
    
    def test_run_models(self):
        """Runs all deterministic models"""
        deterministic_models = dict(inspect.getmembers(pyross.deterministic,
                                                       inspect.isclass))
        traj_dict={}
        for name, model in deterministic_models.items():
            if name.startswith('S'):
                m = model(self.parameters, self.M, self.N)
                x0 = np.array([*self.N, *np.ones(self.M), 
                               *np.zeros(m.nClass -2)], dtype=np.float64).reshape((m.nClass,1))
                traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)

if __name__ == '__main__':
    unittest.main()