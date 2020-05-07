#!python
"""Unittesting for the pyross module. Run as python -m unittest pyross.test."""
import sys
#remove pwd from path that tries to import .pyx files
for i in sys.path:
    if 'pyross' in i or i == '':
        sys.path.remove(i)
# print(sys.path)
import pyross
import unittest
import inspect
import numpy as np
import scipy as sp


class DeterministicTest(unittest.TestCase):
    """testing deterministic.pyx."""
    N = np.asarray([10000], dtype=np.float64)
    M = 1
    alpha = 0
    beta = 0.0071
    gIa = 0.008
    gIs = 0.008
    gI = 0.008
    gE = 0.007
    gIc = 0.1
    gIh = 0.1
    gA = 0
    tE = 0
    tIa = 0
    tIs = 0
    Tf = 100
    Nf = 1000
    fsa = 0
    fh = 1
    sa = 0
    iaa = 0
    hh = 0
    cc = 0
    mm = 0
    tE = 0
    tA = 0
    tIa = 0
    tIs = 0
    kI = 1
    kE = 1
    k = 1
    ep = 0
    parameters = {'N': N, 'M': M, 'alpha': alpha,
                          'beta': beta, 'gIa': gIa, 'gIs': gIs,
                          'gI': gI, 'iaa': iaa,
                          'gE': gE, 'gA': gA, 'tE': tE,
                          'gIc': gIc, 'gIh': gIh, 'fh': fh,
                          'tIa': tIa, 'tIs': tIs, 'fsa': fsa,
                          'sa': sa, 'hh': hh, 'cc': cc,
                          'mm': mm, 'tA': tA, 'tE': tE,
                          'tIa': tIa, 'tIs': tIs, 'kI': kI,
                          'kE': kE, 'ep': ep, 'k': k}


    def __init__(self, *args, **kwargs):
        super(DeterministicTest, self).__init__(*args, **kwargs)
        # self.parameters = self.parameters

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


class StochasticTest(unittest.TestCase):
    """testing stochastic.pyx"""

    def __init__(self, *args, **kwargs):
        super(StochasticTest, self).__init__(*args, **kwargs)
        self.parameters = DeterministicTest.parameters

    def contactMatrix(self, t): return np.identity(self.parameters['M'])

    def test_init_models(self):
        """Initializes all stochastic models"""
        stochastic_models = dict(inspect.getmembers(pyross.stochastic,
                                                        inspect.isclass))
        traj_dict={}
        for name, model in stochastic_models.items():
            if name.startswith('S'):
                params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
                m = model(params, M, N)
                # x0 = np.array([*self.N, *np.ones(self.M),
                #                 *np.zeros(m.nClass -2)], dtype=np.float64).reshape((m.nClass,1))
                # traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)

    def test_run_models(self):
       """Runs all stochastic models"""
       stochastic_models = dict(inspect.getmembers(pyross.stochastic,
                                                       inspect.isclass))
       traj_dict={}
       for name, model in stochastic_models.items():
           if name.startswith('S'):
               params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
               m = model(params, M, N + M*10)
               x0 = np.array([*self.parameters['N'],
                              *np.ones(self.parameters['M'])*10,
                              *np.zeros(m.nClass -2)],
                             dtype=np.float64).reshape((m.nClass,1))
               traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)
    
    def test_stochastic_mean(self):
        """Runs stochastic models a few times and compares mean to 
        deterministic"""
        


if __name__ == '__main__':
    unittest.main()
