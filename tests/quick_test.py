#!python
"""Unittesting for the pyross module. Run as python -m unittest pyross.test."""
import sys
#remove pwd from path that tries to import .pyx files
# for i in sys.path:
#     if 'pyross' in i or i == '':
#         sys.path.remove(i)
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
    gIhp= 0.1
    gIsp= 0.1
    gIcp= 0.1
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
                          'gIsp':gIsp,'gIhp':gIhp,'gIcp':gIcp,
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
            try:
                data = model.simulate(np.zeros(1), np.zeros(1), self.N,
                                      self.contactMatrix, self.Tf,
                                      self.Nf, integrator=integrator)
            except ImportError:
                print(f"{integrator} is not installed, skipping...")
                pass
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
            if name.startswith('S') and not 'Spp':
                m = model(self.parameters, self.M, self.N)

    def test_run_models(self):
        """Runs all deterministic models"""
        deterministic_models = dict(inspect.getmembers(pyross.deterministic,
                                                       inspect.isclass))
        traj_dict={}
        for name, model in deterministic_models.items():
            if name.startswith('S') and not 'Spp':
                m = model(self.parameters, self.M, self.N)
                x0 = np.array([*self.N, *np.ones(self.M),
                               *np.zeros(m.nClass -2)], dtype=np.float64).reshape((m.nClass,1))
                traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)


class StochasticTest(unittest.TestCase):
    """testing stochastic.pyx"""
    nloops=10
    iinfec = 3000
    Tf = 10

    def __init__(self, *args, **kwargs):
        super(StochasticTest, self).__init__(*args, **kwargs)
        self.parameters = DeterministicTest.parameters
        self.stochastic_models = dict(inspect.getmembers(pyross.stochastic,
                                                       inspect.isclass))

    def contactMatrix(self, t): return np.identity(self.parameters['M'])

    def test_init_models(self):
        """Initializes all stochastic models"""
        traj_dict={}
        for name, model in self.stochastic_models.items():
            if name.startswith('S'):
                params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
                m = model(params, M, N)
                # x0 = np.array([*self.N, *np.ones(self.M),
                #                 *np.zeros(m.nClass -2)], dtype=np.float64).reshape((m.nClass,1))
                # traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)

    def test_run_models(self):
       """Runs all stochastic models"""
       traj_dict={}
       for name, model in self.stochastic_models.items():
           
           if name.startswith('S'):
               params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
               m = model(params, M, N + M*10)
               x0 = np.array([*self.parameters['N'],
                              *np.ones(self.parameters['M'])*10,
                              *np.zeros(m.nClass -2)],
                             dtype=np.float64).reshape((m.nClass,1))
               traj_dict[name] = m.simulate(*x0, self.contactMatrix, 100, 100)
    
    def test_stochastic_mean_gillespie(self):
        """Runs stochastic models a few times and compares mean to 
        deterministic"""
        deterministic_models = dict(inspect.getmembers(pyross.deterministic,
                                                        inspect.isclass))
        params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
        for name, model in self.stochastic_models.items():
            if name.startswith('S'):
                mS = model(params, M, N + M*self.iinfec)
                # print(mS.kk)
                mD = deterministic_models[name](params, M, N + M*self.iinfec)
                x0 = np.array([*self.parameters['N'],
                              *np.ones(self.parameters['M'])*self.iinfec,
                              *np.zeros(mS.nClass -2)],
                              dtype=np.float64).reshape((mS.nClass,1))
                trajectories = []
                for i in range(self.nloops):
                    traj =  mS.simulate(*x0, self.contactMatrix, self.Tf, self.Tf)['X']
                    trajectories.append(traj)
                traj_mean = np.mean(trajectories, axis=0)[:-1]
                mean = mD.simulate(*x0, self.contactMatrix, self.Tf, self.Tf)['X']
                absdiff = np.abs(traj_mean -mean)/(N*self.Tf)
                # print(name, np.sum(absdiff[:,:-1]))
                self.assertTrue(np.sum(absdiff[:,:-1])<0.01, 
                                msg=f"{name} model disagreement")

    def test_stochastic_mean_tau(self):
        """Runs stochastic models a few times and compares mean to 
            deterministic using tau leaping"""
        deterministic_models = dict(inspect.getmembers(pyross.deterministic,
                                                        inspect.isclass))
        params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
        for name, model in self.stochastic_models.items():
            if name.startswith('S'):
                mS = model(params, M, N + M*self.iinfec)
                # print(mS.kk)
                mD = deterministic_models[name](params, M, N + M*self.iinfec)
                x0 = np.array([*self.parameters['N'],
                              *np.ones(self.parameters['M'])*self.iinfec,
                              *np.zeros(mS.nClass -2)],
                              dtype=np.float64).reshape((mS.nClass,1))
                trajectories = []
                for i in range(self.nloops):
                    traj =  mS.simulate(*x0, self.contactMatrix, self.Tf, self.Tf,
                                        method='tau_leaping')['X']
                    trajectories.append(traj)
                traj_mean = np.mean(trajectories, axis=0)[:-1]
                mean = mD.simulate(*x0, self.contactMatrix, self.Tf, self.Tf)['X']
                absdiff = np.abs(traj_mean -mean)/(N*self.Tf)
                # print(name, np.sum(absdiff[:,:-1]))
                self.assertTrue(np.sum(absdiff[:,:-1])<0.01, 
                                msg=f"{name} model disagreement")

    def test_stochastic_integrators(self):
        """Compare tau leaping to Gillespie.
            This will fail because there is a problem with SIkR
            Also, difference is an order of magnitude greater than
            Gillespie from the mean.
        """
        self.nloops=10
        params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
        for name, model in self.stochastic_models.items():
            if name.startswith('S'):
                mS = model(params, M, N + M*self.iinfec)
                x0 = np.array([*self.parameters['N'],
                              *np.ones(self.parameters['M'])*self.iinfec,
                              *np.zeros(mS.nClass -2)],
                              dtype=np.float64).reshape((mS.nClass,1))
                gtraj = []
                tautraj = []
                for i in range(self.nloops):
                    gtraj.append(mS.simulate(*x0, self.contactMatrix, self.Tf, self.Tf, 
                                        method='gillespie')['X'])
                    tautraj.append(mS.simulate(*x0, self.contactMatrix, self.Tf, self.Tf, 
                                        method='tau_leaping', epsilon=1E-3)['X'])
                gmean = np.sum(gtraj, axis=0)
                taumean= np.sum(tautraj, axis=0)
                absdiff = np.abs(gmean - taumean)/(N*self.Tf)
                # print(name, np.sum(absdiff), np.shape(gmean), np.shape(taumean))
                self.assertTrue(np.sum(absdiff)<.1, msg=f"{name} model disagreement")


class ControlTest(unittest.TestCase):
    """testing control.pyx"""
    
    def __init__(self, *args, **kwargs):
        super(ControlTest, self).__init__(*args, **kwargs)
        self.parameters = DeterministicTest.parameters
        self.control_models = dict(inspect.getmembers(pyross.control,
                                                       inspect.isclass))

    def contactMatrix(self, t): return np.identity(self.parameters['M'])

    def test_init_models(self):
        """Initializes all control models"""
        for name, model in self.control_models.items():
            if name.startswith('S'):
                params, M, N = self.parameters, self.parameters['M'], self.parameters['N']
                m = model(params, M, N)


class InferenceTest(unittest.TestCase):
    """testing inference.pyx"""
    
    def __init__(self, *args, **kwargs):
        super(InferenceTest, self).__init__(*args, **kwargs)
        self.parameters = DeterministicTest.parameters
        self.control_models = dict(inspect.getmembers(pyross.inference,
                                                       inspect.isclass))
        
    def contactMatrix(self, t): return np.identity(self.parameters['M'])

    def test_init_models(self):
        """Initializes all inference models"""
        for name, model in self.control_models.items():
            if name.startswith('S') and name != "SIR_type":
                params, M, Ni = self.parameters, self.parameters['M'], self.parameters['N']
                N = int(np.sum(Ni))
                fi = Ni/N
                steps = 1
                m = model(params, M, fi, N, steps)


class ForecastTest(unittest.TestCase):
    """testing forcast.pyx"""
    
    def __init__(self, *args, **kwargs):
        super(ForecastTest, self).__init__(*args, **kwargs)
        self.parameters = DeterministicTest.parameters
        self.control_models = dict(inspect.getmembers(pyross.forecast,
                                                       inspect.isclass))
        self.parameters['cov'] = np.identity(2)
        
    def contactMatrix(self, t): return np.identity(self.parameters['M'])

    def test_init_models(self):
        """Initializes all forcast models"""
        for name, model in self.control_models.items():
            if name.startswith('S') and name != "SIR_type":
                params, M, Ni = self.parameters, self.parameters['M'], self.parameters['N']
                N = int(np.sum(Ni))
                fi = Ni/N
                steps = 1
                m = model(params, M, Ni)


class UtilsPythonTest(unittest.TestCase):
    """Testing the minimization function in utils_python.py"""

    def __init__(self, *args, **kwargs):
        super(UtilsPythonTest, self).__init__(*args, **kwargs)

    def test_minimization(self):
        """Test the minimization(...) function in utils_python.py with a few simple examples"""

        # A simple example
        f1 = lambda x, grad=0: 1 + np.linalg.norm(x)**2  
        # A multi-modal example
        f2 = lambda x, grad=0: 1 + np.linalg.norm(x)**2 + 0.1*np.abs(np.sin(4*np.pi*np.linalg.norm(x)))

        # Test global optimisation
        guess = np.array([1.0, 1.0])
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        x, y = pyross.utils_python.minimization(f1, guess, bounds, enable_global=True, enable_local=False,
                                                ftol=1e-4, cma_random_seed=1, verbose=False)
        self.assertTrue(np.abs(y - 1.0) < 1e-3)

        x, y = pyross.utils_python.minimization(f2, guess, bounds, enable_global=True, enable_local=False,
                                                ftol=1e-4, verbose=False, cma_random_seed=2)
        self.assertTrue(np.abs(y - 1.0) < 1e-3)

        # Test local optimisation
        guess = np.array([2.0, 2.0])
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        x, y = pyross.utils_python.minimization(f1, guess, bounds, enable_global=False, enable_local=True,
                                                ftol=1e-5, verbose=False)
        self.assertTrue(np.abs(y - 1.0) < 1e-4)

        # And now combined
        x, y = pyross.utils_python.minimization(f2, guess, bounds, enable_global=True, enable_local=True,
                                        ftol=1e-5, global_ftol_factor=100, verbose=False, cma_random_seed=4)
        self.assertTrue(np.abs(y - 1.0) < 1e-4)


if __name__ == '__main__':
    unittest.main()
