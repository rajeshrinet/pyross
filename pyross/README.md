This folder contains the core files of the PyRoss library.

Each file considers various EPIDEMIOLOGICAL MODELS as described in https://github.com/rajeshrinet/pyross/tree/master/docs

The main files are:
* contactMatrix.pyx - provides prescriitions to evaluate contact matrix and provide various lockdown strategies for a given country

* deterministic.pyx - deterministic simulations of various models

* stochastic.pyx - stochastic simulations of various models

* hybrid.pyx - switched between deterministic and stochastic simulations of various models

* forecast.pyx - forecasts given the parameters

* inference.pyx - Bayesian inference of parameters given trajectories for all the models of PyRoss

* utils.pyx - has miscellaneous functionalities


There are also .pxd files, which correspond to a .pyx file of same name. A .pxd file contains declaration of cdef classes, methods, etc. It is essential when calling PyRoss from another Cython file.
