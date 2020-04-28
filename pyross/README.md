This folder contains the core files of the PyRoss library.

Each file considers various [epidemiological models](https://github.com/rajeshrinet/pyross/tree/master/docs). The library is designed to be model-agnostic. Currently implemented [models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) are  **SIR**, **SIRS**, **SEIR**, **SEI5R**, **SIkR**, **SEkIkR**, **SEAIR**, and **SEAIRQ**. A short description of each file is provided below:


* contact matrix.pyx - evaluate contact matrix for a given country. It also allows for a time-dependent lockdown strategy.  

* control.pyx - provides an option to control the contact matrix as a function of the state. 

* deterministic.pyx - is for deterministic simulations of the above models

* stochastic.pyx - is for stochastic simulations of the models

* hybrid.pyx - allows for an interface between deterministic and stochastic simulations of the models

* forecast.pyx - forecasts trajectories for a given model given the parameters and uncertainty in terms of mean and variance

* inference.pyx - Bayesian inference of parameters given trajectories for all the above models

* utils.pyx - has miscellaneous functionalities


There are also .pxd files, which correspond to a .pyx file of same name. A .pxd file contains declaration of cdef classes, methods, etc. It is essential when calling PyRoss from another Cython file. Read more: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html. 

