This folder contains the core files of the PyRoss library.

Each file considers various [epidemiological models](https://github.com/rajeshrinet/pyross/tree/master/docs). The library is designed to be model-agnostic. Currently implemented [models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) are  **SIR**, **SIRS**, **SEIR**, **SEI5R**, **SIkR**, **SEkIkR**, **SEAIR**, and **SEAIRQ**. A short description of each file is provided below:


* **contactMatrix.pyx**: This file is to evaluate contact matrix of a given country. It also allows for a time-dependent lockdown strategy without any conotrol.  

* **control.pyx**: This file provides an option to control the contact matrix as a function of the state. 

* **deterministic.pyx**: This file is to obtain deterministic trajectories of any of the above models given parameters. 

* **stochastic.pyx**: This file is to obtain stochastic trajectories of of any of the above models given parameters. 

* **hybrid.pyx**: This file allows for an interface between deterministic and stochastic simulations of a given model given parameters. 

* **forecast.pyx**: This file allows for forecasting trajectories of any of the above model given the parameters and its uncertainty in terms of mean and variance. 

* **inference.pyx**: This file is for the Bayesian inference of parameters and latent varibles given trajectories for any of the above models. 

* **utils.pyx**: This file has miscellaneous functionalities


There are also .pxd files, which correspond to a .pyx file of same name. A .pxd file contains declaration of cdef classes, methods, etc. It is essential when calling PyRoss from another Cython file. Read more: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html. 

