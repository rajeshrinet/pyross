This folder contains the core files of the PyRoss library.

Each file considers various [epidemiological models](https://github.com/rajeshrinet/pyross/tree/master/docs). The library is designed to be model-agnostic. Currently implemented [models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) are  **SIR**, **SIRS**, **SEIR**, **SEI5R**, **SIkR**, **SEkIkR**, **SEAIR**, and **SEAIRQ**. A short description of each file is provided below:


* **contactMatrix.pyx** is to evaluate the contact matrix of a given country. It also allows for a time-dependent lockdown strategy without any control.

* **control.pyx** provides an option to control the contact matrix as a function of the state.

* **deterministic.pyx** is to obtain deterministic trajectories of any of the above models given parameters. 

* **stochastic.pyx** is to obtain stochastic trajectories of any of the above models given parameters.

* **hybrid.pyx**: is to allow for an interface to between deterministic and stochastic simulations of any of the above models given parameters.

* **forecast.pyx** allows for forecasting trajectories of any of the above model given the parameters and its uncertainty in terms of mean and variance. 

* **inference.pyx**  is for the Bayesian inference of parameters and latent variables given trajectories for any of the above models.

* **utils.pyx**: This file has miscellaneous functionalities


There are also .pxd files, which correspond to a .pyx file of same name. A .pxd file contains declaration of cdef classes, methods, etc. It is essential when calling PyRoss from another Cython file. Read more: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html. 

