:tocdepth: 2

PyRoss API
==================================


PyRoss is a numerical library for inference, prediction and non-pharmaceutical interventions in age-structured epidemiological compartment models.

The library is designed to be model-agnostic and allows the user to define models in a JSON format. The most common epidemiological models, and several less common ones, come pre-defined with the library. Models can include stages to allow for non-exponentially distributed compartmental residence times. Currently, pre-defined models include ones with multiple infectious (symptomatic, asymptomatic etc) and non-infectious (exposed, asymptomatic etc) classes.

The library supports models formulated stochastically (as chemical master equations) or deterministically (as systems of differential equations). A hybrid algorithm transits dynamically between these depending on the magnitude of the compartmental fluctuations.

Inference on pre-defined or user-defined models is performed using model-adapted Gaussian processes on the epidemiological manifold or its tangent space. This method allows for latent variable inference and fast computation of the model evidence and the Fisher information matrix. These estimates are convolved with the instrinsic stochasticty of the dynamics to provide Bayesian forecasts of the progress of the epidemic.

Non-pharmaceutical interventions are implemented as modifications of the contact structures of the model. Optimised control of these structures, given cost functions, is possible. This feature is being actively developed to be better integrated with the library.

.. image:: ../../examples/banner.png
  :width: 800
  :alt: PyRoss banner

Worked out examples
==========================

* `Example: Deterministic sampling in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/deterministic/ex01-SIR.ipynb>`_
* `Example: Stochastic sampling in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/stochastic/ex1-SIR.ipynb>`_
* `Example: Inference in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex01_inference_SIR.ipynb>`_
* `Example: Inference with latent variables in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex05_inference_latent_SIR.ipynb>`_
* `Example: Simulate any generic compartmental model in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/deterministic/ex16-Spp.ipynb>`_
* `Example: Inference with any generic compartment model in PyRoss <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex_Spp.ipynb>`_

In addition please have a look at the `examples folder <https://github.com/rajeshrinet/pyross/tree/master/examples>`_ for more Jupyter notebook examples.

The examples are classified as:

* `contactMatrix <https://github.com/rajeshrinet/pyross/tree/master/examples/contactMatrix>`_ : shows how to use contact matrix and intervention
* `control <https://github.com/rajeshrinet/pyross/tree/master/examples/control>`_ : shows how to compute time dependent contact matrix which depend of time and state
* `deterministic <https://github.com/rajeshrinet/pyross/tree/master/examples/deterministic>`_ : is for integration of equations of motion in the limit of no stochastic components
* `forecast <https://github.com/rajeshrinet/pyross/tree/master/examples/forecast>`_ : is for forecasting once the parameters are known
* `hybrid <https://github.com/rajeshrinet/pyross/tree/master/examples/hybrid>`_ : is for integration of equations of motion which can switch from deterministic to stochastic
* `inference <https://github.com/rajeshrinet/pyross/tree/master/examples/inference>`_ : shows how to infer parameters and select models given data
* `stochastic <https://github.com/rajeshrinet/pyross/tree/master/examples/stochastic>`_ : is for integration of equations of motion with stochastic components


`Models.pdf <https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf>`_ has a description of the various epidemiological models used in the examples (SIR, SIkR, SEIR, SEkIkR, SEAIR, SEAI5R, etc).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. contents:

.. autoclass:: pyross
    :no-members:
    :no-inherited-members:
    :no-special-members:

Deterministic simulations
==================================

Deterministic simulations with compartment models and age structure.

pyross.deterministic.IntegratorsClass
----------------------------------------

.. autoclass:: pyross.deterministic.IntegratorsClass
    :members:


pyross.deterministic.Spp
--------------------------
Automatic generation of all SIR-type models.

.. autoclass:: pyross.deterministic.Spp
    :members:

pyross.deterministic.SIR
--------------------------
.. autoclass:: pyross.deterministic.SIR
    :members:

pyross.deterministic.SIkR
--------------------------
.. autoclass:: pyross.deterministic.SIkR
    :members:

pyross.deterministic.SEIR
--------------------------
.. autoclass:: pyross.deterministic.SEIR
    :members:

pyross.deterministic.SEkIkR
-----------------------------
.. autoclass:: pyross.deterministic.SEkIkR
    :members:

pyross.deterministic.SEkIkIkR
-------------------------------
.. autoclass:: pyross.deterministic.SEkIkIkR
    :members:

pyross.deterministic.SEI5R
-------------------------------
.. autoclass:: pyross.deterministic.SEI5R
    :members:


pyross.deterministic.SEI8R
-------------------------------
.. autoclass:: pyross.deterministic.SEI8R
    :members:

pyross.deterministic.SEAIR
----------------------------
.. autoclass:: pyross.deterministic.SEAIR
    :members:


pyross.deterministic.SEAI5R
----------------------------
.. autoclass:: pyross.deterministic.SEAI5R
    :members:

pyross.deterministic.SEAI8R
--------------------------------
.. autoclass:: pyross.deterministic.SEAI8R
    :members:

pyross.deterministic.SEAIRQ
------------------------------
.. autoclass:: pyross.deterministic.SEAIRQ
    :members:

pyross.deterministic.SEAIRQ_testing
--------------------------------------
.. autoclass:: pyross.deterministic.SEAIRQ_testing
    :members:



Stochastic simulations
==================================
Stochastic simulations with compartment models and age structure. Has Gillespie and tau-leaping implemented.


pyross.stochastic.stochastic_integration
--------------------------------------------
.. autoclass:: pyross.stochastic.stochastic_integration
    :members:

pyross.stochastic.SIR
----------------------
.. autoclass:: pyross.stochastic.SIR
    :members:

pyross.stochastic.SIkR
------------------------
.. autoclass:: pyross.stochastic.SIkR
    :members:

pyross.stochastic.SEIR
-------------------------
.. autoclass:: pyross.stochastic.SEIR
    :members:

pyross.stochastic.SEI5R
--------------------------
.. autoclass:: pyross.stochastic.SEI5R
    :members:

pyross.stochastic.SEAI5R
----------------------------
.. autoclass:: pyross.stochastic.SEAI5R
    :members:

pyross.stochastic.SEAIRQ
---------------------------
.. autoclass:: pyross.stochastic.SEAIRQ
    :members:

pyross.stochastic.SEAIRQ_testing
-------------------------------------
.. autoclass:: pyross.stochastic.SEAIRQ_testing
    :members:


Hybrid simulations
==================================
Hybrid simulation scheme using a combination of stochastic and determinisitic schemes.

pyross.hybrid.SIR
--------------------
.. autoclass:: pyross.hybrid.SIR
    :members:


Bayesian inference
==================================
Inference for age structured compartment models using the diffusion approximation (via the van Kampen expansion).

pyross.inference.SIR_type
----------------------------
.. autoclass:: pyross.inference.SIR_type
    :members:

pyross.inference.Spp
------------------------
Automatic generation of inference for SIR-type models

.. autoclass:: pyross.inference.Spp
    :members:

pyross.inference.SIR
------------------------
.. autoclass:: pyross.inference.SIR
    :members:

pyross.inference.SEIR
-------------------------
.. autoclass:: pyross.inference.SEIR
    :members:

pyross.inference.SEAI5R
--------------------------
.. autoclass:: pyross.inference.SEAI5R
    :members:

pyross.inference.SEAIRQ
--------------------------
.. autoclass:: pyross.inference.SEAIRQ
    :members:

Control with Non-Pharmaceutical interventions
=================================================


pyross.control.control_integration
------------------------------------
.. autoclass:: pyross.control.control_integration
    :members:

pyross.control.SIR
------------------------------------
.. autoclass:: pyross.control.SIR
    :members:

pyross.control.SEkIkIkR
------------------------------------
.. autoclass:: pyross.control.SEkIkIkR
    :members:

pyross.control.SIRS
---------------------
.. autoclass:: pyross.control.SIRS
    :members:

pyross.control.SEIR
---------------------
.. autoclass:: pyross.control.SEIR
    :members:

pyross.control.SEI5R
----------------------
.. autoclass:: pyross.control.SEI5R
    :members:

pyross.control.SIkR
----------------------
.. autoclass:: pyross.control.SIkR
    :members:

pyross.control.SEkIkR
------------------------
.. autoclass:: pyross.control.SEkIkR
    :members:


pyross.control.SEAIR
-----------------------
.. autoclass:: pyross.control.SEAIR
    :members:

pyross.control.SEAI5R
-----------------------
.. autoclass:: pyross.control.SEAI5R
    :members:

pyross.control.SEAIRQ
-----------------------
.. autoclass:: pyross.control.SEAIRQ
    :members:


Forecasting
==================================
Forecasting with the inferred parameters, error bars and, if there are latent variables, inferred initial conditions.


pyross.forecast.SIR
---------------------
.. autoclass:: pyross.forecast.SIR
    :members:

pyross.forecast.SIR_latent
------------------------------
.. autoclass:: pyross.forecast.SIR_latent
    :members:

pyross.forecast.SEIR
---------------------
.. autoclass:: pyross.forecast.SEIR
    :members:

pyross.forecast.SEIR_latent
--------------------------------
.. autoclass:: pyross.forecast.SEIR_latent
    :members:

pyross.forecast.SEAIRQ
----------------------------
.. autoclass:: pyross.forecast.SEAIRQ
    :members:

pyross.forecast.SEAIRQ_latent
------------------------------------
.. autoclass:: pyross.forecast.SEAIRQ_latent
    :members:

pyross.forecast.SEAI5R
----------------------------
.. autoclass:: pyross.forecast.SEAI5R
    :members:

pyross.forecast.SEAI5R_latent
----------------------------------
.. autoclass:: pyross.forecast.SEAI5R_latent
    :members:


Contact matrix class
==================================
Generates contact matrix for given interventions

pyross.contactMatrix.ContactMatrixFunction
----------------------------------------------
.. autoclass:: pyross.contactMatrix.ContactMatrixFunction
    :members:
