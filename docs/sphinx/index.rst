PyRoss API
==================================
.. image:: ../../examples/others/banner.jpg
  :width: 800
  :alt: PyRoss banner

PyRoss is a numerical library for inference, prediction and non-pharmaceutical interventions in age-structured epidemiological compartment models. The library is designed to be model-agnostic and allows the user to define models using a Python dictionary.

           
The library supports models formulated stochastically (as chemical master equations) or deterministically (as systems of differential equations). Inference on pre-defined or user-defined models is performed using model-adapted Gaussian processes on the epidemiological manifold or its tangent space. This method allows for latent variable inference and fast computation of the model evidence and the Fisher information matrix. These estimates are convolved with the instrinsic stochasticty of the dynamics to provide Bayesian forecasts of the progress of the epidemic.


Installation
------------

From a checkout of PyRoss GitHub repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the recommended way as it downloads a whole suite of examples along with the package. 

Install PyRoss and an extended list of dependencies using
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    >> git clone https://github.com/rajeshrinet/pyross.git
    >> cd pyross
    >> pip install -r requirements.txt
    >> python setup.py install

Install PyRoss and an extended list of dependencies, via `Anaconda <https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html>`__, in an `environment <https://github.com/rajeshrinet/pyross/blob/master/environment.yml>`__ named ``pyross``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    >> git clone https://github.com/rajeshrinet/pyross.git
    >> cd pyross
    >> make env
    >> conda activate pyross
    >> make

Via pip
~~~~~~~

Install the latest `PyPI <https://pypi.org/project/pyross>`__ version

.. code:: bash

    >> pip install pyross

See also installation instructions and more details in the `README.md <https://github.com/rajeshrinet/pyross/blob/master/README.md>`_ on GitHub.

Tutorial examples
==========================

Please have a look at the `examples folder <https://github.com/rajeshrinet/pyross/tree/master/examples>`_ for Jupyter notebook examples on GitHub.


API Reference
=============

.. toctree::
   :maxdepth: 1

   deterministic
   stochastic
   hybrid
   inference
   control
   contactMatrix
   forecast
   evidence
   tsi 
