Bayesian inference
==================================
Inference for age structured compartment models using the diffusion approximation (via the van Kampen expansion).
See this `paper <https://arxiv.org/abs/2010.11783>`_ for more details on the method.

There are two ways to do inference: manifold method (sec 3.3 in the report) and tangent space method (sec 3.4 in the report).
In various degrees of `less robust but fast` to `more robust but slow`:

* tangent space method.
* manifold method with few internal steps and fast integration method (`det_method` = `RK2`, `lyapunov_method` = `euler`).
* manifold method with large number of internel steps and robust integration method (`solve_ivp` from scipy library).

============================= ==========================================================
Methods for full data
============================= ==========================================================
infer                         Infers epidemiological and control parameters given all information.
infer_mcmc                    Explore the posterior distribution given all information.
infer_nested_sampling         Compute the model evidence (and generate posterior samples) given all information.
obtain_minus_log_p            Computes -log(p) of a fully observed trajectory.
compute_hessian               Computes the Hessian of -log(p).
nested_sampling_inference     Compute the log-evidence and weighted samples.
============================= ==========================================================

================================ ===========================================================
Methods for partial data
================================ ===========================================================
latent_infer                     Infers epidemiological and control parameters and initial conditions.
latent_infer_mcmc                Explore the posterior distribution.
latent_infer_nested_sampling     Compute the model evidence (and generate posterior samples).
minus_logp_red                   Computes -log(p) of a partially observed trajectory.
compute_hessian_latent           Computes the Hessian of -log(p).
nested_sampling_latent_inference Compute the log-evidence and weighted samples.
================================ ===========================================================


======================= ===================================================================
Sensitivity analysis
======================= ===================================================================
FIM                     Computes the Fisher Information Matrix of the stochastic model.
FIM_det                 Computes the Fisher Information Matrix of the deterministic model.
sensitivity             Computes the normalized sensitivity measure
======================= ===================================================================

======================= ===========================================================
Helper function
======================= ===========================================================
integrate               A wrapper around 'simulate' in pyross.deterministic.
set_params              Sets parameters.
set_det_method          Sets the integration method of the deterministic equation
set_lyapunov_method     Sets the integration method of the Lyapunov equation
set_det_model           Sets the internal deterministic model
set_contact_matrix      Sets the contact matrix
fill_params_dict        Fills and returns a parameter dictionary
get_mean_inits          Constructs full initial conditions from the prior dict
======================= ===========================================================

The functions are documented under the parent class `SIR_type`.

SIR_type
----------------------------
.. autoclass:: pyross.inference.SIR_type
    :members:

Model
------------------------
.. autoclass:: pyross.inference.Model
    :members:
* `Link to example notebook <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex01b_Model.ipynb>`_

Spp
------------------------
.. autoclass:: pyross.inference.Spp
    :members:
* `Link to example notebook <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex02a_Spp_inference.ipynb>`_

SIR
------------------------
.. autoclass:: pyross.inference.SIR
    :members:

SEIR
-------------------------
.. autoclass:: pyross.inference.SEIR
    :members:

SEAIRQ
--------------------------
.. autoclass:: pyross.inference.SEAIRQ
    :members:

SppQ
------------------------
.. autoclass:: pyross.inference.SppQ
    :members:

