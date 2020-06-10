Bayesian inference
==================================
Inference for age structured compartment models using the diffusion approximation (via the van Kampen expansion).
See our `preprint <https://arxiv.org/abs/2005.09625>`_ for more details on the method.

There are two ways to do inference: manifold method (sec 3.3 in the report) and tangent space method (sec 3.4 in the report).
Currently, all of the methods listed below are implemented for manifold, and `infer_parameters`, 'latent_infer_parameters',
'obtain_minus_log_p', 'minus_logp_red' are implemented for tangent space.

In various degrees of `less robust but fast` to `more robust but slow`:

* tangent space method.
* manifold method with few internal steps and fast integration method (`det_method` = `RK2`, `lyapunov_method` = `euler`).
* manifold method with large number of internel steps and robust integration method (`solve_ivp` from scipy library).

======================= ==========================================================
Methods for full data
======================= ==========================================================
infer_parameters        Infers epidemiological parameters given all information.
infer_control           Infers control parameters.
obtain_minus_log_p      Computes -log(p) of a fully observed trajectory.
compute_hessian         Computes the Hessian of -log(p).
======================= ==========================================================

========================== ===========================================================
Methods for partial data
========================== ===========================================================
latent_infer_parameters    Infers parameters and initial conditions.
latent_infer_control       Infers control parameters.
minus_logp_red             Computes -log(p) of a partially observed trajectory
compute_hessian_latent     Computes the Hessian of -log(p).
========================== ===========================================================

======================= ===========================================================
Helper function
======================= ===========================================================
integrate               A wrapper around 'simulate' in pyross.deterministic.
set_params              Sets parameters.
set_det_method          Sets the integration method of the deterministic equation
set_lyapunov_method     Sets the integration method of the Lyapunov equation
make_det_model          Makes a pyross.deterministic model of the same class
fill_params_dict        Fills and returns a parameter dictionary
fill_initial_conditions Generates full initial condition with partial info.
======================= ===========================================================

The functions are documented under the parent class `SIR_type`.

SIR_type
----------------------------
.. autoclass:: pyross.inference.SIR_type
    :members:

Spp
------------------------
.. autoclass:: pyross.inference.Spp
    :members:

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
