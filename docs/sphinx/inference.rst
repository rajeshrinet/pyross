Bayesian inference
==================================
Inference for age structured compartment models using the diffusion approximation (via the van Kampen expansion).

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
Automatic generation of inference for SIR-type models

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

SEAI5R
--------------------------
.. autoclass:: pyross.inference.SEAI5R
    :members:

SEAIRQ
--------------------------
.. autoclass:: pyross.inference.SEAIRQ
    :members:
