Evidence
========
Additional functions for computing the evidence of a pyross compartment model.

This is an extension of pyross.inference. 
Evidence computation via nested sampling is already directly implemented in the inference module.
However, for large-scale (high-dimensional) inference problems, nested sampling can become very slow.
In this module, we implement two additional ways to compute the evidence that work whenever the MCMC simulation of the posterior distribution is feasible.
See the `ex-evidence.ipynb notebook <https://github.com/rajeshrinet/pyross/blob/master/examples/inference/ex-evidence.ipynb>`_ for a code example of all ways to compute the evidence.


.. automodule:: pyross.evidence
    :members:
