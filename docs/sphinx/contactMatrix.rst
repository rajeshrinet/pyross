Contact matrix
==================================
Classes and methods to compute contact matrix of a meta-population. 

Contact Matrix Function
----------------------------------------------
Generates contact matrix for given interventions

.. autoclass:: pyross.contactMatrix.ContactMatrixFunction
    :members:

Spatial Contact Matrix
---------------------------------------------------
Approximates the spatial contact matrix given the locations, populations and areas of
the geographical regions and the overall age structured contact matrix.

.. autoclass:: pyross.contactMatrix.SpatialContactMatrix
    :members:

.. automethod:: pyross.contactMatrix.getCM 


.. automethod:: pyross.contactMatrix.characterise_transient


