Contact matrix
==================================
Classes and methods to compute contact matrix of a meta-population. The contact matrix :math:`C_{ij}` denotes the average number of contacts made per day by an individual in class :math:`i` with an individual in class :math:`j`. Clearly, the total number of contacts between group :math:`i` to group :math:`j` must equal the total number of contacts from group :math:`j` to group :math:`i`, and thus, for populations of fixed size the contact matrices obey the reciprocity relation :math:`N_{i}C_{ij}=N_{j}C_{ji}`. Here :math:`N_i` is the population in group :math:`i`. 

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


.. automethod:: pyross.contactMatrix.characterise_transient


