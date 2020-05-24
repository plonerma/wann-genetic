Numpy Implementation of Signal Propagation in Neural Networks
=============================================================


Feed-Forward Networks
---------------------

The final activation state can be reached by multiplying the initial activation
vector (only input nodes and bias is set) with the weight matrix and applying
the activation functions. In the implementation only the columns of matrix are
used that correspond to the nodes that are currently calculated. The order of
calculation is determined by the layers (calculated during genes expression by
hierarchically sorting the hidden nodes). Additionaly, only the rows are used
that correspond to nodes in layers prior to the current layer. All other rows
have to contain zero.


.. note::

  During expression the nodes are topologically sorted. Therefore, the index of
  a node in the network does not correspond to the index of the same node in
  the gene.


.. figure:: _static/figure_stored_weights.svg
   :height: 600pt
   :alt: stored weight matrix

   Weight matrix of the network. Only the are marked with the green dashed
   lines are stored in memory all other values have to be zero and don't need
   to be stored explicitly (this is relevant when looking at the indexing in
   the implementation of the signal propagation algorithm).


Recurrent Networks
------------------

For recurrent networks, two weight matrices are generated. The first
corresponds to the weight matrix in a feed-forward network. It also has to be
acyclic and determines a topological sorting of the nodes. The second matrix
represents the recurrent connections. The recurrent weight matrix does not need
to be acyclic since the signal will not be used during forward feeding but in
the next step of element processing. Additionally, the recurrent weight matrix
can contain connections from output nodes to hidden nodes or output nodes.
