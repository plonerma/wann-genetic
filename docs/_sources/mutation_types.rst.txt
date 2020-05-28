Mutation Types
==============

This document describes some of the possible mutation types. Check doc:`params`
for implemented types and how to configure them.


New Edge
---------

The `New Edge` mutation adds a new edge to the genes. There are two
implementations for this mutation: `layer based` and `layer agnostic`.


New Node
---------

In this mutation an existing enabled add is picked. A new node is added to the
genes and is connected to the source and target node of the picked edge
(maintaining the signal flow in the network). The picked egdge is the disabled.


Reenable Edge
--------------

A random disabled edge is selected an reenabled.


Change Activation
------------------

The activation function of a randomly picked node is changed.


Disable Edge
--------------

.. warning::

  This mutation type has not been implemented yet.

Enabling the genetic algorithm to disable edges could help producing prune
networks.

.. note::

  If edges are disabled that connect a source node with no other outgoing edges
  or a target node with no other incoming edges, the resulting network will
  contain dead ends. So either, affected nodes are also removed (this could
  lead to larger implications for the network) or this possibility is excluded.

.. note::

  It might make sense to prefer disabling edges that span across multiple
  layers since this could potentially reduce the maximum size of the activation
  vector (see :doc:`torch_network`).

  At the moment layers are arranged by topologically sorting the nodes starting
  at input layer. This ordering essentially produces an ordering for earliest
  possible computation. Vice-versa, an ordering starting with the output layer
  would produce layers sorted by latest possible computation.

  The latter approach would lead to less required memory in the
  :doc:`torch_network`, since the output of nodes will only be concatenated
  once it has actually been computed.

  A more efficient implementation might order the nodes so that a minimum size
  of memory is required. If the two or more input nodes of a node `k` are
  only required by node `k`, then calculating node `k` will reduce the size of
  the activation vector. Ordering the nodes optimally is not a trivial task
  and might take more time than the actual computation (if only few
  sample/weight pairs are to be calculated and the network does not enter the
  elite).


Change Edge Sign
-----------------

The sign of a randomly selected enabled edge is changed.

.. seealso::

  This mutation type is only enabled when corresponding parameters are set (see
  :doc:`params`). If this option is disabled, all edges will have a positive
  sign.


Add Recurrent Edge
-------------------

A new recurrent edge is added to the genes. Any two nodes can be connected as
long as the same connection does not already exist.
