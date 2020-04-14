import numpy as np

# Tuple: https://stackoverflow.com/a/47534998
from typing import Tuple, NamedTuple

GeneEncoding = Tuple[Tuple[str, np.dtype], ...]


class Genotype:
    # data
    edges : np.array
    nodes : np.array

    n_in : int
    n_out : int

    edge_encoding : GeneEncoding = (
        ('id', np.dtype(int)), # innovation number
        ('src', np.dtype(int)), # node id
        ('dest', np.dtype(int)), # node id
    #    ('weight', np.dtype(int)), # WANN for now
        ('enabled', np.dtype(bool)), # node id
    )
    node_encoding : GeneEncoding = (
        ('id', np.dtype(int)),
        ('out', np.dtype(bool)), # don't store ins and bias as genes
        ('func', np.dtype(int)) # activation function
    )

    def node_out_factory(self, id):
        node = np.zeros(1, dtype=list(self.node_encoding))
        node['id'] = id
        node['out'] = True
        return node

    @property
    def n_static(self):
        """Nodes that will always be in the network (all but the hidden nodes)."""
        return self.n_in + self.n_out + 1 # bias

    def __init__(self, *, edges, nodes, n_in, n_out):
        self.edges=np.array(edges, dtype=list(self.edge_encoding))
        self.nodes=np.array(nodes, dtype=list(self.node_encoding))
        # sort entries by id
        self.nodes = self.nodes[np.argsort(self.nodes['id'])]

        # n_out == number of output nodes in genes
        assert np.sum(self.nodes['out']) == n_out
        # indices of output nodes are continous and start at right id
        assert (n_out == 0  or (
            np.all(self.nodes['id'][:n_out] == n_in + 1 + np.arange(n_out))
            and np.all(self.nodes['out'][:n_out])
        ))

        self.n_in = n_in
        self.n_out = n_out

    def __str__(self):
        return "\n".join([
            "=== Nodes ===",
            str(self.nodes),
            "",
            "=== Edges ===",
            str(self.edges)
        ])

    def __eq__(self, other):
        return (
                self.n_in == other.n_in
            and self.n_out == other.n_out
            and np.all(self.edges == other.edges)
            and np.all(self.nodes == other.nodes)
        )

    @classmethod
    def base(cls, n_in, n_out):
        """Create new base gene for given encodings."""
        # start without any edges
        edges = np.array([], dtype=list(cls.edge_encoding))

        # start only with output nodes (input and bias nodes are implicit)
        nodes = np.zeros(n_out, dtype=list(cls.node_encoding))
        nodes['out'] = True

        # reserve first n_in + 1 ids for implicit input and bias nodes
        nodes['id'] = np.arange(n_out) + n_in + 1

        return cls(edges=edges, nodes=nodes, n_in=n_in, n_out=n_out)

    @classmethod
    def full_initial(cls, n_in, n_out, prob_enabled=1):
        """Create new base gene with all input nodes connected to the output nodes."""
        # connect all input (and bias) nodes to all output nodes
        n_edges = (n_in+1)*n_out

        edges = np.zeros(n_edges, dtype=list(cls.edge_encoding))
        edges['id'] = np.arange(n_edges)
        edges['src'] = np.tile(np.arange(n_in+1), n_out)
        edges['dest'] = np.repeat(np.arange(n_out), n_in+1)
        edges['enabled'] = np.random.rand(n_edges) < prob_enabled

        # start only with output nodes (input and bias nodes are implicit)
        nodes = np.zeros(n_out, dtype=list(cls.node_encoding))
        nodes['out'] = True

        # reserve first n_in + 1 ids for implicit input and bias nodes
        nodes['id'] = np.arange(n_out) + n_in + 1

        return cls(edges=edges, nodes=nodes, n_in=n_in, n_out=n_out)

    def copy(self):
        return self.__class__(
            edges=np.copy(self.edges), nodes=np.copy(self.nodes),
            n_in=self.n_in, n_out=self.n_out
        )
