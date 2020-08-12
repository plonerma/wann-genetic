import numpy as np

# Tuple: https://stackoverflow.com/a/47534998
from typing import Tuple, NamedTuple

GeneEncoding = Tuple[Tuple[str, np.dtype], ...]


class Genes:
    """Genetic encoding of Feed Forward Networks."""
    # data
    edges: np.array
    nodes: np.array

    n_in: int
    n_out: int

    edge_encoding: GeneEncoding = (
        # innovation number
        ('id', np.dtype(int)),

         # id of source node (any but output)
        ('src', np.dtype(int)),

        # id of destination node (either hidden or output)
        ('dest', np.dtype(int)),

        # in {-1,+1} if negative signs allowed, else 1
        ('sign', np.dtype(int)),

        # sign needs to be retained even if disabled, in case the edge is
        # reenabled
        ('enabled', np.dtype(bool))
    )
    node_encoding: GeneEncoding = (
        ('id', np.dtype(int)),

        # input and bias nodes are not stored in genes, since no activation
        # function is required (ids are still reserved for these nodes)
        ('out', np.dtype(bool)),

        # int representation of activation function
        ('func', np.dtype(int))
    )

    def node_out_factory(self, id):
        node = np.zeros(1, dtype=list(self.node_encoding))
        node['id'] = id
        node['out'] = True
        return node

    @property
    def n_static(self):
        """Nodes that will always be in the network (all but the hidden nodes)."""
        return self.n_in + self.n_out + 1  # bias

    def __init__(self, *, edges, nodes, n_in, n_out):
        self.edges = np.array(edges, dtype=list(self.edge_encoding))
        self.nodes = np.array(nodes, dtype=list(self.node_encoding))
        # sort entries by id
        self.nodes = self.nodes[np.argsort(self.nodes['id'])]

        # n_out == number of output nodes in genes
        assert np.sum(self.nodes['out']) == n_out
        # indices of output nodes are continous and start at right id
        assert (n_out == 0 or (
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
    def empty_initial(cls, n_in, n_out, n_funcs, initial_func='random'):
        """Create new base gene for given encodings."""
        # start without any edges
        edges = np.array([], dtype=list(cls.edge_encoding))

        # start only with output nodes (input and bias nodes are implicit)
        nodes = np.zeros(n_out, dtype=list(cls.node_encoding))
        nodes['out'] = True
        nodes['func'] = np.random.randint(n_funcs, size=n_out) if initial_func == 'random' else initial_func

        # reserve first n_in + 1 ids for implicit input and bias nodes
        nodes['id'] = np.arange(n_out) + n_in + 1

        return cls(edges=edges, nodes=nodes, n_in=n_in, n_out=n_out)

    @classmethod
    def full_initial(cls, n_in, n_out, n_funcs, prob_enabled=1, negative_edges_allowed=False, initial_func='random'):
        """Create new base gene with all input nodes connected to the output nodes."""
        # connect all input (and bias) nodes to all output nodes
        n_edges = (n_in+1)*n_out

        edges = np.zeros(n_edges, dtype=list(cls.edge_encoding))

        edges['id'] = 0  # initial edges dont keep an id

        edges['src'] = np.tile(np.arange(n_in+1), n_out)
        edges['dest'] = np.repeat(n_in+1 + np.arange(n_out), n_in+1)

        edges['enabled'] = np.random.rand(n_edges) < prob_enabled

        if negative_edges_allowed:
            edges['sign'] = np.random.choice([-1,+1], n_edges)
        else:
            edges['sign'] = 1

        # start only with output nodes (input and bias nodes are implicit)
        nodes = np.zeros(n_out, dtype=list(cls.node_encoding))

        nodes['func'] = np.random.randint(n_funcs, size=n_out) if initial_func == 'random' else initial_func

        nodes['out'] = True

        # reserve first n_in + 1 ids for implicit input and bias nodes
        nodes['id'] = np.arange(n_out) + n_in + 1

        return cls(edges=edges, nodes=nodes, n_in=n_in, n_out=n_out)

    def copy(self):
        return self.__class__(
            edges=np.copy(self.edges), nodes=np.copy(self.nodes),
            n_in=self.n_in, n_out=self.n_out
        )


class RecurrentGenes(Genes):
    """Genetic encoding of Recurrent Networks."""
    edge_encoding: GeneEncoding = (
        # innovation number
        ('id', np.dtype(int)),

        # id of source node (any but output)
        ('src', np.dtype(int)),

        # id of destination node (either hidden or output)
        ('dest', np.dtype(int)),

        # in {-1,+1} if negative signs allowed, else 1
        ('sign', np.dtype(int)),

        # sign needs to be retained even if disabled, in case the edge is
        # reenabled
        ('enabled', np.dtype(bool)),

        ('recurrent', np.dtype(bool)),
    )
