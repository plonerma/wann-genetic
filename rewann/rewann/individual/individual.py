import numpy as np

class Individual:
    """ Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    genes = None
    network = None
    performance = None

    from .genes import Genotype
    from .network import Network
    from ..tasks import Performance

    from .genetic_operations import mutation, crossover

    def __init__(self, genes=None, network=None, performance=None):
        self.genes = genes
        self.network = network
        self.performance = performance

    # Translations

    def express(self):
        """Translate genes to network."""
        assert self.genes is not None
        if self.network is None:
            self.network = self.Network.from_genes(self.genes)

    def apply(self, *args, **keargs):
        if self.network is None:
            self.express()
        return self.network.apply(*args, **keargs)

    def evaluation(self, task):
        if self.performance is None:
            self.performance = task.evaluate_individual(self)
        return self.performance

    @classmethod
    def base(cls, *args, **kwargs):
        return cls(genes=cls.Genotype.base(*args, **kwargs))

    # Serialization

    def serialize(self):
        return self.genes.serialize()

    @classmethod
    def deserialize(cls, d : dict):
        return cls(
            genes=cls.Genotype.deserialize(d),
        )
