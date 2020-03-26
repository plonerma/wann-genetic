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
    from .metrics import ClassificationRecord as Performance

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

    def evaluate(self, env):
        if self.performance is None:
            self.performance = self.Performance(env.task.n_out, env=env)

        shared_weight = env['sampling', 'current_weight']

        if self.performance.has_weight(shared_weight):
            y_pred = self.apply(env.task.x, func='argmax', w=shared_weight)
            self.performance.stack_predictions(
                y_true=env.task.y_true, y_pred=y_pred, w=shared_weight)

    @property
    def fitness(self):
        return self.performance.get_metrics('avg_accuracy')

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
