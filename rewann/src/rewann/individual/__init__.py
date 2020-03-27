import numpy as np

class Individual:
    """ Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    genes = None
    network = None
    record = None

    from .genes import Genotype
    from .network import Network
    from .metrics import ClassificationRecord as Record

    from .genetic_operations import mutation

    def __init__(self, genes=None, network=None, record=None):
        self.genes = genes
        self.network = network
        self.record = record

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
        self.express()
        if self.record is None:
            self.record = self.Record(env.task.n_out, env=env)

        shared_weight = env['sampling', 'current_weight']

        if not self.record.has_weight(shared_weight):
            self.record.stack_predictions(
                y_true=env.task.y_true,
                y_pred=self.apply(env.task.x, func='argmax', w=shared_weight),
                w=shared_weight)

    @property
    def fitness(self):
        return self.record.get_metrics('avg_accuracy')

    @classmethod
    def base(cls, *args, **kwargs):
        return cls(genes=cls.Genotype.base(*args, **kwargs))

    # Serialization

    def serialize(self):
        d = dict(genes=self.genes.serialize())
        if self.record:
            d['record']=self.record.serialize()
        return d

    @classmethod
    def deserialize(cls, d : dict):
        d = dict(genes=cls.Genotype.deserialize(d['genes']))
        if 'record' in d:
            d['record'] = cls.Record.deserialize(d['record'])
        return cls(**d)
