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

    def __init__(self, genes=None, network=None, performance=None):
        self.genes = genes
        self.network = network
        self.performance = performance

    # Serialization

    def serialize(self, include=('genes', 'network', 'performance')):
        """Save all available representations.

        Returns dictionary which can be saved as a json file.
        """
        d = dict()
        if self.genes is not None and 'genes' in include:
            d['genes'] = self.genes.serialize()

        if self.network is not None and 'network' in include:
            d['network'] = self.network.serialize()

        if self.performance is not None and 'performance' in include:
            d['performance'] = self.performance

        return d

    @classmethod
    def deserialize(cls, d : dict):
        return cls(
            genes=cls.Genotype.deserialize(d.get('genes', None)),
            network=cls.Network.deserialize(d.get('network', None)),
            performance=d.get('performance', None)
        )


    # Translations

    def express(self):
        """Translate genes to network."""
        assert self.genes is not None
        self.network = self.Network.from_genes(self.genes)

    def evaluate(self, task):
        """Evaluate performance on task."""
        # If genes have not been translated to network yet, do that
        if self.network is None:
            self.express()
        pass
