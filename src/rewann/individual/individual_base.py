import numpy as np

from .util import num_used_activation_functions

import logging


class IndividualBase:
    """Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    from .genes import Genes as Genotype

    def __init__(self, genes=None, network=None,
                 id=None, birth=None, parent=None, mutations=0):
        """Initialize individual.

        Parameters
        ----------
        genes : rewann.Genes
        network : rewann.Network, optional
        id : int, optional
        birth: int, optional
            Index of the generation the individual was created
        parent : int, optional
            Id of the parent individual (self is result of a mutation on parent)
        mutations : int, optional
            Length of the chain of mutations that led to this individual
        measurements : dict
            Measurements that have already been made on the individual (might be overwritten)

        """
        self.genes = genes
        self.network = network
        self.id = id
        self.birth = birth
        self.parent = parent
        self.front = -1
        self.mutations = mutations
        self.measurements = None

    # Translations

    def express(self):
        """Translate genes to network."""
        if self.network is None:
            assert self.genes is not None
            self.network = self.Phenotype.from_genes(self.genes)

    def get_measurements(self, *args, **kwargs):
        assert self.network is not None
        return self.network.get_measurements(*args, **kwargs)

    def record_measurements(self, *, weights, measurements):
        if self.measurements is None or 'n_evaluations' not in self.measurements:
            self.measurements = dict(n_evaluations=0)

        ms = self.measurements
        ms_new = measurements

        for k, v in measurements.items():

            k_max, k_min, k_mean = f'{k}.max', f'{k}.min', f'{k}.mean'

            if ms['n_evaluations'] < 1:
                ms[k_max ] = np.max(v)
                ms[k_min ] = np.min(v)
                ms[k_mean] = np.mean(v)
            else:
                ms[k_max] = max(np.max(v), ms[k_max])
                ms[k_min] = min(np.min(v), ms[k_min])
                ms[k_mean] = ((
                        np.sum(v)
                        + ms[k_mean] * ms['n_evaluations']
                    ) / (ms['n_evaluations'] + len(weights)))

        ms['n_evaluations'] += len(weights)

    def metadata(self, current_gen=None):
        data = dict(
            n_hidden=self.network.n_hidden,
            n_layers=self.network.n_layers,
            id = self.id,
            birth = self.birth,
            n_mutations = self.mutations,
            n_enabled_edges=np.sum(self.genes.edges['enabled'] == True),
            n_disabled_edges=np.sum(self.genes.edges['enabled'] == False),
            n_total_edges=len(self.genes.edges),
            front=self.front,
            age=None if current_gen is None else (current_gen - self.birth)
        )
        data.update(
            num_used_activation_functions(
                self.genes.nodes, self.network.available_act_functions))
        return data

    def get_data(self, *keys, as_list=False, **kwargs):
        try:
            if len(keys) == 1:
                if keys[0] in self.measurements:
                    return self.measurements[keys[0]]
                else:
                    return self.metadata()[keys[0]]

            else:
                data = self.metadata(**kwargs)
                data.update(self.measurements)

                if len(keys) == 0:
                    return data
                else:
                    if as_list:
                        return [data[k] for k in keys]
                    else:
                        return {k: data[k] for k in keys}
        except KeyError as e:
            logging.debug(self.measurements.keys())
            raise e


    @classmethod
    def empty_initial(cls, *args, **kwargs):
        """Create an initial individual with no edges and no hidden nodes."""
        return cls(genes=cls.Genotype.empty_initial(*args, **kwargs), birth=0, id=0)

    @classmethod
    def full_initial(cls, *args, id=0, **kwargs):
        """Create an initial individual with no hidden nodes and fully connected input and output nodes (some edges are randomly disabled)."""
        return cls(genes=cls.Genotype.full_initial(*args, **kwargs), birth=0, id=id)


class RecurrentIndividualBase(IndividualBase):
    from .genes import RecurrentGenes as Genotype
