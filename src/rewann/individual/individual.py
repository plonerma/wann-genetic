import numpy as np
import pandas as pd

from .measures import apply_measures, num_used_activation_functions

import logging

def expressed(func):
    def exp_func(self, *args, **kwargs):
        self.express()
        return func(self, *args, **kwargs)
    return exp_func

class Individual:
    """Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    from .genes import Genes as Genotype
    from .network import Network as Phenotype

    from .genetic_operations import mutation

    # can be changed via selection.recorded_metrics
    recorded_measures = 'accuracy', 'kappa', 'log_loss'

    def __init__(self, genes=None, network=None, measurements=None, id=None, birth=None, parent=None, mutations=0):
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
        self._measurements = measurements

    # Translations

    def express(self):
        """Translate genes to network."""
        if self.network is None:
            assert self.genes is not None
            self.network = self.Phenotype.from_genes(self.genes)

    @expressed
    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    def record_measurements(self, *, weights, y_true, y_raw, record_raw=False):
        assert len(y_raw.shape) == 3 # weights, samples, nodes

        y_true = [y_true] * len(weights)
        y_labels = [np.arange(self.genes.n_out)] * len(weights)

        if record_raw:
            self._measurements = {
                'weight': weights,
                'y_true': y_true,
                'y_raw': y_raw,
                'n_evaluations': len(weights)
            }

        else:
            if self._measurements is None or 'n_evaluations' not in self._measurements:
                self._measurements = dict(n_evaluations=0)

            try:
                ms_new = apply_measures({
                    'weight': weights,
                    'y_true': y_true,
                    'y_labels': y_labels,
                    'y_raw': y_raw
                }, self.recorded_measures)
            except Exception as e:
                logging.warning(weights)
                logging.warning(y_true)
                logging.warning(y_raw)
                raise e

            ms = self._measurements

            for m in self.recorded_measures:

                k_max, k_min, k_mean = f'{m}.max', f'{m}.min', f'{m}.mean'

                if ms['n_evaluations'] < 1:
                    ms[k_max ] = np.max(ms_new[m])
                    ms[k_min ] = np.min(ms_new[m])
                    ms[k_mean] = np.mean(ms_new[m])
                else:
                    ms[k_max] = max(np.max(ms_new[m]), ms[k_max])
                    ms[k_min] = min(np.min(ms_new[m]), ms[k_min])
                    ms[k_mean] = ((
                            np.sum(ms_new[m])
                            + ms[k_mean] * ms['n_evaluations']
                        ) / (ms['n_evaluations'] + len(weights)))

            ms['n_evaluations'] += len(weights)


    @expressed
    def measurements(self, *measures, current_gen=None, as_list=False, as_dict=False):
        if not measures:
            measures = ('kappa.max', 'kappa.mean', 'kappa.min',
                            'n_hidden', 'n_edges')

        values = dict(
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

        values.update(
            num_used_activation_functions(
                self.genes.nodes, self.network.available_act_functions))

        values.update(self._measurements)

        if 'y_raw' in values:
            post_funcs = {
                'min': np.min,
                'max': np.max,
                'mean': np.mean
            }

            base_measures = set()
            post = dict()
            for m in measures:
                if m in values:
                    continue
                if '.' in m:
                    base, p = m.split('.')
                    base_measures.add(base)
                    post[m] = lambda values: post_funcs[p](values[base])
                else:
                    base_measures.add(m)
            values = apply_measures(values, base_measures)
            values.update({
                k: v(values) for k, v in post.items()
            })


        try:
            if len(measures) == 1 and not as_dict and not as_list:
                return values[measures[0]]
            elif as_list and not as_dict:
                return [values[k] for k in measures]
            elif not as_list:
                return {k: values[k] for k in measures}
            else:
                raise RuntimeWarning("as_list and as_dict are mutually exclusive")
        except KeyError as e:
            logging.warning(str(list(values.keys())))
            logging.warning(str(list(self._measurements.keys())))
            raise e

    def measurements_df(self, *measures):
        values = self._measurements
        values = apply_measures(values, measures)
        if 'weight' not in measures:
            measures = measures +  ('weight',)
        values = {k:v for k,v in values.items() if k in measures}
        return pd.DataFrame(data=values).sort_values(by=['weight'])

    @classmethod
    def empty_initial(cls, *args, **kwargs):
        """Create an initial individual with no edges and no hidden nodes."""
        return cls(genes=cls.Genotype.empty_initial(*args, **kwargs), birth=0, id=0)

    @classmethod
    def full_initial(cls, *args, id=0, **kwargs):
        """Create an initial individual with no hidden nodes and fully connected input and output nodes (some edges are randomly disabled)."""
        return cls(genes=cls.Genotype.full_initial(*args, **kwargs), birth=0, id=id)


class RecurrentIndividual(Individual):
    from .genes import RecurrentGenes as Genotype
    from .network import RecurrentNetwork as Phenotype

    def record_measurements(self, *, y_true, y_raw, **kwargs):
        assert len(y_raw.shape) == 4 # weights, samples, sequence elements, nodes
        #w, s, e, n = y_raw.shape

        valid = ~np.isnan(y_true)  # only use predictions, where labels are set
        y_true = y_true[valid]
        y_raw = y_raw[:, valid, :]

        #assert len(y_raw.shape) == 4

        #y_raw = np.reshape(y_raw, (w, -1, n))
        super().record_measurements(
            y_raw=y_raw,
            y_true=y_true,
            **kwargs)
