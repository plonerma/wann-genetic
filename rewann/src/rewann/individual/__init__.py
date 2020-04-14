import numpy as np
import pandas as pd

from .metrics import apply_metrics

import logging

def expressed(func):
    def exp_func(self, *args, **kwargs):
        self.express()
        return func(self, *args, **kwargs)
    return exp_func

class Individual:
    """ Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    from .genes import Genotype
    from .network import Network

    from .genetic_operations import mutation

    recorded_metrics = 'accuracy', 'kappa', 'log_loss'

    def __init__(self, genes=None, network=None, metric_values=None, id=None, birth=None, parent=None):
        self.genes = genes
        self.network = network
        self.id = id
        self.birth = birth
        self.parent = parent
        if metric_values is None:
            self._metric_values = dict()
            self._metric_values['n_evaluations'] = 0
        else:
            self._metric_values = metric_values

    # Translations

    def express(self):
        """Translate genes to network."""
        if self.network is None:
            assert self.genes is not None
            self.network = self.Network.from_genes(self.genes)

    @expressed
    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    @expressed
    def evaluate(self, weights, x, y_true):
        if not isinstance(weights, np.ndarray):
            weights = np.array([weights])

        y_probs = self.apply(x, func='softmax', weights=weights)

        self.record_metrics(weights, y_true, y_probs)


    def record_metrics(self, weights, y_true, y_probs):
        y_preds = np.argmax(y_probs, axis=-1)

        for y_prob, y_pred, weight in zip(y_probs, y_preds, weights):

            values = dict(
                y_prob=y_prob,
                y_pred=y_pred,
                y_true=y_true,
                labels=list(range(self.genes.n_out))
            )
            metrics = apply_metrics(values, self.recorded_metrics)

            values = self._metric_values

            for r in self.recorded_metrics:
                k_max, k_min, k_mean = f'{r}.max', f'{r}.min', f'{r}.mean'

                if values['n_evaluations'] < 1:
                    values[k_max] = metrics[r]
                    values[k_min] = metrics[r]
                    values[k_mean] = metrics[r]
                else:
                    values[k_max] = max(metrics[r], values[k_max])
                    values[k_min] = min(metrics[r], values[k_min])
                    values[k_mean] = (
                        (metrics[r] + values[k_mean] * values['n_evaluations'])
                        / (values['n_evaluations'] + 1))

            self._metric_values['n_evaluations'] += 1

    @expressed
    def metrics(self, *metric_names, current_gen=None, as_list=False):
        if not metric_names:
            metric_names = ('kappa.max', 'kappa.mean', 'kappa.min',
                            'n_hidden', 'n_edges')

        metric_values = dict(
            n_hidden=self.network.n_hidden,
            n_edges=len(self.genes.edges),
            age=None if current_gen is None else (current_gen - self.birth)
        )

        metric_values.update(self._metric_values)

        if as_list:
            return [metric_values[k] for k in metric_names]
        else:
            return {k: metric_values[k] for k in metric_names}


    @property
    def metric_values(self):
        return pd.DataFrame(data=self._metric_values)

    @classmethod
    def base(cls, *args, **kwargs):
        return cls(genes=cls.Genotype.base(*args, **kwargs), birth=0, id=0)

    @classmethod
    def full_initial(cls, *args, id=0, **kwargs):
        return cls(genes=cls.Genotype.full_initial(*args, **kwargs), birth=0, id=id)
