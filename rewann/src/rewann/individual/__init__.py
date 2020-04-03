import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


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

    genes = None
    network = None
    prediction_records = None

    from .genes import Genotype
    from .network import Network

    from .genetic_operations import mutation

    def __init__(self, genes=None, network=None, prediction_records=None, id=None, birth=None):
        self.genes = genes
        self.network = network
        self.prediction_records = prediction_records
        self.id = id
        self.birth = birth

    # Translations

    def express(self):
        """Translate genes to network."""
        if self.network is None:
            assert self.genes is not None
            self.network = self.Network.from_genes(self.genes)

    @expressed
    def apply(self, *args, **keargs):
        return self.network.apply(*args, **keargs)

    @expressed
    def evaluate(self, env, average_over_sampled_weights=True):
        if self.prediction_records is None:
            # stack for storing confusion matrices
            cm_list = list()

            # weights used during evaluations
            weight_list = list()
        else:
            cm_list, weight_list = self.prediction_records

        weights = env['sampling', 'current_weight']

        if not isinstance(weights, np.ndarray):
            weights = np.array([weights])

        y_preds = self.apply(env.task.x, func='argmax', weights=weights)


        if average_over_sampled_weights:
            # though this changes metrics, it is much faster
            cm = confusion_matrix(np.tile(env.task.y_true, y_preds.shape[0]),
                                  y_preds.flatten(),
                                  labels=list(range(env.task.n_out)),
                                  normalize='all')
            cm_list.append(cm)
            weight_list.append(weights)
        else:
            for y_pred, weight in zip(y_preds, weights):
                cm = confusion_matrix(np.tile(env.task.y_true, y_preds.shape[0]),
                                      y_preds.flatten(),
                                      labels=list(range(env.task.n_out)),
                                      normalize='all')
                cm_list.append(cm)
                weight_list.append(weight)

        self.prediction_records = cm_list, weight_list

    @property
    def fitness(self):
        fitness_metric = 'median:accuracy'
        f = self.get_prediction_metrics(fitness_metric)[fitness_metric]
        return f

    @expressed
    def metrics(self, *metrics, current_gen=None, as_list=False):
        if not metrics:
            metrics = ('max:kappa', 'mean:kappa', 'min:kappa',
                       'n_hidden', 'n_edges')

        base_metrics = dict(
            n_hidden=self.network.n_hidden,
            n_edges=len(self.genes.edges),
            n_evaluations=len(self.prediction_records[0]),
            age=None if current_gen is None else (current_gen - self.birth)
        )

        cm_list, weight_list = data=self.prediction_records

        values = dict(
            cm_stack=np.array(cm_list), weights=np.array([weight_list])
        )

        base_metrics.update(apply_metrics(values, [m for m in metrics if not m in base_metrics]))

        if as_list:
            return [base_metrics[k] for k in metrics]
        else:
            return {k: base_metrics[k] for k in metrics}

    @classmethod
    def base(cls, *args, **kwargs):
        return cls(genes=cls.Genotype.base(*args, **kwargs), birth=0, id=0)
