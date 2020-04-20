import toml
import logging
import numpy as np
import pandas as pd
import os
import subprocess
import logging


from multiprocessing import Pool



from .tasks import select_task
from .evolution import evolution, update_hof
from .util import get_version, TimeStore

class Environment:
    from rewann.individual import Individual as ind_class
    from .util import (default_params, setup_params, open_data,
                       store_gen, store_gen_metrics, load_pop, load_gen_metrics,
                       stored_populations, stored_indiv_metrics, store_hof, load_hof,
                       load_indiv_metrics,
                       env_path)

    def __init__(self, params):
        self.setup_params(params)

        self.metrics = list()
        self.pool= None
        self.data_file = None

        self.hall_of_fame = list()

        use_test_samples = True

        # choose task
        self.task = select_task(self['task', 'name'])

    def seed(self):
        np.random.seed(self['sampling', 'seed'])

    @property
    def elite_size(self):
        return int(np.floor(self['selection', 'elite_ratio'] * self['population', 'size']))

    def sample_weights(self, n=None):
        if n is None:
            n = self['sampling']['num_weight_samples_per_iteration']

        dist = self['sampling', 'distribution'].lower()

        if dist == 'one':
            w = 1

        elif dist == 'uniform':
            l = self['sampling', 'lower_bound']
            u = self['sampling', 'upper_bound']
            assert l is not None and u is not None

            w = np.random.uniform(l, u, size=n)

        self['sampling', 'current_weight'] = w
        return w

    def setup_pool(self, n=None):
        if n is None:
            n = self['config', 'num_workers']
        if n == 1:
            self.pool = None
        else:
            self.pool = Pool(n)

    def pool_map(self, func, iter):
        if self.pool is None:
            return map(func, iter)
        else:
            return self.pool.imap(func, iter)

    def setup_optimization(self):
        log_path = self.env_path(self['storage', 'log_filename'])
        logging.info (f"Check log ('{log_path}') for details.")

        logger = logging.getLogger()

        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        if self['debug']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logging.info(f"Package version {get_version()}")

        p = os.path.abspath(self['experiment_path'])
        logging.info(f'Saving data at {p}.')

        logging.debug('Loading training dataset')
        self.task.load_training()

        n_samples = len(self.task.y_true)
        logging.debug(f'{n_samples} samples in training data set.')

        # log used parameters
        params_toml = toml.dumps(self.params)
        logging.debug(f"Running experiments with the following parameters:\n{params_toml}")

        with open(self.env_path('params.toml'), 'w') as f:
            params = dict(self.params)
            params['is_report'] = True # mark stored params as part of a report
            toml.dump(params, f)

        self.seed()


    def run(self):
        # set up logging, write params
        self.setup_optimization()

        # set up pool of workers
        self.setup_pool()

        with self.open_data('w'):
            # run optimization
            self.optimize()

        with self.open_data('r'):
            # evaluate individuals in hall of fame
            self.post_optimization()

    def optimize(self):
        generations = evolution(self)

        logging.info("Starting evolutionary algorithm")

        ts = TimeStore()

        ts.start()
        for _ in range(self['population', 'num_generations']):

            ts.start()

            w = self.sample_weights()
            logging.debug(f'Sampled weight {w}')

            gen, pop = next(generations)

            self.hall_of_fame = update_hof(self, pop)

            ts.stop()

            avg = (ts.total / gen)
            expected_time = (self['population', 'num_generations'] - gen) * avg
            logging.info(f'Completed generation {gen}; {ts.dt:.02}s elapsed, {avg:.02}s avg, {ts.total:.02}s total. '
                         f'Expected time remaining: {expected_time:.02}s')

            self.store_data(gen, pop)
        self.last_population = pop

    def post_optimization(self):
        pass

    def store_data(self, gen, pop):
        gen_metrics = self.generation_metrics(gen=gen, population=pop)
        gen_metrics, indiv_metrics = self.generation_metrics(gen=gen, population=pop, return_indiv_metrics=True)

        logging.info(f"#{gen} mean, min log_loss: {gen_metrics['MAX:log_loss.mean']:.2}, {gen_metrics['MIN:log_loss.min']:.2}")

        logging.debug(f"Best mean accuracy: {gen_metrics['best_mean_acc']}")

        self.metrics.append(gen_metrics)

        if gen % self['storage', 'commit_population_freq'] == 0:
            elite_size = int(np.floor(self['selection', 'elite_ratio'] * self['population', 'size']))
            self.store_gen(gen, population=pop[:elite_size], indiv_metrics=indiv_metrics)
            self.store_gen_metrics(pd.DataFrame(data=self.metrics))
            self.store_hof()

    def generation_metrics(self, population, gen=None, return_indiv_metrics=False):
        if gen is None:
            gen = self['population', 'num_generations']

        names = 'kappa', 'accuracy', 'log_loss'
        pfs = 'max', 'mean', 'min'

        metric_names = [
            'n_hidden', 'n_edges', 'n_evaluations', 'age', 'front', 'n_layers'] + [
            f'{m}.{p}' for p in pfs for m in names
        ]

        individual_metrics = pd.DataFrame(data=[
            ind.metrics(*metric_names, current_gen=gen) for ind in population
        ])

        metrics = dict(
            num_unique_individuals=len(set(population)),

            num_individuals=len(population),

            best_mean_acc=self.hall_of_fame[0].metrics('accuracy.mean'),

            # number of inds without edges
            num_no_edge_inds=np.sum(individual_metrics['n_edges'] == 0),

            # number of inds without hidden nodes
            num_no_hidden_inds=np.sum(individual_metrics['n_hidden'] == 0),

            # individual with the most occurences
            biggest_ind=max([population.count(i) for i in set(population)]),
            covariance_mean_kappa_n_edges=individual_metrics['n_edges'].cov(individual_metrics['kappa.mean'])
        )

        for name, values in individual_metrics.items():
            metrics[f'MAX:{name}'] = values.max()
            metrics[f'MIN:{name}'] = values.min()
            metrics[f'MEAN:{name}'] = values.mean()
            metrics[f'MEDIAN:{name}'] = values.median()

        if return_indiv_metrics:
            return metrics, individual_metrics
        else:
            return metrics

    # magic methods for direct access of parameters
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params
        for k in keys: d = d[k]
        return d

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = [keys]

        *first_keys, last_key = keys
        d = self.params
        for k in first_keys:
            d[k] = d.get(k, dict())
            d = d[k]
        d[last_key] = value

    def __contains__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params

        for k in keys:
            try: d = d[k]
            except: return False
        return True
