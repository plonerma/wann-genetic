import toml
import logging
import numpy as np
import pandas as pd
import os

from wann_genetic import Individual, RecurrentIndividual


from wann_genetic.tasks import select_task

from wann_genetic import GeneticAlgorithm

from .util import get_version, TimeStore

from .evaluation_util import (get_objective_values, update_hall_of_fame,
                              make_measurements)

from wann_genetic.util import ParamTree
from wann_genetic.postopt import Report


class Environment(ParamTree):
    """Environment for executing training and post training evaluations.

    Takes care of process pool, reporting, and experiment parameters.

    Parameters
    ----------
    params : dict or str
        Dictionary containing the parameters or a path to a parameters spec
        file.
    """

    from .util import (default_params, setup_params, open_data,
                       store_gen, store_gen_metrics, load_gen_metrics,
                       stored_populations, stored_indiv_measurements,
                       store_hof, load_hof, load_pop,
                       load_indiv_measurements,
                       env_path)

    def __init__(self, params):
        """Initialize an environment for training or post training analysis."""
        super().__init__()

        self.setup_params(params)

        self.metrics = list()
        self.pool = None
        self.data_file = None

        self.hall_of_fame = list()

        # choose task
        self.task = select_task(self['task', 'name'])

        # choose adequate type of individuals
        if self['config', 'backend'].lower() == 'torch':
            import wann_genetic.individual.torch as backend
        else:
            import wann_genetic.individual.numpy as backend

        if self.task.is_recurrent:
            self.ind_class = backend.RecurrentIndividual
        else:
            self.ind_class = backend.Individual

        # only use enabled activations functions
        available_funcs = self.ind_class.Phenotype.available_act_functions
        enabled_acts = self['population', 'enabled_activation_funcs']

        if self['population', 'enabled_activation_funcs'] != 'all':
            self.ind_class.Phenotype.enabled_act_functions = [
                available_funcs[i] for i in enabled_acts
            ]

    def seed(self, seed):
        """Set seed to `seed` or from parameters.

        Parameters
        ----------
        seed : int
            Seed to use.
        """
        np.random.seed(seed)

    @property
    def elite_size(self):
        """Size of the elite (:math:`population\\ size * elite\\ ratio`)."""
        return int(np.floor(self['selection', 'elite_ratio']
                            * self['population', 'size']))

    def sample_weights(self, n=None):
        if n is None:
            n = self['sampling']['num_weights_per_iteration']

        dist = self['sampling', 'distribution'].lower()

        if dist == 'one':
            w = 1

        elif dist == 'uniform':
            lower = self['sampling', 'lower_bound']
            upper = self['sampling', 'upper_bound']
            assert lower is not None and upper is not None

            w = np.random.uniform(lower, upper, size=n)

        elif dist == 'linspace':
            lower = self['sampling', 'lower_bound']
            upper = self['sampling', 'upper_bound']
            assert lower is not None and upper is not None

            w = np.linspace(lower, upper, size=n)

        elif dist == 'lognormal':
            mu = self['sampling', 'mean']
            sigma = self['sampling', 'sigma']
            assert mu is not None and sigma is not None

            w = np.random.lognormal(mu, sigma, size=n)

        elif dist == 'normal':
            mu = self['sampling', 'mean']
            sigma = self['sampling', 'sigma']
            assert mu is not None and sigma is not None

            w = np.random.normal(mu, sigma, size=n)

        else:
            raise RuntimeError(f'Distribution {dist} not implemented.')
        self['sampling', 'current_weight'] = w
        return w

    def setup_pool(self, n=None):
        """Setup process pool."""
        if n is None:
            n = self['config', 'num_workers']
        if n == 1:
            self.pool = None
        else:
            if self['config', 'backend'].lower() == 'torch':
                logging.info('Using torch multiprocessing')
                from torch.multiprocessing import Pool
                self.pool = Pool(n)
            else:
                logging.info('Using usual multiprocessing')
                from multiprocessing import Pool
                self.pool = Pool(n)

    def pool_map(self, func, iter):
        if self.pool is None:
            return map(func, iter)
        else:
            return self.pool.imap(func, iter)

    def setup_optimization(self):
        """Setup everything that is required for training (eg. loading test
        samples).
        """
        log_path = self.env_path(self['storage', 'log_filename'])
        logging.info(f"Check log ('{log_path}') for details.")

        logger = logging.getLogger()

        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        if self['config', 'debug']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logging.info(f"Package version {get_version()}")

        p = os.path.abspath(self['experiment_path'])
        logging.info(f'Saving data at {p}.')

        logging.debug('Loading training dataset')
        self.task.load_training(env=self)

        # log used parameters
        params_toml = toml.dumps(self.params)
        logging.debug(f"Running experiments with the following parameters:\n"
                      f"{params_toml}")

        with open(self.env_path('params.toml'), 'w') as f:
            params = dict(self.params)
            # mark stored params as part of a report
            params['is_report'] = True
            toml.dump(params, f)



    def run(self):
        """Run optization and post optimization (if enabled)."""
        # set up logging, write params
        self.setup_optimization()

        # set up pool of workers
        self.setup_pool()

        with self.open_data('w'):
            # run optimization
            self.optimize()

        if self['postopt', 'run_postopt']:
            with self.open_data('r'):
                # evaluate individuals in hall of fame
                self.post_optimization()
        if self.pool is not None:
            logging.info('Closing pool')
            self.pool.close()

    def optimize(self):
        logging.info("Starting evolutionary algorithm")

        ts = TimeStore()

        alg = GeneticAlgorithm(self)

        first_generation = True

        self.seed(self['sampling', 'seed'])

        ts.start()
        for gen in np.arange(self['population', 'num_generations']) + 1:

            ts.start()

            pop = alg.ask()

            seed = self['sampling', 'post_init_seed']
            if (first_generation and not isinstance(seed, bool)):
                self.seed(seed)

            # evaluate indivs

            weights = self.sample_weights()

            logging.debug(f'Sampled weight {weights}')

            make_measurements(self, pop, weights=weights)

            obj_values = np.array([
                get_objective_values(ind, self.objectives)
                for ind in pop
            ])

            alg.tell(obj_values)

            logging.debug('Updating hall of fame')
            self.hall_of_fame = update_hall_of_fame(self, pop)

            ts.stop()

            avg = (ts.total / gen)
            expected_time = (self['population', 'num_generations'] - gen) * avg
            logging.info(f'Completed generation {gen}; {ts.dt:.02}s elapsed, {avg:.02}s avg, {ts.total:.02}s total. '
                         f'Expected time remaining: {expected_time:.02}s')

            self.store_data(gen, pop, dt=ts.dt)
        self.last_population = pop
        self.store_hof()

    def post_optimization(self):
        r = Report(self).run_evaluations(  # run evaluations on test data
            num_weights=self['postopt', 'num_weights'],
            num_samples=self['postopt', 'num_samples']  # all
        )

        if self['postopt', 'compile_report']:
            r.compile()  # plot metrics, derive stats
        else:
            r.compile_stats()  # at least derive and store stats

    def store_data(self, gen, pop, dt=-1):
        gen_metrics, indiv_metrics = self.population_metrics(
            gen=gen, population=pop, return_indiv_measurements=True, dt=dt)

        metric, metric_sign = self.hof_metric
        p = ("MAX" if metric_sign > 0 else "MIN")
        metric_value = gen_metrics[f"{p}:{metric}"]

        logging.info(f"#{gen} {p}:{metric}: {metric_value:.2}")

        self.metrics.append(gen_metrics)

        commit_freq = self['storage', 'commit_elite_freq']

        if (commit_freq > 0 and gen % commit_freq == 0):

            self.store_gen(
                gen, population=pop[:self.elite_size],
                indiv_metrics=indiv_metrics)

        commit_freq = self['storage', 'commit_metrics_freq']

        if (commit_freq > 0 and gen % commit_freq == 0):
            self.store_gen_metrics(pd.DataFrame(data=self.metrics))

    def population_metrics(self, population, gen=None, dt=-1,
                           return_indiv_measurements=False):
        """Get available measurements for all individuals in the population and
        calculate statistical key metrics.

        The statistical key metrics include:

        `Q_{0, 4}`
            The quartiles :math:`\\{1, 2, 3\\}` as well as the minimum and
            maximum :math:`(0,4)`.
        `MEAN`, `STD`
            Mean and standard deviation.
        `MIN`, `MEDIAN`, `MAX`
            Equal to `Q_0`, `Q_2`, `Q_3`

        Parameters
        ----------
        population : [wann_genetic.Individual]
            List of individuals that constitute the population.
        gen : int
            Current generation index (required for caluclating the individuals
            age).
        return_indiv_measurements : bool, optional
            Whether to return the individual measurements as well.

        Returns
        -------
        dict
            Dictionary of the produced measurements (cross product of key
            metrics and a list of individual measurements).

        """
        if gen is None:
            gen = self['population', 'num_generations']

        rows = list()
        for ind in population:
            data = ind.metadata(current_gen=gen)
            data.update(ind.measurements)
            rows.append(data)

        indiv_measurements = pd.DataFrame(data=rows)

        metrics = dict(
            num_unique_individuals=len(set(population)),

            num_individuals=len(population),

            delta_time=dt,

            # number of inds without edges
            num_no_edge_inds=np.sum(indiv_measurements['n_enabled_edges'] == 0),

            # number of inds without hidden nodes
            num_no_hidden_inds=np.sum(indiv_measurements['n_hidden'] == 0),

            # individual with the most occurences
            biggest_ind=max([population.count(i) for i in set(population)]),
        )

        for name, values in indiv_measurements.items():
            metrics[f'Q_0:{name}'] = metrics[f'MIN:{name}'] = values.min()
            metrics[f'Q_1:{name}'] = np.quantile(values, .25)
            metrics[f'Q_2:{name}'] = metrics[f'MEDIAN:{name}'] = values.median()
            metrics[f'Q_3:{name}'] = np.quantile(values, .75)
            metrics[f'Q_4:{name}'] = metrics[f'MAX:{name}'] = values.max()

            metrics[f'MEAN:{name}'] = values.mean()
            metrics[f'STD:{name}'] = values.std()

        if return_indiv_measurements:
            return metrics, indiv_measurements
        else:
            return metrics


def run_experiment():
    """Execute an experiment (see :doc:`cli`)."""
    import argparse

    parser = argparse.ArgumentParser(description='Post Optimization')

    parser.add_argument('path', type=str, help='path to experiment specification')

    parser.add_argument('--comment', type=str, help='add comment field to params.', default=None)

    args = parser.parse_args()
    env = Environment(args.path)

    if args.comment is not None:
        env['comment'] = args.comment
    env.run()

    logging.info(f'Completed excution.')
