import toml

from ..tasks import select_task

from .util import FsInterface, default_params_path, nested_update
from .population import Population

import logging

class Experiment:
    default_params = toml.load(default_params_path)

    def __init__(self, params):
        self.params = dict(self.default_params)

        print(self.params)

        if not isinstance(params, dict):
            params = toml.load(params)

        self.update_params(params)
        task_name = self['task', 'name']
        self.task = select_task(task_name)


        if self['experiment_name'] is None:
            self['experiment_name'] = '{}_run'.format(task_name)

        self.fs = FsInterface.new_path(self['experiment_name'], base_path=self['config', 'data_base_path'])

        params_toml = toml.dumps(self.params)
        logging.info(f"Running experiments with the following parameters:\n{params_toml}\n---")

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            d = self.params
            for k in keys:
                d = d.get(k)
                if d is None:
                    return None
            return d
        else:
            return self.params.get(keys)

    def __setitem__(self, keys, value):
        if isinstance(keys, tuple):
            assert len(keys) > 0

            d = self.params
            for k in keys[:-1]:
                print (k, d)
                d[k] = d.get(k, dict())
                d = d[k]
            d[keys[-1]] = value
        else:
            self.params[keys] = value

    def update_params(self, params):
        self.params = nested_update(self.params, params)

    @property
    def path(self):
        return self.fs.base_path

    def run(self):
        self.run_evolution()

    def run_evolution(self):
        pop = Population.initial(self.task.n_dims, self.task.n_classes, params=self)
        pop.evaluate(self.task)

        for gen in range(self['population', 'num_generations']):
            pop.evolve(params=self)
            pop.evaluate(self.task)
            self.store_results(gen, pop)

    def store_results(self, gen, pop):
        if gen % 5:
            return
        print(f"""

=== gen #{gen} ===
accuracy: {pop.performance.accuracy}

            """.strip())

        print (pop.performance.confusion_matrix)
        self.fs.commit_generation(pop, include_population=True)
