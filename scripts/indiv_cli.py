import sys, cmd, cli_ui

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Cairo')

from line_parse_util import parse_line

import argparse

from rewann import Environment
from rewann.environment.util import load_ind, load_hof
from rewann.environment.evolution import evaluate_inds
from rewann.vis import draw_graph, draw_weight_matrix, node_names

def ask_int(*args, **kwargs):
    i = cli_ui.ask_string(*args, **kwargs)
    try:
        return int(i)
    except ValueError:
        cli_ui.info(cli_ui.red, "Please enter an integer!")
        return None

def ask_choice_default(*prompt, choices, default=None, **kwargs):
    default = choices[default]
    s = cli_ui.ask_choice(*prompt, f'[default: {default}]', choices=choices, **kwargs)
    return default if s is None else s

class IndivExplorer(cmd.Cmd):
    intro = """Individual explorer.
Start with selecting an individual ('select').
Type help or ? to list commands.
"""
    prompt = '> '

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.gens_with_pop = self.env.stored_indiv_metrics()
        self.selected_ind = None
        self._current_subplots = None, None

    def emptyline(self): return

    def warning(self, *elements):
        cli_ui.info(*([cli_ui.red] + elements))

    def subplots(self, keep=False):
        if None in self._current_subplots:
            if keep:
                self._current_subplots = plt.subplots()
                return self._current_subplots
            else:
                return plt.subplots()
        else:
            sp = self._current_subplots
            if not keep:
                self._current_subplot = None, None
            return sp

    @parse_line
    def do_evaluate(self, weights : int=None, samples : int=None):
        ind = self.selected_ind

        if weights is None:
            weights = ask_int('number of weights', default=100)
        if samples is None:
            samples = ask_int('number of samples', default=100)

        if weights is None or samples is None:
            return

        env.task.load_test()
        env.sample_weights(weights)
        evaluate_inds(env, [ind], n_samples=samples, reduce_values=False, use_test_samples=True)

    @parse_line
    def do_select(self, hof_index=-1):
        if hof_index >= 0:

            hof = load_hof(self.env)

            try:
                self.selected_ind = hof[hof_index]
            except IndexError:
                self.selected_ind = None

                self.warning(f'{hof_index} is invalid index (hof has {len(hof)} entries).')
        else:
            sel = ask_choice_default('Select from', choices=['gen', 'hof'], default=-1, sort=False)

            if sel == 'gen':
                self.select_from_gen()
            elif sel == 'hof':
                self.select_from_hof()
            else:
                self.warning(f'invalid selection {sel}')

    def select_from_gen(self):
        gens = self.gens_with_pop
        gen = ask_choice_default("Select a generation", choices=gens, default=-1, sort=False)

        pop = self.env.load_pop(gen)

        individuals = {ind.id: ind for ind in pop}
        i = ask_choice_default('Select and individual', choices=list(individuals.keys()), default=0, sort=False)

        ind = individuals[i]
        self.selected_ind = ind

    def select_from_hof(self):
        hof = load_hof(self.env)
        for i in hof:
            print(i, i.id)
        individuals = {str(ind.id): ind for ind in hof}
        i = ask_choice_default('Select and individual', choices=list(individuals.keys()), default=0, sort=False)
        ind = individuals[i]
        self.selected_ind = ind

    @property
    def ind_metrics(self):
        if self.selected_ind is not None:
            return self.selected_ind.metric_values
        else:
            return None

    @parse_line
    def do_print(self, names : list):
        data = [((k,), (v,)) for k,v in self.selected_ind.metrics(*names, as_dict=True).items()]
        cli_ui.info_table(data, headers=['key', 'value'])

    @parse_line
    def do_plot(self, names : list, save_to="", keep_plotting=False, title=""):
        if self.ind_metrics is None:
            self.warning('No individual selected.')
            return

        fig, ax = self.subplots(keep=keep_plotting)

        ind_metrics = self.ind_metrics.sort_values(by=['weight'])

        cli_ui.info_3(f"Plotting metrics {', '.join(names)}")

        plotted_any = False

        for m in names:
            if m not in ind_metrics:
                cli_ui.info(cli_ui.red, f'{m} not in metrics')
                continue
            plotted_any = True
            ind_metrics.plot(kind='line',x='weight',y=m, ax=ax)

        if title:
            fig.suptitle(f"Plot for individual #{self.selected_ind.id}")

        if save_to:
            fig.savefig(save_to)
            cli_ui.info_3(f"Saving to '{save_to}'")
        elif plotted_any and not keep_plotting:
            fig.show()

    def do_plot_network(self, args):
        fig, ax = plt.subplots()
        draw_graph(self.selected_ind.network, ax)
        fig.suptitle(f"Network of individual #{self.selected_ind.id}")
        fig.show()

    def do_EOF(self, arg):
        print("\nBye.")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('experiment_path', type=str,
                        help='path to experiment')
    parser.add_argument('--script',default=None,
                        help='run commands from script (instead of interactive mode)')

    args = parser.parse_args()

    env = Environment(args.experiment_path)

    with env.open_data('r'):
        ie = IndivExplorer(env)

        if not args.script:
            sys.exit(ie.cmdloop())
        else:
            with open(args.script) as f:
                ie.cmdqueue.extend(f.read().splitlines())
            ie.cmdqueue.extend(['EOF'])
            sys.exit(ie.cmdloop())
