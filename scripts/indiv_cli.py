import sys, cmd, cli_ui

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Cairo')



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

def parse_list(args):
    l = (x.strip() for x in args.strip().split(' '))
    return [x for x in l if len(x) > 0]

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

    def emptyline(self): return

    def evaluate_ind(self):
        ind = self.selected_ind

        n_weights = ask_int('number of weights', default=100)
        n_samples = ask_int('number of samples', default=100)
        if n_weights is None or n_samples is None:
            return

        env.task.load_test()
        env.sample_weights(n_weights)
        evaluate_inds(env, [ind], n_samples=n_samples, reduce_values=False, use_test_samples=True)

    def do_select(self, args):
        args = args.strip().lower()
        if args not in ('gen', 'hof'):
            args = ask_choice_default('Select from', choices=['gen', 'hof'], default=-1, sort=False)

        if args == 'gen':
            self.select_from_gen()
        elif args == 'hof':
            self.select_from_hof()
        else:
            cli_ui.info(cli_ui.red, f'invalid selection {args}')

    def select_from_gen(self):
        gens = self.gens_with_pop
        gen = ask_choice_default("Select a generation", choices=gens, default=-1, sort=False)

        pop = self.env.load_pop(gen)

        individuals = {ind.id: ind for ind in pop}
        i = ask_choice_default('Select and individual', choices=list(individuals.keys()), default=0, sort=False)

        ind = individuals[i]
        self.selected_ind = ind
        self.evaluate_ind()

    def select_from_hof(self):
        hof = load_hof(self.env)
        for i in hof:
            print(i, i.id)
        individuals = {str(ind.id): ind for ind in hof}
        i = ask_choice_default('Select and individual', choices=list(individuals.keys()), default=0, sort=False)
        ind = individuals[i]
        self.selected_ind = ind
        self.evaluate_ind()


    @property
    def ind_metrics(self):
        if self.selected_ind is None:
            self.do_select('')
        return self.selected_ind.metric_values

    def do_print(self, args):
        names = parse_list(args)

        data = [((k,), (v,)) for k,v in self.selected_ind.metrics(*names, as_dict=True).items()]
        cli_ui.info_table(data, headers=['key', 'value'])

    def do_plot(self, args):
        names = parse_list(args)

        fig, ax = plt.subplots()
        ind_metrics = self.ind_metrics
        ind_metrics = ind_metrics.sort_values(by=['weight'])
        cli_ui.info_3(f"Plotting metrics {', '.join(names)}")

        plotted_any = False

        for m in names:
            if m not in ind_metrics:
                cli_ui.info(cli_ui.red, f'{m} not in metrics')
                continue
            plotted_any = True
            ind_metrics.plot(kind='line',x='weight',y=m, ax=ax)

        if not plotted_any: return

        fig.suptitle(f"Plot for individual #{self.selected_ind.id}")
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
    env = Environment(sys.argv[1])
    with env.open_data('r'):
        IndivExplorer(env).cmdloop()
