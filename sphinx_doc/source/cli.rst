Command Line Interfaces
=======================

On installation, rewann registers these CLI entry points (using the ``--help`` option displays usage):

``run_experiment [-h] [--comment COMMENT] PATH``

  Used for executing a single experiment (placed at ``PATH``) (see :doc:`getting_started`).

  Optional:

    -h, --help         show help message
    --comment COMMENT  add comment field to params


``compile_report [-h] [--path PATH] [--weights WEIGHTS] [--samples SAMPLES]``

  Run evaluations on best individuals (stored in hall of fame) and compile a report.

  Optional:

    -h, --help             show help message
    --path PATH            path to experiment (defaults to current working directory)
    --weights WEIGHTS, -w  number of weights to use during evaluation
    --samples SAMPLES, -s  number of samples to use



``draw_network [-h] [--path PATH] [--function_names] [--function_plots] [--names] ID``

  Draw network plot for individual with given id.

  Optional:

    -h, --help            show this help message and exit
    --path PATH, -p PATH  path to experiment (defaults to current working directory)
    --function_names, -l  use names of activation as node labels
    --function_plots, -g  use plot functions on nodes
    --names, -n           use names of nodes as labels


``generate_experiment_series [-h] [--build_dir BUILD_DIR] SERIES_SPEC_FILE``

  Simple templating tool to generate a series of experiments from a series specification file (see :doc:`experiment_series` for details).

  Optional:

    -h, --help                 show this help message and exit
    --build_dir BUILD_DIR, -o  output directory for experiment specification files (defaults to 'build')
