Getting Started
===============

This guide is supposed to provide all the information required to run first experiments. For generating series of experiments see :doc:`experiment_series`.


Installation
------------

The `wann_genetic` package can be install from the repository via pip. It is recommended to install the package in a virtual environment as the installation will add binaries to the system path (see :doc:`cli`). Make sure to create the environment with python3.


Installation with virtualenv
............................

.. highlight:: bash

::

  # create the virtual environment
  python3 -m virtualenv -p python3 venv

  # activate the virtual environment (can be deactivated using 'deactivate')
  . venv/bin/activate

  # install the package
  pip install git+ssh://git@github.com/plonerma/wann_genetic.git


Execution of a first experiment
-------------------------------

All specification of wich task to train on and which hyper parameters are to be used is stored in one `TOML <https://github.com/toml-lang/toml>`_ file.
Parameters that are not set in the spec file will take on default values. For a overview over the available parameters and default values see :doc:`params`.

One simple available task is the *iris* task. A minimal specification could look like this:

.. highlight:: toml

::

  [task]
    name='iris'

  [population]
    num_generations=200  # number of iterations
    size=100  # number of individuals in the population

  [sampling]
    # this section determines how weights are sampled
    seed=0
    distribution='lognormal'
    mean=0
    sigma=0.25
    num_weight_samples_per_iteration=5

  [postopt]
    # this section determines how the best individuals (collected in the
    # hall_of_fame) will be evaluated once the training has been completed
    run_postopt=true

    # compile a human-readable report
    compile_report=true

    # evaluate best individuals with 100 weights sampled from the specified random distribution
    num_weights=100

    # use all test sampels for evaluation (on iris at the moment training and test samples are identical)
    num_samples=-1


Store this file (eg. in `iris.toml`) and invoke wann_genetic execution of the experiment via:

.. highlight:: bash

::

  run_experiment iris.toml

For each execution a subdirectory is created in :file:`data` directory (see :doc:`params`). The experiment data directory will contain the following file structure:

::

  │   # contains some of the individuals produced during training
  ├── data.hdf5
  │
  │   # log output of training and post training evaluation
  ├── execution.log
  │
  |   # generation-wise population metrics (can be used to track performance during evaluation)
  ├── metrics.json
  │
  │   # the parameters that were used for execution (with all default values inserted)
  ├── params.toml
  │
  │   # data produced by post training evaluation
  └── report
      │
      │   # statistical measures on the best individuals (used for inter experiment comparisons)
      ├── stats.json
      │
      │   # human-readable report (including figures and statistical data)
      ├── report.md
      │
      │   # contains all the plot figures required for the report
      └── media


To produce a html document from the markdown file, `pandoc <https://pandoc.org/>`_ can be used. To create a pdf using pandoc the svg images might need to be converted to pdfs first (inside the ``report`` subdirectory do):

.. highlight: bash

::

  for svg in $(echo "media/*.svg")
  do
  	pdf="${svg%.*}.pdf"
  	echo "$svg - > $pdf"
  	rsvg-convert -f pdf -o "${pdf}" "${svg}"
  done

Example report
--------------

The resulting report might look like this example report:

.. toctree::
   :maxdepth: 2

   example_report
