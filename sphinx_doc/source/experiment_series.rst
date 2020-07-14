Generating Series of Experiments
================================

In order to compare the impact of the various hyper parameters, it is necessary to compare multiple runs of the experiment with slightly changing parameters.
The ``generate_experiment_series`` command can be used to generate a list of experiments from a `base` and a `specification` file. The `base` file is used to define parameters that are shared within a series of experiments (eg. you might want to run multiple parameters on the same task - the task name must then be placed in the `base` file).

The `specification` file defines a the parameters that will be changed between the experiment and the values they will take on. Every variable/value combination (cross product) will then be created. The command line tool will output a list of files that can then be used to run these experiments (eg. using a simple for loop):

.. highlight:: bash

::

  build_dir='build'

  # iterate through all files
  for experiment in $(echo "${build_dir}/*")
  do

    # if you are executing experiments on a shared machine, you might want to prefix this command
    # with 'nice -19'

    run_experiment "${experiment}"

    if [ $? -ne 0 ]
    then
      echo "Something went wrong during $experiment"
      exit 1
    fi

    echo "Completed execution of experiment $experiment"
  done


Experiment Series Specification
--------------------------------

While the `base` file will look like any other experiment specification file (all parameters that are variable will be overwritten), the specification of the experiment series looks different.

To explain the usage of the experiment series specification consider the example of comparing different random distributions for sampling the base weighs for the neural network.

One of the variables is the uniform distribution (including the parameters that define it), the other is the random seed. If you just want to compare one experiment for each distribution, the seed would not need to be changed. In order to get more samples and to draw more certain statistical conclusions, changing the seed is a useful tool.

A variable is defined by adding a section to the spec file:

.. highlight:: toml

::

  [seed]
    values=[0,1,2,3,4]
    key=['sampling', 'seed']

The name of the section 'seed' defines the variable. The `values` fields specifies the values the parameter take on. The `key` field determine which field in the experiment specification file will be overwritten (in this case the field `seed` in the section `sampling`).

To define the distribution a single value is not sufficient (we also need the parameters).

.. highlight:: toml

::

  [distribution]
    fmt="uniform {lower_bound},{upper_bound}"
    key=['sampling']
    values=[
      {lower_bound =-2, upper_bound = 2, distribution = "uniform"},
      {lower_bound = 1, upper_bound = 5, distribution = "uniform"}
    ]




If the values are dictionaries (maps), the target parameters (determined by key) are not entirely overwritten, instead only the field present in the dictionary are.

The `fmt` field determines how the name part for this variable (it is processed via  the python function ``fmt.format``). If not `fmt` string is defined the python ``str`` function is applied to the value.
All the name part for all the variable are then combined to determine the name of the experiment. This can be controlled via the `experiment_name` field (top level / not in any section) - `experiment_name` acts as a template:

.. highlight:: toml

::

  experiment_name="Iris edge signs {distribution} {seed}"

Lastly, a field is required for referencing the base parameters:

.. highlight:: toml

::

  base_params="base.toml"

Keeping series of experiments each in their own subdirectory, calling the base params file 'base.toml' and the specification file 'spec.toml' is recommended as it makes analysis much faster with the provided tools (see :ref:`series_analysis`).


If dictionary values with different field are used, a `fmt` string can can also be define for each dictionary. Consider the example of comparing different types of probability distributions:

.. highlight:: toml

::

  [distribution]
    key=['sampling']
    values=[
      {lower_bound =-2, upper_bound = 2, distribution = "uniform", _fmt="uniform {lower_bound},{upper_bound}"},
      {lower_bound = 1, upper_bound = 5, distribution = "uniform", _fmt="uniform {lower_bound},{upper_bound}"},
      {mean = 0, sigma = 0.5, distribution = "lognormal", _fmt="lognormal {mean},{sigma}"}
    ]


A final spec file might look like this:

.. highlight:: toml

::

  experiment_name="Iris edge signs {distribution} {seed}"
  base_params="base.toml"

  [distribution]
    key=['sampling']
    values=[
      {lower_bound =-2, upper_bound = 2, distribution = "uniform", _fmt="uniform {lower_bound},{upper_bound}"},
      {lower_bound = 1, upper_bound = 5, distribution = "uniform", _fmt="uniform {lower_bound},{upper_bound}"},
      {mean = 0, sigma = 0.5, distribution = "lognormal", _fmt="lognormal {mean},{sigma}"}
    ]

  [seed]
    values=[0,1,2,3,4]
    key=['sampling', 'seed']


This specification file would produce a series of 15 experiments (5 seeds * 3 distribution values).



.. _series_analysis:

Analysis of Experiment Series
-----------------------------


In order to easily compare the results of the series, use the ``load_experiment_series`` function:

.. highlight:: python

::

  from wann_genetic.tools import load_series_stats

  df = load_series_stats("path_to_series_spec")

Pass the path to a copy of the series spec file to the function. It expects the experiment data to be in a directory 'data' in the same directory. Use the `data_path` argument to specify another location if necessary. If you want to read in additional values from the params files, use the params_map argument (containing a mapping of `column`: list of keys determining the field in the parameters).
