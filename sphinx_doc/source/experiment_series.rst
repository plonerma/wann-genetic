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
