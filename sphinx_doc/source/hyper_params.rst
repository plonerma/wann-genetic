Hyperparamaters
===============

All configuration of the evolutionary process is stored in the experiment specification file usign the `TOML <https://github.com/toml-lang/toml>`_ format.

If a parameter has not been defined in the experiment specification file, the algorithm will fall back to parameters found in *rewann/environment/default.toml*.

Using a base and a series spec file, series of experiments with varying hyper parameters can be generated (refer to :doc:`experiment_series`).

The following sections give an overview over the available configuration fields (this is subject to change):

task
----

.. table::
  :widths: 35 20 45
  :width: 100%

  ==========  =======  ===========
  Parameter   Default  Description
  ==========  =======  ===========
  ``name``    'iris'   Task to train on (available tasks are defined in ``rewann.environment.tasks``).
  ==========  =======  ===========


sampling
---------

.. table::
  :widths: 35 20 45
  :width: 100%

  ========================================  =========  ===============================================================
  Parameter                                 Default    Description
  ========================================  =========  ===============================================================
  ``seed``                                  0          Initial seed for random number generator
  ``post_init_seed``                        false      If not false, set seed after initial population was generated (used to determine how much influence the initial pop has).
  ``distribution``                          'uniform'  Distribution to use for weight sampling.
  ``lower_bound``                           -2         Lower bound of uniform distribution (if any other distribution is used, this param is ignored).
  ``upper_bound``                           2          Upper bound of uniform distribution
  ``mean``                                  0          Mean used for the normal and log normal distribution (ignored if an uniform distribution is used).
  ``sigma``                                 1          Sigma used for the normal and log normal distribution.
  ``num_weight_samples_per_iteration``      1          Number of weights to use for evaluation within one iteratiion (or generation).
  ``num_training_samples_per_iteration``    -1         Number of training samples to use for one evaluation (a random subset is generated) - if a negative number is provided, all training samples are used (this is not a valid option when training on tasks that generate samples).
  ``hof_evaluation_iterations``             20
  ========================================  =========  ===============================================================


population
----------

.. table::
  :widths: 35 20 45
  :width: 100%

  ========================================  ===============  ======================
  Parameter                                 Default          Description
  ========================================  ===============  ======================
  ``num_generations``                       20               Number of iterations.
  ``size``                                  64
  ``initial_enabled_edge_probability``      0.05
  ``initial_genes``                         'full'
  ``hof_size``                              10
  ``enable_edge_signs``                     false
  ``enabled_activation_functions``          'all'
  ========================================  ===============  ======================

selection
---------

.. table::
  :widths: 35 20 45
  :width: 100%

  =====================  =================================================  ===============================================================
  Parameter              Default                                            Description
  =====================  =================================================  ===============================================================
  ``use_tournaments``    true
  ``elite_ratio``        0.1
  ``culling_ratio``      0.2                                                Don't use the worst 20% of the population for further variations
  ``tournament_size``    5
  ``objectives``         ['-log_loss.min', '-log_loss.mean', '-n_hidden']
  ``recorded_metrics``   ['accuracy', 'kappa', 'log_loss']
  ``hof_metric``         'accuracy.mean'
  =====================  =================================================  ===============================================================



mutation
--------
The following sections define the types of mutations that are possible and how frequently they occur.
The probability of all enabled mutation types will be normalized to a sum of one.


mutation.new_edge
.................
.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.20
  ``strategy``     'layer_agnostic'
  ===============  ================  ===========

mutation.new_node
.................
.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.25
  ===============  ================  ===========

mutation.reenable_edge
......................
.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.05
  ===============  ================  ===========


mutation.change_activation
..........................
.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.5
  ===============  ================  ===========


mutation.change_edge_sign
.........................
This mutation type will be ignored if edges signs are disabled.

.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.2
  ===============  ================  ===========



mutation.add_recurrent_edge
...........................
This mutation type will be ignored if task does not require recurrence


.. table::
  :widths: 35 20 45
  :width: 100%

  ===============  ================  ===========
  Parameter        Default           Description
  ===============  ================  ===========
  ``probability``  0.3
  ===============  ================  ===========


postopt
-------
.. table::
  :widths: 35 20 45
  :width: 100%

  ==================  =======  ===============================================================
  Parameter           Default  Description
  ==================  =======  ===============================================================
  ``run_postopt``     true     Do post optimization evaluation.
  ``compile_report``  true     Compile a report with statistical data and figures.
  ``num_weights``     100      Number of weights to use during post optimization evaluation.
  ``num_samples``     1000     Number of samples to use during post optimization evaluation.
  ==================  =======  ===============================================================


config
------
.. table::
  :widths: 35 20 45
  :width: 100%

  ==================  =======  ===============================================================
  Parameter           Default  Description
  ==================  =======  ===============================================================
  ``num_worker``      true     Number of parallel processes to use.
  ``debug``           true     Set log level to debug.
  ==================  =======  ===============================================================



storage
-------

.. table::
  :widths: 35 20 45
  :width: 100%

  ========================================  ===============  ==============================================
  Parameter                                 Default          Description
  ========================================  ===============  ==============================================
  ``data_base_path``                        'data'
  ``log_filename``                          'execution.log'
  ``include_prediction_records``            false
  ``commit_elite_freq``                     -1
  ``commit_metrics_freq``                   10
  ========================================  ===============  ==============================================
