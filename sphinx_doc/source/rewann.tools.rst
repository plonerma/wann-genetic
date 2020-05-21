rewann.tools package
====================

Submodules
----------

rewann.tools.compare\_experiments module
----------------------------------------

.. automodule:: rewann.tools.compare_experiments
   :members:
   :undoc-members:
   :show-inheritance:

rewann.tools.cli module
-----------------------------------------

.. automodule:: rewann.tools.cli
   :members:
   :undoc-members:
   :show-inheritance:

rewann.tools.experiment\_series module
--------------------------------------

.. automodule:: rewann.tools.experiment_series
   :members:
   :undoc-members:
   :show-inheritance:


Example usage
.............

::

  >>> from rewann.tools import ExperimentSeries
  >>> spec = dict(
  ...   experiment_name="a: {var_a}; b: {var_b}",
  ...   var_a=dict(key=['section_a', 'a'], values=[0,1,2]),
  ...   var_b=dict(key=['section_b', 'b'], values=['x', 'y', 'z'])
  ... )
  >>> series = ExperimentSeries(spec)
  >>> series.num_configurations()
  9
  >>> series.configuration_name((1,2))
  'a: 1; b: z'

Module contents
---------------

.. automodule:: rewann.tools
   :members:
   :undoc-members:
   :show-inheritance:
