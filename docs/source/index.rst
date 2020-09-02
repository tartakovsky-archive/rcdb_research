rcdb_research
-------------

===============
Installation
===============
**Regular installation**::

    pip install rcdb_research


**Installation for development**::

    git clone https://github.com/3jane/rcdb_research.git
    cd rcdb_research
    pip install -e .[dev]  # note: make sure you are using pip>=20

==================
Bars
==================
.. automodule:: rcdb_research.bars
    :members:

==================
Feature importance
==================
.. automodule:: rcdb_research.feature_importance
    :members:

.. automodule:: rcdb_research.feature_importance.ensemble_feature_importance
    :members:

.. automodule:: rcdb_research.feature_importance.mean_decrease_accuracy
    :members:

.. automodule:: rcdb_research.feature_importance.mean_decrease_impurity
    :members:

.. automodule:: rcdb_research.feature_importance.mutual_information
    :members:

=================
Feature selection
=================
.. automodule:: rcdb_research.feature_selection
    :members:

.. automodule:: rcdb_research.feature_selection.ensemble_feature_selection
    :members:

.. automodule:: rcdb_research.feature_selection.select_k_best
    :members:
