.. _API:

API Reference
================================================================================

.. note::
    API documentation is under construction. Current documentation is focused
    on core functionality.

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference <self>

.. toctree::
    :maxdepth: 2
    :caption: Core Functionality
    :hidden:

    modules/cell_annotation
    modules/cell_embedding
    modules/cell_query
    modules/interpreter

.. toctree::
    :maxdepth: 2
    :caption: Model Training
    :hidden:

    modules/data_models
    modules/nn_models
    modules/training_models
    modules/triplet_selector
    modules/zarr_data_models
    modules/zarr_dataset

.. toctree::
    :maxdepth: 2
    :caption: Utilities
    :hidden:

    modules/ontologies
    modules/utils
    modules/visualizations

Core Functionality
--------------------------------------------------------------------------------

These modules provide functionality for utilizing SCimilarity embeddings for a
variety of tasks, including cell type annotation, cell queries, and gene
attribution scoring.

* :mod:`scimilarity.cell_annotation`