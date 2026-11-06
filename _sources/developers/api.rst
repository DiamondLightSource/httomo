.. _api:

API
===

This reference is generated from HTTomo's Python modules. Most of these
modules support the framework's internal execution machinery rather than the
scientific methods used in a YAML pipeline. Backend method developers should
normally begin with :ref:`developers_add_own_method` and
:ref:`developers_httomo_backends`; use this API when extending HTTomo itself or
integrating with its runner interfaces.

Module overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Responsibility
   * - ``httomo.base_block``
     - Provides ``BaseBlock``, the default implementation of block data access
       and CPU/GPU transfer behaviour, including access to angles, darks and
       flats.
   * - ``httomo.block_interfaces``
     - Defines the protocols that a processable block must satisfy: data and
       auxiliary-data access, global indexing, and transfer between CPU and GPU
       memory.
   * - ``httomo.cli``
     - Implements the ``check``, ``memory-check`` and ``run`` commands, converts
       command-line options, prepares the output directory and starts the
       appropriate runner.
   * - ``httomo.cli_utils``
     - Contains command-line support functions, principally detection of
       parameter sweeps in YAML files, JSON strings and parsed configurations.
   * - ``httomo.darks_flats``
     - Describes where dark-field and flat-field images are stored and loads or
       selects those calibration images for a loader.
   * - ``httomo.data``
     - Contains RAM- and HDF5-backed dataset stores, MPI redistribution helpers,
       padding operations and the stores used during parameter sweeps.
   * - ``httomo.globals``
     - Holds process-wide runtime settings populated by the CLI, such as the
       output directory, selected GPU, chunking, compression and reconstruction
       filename options.
   * - ``httomo.loaders``
     - Creates loader implementations from pipeline configuration. The standard
       tomography loader reads projections, angles, darks and flats from
       HDF5/NeXus input.
   * - ``httomo.logger``
     - Configures concise terminal and ``user.log`` output, detailed
       ``debug.log`` output and optional GELF syslog reporting.
   * - ``httomo.methods``
     - Provides framework-owned pipeline operations, including global
       statistics calculation and block-wise writing of intermediate HDF5
       datasets.
   * - ``httomo.method_wrappers``
     - Selects and constructs the adapter that invokes each backend function.
       Specialised wrappers handle reconstruction, rotation, image output,
       statistics and other non-generic method behaviour.
   * - ``httomo.monitors``
     - Constructs summary and benchmark monitors and combines multiple monitors
       into one reporting interface.
   * - ``httomo.preview``
     - Represents input cropping selections, checks their bounds and calculates
       the selected indices and resulting global data shape.
   * - ``httomo.runner``
     - Contains the main execution engine and its contracts: pipelines,
       sections, block splitting, dataset blocks and stores, loaders, method
       wrappers, monitoring, side-output references and GPU utilities.
   * - ``httomo.sweep_runner``
     - Parses and executes single-parameter sweeps, divides sweep values among
       MPI ranks, manages sweep stages and side outputs, and stores sweep
       results.
   * - ``httomo.transform_layer``
     - Rewrites an executable pipeline before it runs by inserting framework
       operations such as data reduction, data checking, intermediate saving
       and sweep image output.
   * - ``httomo.transform_loader_params``
     - Converts loader values parsed from YAML or JSON into validated internal
       configurations for previews, angles, calibration images and continuous
       scan subsets.
   * - ``httomo.types``
     - Defines shared type aliases, notably the generic array type used for
       either NumPy CPU arrays or CuPy GPU arrays.
   * - ``httomo.ui_layer``
     - Converts a user-facing YAML or JSON pipeline into the internal immutable
       ``Pipeline``, constructing its loader and method wrappers and resolving
       output references.
   * - ``httomo.utils``
     - Supplies shared array, timing, logging, error-handling, snapshot and
       block-size helpers, as well as NumPy/CuPy backend selection.
   * - ``httomo.yaml_checker``
     - Validates pipeline YAML structure, loader position, methods, parameters,
       side-output references and, when input data is supplied, referenced HDF5
       paths.

Generated reference
-------------------

.. autosummary::
    :recursive:
    :toctree: generated

    httomo.base_block
    httomo.block_interfaces
    httomo.cli
    httomo.cli_utils
    httomo.darks_flats
    httomo.data
    httomo.globals
    httomo.loaders
    httomo.logger
    httomo.methods
    httomo.method_wrappers
    httomo.monitors
    httomo.preview
    httomo.runner
    httomo.sweep_runner
    httomo.transform_layer
    httomo.transform_loader_params
    httomo.types
    httomo.ui_layer
    httomo.utils
    httomo.yaml_checker
