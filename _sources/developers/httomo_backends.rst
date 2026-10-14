.. _developers_httomo_backends:

Integrating methods with ``httomo-backends``
*********************************************

After writing and testing a processing function in a backend library, describe
the function in `httomo-backends
<https://github.com/DiamondLightSource/httomo-backends>`_ before trying to use
it in an HTTomo pipeline. This page explains that integration step.

Where ``httomo-backends`` fits
==============================

The three projects have different responsibilities:

* a backend library, such as HTTomolib, HTTomolibGPU or TomoPy, implements the
  processing function;
* ``httomo-backends`` describes how HTTomo should execute that function and
  provides its pipeline template; and
* HTTomo imports the function, divides the data into suitable blocks and calls
  it with the parameters supplied by the pipeline.

Adding a function to a backend library therefore does not automatically make it
available in HTTomo. The corresponding ``httomo-backends`` change is what makes
the function discoverable and safe for HTTomo to schedule.

``httomo-backends`` supplies two related descriptions of every supported
method:

Runtime metadata
   The method database records the data pattern, implementation type, output
   shape behaviour, padding and memory requirements. HTTomo reads this
   information while constructing and executing pipeline sections.

Pipeline template
   A generated YAML file exposes the method name, module path and configurable
   parameters to pipeline authors. These files form the
   :ref:`reference_templates` catalogue.

For background on this separation, see the `upstream method-information guide
<https://diamondlightsource.github.io/httomo-backends/backends/method_info.html>`_.

Before starting
===============

Complete the backend-library work first. The new function should:

* be importable from its intended public module;
* be included in that module's ``__all__`` list, because the template generator
  inspects ``__all__``;
* accept the data array and processing options as explicit function arguments;
* have meaningful defaults for optional arguments; and
* have tests in the backend library.

Install the backend library and ``httomo-backends`` from their development
checkouts in the same environment. This allows the generator and tests to
inspect the newly added function rather than an older installed release.

Integration workflow
====================

The examples below use a function called ``new_filter`` in
``httomolibgpu.prep.stripe``. Substitute the real package, module and function
names.

1. Register a new module when necessary
---------------------------------------

If the function is in a module that is already listed, skip this step.
Otherwise add its full import path to the backend's module list:

.. code-block:: text

   httomo_backends/methods_database/packages/backends/
   └── httomolibgpu/
       └── httomolibgpu_modules.yaml

For example:

.. code-block:: yaml

   - httomolibgpu.prep.stripe

The module list tells the template generator which modules to inspect. It does
not contain method metadata.

2. Add the method to the method database
----------------------------------------

Edit the library file for the backend:

.. code-block:: text

   httomo_backends/methods_database/packages/backends/
   ├── httomolib/httomolib.yaml
   ├── httomolibgpu/httomolibgpu.yaml
   └── tomopy/tomopy.yaml

Mirror the Python module hierarchy below the package name. For
``httomolibgpu.prep.stripe.new_filter``, add an entry under ``prep``, then
``stripe``:

.. code-block:: yaml

   prep:
     stripe:
       new_filter:
         pattern: sinogram
         output_dims_change: false
         implementation: gpu_cupy
         save_result_default: false
         padding: false
         memory_gpu:
           multiplier: 2.0
           method: direct

Choose each value from the behaviour of the function, not from a similar
method's name.

.. list-table:: Method metadata
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``pattern``
     - ``projection`` processes projection slices, ``sinogram`` processes
       sinogram slices, and ``all`` can use the current data orientation.
   * - ``output_dims_change``
     - Set to ``true`` when the two non-slice dimensions can change. A
       supporting function must then calculate the output dimensions.
   * - ``implementation``
     - ``cpu`` runs with NumPy arrays, ``gpu`` manages its own GPU transfer,
       and ``gpu_cupy`` accepts and returns CuPy arrays.
   * - ``save_result_default``
     - Whether HTTomo should save this method's result when the pipeline entry
       does not provide ``save_result``. This is normally ``false`` for
       intermediate processing and ``true`` for a final reconstruction.
   * - ``padding``
     - Set to ``true`` when independently processed blocks need overlapping
       slices. A supporting function must calculate the overlap.
   * - ``memory_gpu``
     - Use ``None`` for CPU methods. GPU methods use either a direct per-slice
       multiplier or a supporting function for a parameter-dependent or
       iterative calculation.

The metadata is operational: an incorrect pattern can introduce the wrong
re-slicing behaviour, while an underestimated memory requirement can cause a
GPU out-of-memory failure. Add a test for every value that affects scheduling
or memory estimation.

3. Add supporting functions when metadata is not enough
-------------------------------------------------------

Simple properties belong in the library YAML file. Calculations that depend on
array shape, data type or method parameters belong in Python supporting
functions. Their package and module hierarchy must match the backend function:

.. code-block:: text

   httomo_backends/methods_database/packages/backends/httomolibgpu/
   └── supporting_funcs/prep/stripe.py

HTTomo locates these functions by name. Implement the functions required by
the metadata:

.. list-table:: Supporting-function names
   :header-rows: 1
   :widths: 36 64

   * - Situation
     - Required name
   * - Parameter-dependent GPU memory
     - ``_calc_memory_bytes_<method>``
   * - Iterative GPU memory search
     - ``_calc_memory_bytes_for_slices_<method>``
   * - ``output_dims_change: true``
     - ``_calc_output_dim_<method>``
   * - ``padding: true``
     - ``_calc_padding_<method>``

For example, a method whose output width is controlled by ``new_width`` could
provide:

.. code-block:: python

   def _calc_output_dim_new_filter(non_slice_dims_shape, **kwargs):
       return non_slice_dims_shape[0], kwargs["new_width"]

Follow the signatures and return types of an existing supporting function with
the same purpose. For memory calculations, test the estimate against measured
peak allocation over representative shapes and parameter values. See
:ref:`developers_memorycalc` for the HTTomo memory model.

4. Generate the pipeline template
---------------------------------

Run the template generator from the root of the ``httomo-backends`` checkout.
For HTTomolibGPU, use:

.. code-block:: console

   $ python httomo_backends/scripts/yaml_templates_generator.py \
       -i httomo_backends/methods_database/packages/backends/httomolibgpu/httomolibgpu_modules.yaml \
       -o httomo_backends/yaml_templates/httomolibgpu

Use the equivalent ``httomolib`` or ``tomopy`` paths for another backend. The
backend package must be installed in the active environment.

The expected file for the example is:

.. code-block:: text

   httomo_backends/yaml_templates/httomolibgpu/
   └── httomolibgpu.prep.stripe/new_filter.yaml

The generator obtains parameters and defaults from the Python signature. Review
the result rather than assuming it is ready:

* the ``method`` and ``module_path`` must identify the new function;
* the input data argument must not appear under ``parameters``;
* optional parameters should retain useful defaults;
* mandatory user parameters should be marked ``REQUIRED``; and
* any side outputs and references must follow the HTTomo pipeline format.

If the generator handles a signature incorrectly, update the generator rules
or the function interface as appropriate, then regenerate. Avoid maintaining a
manual template that will be overwritten during a later documentation or
release build. See the `template-generator documentation
<https://diamondlightsource.github.io/httomo-backends/utilities/yaml_generator.html>`_
for more detail.

5. Test the integration
-----------------------

At minimum, query every new metadata property in a unit test. The following
example catches misspelled module paths and incorrectly nested library entries:

.. code-block:: python

   from httomo_backends.methods_database.query import (
       MethodsDatabaseQuery,
       Pattern,
   )


   def test_new_filter_metadata():
       query = MethodsDatabaseQuery(
           "httomolibgpu.prep.stripe", "new_filter"
       )
       assert query.get_pattern() == Pattern.sinogram
       assert query.get_implementation() == "gpu_cupy"
       assert query.get_output_dims_change() is False

Run the method-database tests and the relevant backend tests:

.. code-block:: console

   $ pytest tests/test_method_query.py
   $ pytest tests/test_httomolibgpu.py

GPU memory tests require a suitable GPU environment. Test any new supporting
function directly, including boundary shapes and parameters that maximise its
memory or padding requirement.

Finally, install the modified ``httomo-backends`` checkout alongside HTTomo,
put the generated method entry into a small pipeline, and validate it:

.. code-block:: console

   $ python -m httomo check pipeline.yaml input.nxs

Run the pipeline on a small representative dataset as an end-to-end check. For
a GPU method, include enough variation in shape and parameters to exercise its
memory estimate.

6. Submit and release in dependency order
-----------------------------------------

The backend function must be available before the ``httomo-backends`` change
that imports and describes it can be released. Keep the two changes linked in
their pull-request descriptions and state the minimum compatible backend
version when needed.

A method is ready for HTTomo users when all of the following are true:

* the backend function is released and importable;
* its runtime metadata and any supporting functions are tested;
* its generated YAML template is present and correct;
* a representative pipeline passes ``httomo check``; and
* the compatible ``httomo-backends`` version is available to HTTomo.

Common mistakes
===============

Method is present in the library but has no template
   Confirm that its module is in ``<backend>_modules.yaml`` and that the
   function is exported through the module's ``__all__`` list, then regenerate
   the templates.

``KeyError`` when HTTomo constructs the method
   Check that the hierarchy in ``<backend>.yaml`` exactly matches the full
   module path and method name, including capitalisation.

Supporting function cannot be imported
   Check both the directory hierarchy and the required function-name prefix.
   They are derived from the backend module path and method name.

Pipeline validates but fails on larger data
   Revisit the GPU memory estimate, padding and output-dimension calculation.
   Template validation checks the interface; it cannot prove that runtime
   resource estimates are conservative.
