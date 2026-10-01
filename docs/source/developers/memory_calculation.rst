.. _developers_memorycalc:

Memory estimation and block sizing
**********************************

HTTomo processes a section in blocks. For a section containing GPU methods,
the block length is limited by the peak GPU memory required by every method in
that section. A correct estimate must be conservative enough to prevent an
out-of-memory failure without making blocks unnecessarily small.

Memory-estimation metadata and helper functions belong in
``httomo-backends``, not in the processing library or HTTomo itself. Read
:ref:`developers_httomo_backends` before adding an estimator.

.. important::

   The former ``calc_max_slices`` extension API is no longer used. Do not add a
   ``calc_max_slices`` function to a processing method. Estimators now report
   memory in bytes through ``httomo-backends``; HTTomo calculates the block
   length from those values. Estimators also no longer return an output data
   type.

How HTTomo selects a block length
=================================

Before executing a section, HTTomo:

#. clears the CuPy memory pool and FFT plan cache;
#. queries the available memory on the selected GPU;
#. reserves a 10% safety margin;
#. applies ``--max-memory`` as an additional per-process upper limit when that
   option is non-zero;
#. asks each GPU method with ``memory_gpu`` metadata how many slices fit; and
#. uses the smallest result for the whole section, capped by the number of
   slices in the process's chunk.

The non-slice dimensions are passed through the methods in execution order. If
a method changes the output dimensions, the next estimator receives those
updated dimensions. This is why an output-dimension helper is as important as
the memory estimator for a size-changing method.

CPU methods do not use this GPU estimator. A CPU-only section uses the
``--max-cpu-slices`` limit instead. In a mixed section, the registered GPU
methods determine the memory-based limit.

If a section uses overlap padding, at least one core slice plus the leading and
trailing padding must fit. HTTomo stops before execution if the calculated
block is smaller than that minimum.

Where the configuration lives
==============================

Every supported method has an entry in its backend library file, for example:

.. code-block:: text

   httomo_backends/methods_database/packages/backends/
   └── httomolibgpu/httomolibgpu.yaml

GPU methods define ``memory_gpu``. CPU methods use ``memory_gpu: None``.

.. code-block:: yaml

   prep:
     normalize:
       minus_log:
         pattern: all
         output_dims_change: false
         implementation: gpu_cupy
         save_result_default: false
         padding: false
         memory_gpu:
           multiplier: 3.0
           method: direct

Do not leave ``memory_gpu`` unset for a GPU method. HTTomo does not apply a
method-specific block-size constraint when this metadata is absent.

Choose an estimation strategy
==============================

``memory_gpu.method`` selects one of three strategies.

.. list-table:: GPU memory-estimation strategies
   :header-rows: 1
   :widths: 18 34 48

   * - Strategy
     - Use it when
     - Estimator contract
   * - ``direct``
     - Peak memory is proportional to the number of elements in one input
       slice.
     - Store a conservative multiplier directly in the library YAML file.
   * - ``module``
     - Memory is still linear in the number of slices, but the per-slice or
       fixed cost depends on shape, data type or method parameters.
     - Implement ``_calc_memory_bytes_<method>`` in the matching supporting
       module.
   * - ``iterative``
     - Total memory is non-linear in the number of slices, or a backend already
       provides a whole-block peak-memory calculation.
     - Implement ``_calc_memory_bytes_for_slices_<method>`` in the matching
       supporting module.

The supporting-module hierarchy must mirror the processing function's module
path. For example, helpers for ``httomolibgpu.prep.phase.paganin_filter`` live
in:

.. code-block:: text

   httomo_backends/methods_database/packages/backends/httomolibgpu/
   └── supporting_funcs/prep/phase.py

Direct estimates
================

Use ``direct`` when a single multiplier accurately describes the complete peak
allocation per input slice:

.. code-block:: yaml

   memory_gpu:
     multiplier: 3.0
     method: direct

HTTomo calculates:

.. code-block:: python

   bytes_per_slice = (
       multiplier
       * non_slice_dims_shape[0]
       * non_slice_dims_shape[1]
       * dtype.itemsize
   )
   max_slices = available_memory // bytes_per_slice

The multiplier represents the full peak allocation, not only temporary
workspace. Include the input, output and all arrays that coexist at the peak.
For example, an in-place operation with no additional allocation may use a
multiplier near ``1.0``; a method holding the input, output and another
full-sized temporary array needs at least ``3.0``.

Use a supporting function instead if the ratio changes with dimensions,
parameters or fixed-size allocations.

Module estimates
================

Use ``module`` for a memory model consisting of a per-slice cost and a fixed
cost:

.. code-block:: yaml

   memory_gpu:
     multiplier: None
     method: module

Implement this function in the matching supporting module:

.. code-block:: python

   def _calc_memory_bytes_new_filter(
       non_slice_dims_shape: tuple[int, int],
       dtype: np.dtype,
       **kwargs,
   ) -> tuple[int, int]:
       """Return (bytes_per_slice, fixed_bytes)."""
       input_bytes = np.prod(non_slice_dims_shape) * dtype.itemsize
       output_bytes = input_bytes
       fixed_bytes = 8 * 1024**2
       return int(input_bytes + output_bytes), fixed_bytes

HTTomo then calculates:

.. code-block:: python

   max_slices = (
       available_memory - fixed_bytes
   ) // bytes_per_slice

``non_slice_dims_shape`` contains the two dimensions that do not form the
current block length. ``kwargs`` contains the configured method parameters,
with side-output references resolved to their runtime values.

``bytes_per_slice`` must include allocations that grow linearly with the block
length. ``fixed_bytes`` covers allocations that occur once per method call,
such as a filter or other workspace independent of the number of slices. Both
values describe the peak, so do not add allocations that cannot coexist.

Iterative estimates
===================

Use ``iterative`` when a single per-slice value is not valid:

.. code-block:: yaml

   memory_gpu:
     multiplier: None
     method: iterative

Implement a whole-block estimator:

.. code-block:: python

   def _calc_memory_bytes_for_slices_new_filter(
       dims_shape: tuple[int, int, int],
       dtype: np.dtype,
       **kwargs,
   ) -> int:
       """Return peak bytes for the complete candidate block."""
       return calculate_peak_bytes(dims_shape, dtype=dtype, **kwargs)

HTTomo inserts a candidate block length into ``dims_shape`` at the section's
slicing dimension and calls the estimator repeatedly. It first makes a linear
approximation and then searches for a safe value. The search can stop once an
estimate uses at least 90% of the available memory, so the result is a safe
approximation rather than necessarily the exact largest possible block.

An iterative estimator must therefore:

* return the total peak bytes for the complete candidate block;
* be monotonically non-decreasing as the candidate slice count increases;
* be deterministic and safe to call repeatedly; and
* account for the supplied dimensions, data type and relevant parameters.

If the estimator raises an exception for a candidate size, HTTomo treats that
candidate as too large. Do not use exceptions as the normal calculation path.

Data types and allocations
==========================

During section sizing, HTTomo currently passes ``float32`` to GPU memory
estimators because input data is converted to floating point after loading.
The old estimator API propagated an output dtype; the current one does not.

If a method creates arrays with another dtype, include their actual byte sizes
inside its multiplier or supporting function. Use ``np.dtype(...).itemsize``
rather than assuming four bytes per element.

Account for every allocation that can be live at the peak, including where
applicable:

* the input and output arrays;
* dtype conversions and contiguous copies;
* padded arrays and overlap regions;
* temporary workspaces;
* FFT plans and FFT outputs;
* reconstruction buffers; and
* parameter-dependent fixed arrays.

Follow the allocation lifetime through the backend implementation. Summing all
allocations made during the function can substantially overestimate the peak
when earlier arrays are released before later ones are created.

Testing an estimator
====================

Test the metadata and the estimate separately.

Metadata test
-------------

Query the method through ``MethodsDatabaseQuery`` so that an incorrect YAML
hierarchy, method name or strategy is detected:

.. code-block:: python

   from httomo_backends.methods_database.query import MethodsDatabaseQuery


   def test_new_filter_memory_metadata():
       query = MethodsDatabaseQuery(
           "httomolibgpu.prep.stripe", "new_filter"
       )
       requirement = query.get_memory_gpu_params()
       assert requirement is not None
       assert requirement.method == "module"
       assert requirement.multiplier is None

Peak-memory test
----------------

Measure the processing function's peak GPU allocation with representative
inputs, then compare it with the estimate. For a module estimator:

.. code-block:: python

   bytes_per_slice, fixed_bytes = _calc_memory_bytes_new_filter(
       data.shape[1:], data.dtype, **parameters
   )
   estimated_peak = data.shape[0] * bytes_per_slice + fixed_bytes

The estimate must not be below the measured peak. Also keep it close enough to
the measurement to avoid unnecessarily small blocks. Existing estimators often
use a tolerance around 20%, but the appropriate tolerance depends on the
method and must be stated explicitly in its test.

Cover more than one block length and include the dimensions, dtypes and
parameters that change memory use. Test boundary cases such as padding,
optional workspaces and the largest supported reconstruction shape.

For an iterative estimator, additionally verify that:

* the estimate is monotonic over the tested slice counts;
* the block length returned by HTTomo fits in the available memory; and
* either the next slice count does not fit or the selected block already uses
  at least 90% of the available memory.

Run the relevant ``httomo-backends`` unit and GPU tests, followed by a small
end-to-end HTTomo pipeline. A template or metadata test alone cannot detect an
underestimated runtime allocation.
