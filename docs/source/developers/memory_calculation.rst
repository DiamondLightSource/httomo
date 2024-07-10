.. _developers_memorycalc:

GPU memory calculations
***********************

The ``calc_max_slices`` function must have the following signature::

    def calc_max_slices(slice_dim: int,
                        other_dims: Tuple[int, int],
                        dtype: np.dtype,
                        available_memory: int,
                        **kwargs) -> Tuple[int, np.dtype]: ...

The ``httomo`` package will call this function, passing in the dimension along which it will slice
(``0`` for projection, ``1`` for sinogram), the other dimensions of the data array shape,
the data type for the input, and the available memory on the GPU for method execution.
Additionally it passes all other parameters of the method in the ``kwargs`` argument,
which can be used by the function in case parameters determine the memory consumption.
The function should calculate how many slices along the slicing dimension it can fit into the given memory.
Further, it returns the output datatype of the method (given the input ``dtype`` argument),
which ``httomo`` will use for calling subsequent functions.

**Example (All Patterns):**

The ``httomo`` package will have to determine the number of slices along the projection dimension
it can fit, given the other two dimension sizes. For example:

* ``data.shape`` is ``[x, 10, 20]``, and ``httomo`` needs to determine the value for ``x``
* The data type for ``data`` is ``float32`` (which means 4 bytes are needed per element)
* Assume the available free memory on the GPU is 14,450 bytes
* It will call the function as::

    max_slices, outdtype = my_method.meta.calc_max_slices(0, (10, 20), np.float32(), 14450, **method_args)

* The developer of the given method needs to provide this function implementation,
  and it needs to calculate the maximum number of slices it can fit.
* Assuming that the method is very simple and does not need any local temporary memory,
  requiring only space for the input and output array, it could be implemented as follows::

    def _my_calc_max_slices(slice_dim: int,
                            other_dims: Tuple[int, int],
                            dtype: np.dtype,
                            available_memory: int,
                            **kwargs) -> int:
        input_mem_per_slice = other_dims[0] * other_dims[1] * dtype.nbytes
        output_mem_per_slice = input_mem_per_slice
        max_slices = available_memory // (input_mem_per_slice + output_mem_per_slice)
        return max_slices, dtype

  (note that `//` denotes integer division, which rounds towards zero)
* If the method needs extra memory internally, this should be taken into account in the implementation.
  There are several methods in this respository which can serve as an example.

* With the example data given above, the function would determine that 9 slices can fit:

  * call ``calc_max_slices(0, (10, 20), np.float32(), 14450, **method_args)``
  * ``input_mem_per_slice = 800``
  * ``output_mem_per_slice = 800``
  * => ``max_slices = 14450 // 1600 = 9``


Max Slices Tests
----------------

In order to test that the slice calculation function is reflecting reality, each method has a
unit test implemented that verifies that the calculation is right (within bounds).
That is, it tests that the estimated slices are between 80% and 100% of the actually used slices.
These tests also help to keep the memory estimation functions in sync with the implementation.

The strategy for testing is the other way around:

* We first run the actual method, given a specific data set, and record the maximum memory actually
  used by the method.
* Then, retrospectively, we call the ``calc_max_slices`` estimator function and pass in this memory
  as the ``available_memory`` argument. So we're asking the estimation function to assume that
  the memory available is the actually used memory in the method call.
* The estimated number of slices should then be less or equal to the actual slices used earlier.
* To make sure the function is not too conservative, we're checking that it returns at least 80%
  of the slices that actually fit




