.. _howto_process_list:

Configure efficient pipelines
=============================

Here we focus on several important aspects which can be helpful while configuring a
process list. In order to construct more efficient pipelines one needs to be
familiar with :ref:`pl_conf_order`, :ref:`info_reslice`, and :ref:`info_sections`.

.. _pl_conf_order:

Method pattern and method order
-------------------------------

An HTTomo pipeline consists of multiple methods ordered sequentially and is
executed in the given serial order (meaning that there is no branching in HTTomo
pipelines). Behind the scenes HTTomo will take care of providing the input data
for each method, and passing the output data of each method to the next method.

Different methods require data to be provided in different orientations (ie, the
direction of slicing an array). In order to satisfy those requirements, the notion
of a method having a *pattern* was introduced in HTTomo, i.e., every method has a
pattern associated with it. So far HTTomo supports three types of patterns:
:code:`projection`, :code:`sinogram`, and  :code:`all`.

.. note:: Transitioning between methods that change the pattern from
   :code:`projection` to :code:`sinogram` or vice versa will trigger a costly
   :ref:`info_reslice` operation. Methods with pattern :code:`all` inherit the
   pattern of the previous method.

In order to minimise the amount of reslice operations it is best to group methods
together based on the pattern. For example, putting methods that work with
projections in one group, and methods that work with sinograms in another group. It
may not always be possible to group the methods in such way, especially with longer
pipelines. However, it's useful to keep this in mind if one seeks the most
computationally efficient pipeline.

The pattern of any supported method can be found in :ref:`pl_library`.

.. note:: Currently, HTTomo loaders use the :code:`projection` pattern by default,
   therefore it's best for efficiency purposes that the first method after the
   loader has the :code:`projection` pattern. It is also recommended to place
   :ref:`centering` methods right after the loader.

.. _pl_library:

Library files
-------------

Here is the list of :ref:`pl_library` for backends where patterns and other fixed arguments for methods are specified. When HTTomo operates
with a certain method it always refers to its library file in order get the specific requirements for that method.

.. dropdown:: TomoPy's library file

    .. literalinclude:: ../../../../httomo/methods_database/packages/external/tomopy/tomopy.yaml

.. dropdown:: Httomolibgpu's library file

    .. literalinclude:: ../../../../httomo/methods_database/packages/external/httomolibgpu/httomolibgpu.yaml

.. dropdown:: Httomolib's library file

    .. literalinclude:: ../../../../httomo/methods_database/packages/external/httomolib/httomolib.yaml

.. _pl_grouping:

Grouping CPU/GPU methods
------------------------

There are different implementations of methods in :ref:`backends_list`, and can be
classified into three categories:

- :code:`cpu` methods. These are traditional CPU implementations in Python or other
  compiled languages. The exposed TomoPy functions are mostly pure CPU.
- :code:`gpu` methods. These are methods that use GPU devices and require an input
  array in CPU memory (e.g. Numpy ndarray).
- :code:`gpu_cupy` methods. These are a special group of methods, mostly from the
  `HTTomolibgpu <https://github.com/DiamondLightSource/httomolibgpu>`_ library,
  that are executed on GPU devices using the CuPy API. The main difference between
  :code:`gpu_cupy` methods and :code:`gpu` methods is that :code:`gpu_cupy` methods
  require CuPy arrays as input instead of Numpy arrays. The CuPy arrays are then
  kept in GPU memory across any consecutive :code:`gpu_cupy` methods until they are
  requested back on the CPU. This approach allows more flexibility with the
  sequences of GPU methods, as they can be chained together for more efficient
  processing.

.. note:: If GPUs are available to the user, it is recommended to use
   :code:`gpu_cupy` or :code:`gpu` methods in process lists. The methods themselves
   are usually optimised for performance and HTTomo will take care of chaining the
   methods together to avoid unnecessary CPU-GPU data transfers.

The implementation of any supported method can be found in :ref:`pl_library`.

Minimise writing to disk
------------------------

HTTomo does not require :ref:`save-result-examples` by default. If the result of a
method is not needed as a separate file, then there is no reason for it to be
written to disk. This is because saving intermediate files can significantly slow
down the execution time.
