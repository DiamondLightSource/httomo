.. _howto_process_list:

Configure efficient pipelines
=============================

Here we focus on several important aspects which can be helpful while configuring the process list. 
In order to construct more efficient pipelines one need to know :ref:`pl_conf_order`, :ref:`info_reslice`, and :ref:`info_sections`. 

.. _pl_conf_order:

More on methods order 
---------------------

The pipelines in HTTomo consist of multiple methods stacked together and executed in a serial order. Behind the scenes, 
HTTomo will take care of providing the data for the method's input/output. 

Different methods require data to be provided in different orientations (array slices). In order
to satisfy those requirements, we introduced the notion of *pattern* in HTTomo, i.e., every method has a pattern associated with it.
So far HTTomo supports three types of patterns: :code:`projection`, :code:`sinogram`, and  :code:`all`. 

.. note:: The methods that change the pattern from *projection* to *sinogram* or vice versa will trigger a costly :ref:`info_reslice` operation. Methods with pattern *all* inherit the pattern of the previous method.

In order to minimise the amount of reslice operations it is better to group methods together based on the pattern. 
For example, methods that work with projections in one group and methods that work with sinograms in another group. 
Worth noting that it might not be always possible to group the methods such way, especially the longer pipelines. It, however,
useful to keep that in mind if one seeks the most computationally efficient pipeline. The user can check the pattern of the 
method in :ref:`pl_library`.

.. note:: Currently HTTomo loaders use *projection* pattern by default, therefore the first method after the loader will be working with the *projection* pattern. It is also recommended to place :ref:`centering` methods right after the loader.

.. _pl_library:

Library files
-------------

The :ref:`pl_library` demonstrate the library files for backends where patterns are specified. 

.. dropdown:: TomoPy's library file

    .. literalinclude:: ../../../../httomo/methods_database/packages/external/tomopy/1.14/tomopy.yaml    

.. dropdown:: Httomolibgpu's library file
    
    .. literalinclude:: ../../../../httomo/methods_database/packages/external/httomolibgpu/1.2/httomolibgpu.yaml

.. dropdown:: Httomolib's library file
    
    .. literalinclude:: ../../../../httomo/methods_database/packages/external/httomolib/1.2/httomolib.yaml

.. _pl_grouping:

Grouping CPU/GPU methods
------------------------

There are different implementations of methods in :ref:`backends_list`, we can classify them into 3 categories: 

- :code:`cpu` methods. These are traditional CPU implementations in Python or other compiled languages. The exposed TomoPy functions are mostly pure CPU. 
- :code:`gpu` methods. These are the methods that use GPU devices and require an array (e.g. Numpy ndarray) in the CPU memory as an input.
- :code:`gpu_cupy` methods. A special group of methods mostly from the `HTTomolibgpu <https://github.com/DiamondLightSource/httomolibgpu>`_ library that use CuPy API and also executed on the GPU devices. The main difference of :code:`gpu_cupy` methods from the :code:`gpu` methods is that they operate on CuPy arrays instead of Numpy arrays. CuPy arrays are kept in the GPU memory until they are requested back on the CPU. This approach allows us to be more flexible with the sequences of GPU methods as we can chain them together for more efficient processing. 

.. note:: If the GPU processing is possible, it is recommended to employ :code:`gpu_cupy` or :code:`gpu` methods in the process lists. The methods themselves are usually optimised for the performance and HTTomo will take care of chaining the methods together to avoid unnecessary CPU-GPU data transfers.

The user can check the implementation of the method in :ref:`pl_library`.

Minimise saving on disk
-----------------------

HTTomo does not require :ref:`save-result-examples` by default. If the result of the method is not needed as a separate file,
then there is no reason for it to be saved on the hard disk. Saving the intermediate files can significantly slow down the execution time.
