.. _developers_howtocontribute:

How to contribute
*****************

For those who are interested in contributing to HTTomo, we provide here steps to follow. All additional enquires can be left in
the issues section on `HTTomo's Github page <https://github.com/DiamondLightSource/httomo/issues>`_.

1. Write a new data processing method in Python.

   One needs to write a method and make it accessible in either a separate library or integrating
   it into the list of already :ref:`backends_list`. The latter option is preferred as some of the packages are
   maintained by HTTomo developers which will provide support during the integration.

2. Expose the method in the library file in HTTomo

   Then one needs to expose that method to HTTomo by editing the :ref:`pl_library`. You would need to specify
   the main static descriptors of that method, such as, :code:`pattern`, :code:`implementation`, etc. If the implementation is :code:`cpu` only,
   then :code:`memory_gpu` must be set to :code:`None`. However, if the method requires GPU, then you would need to provide more information so
   that HTTomo's framework would account for the memory use on the device. See :code:`HTTomolibgpu` library file for that.
   In a simple case, one can calculate the memory directly by providing multipliers in the library file. When memory
   calculation is more complicated, one needs to add a Python script that does this calculation. See more in :ref:`developers_memorycalc`.

3. Check the wrapper type

   Every method is executed by using :ref:`info_wrappers`. Check that the method fits the existing wrapper type, and if not, then possibly more work required
   to accommodate it. In most of the cases the method should fit the existing types.

4. Generate the Yaml template

   HTTomo's UI requires :ref:`reference_templates` to execute the created method. One can either construct that YAML template manually or employ
   `YAML generator <https://diamondlightsource.github.io/httomo-backends/utilities/yaml_generator.html>`_.




