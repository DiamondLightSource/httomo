.. _info_sections:

Sections
--------

Sections is the fundamental concept of the HTTomo's framework which is related to how the I/O operations and processing of data is organised.

.. note:: The main purpose of a section is to organise the data input/output workflow, as well as, chain together the methods so that the constructed pipeline is computationally efficient.

To better understand the purpose of the section it is also useful to read information about :ref:`chunks_data`, :ref:`blocks_data` and :ref:`info_memory_estimators`.

Bellow we present different situations that can lead to the sections being organised in a specific manner.

.. _fig_sec1:
.. figure::  ../../_static/sections/sections1.png
    :scale: 40 %
    :alt: Sections in pipelines

    Here is a typical pipeline with a loader (`L`), 5 methods (`M`), and 4 data transfer operations (`T`) between methods.

Sections are created when:

1. :ref:`info_reslice` is needed, which is related to the change of pattern.
2. The output of the method needs to be saved to the disk.
3. The :ref:`side_output` is required by one of the methods.

Example 1: Sections with re-slice
=================================

.. _fig_sec2:
.. figure::  ../../_static/sections/sections2.png
    :scale: 50 %
    :alt: Sections in pipelines

    Let us say that the pattern in methods `M`\ :sub:`1-3` is *projection* and methods in `M`\ :sub:`4-5` belong to *sinogram* pattern.
    This will result in two sections created and also :ref:`info_reslice` operation in the data transfer `T`\ :sub:`3` layer.

Example 2 : Sections with re-slice and data saving
==================================================

.. _fig_sec3:
.. figure::  ../../_static/sections/sections3.png
    :scale: 50 %
    :alt: Sections in pipelines

    In addition Example 1 situation, let us assume that we want to save the result of `M`\ :sub:`2` method to the disk.
    This means that even though `M`\ :sub:`1-3` methods can be performed on the GPU, the data will be transferred to CPU.
    The pipeline will be further fragmented to introduce another section, so that the data transfer `T`\ :sub:`2` layer also saves the data on the
    disk, as well as, taking care to return the data back on the GPU for the method `M`\ :sub:`3`.

Example 3 : Sections with side outputs
======================================

.. _fig_sec4:
.. figure::  ../../_static/sections/sections4.png
    :scale: 50 %
    :alt: Sections in pipelines

    Consider, again, Example 1. Here, however, `M`\ :sub:`5` requests the :ref:`side_output` of the method `M`\ :sub:`4`.
    This, for example, can be because the reconstruction method `M`\ :sub:`5` requires the :ref:`centering` value of `M`\ :sub:`4`, where
    this value is calculated. This divides `M`\ :sub:`4` and `M`\ :sub:`5` into separate sections. Also notice that `M`\ :sub:`1` needs the data
    to be saved on disk, so in total, it is a pipeline with 4 sections in it.

.. note:: It can be seen that creating more sections in pipelines is to be avoided when building an efficient pipeline. Creating a section usually leads to synchronisation of all processes on the CPU and potentially, if not enough memory, through-disk operations.
