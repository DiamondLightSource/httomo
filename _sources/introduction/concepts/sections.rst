.. _info_sections:

Sections
========

A *section* is a consecutive group of pipeline methods that operate on the same
data pattern or slicing orientation. HTTomo creates sections automatically to
organise data processing and movement efficiently.

.. _fig_sections_concept:

.. figure:: ../../_static/sections/sections_concept.png
   :alt: An HTTomo pipeline divided into sections
   :align: center
   :width: 100%

   Methods are grouped by data pattern. A pattern change, data saving or
   side-output dependency introduces a new section.

Processing sections
~~~~~~~~~~~~~~~~~~~

HTTomo processes one section at a time. It calculates a :ref:`blocks_data`
size that satisfies the memory requirements of every method in the section,
using :ref:`info_memory_estimators`, then passes each block through those
methods in sequence.

A new section begins when:

- the data pattern changes between projections and sinograms;
- a method depends on a side output produced by an earlier method;
- a method's output needs to be saved to disk; or
- another method requiring padded data is encountered.

When the data pattern changes, HTTomo also :ref:`re-slices <info_reslice>` the
data before processing the next section.

Example
~~~~~~~

In :numref:`fig_sections_concept`, Methods 1 and 2 operate on projections and
form the first section. Method 3 requires sinograms, so HTTomo re-slices the
data and starts a second section. Method 4 depends on a side output from Method
3, creating a third section that also contains Method 5.

Performance considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Section boundaries may require process synchronisation, data redistribution or
temporary storage. Pipelines with fewer sections are therefore generally more
efficient, although some boundaries are required by the selected methods.
