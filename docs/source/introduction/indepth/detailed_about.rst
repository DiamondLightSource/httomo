.. _detailed_about:

Detailed concepts
+++++++++++++++++

Here we present more detailed concepts of HTTomo's framework, such as,
:ref:`info_sections`, :ref:`info_reslice`, :ref:`info_data` (:ref:`chunks_data`, :ref:`blocks_data`), :ref:`info_wrappers`,
:ref:`info_memory_estimators`, and others.

.. toctree::
   :maxdepth: 2

   sections
   reslice
   wrappers
   memory_estimators

.. _info_data:

Data terminology
----------------

HTTomo's framework deals with handling all the data involved in executing the
pipeline. "Data handling" in HTTomo involves operations such as splitting data up
into pieces, passing the pieces of data into methods, gathering data up and
re-splitting into a different set of pieces, and more.

The concept of data being split into smaller pieces is a common theme in HTTomo,
and there naturally arose two main "levels" of data splitting. These two levels
have been given names, as an easy way to give some context to a piece of data when
referring to it.

One level of data splitting produces a piece of data called a *chunk* (not to be
confused with the term "chunk" used in the hdf5 data format), and the other level
of data splitting produces a piece of data called a *block*.


.. toctree::
   :maxdepth: 2

   chunks
   blocks