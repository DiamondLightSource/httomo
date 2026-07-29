.. _howto_run:

Running HTTomo
--------------

The next section gives an overview of the commands to quickly get started running
HTTomo.

For those interested in learning about the different ways HTTomo can be configured
to run, there is the :ref:`run-httomo-indepth` section.

Quick Overview of Running HTTomo
================================

Required inputs
+++++++++++++++

In order to run HTTomo you require a data file (an HDF5 file) and a YAML process
list file that describes the desired processing pipeline. For information on
getting started creating this YAML file, please see :ref:`howto_process_list`
and also ready-to-be-used :ref:`tutorials_pl_templates`.

Running HTTomo Inside or Outside of Diamond
+++++++++++++++++++++++++++++++++++++++++++

As HTTomo was developed at the Diamond Light Source, there have been some extra
efforts to accommodate the users at Diamond (for example, aliases for commands
and launcher scripts). As such, there are some differences as to how one would run
HTTomo at Diamond vs. outside of Diamond, and the guidance on running HTTomo has
been split into two sections accordingly.

Additionally, HTTomo is able to run in serial or in parallel depending on what
computer hardware is available to the user, so some sections have been further
split into these two subsections where relevant.

.. toctree::
   :maxdepth: 2

   how_to_run/at_diamond
   how_to_run/outside_diamond
   how_to_run/run_in_depth

