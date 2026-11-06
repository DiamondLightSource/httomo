.. _howto_process_list:
.. _how_to_configure_pipeline:

Configure a pipeline
********************

An HTTomo pipeline is an ordered sequence of loading and processing operations
defined in YAML. See
:ref:`explanation_process_list` for an introduction to pipelines and
:ref:`explanation_templates` for the structure of method templates.

Choose an editor
----------------

Use a text editor with YAML syntax highlighting and indentation support.
Suitable editors include:

* `VS Code for the Web <https://vscode.dev/>`_ Online YAML editing
* `Visual Studio Code <https://code.visualstudio.com/>`_
* `PyCharm <https://www.jetbrains.com/pycharm/>`_
* `Sublime Text <https://www.sublimetext.com/>`_
* `Notepad++ <https://notepad-plus-plus.org/>`_ for Windows
* `Vim <https://www.vim.org/>`_ or `Neovim <https://neovim.io/>`_ for
  terminal-based editing

Build the pipeline
------------------

#. Start with an :ref:`HTTomo loader <reference_loaders>`.
#. Copy the required processing methods from
   :ref:`reference_templates` into the same YAML file.
#. Arrange the methods in execution order. HTTomo runs them sequentially from
   top to bottom.
#. Edit each method's parameters for the input data and processing task. Consult
   the relevant library documentation for parameter details.


.. _utilities_yamlchecker:

Validate the pipeline
---------------------

.. note::

   Pipeline validation is integrated into the launcher at Diamond Light Source.
   See :ref:`howto_run_at_diamond`.

Check the pipeline regularly while editing it:

.. code-block:: console

   $ python -m httomo check pipeline.yaml

Supplying the input file also validates the dataset paths used by the loader:

.. code-block:: console

   $ python -m httomo check pipeline.yaml input.nxs

Always validate the completed pipeline before running it. See
:ref:`run-httomo-indepth` for the complete ``check`` command syntax.

What validation checks
++++++++++++++++++++++

The checker reports an error when:

* the YAML syntax or indentation is invalid;
* the first pipeline entry is not an HTTomo loader;
* a method or module path is unknown;
* a parameter name or value has the wrong type;
* a required parameter is missing;
* a side-output reference is invalid; or
* a referenced HDF5 dataset does not exist when an input file is supplied.

When a loader parameter is set to ``auto``, supplying the input file also
checks that HTTomo can find an NXtomo entry. See :ref:`nxtomo_discovery`.

Common validation errors
++++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Message or symptom
     - What to check
   * - YAML cannot be parsed
     - Use spaces rather than tabs and align fields at the same nesting level.
   * - Method is not valid
     - Copy its ``method`` and ``module_path`` from :ref:`reference_templates`.
   * - Parameter is unknown or has the wrong type
     - Compare the ``parameters`` mapping with the method template.
   * - Dataset path is not valid
     - Check the loader paths in the input HDF5 file, or use NXtomo automatic
       discovery where supported.
