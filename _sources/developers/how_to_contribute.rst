.. _developers_howtocontribute:

How to contribute
*****************

Use this workflow to expose a new processing method through HTTomo. For changes
to HTTomo itself, open a pull request with focused tests and documentation. If
the scope is unclear, discuss it first in the `HTTomo issue tracker
<https://github.com/DiamondLightSource/httomo/issues>`_.

1. Put the code in the right project
====================================

Implement the processing function in a backend library, such as HTTomolib,
HTTomolibGPU or TomoPy. Keep HTTomo responsible for orchestration rather than
scientific algorithms. See :ref:`backends_list` for the supported libraries.

Add tests and make the function importable from its public module before
starting the HTTomo integration.

2. Confirm that HTTomo can call it
==================================

Prefer a function that accepts an array and explicit parameters and returns the
processed array. Check that it fits an existing :ref:`wrapper <info_wrappers>`.
Only add or modify a wrapper when the function's interface cannot use an
existing one. See :ref:`developers_add_own_method` for the basic method
requirements.

3. Register it in ``httomo-backends``
=====================================

Add the method's execution metadata, any required memory or shape calculations,
and its generated pipeline template. Follow
:ref:`developers_httomo_backends` for the complete procedure.

Do not add scientific processing code to ``httomo-backends``. That repository
describes how HTTomo should execute the backend function; it does not implement
the function itself.

4. Test the integration
=======================

Before submitting the changes:

* run the backend library's tests;
* test every new ``httomo-backends`` metadata or supporting-function entry;
* review the generated YAML template;
* validate a representative pipeline with
  ``python -m httomo check pipeline.yaml input.nxs``; and
* run that pipeline on a small representative dataset.

Include CPU or GPU tests appropriate to the implementation. Memory estimates,
padding and output-shape calculations should cover boundary values, not only a
single typical input.

5. Submit linked changes
========================

Submit changes to the backend library before, or together with, the associated
``httomo-backends`` change. Link the pull requests and identify any minimum
compatible package versions.

Update HTTomo itself only when the new method requires a framework change, such
as a new wrapper capability. Include the relevant tests and documentation in
that pull request.

A contribution is complete when the backend method is tested and importable,
its metadata and template are available from ``httomo-backends``, and a small
HTTomo pipeline validates and runs successfully.
