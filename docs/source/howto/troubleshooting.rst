.. _troubleshooting:

Troubleshooting
===============

Start by reading ``user.log`` in the run directory. If it does not explain the
failure, inspect ``debug.log`` and rerun pipeline validation with the input
file:

.. code-block:: console

   $ python -m httomo check pipeline.yaml input.nxs

Pipeline validation fails
-------------------------

Check YAML indentation, method names, parameter names and required values.
When an input file is supplied, HTTomo also verifies explicit HDF5 dataset
paths. See :ref:`utilities_yamlchecker` and :ref:`pipeline_file_reference`.

HTTomo cannot find NXtomo data
------------------------------

Inspect the input hierarchy and confirm that the NXtomo ``NX_class`` and
``definition`` attributes are present. If automatic discovery is not suitable,
set ``data_path``, ``image_key_path`` and ``rotation_angles`` explicitly. See
:ref:`nxtomo_discovery` and :ref:`create_nxtomo`.

MPI or parallel HDF5 fails at startup
-------------------------------------

Confirm that HTTomo, ``mpi4py`` and ``h5py`` use compatible MPI libraries and
that h5py reports parallel support:

.. code-block:: console

   $ python -c "import h5py; print(h5py.get_config().mpi)"

The command must print ``True``. Inconsistent MPI installations commonly cause
import errors, immediate termination or hangs during file access.

CUDA or CuPy cannot see a GPU
-----------------------------

Check the NVIDIA driver and CuPy runtime before running HTTomo:

.. code-block:: console

   $ nvidia-smi
   $ python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

The installed CuPy CUDA package must be compatible with the system driver. Use
a CPU pipeline when no CUDA-capable GPU is available.

GPU memory is exhausted
-----------------------

An out-of-memory error can be caused by either the pipeline block size or other
processes already using the device.

#. Run ``nvidia-smi`` and stop unrelated jobs using the selected GPU.
#. In a multi-process run, assign one MPI rank to each GPU. Do not allow several
   ranks to select the same device unless the pipeline and hardware were sized
   for that arrangement.
#. Reduce the input :term:`preview`, particularly ``detector_y`` while testing.
#. Set a smaller per-process ceiling, for example ``--max-memory 12G``. For GPU
   sections this caps the budget used to calculate the block size. It may also
   cause section data to use slower disk-backed storage.
#. If one method still fails while others fit, its GPU memory estimator may be
   inaccurate. Record the method, parameters, input shape and ``debug.log`` when
   reporting the problem.

See :ref:`memory_and_performance` for user guidance and
:ref:`developers_memorycalc` when diagnosing an estimator.

Pipeline and method templates do not match
------------------------------------------

Errors about an unknown method or parameter often mean that a pipeline was
generated for a different HTTomo or ``httomo-backends`` version. Use
:ref:`versioned_downloads` for a released HTTomo version and check the
:ref:`compatibility` notes before updating one component independently.

The output directory cannot be created or written
--------------------------------------------------

HTTomo creates a run directory below ``OUT_DIR``. Confirm that the parent
directory exists where required by the scheduler, is writable by every MPI
rank and has enough free space for requested intermediate files. With a
multi-node run, use storage visible from every node. The directory supplied to
``--reslice-dir`` must already exist and be writable.

Re-slicing is unexpectedly slow
-------------------------------

A change between projection and sinogram processing can require temporary
disk-backed data. Put ``--reslice-dir`` on fast storage accessible to every
participating process. See :ref:`info_reslice` and the
:ref:`command-line reference <run-httomo-indepth>`.

Finding more information
------------------------

The :ref:`info_logger` page explains common progress messages and generated
files. When reporting a reproducible problem, include the HTTomo version,
pipeline, relevant log excerpt, execution command and hardware configuration.
