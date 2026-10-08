About HTTomo
************

HTTomo is a Python framework for high-throughput processing and reconstruction
of parallel-beam tomography data. It is developed at `Diamond Light Source
<https://www.diamond.ac.uk/>`_ to support increasing data rates, including
those anticipated from the `Diamond-II upgrade
<https://www.diamond.ac.uk/Home/About/Vision/Diamond-II.html>`_.

HTTomo can divide large three-dimensional datasets into memory-sized chunks and
process them in parallel across CPUs or multiple GPUs. Performance comes from
distributed I/O, MPI-based in-memory operations, GPU-accelerated methods and
CuPy device-to-device processing.

The framework orchestrates methods supplied by external CPU and GPU processing
libraries rather than implementing the scientific methods itself. Users combine
those methods in reusable YAML :term:`pipelines <pipeline>`.

.. figure:: ../_static/httomo-workflow-dark.png
   :scale: 40 %
   :alt: Simple tomographic pipeline

   Parallel-beam projection data can be divided between several workers for
   processing and reconstruction.
