.. _chunks_data:

Chunks
======

Definition
~~~~~~~~~~

When data is split into pieces and distributed among the MPI processes (one piece
per process), these pieces are called *chunks*.

Motivation: distributing data across MPI processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTomo is able to run with multiple processes using MPI. The main idea is that
HTTomo is given input data, and each process gets a subset of the input data. Thus,
the input data is split, and each MPI process gets one piece. The pieces that the
input data is split into are called *chunks*. So, in this terminology, each MPI
process has one chunk to work with. (Again, the term "chunk" here shouldn't be
confused with the hdf5 notion of a chunk!)

How are chunk shapes calculated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chunk shape calculation is simple and is based on:

- the number of MPI processes HTTomo is launched with
- the shape of the "full data"

The data is split such that each chunk is as close to the same shape as other
chunks. If the data is being split as projections, then each MPI process gets a
chunk with roughly the same number of projections. Similarly, if the data is being
split as sinograms, then each MPI process gets a chunk with roughly the same number
of sinograms.

.. note:: If the data doesn't split evenly, then the MPI process with the largest
   rank is the one that gets a chunk with the shape that's the "odd one out"

Example
~~~~~~~

Consider 3D input data with shape :code:`(180, 128, 160)` (ie, 180 projections,
where each projection has dimensions :code:`(128, 160)`), and running HTTomo with
two MPI processes.

If evenly splitting the data along the first axis of length :code:`180`, this
results in two pieces, each with shape :code:`(90, 128, 160)`. Each piece would be
referred to as a "chunk" in HTTomo, and each MPI process would get one of these
:code:`(90, 128, 160)` shaped chunks.
