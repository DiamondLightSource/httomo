====================================================
The list of supported backends
====================================================

HTTomo currently supports several software packages that are used as backends to perform data processing and reconstruction. The support list of packages will be growing to meet users demands.

If the package has a modular structure with an easy access to every method, e.g., `TomoPy <https://tomopy.readthedocs.io>`_ software or `scikit <https://scikit-image.org/>`_ library, then the integration process is `straightforward <https://diamondlightsource.github.io/httomo/explanation/faq.html>`_.  
More complicated in structure packages would need an additional wrapping, see for instance the reconstruction method in the `HTTomolib <https://github.com/DiamondLightSource/httomolib/blob/master/httomolib/recon/algorithm.py#L72>`_ library. 


TomoPy software (CPU)
---------------------------------
`TomoPy <https://tomopy.readthedocs.io>`_ is an open-source Python package for tomographic data processing and image reconstruction developed at `The Advanced Photon Source <https://www.aps.anl.gov/>`_ in Illinois, USA. 
Being active since 2013 it gained a `large audience <https://github.com/tomopy/tomopy>`_ of users and contributors in tomographic imaging community.

* An open-source package in Python and C for data data processing and reconstruction.  With a few exceptions, TomoPy is mostly CPU processing library. In HTTomo we expose only CPU modules of TomoPy. 
* It is CPU-multithreaded package. HTTomo controls parallelisation through MPI on a higher level and also supports local CPU multithreading, which TomoPy offers, for every MPI process.
* It is a library of stand-alone methods which can be easily integrated into HTTomo. Notably not all TomoPy methods are integrated in HTTomo because of the I/O nature of some modules. Please see the list of available TomoPy `templates <https://diamondlightsource.github.io/httomo/reference/templates.html#tomopy-modules>`_.

HTTomolib library (GPU)
------------------------------------
`HTTomolib <https://github.com/DiamondLightSource/httomolib>`_ is a Python library of GPU accelerated methods written with the help of `CuPy <https://cupy.dev/>`_ API and CUDA language.

* Most of the original methods have been taken from TomoPy or `Savu <https://github.com/DiamondLightSource/Savu>`_ software and then optimised and GPU-accelerated.
* Fully modular library and methods can be used stand-alone, however the GPU memory distribution feature of HTTomo will not be available. Meaning that the methods can silently fail if the GPU memory is overflowed.


*HTTomo releases usually support a specific version of the backend package. Therefore please check that the YAML templates are associated with the version advertised in the release.*