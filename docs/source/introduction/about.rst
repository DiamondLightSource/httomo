HTTomo concept
*********************************************

    HTTomo stands for High Throughput Tomography pipeline for processing and reconstruction of parallel-beam tomography data. 
    The `HTTomo project <https://github.com/DiamondLightSource/httomolib>`_ initiated in 2022 at `Diamond Light source  <https://www.diamond.ac.uk/>`_ by Data Analysis Group and it is written in Python.
    With `Diamond-II  <https://www.diamond.ac.uk/Home/About/Vision/Diamond-II.html>`_ upgrade approaching, there is a
    need to be able to process bigger data in larger quantities and with high fidelity. With the support of modern developments in
    the field of High Performance Computing and multi-GPU processing, it is possible to enable faster data streaming and higher throughput for big data.

    The main concept of HTTomo is to split the data into three-dimensional chunks and process them in parallel. The speed is gained using
    the optimised I/O modules, in-memory reslicing operations using MPI protocols, and a capability of device-to-device GPU processing using `CuPy <https://cupy.dev/>`_ library.  
    HTTomo orchestrates the optimal data splitting driven by the available GPU memory, which makes possible processing and reconstruction of big data on smaller GPU cards. 
    
    HTTomo is a User Interface (UI) package and does not contain any data processing methods but rather utilises other libraries as backends. 
    Currently we support `TomoPy <https://tomopy.readthedocs.io>`_ (CPU methods) and `HTTomolib <https://github.com/DiamondLightSource/httomolib>`_ (mostly GPU methods) libraries. 
    It is possible, however, to enable any other modular library of data processing 
    methods in HTTomo using YAML `templates <https://diamondlightsource.github.io/httomo/templates.html>`_.
    
    A complex data analysis pipelines can be build by stacking together YAML templates which are provided at the documentation page. `Examples <https://diamondlightsource.github.io/httomo/examples.html>`_ 
    of ready to be used full pipelines are also provided. 
    
