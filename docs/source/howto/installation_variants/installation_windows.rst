.. _installation_windows:

Windows
*******

Although the libraries (backends) can be installed on Windows natively, the HTTomo framework requires Linux. For Windows 10 and newer one can install Linux through `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ and run HTTomo there. 
WSL supports also CUDA-compatible GPU devices, so the GPU methods will also be working smoothly. 

Installation steps
==================

Steps 1-3 are following the official WSL installation provided `here <https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command>`_.

1. Open Terminal as Admin: Right-click Start, select "Windows PowerShell (Admin)" or "Terminal (Admin)".
2. Run Install Command: Type :code:`wsl --install` and hit Enter. Reboot PC.
3. Open Promt and type :code:`wsl` to start WSL.

.. dropdown:: Troubleshooting: No Internet connection in WSL.

	Perform the steps as described in `here <https://stackoverflow.com/a/67756837>`_.

4. Get the latest Miniforge for Linux using this `link <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh>`_.
5. Get into the **base** environment of conda by :code:`source /path/to/.bashrc`. 
6. Install HTTomo and dependencies by following the :ref:`installation_main` notes for Linux.
7. Optional step. :ref:`run_tests` to make sure that everything working.

.. dropdown:: Troubleshooting: HTTomoLib library requires a :code:`gcc` compiler.

	Install the :code:`gcc` compiler with :code:`sudo apt-get install gcc`.

.. dropdown:: Troubleshooting: HTTomo doesn't run on a GPU with an older architecture.

	If there is a GPU with an older GPU architecture then try:

	a. :code:`conda install -c conda-forge cupy==12.3.6 openmpi==4.1.6 h5py[build=*openmpi*] python>=3.10 numpy astra-toolbox aiofiles click graypy loguru nvtx pillow pyyaml scikit-image scipy tqdm hdf5plugin pip pywavelets`

	b. :code:`pip install tomobar httomolib httomolibgpu httomo-backends --no deps`

	c. Go to :code:`numpy1` branch on the cloned HTTomo repository and run :code:`pip install . --no-deps` to install a compatible older version.