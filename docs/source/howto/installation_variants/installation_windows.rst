.. _installation_windows:

Windows
*******

HTTomo requires Linux and cannot run directly on Windows. On supported
versions of Windows, HTTomo can be run using Windows Subsystem for Linux 2
(WSL 2).

These instructions require Windows 10 version 2004 (build 19041) or later, or
Windows 11. GPU acceleration additionally requires a supported NVIDIA GPU and
an NVIDIA Windows driver that supports CUDA in WSL.

See the following documentation before continuing:

- `Install WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_
- `Enable NVIDIA CUDA on WSL
  <https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>`_

Installation steps
==================

1. Open PowerShell or Windows Terminal as an administrator.

2. Install WSL and its default Ubuntu distribution:

   .. code-block:: powershell

      wsl --install

3. Restart Windows when prompted. Then open Ubuntu from the Start menu and
   complete the initial Linux user setup.

4. Confirm that the distribution is using WSL 2:

   .. code-block:: powershell

      wsl --list --verbose

   If necessary, replace ``Ubuntu`` below with the distribution name shown by
   the preceding command:

   .. code-block:: powershell

      wsl --set-version Ubuntu 2

5. Inside the WSL terminal, update the package index and install the required
   build tools:

   .. code-block:: console

      $ sudo apt update
      $ sudo apt install build-essential wget

6. Download and install Miniforge inside WSL:

   .. code-block:: console

      $ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
      $ bash Miniforge3-Linux-x86_64.sh
      $ source ~/.bashrc

   The installer shown above is for x86-64 systems. Select a different
   `Miniforge installer
   <https://github.com/conda-forge/miniforge#requirements-and-installers>`_
   when using another architecture.

7. Follow the Conda instructions in :ref:`installation_main` to create an
   environment and install HTTomo.

8. Run the verification commands in :ref:`installation_main`.

.. dropdown:: Troubleshooting: WSL has no network connection

   Follow Microsoft's
   `WSL troubleshooting guidance
   <https://learn.microsoft.com/en-us/windows/wsl/troubleshooting>`_.

.. dropdown:: Troubleshooting: A compiler is missing

   Install the standard Ubuntu build tools:

   .. code-block:: console

      $ sudo apt update
      $ sudo apt install build-essential

.. dropdown:: Troubleshooting: HTTomo cannot use the GPU

   First confirm that the GPU is visible inside WSL:

   .. code-block:: console

      $ nvidia-smi

   If this command fails, update the NVIDIA driver installed on Windows and
   follow Microsoft's
   `CUDA on WSL guidance
   <https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>`_.

   Do not install a Linux NVIDIA display driver inside WSL.
