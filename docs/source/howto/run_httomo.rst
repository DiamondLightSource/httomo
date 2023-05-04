How to run HTTomo
---------------------------------------------

To run Httomo you require a data file (an hdf5 file) and a YAML template.
After HTTomo is successfully installed into your conda environment you will have an access to the following commands bellow from your UNIX shell:


TODO:


+-------------------+---------------------------------------+----------------------------------------------+
|    Alias          |            Description                |             Required input parameters        |
+===================+=======================================+==============================================+
| savu_config       | Create or amend process lists         |                                              |
+-------------------+---------------------------------------+----------------------------------------------+
|   savu            | Run single threaded Savu              | <data_path> <process_list_path> <output_path>|
+-------------------+---------------------------------------+----------------------------------------------+
| savu_mpijob_local | Run multi-threaded Savu on your PC    | <data_path> <process_list_path> <output_path>|
+-------------------+---------------------------------------+----------------------------------------------+
|  savu_mpi         | Run mpi Savu across the cluster       | <data_path> <process_list_path> <output_path>|
+-------------------+---------------------------------------+----------------------------------------------+
| savu_mpi_preview  | Run mpi Savu across 1 node            | <data_path> <process_list_path> <output_path>|
+-------------------+---------------------------------------+----------------------------------------------+

Optional arguments:

+--------+----------------------------+-----------------------+--------------------------------------------------+
|  short |         long               |       argument        |                   Description                    |
+========+============================+=======================+==================================================+
|  -f    |    **--folder**            |      folder_name      | Override the output folder name                  |
+--------+----------------------------+-----------------------+--------------------------------------------------+
|  -d    |    **--tmp**               |      path_to_folder   | Store intermediate files in this (temp) directory|
+--------+----------------------------+-----------------------+--------------------------------------------------+
|  -l    |     **--log**              |      path_to_folder   | Store log files in this directory                |
+--------+----------------------------+-----------------------+--------------------------------------------------+
| -v, -q | **--verbose**, **--quiet** |                       | Verbosity of output log messages                 |
+--------+----------------------------+-----------------------+--------------------------------------------------+


.. note:: memory/disc reslice note?
