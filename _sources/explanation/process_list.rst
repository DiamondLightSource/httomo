.. _explanation_process_list:

What is a process list?
------------------------

A process list is a YAML file (see :ref:`explanation_yaml`), which is required to execute the processing of data in HTTomo. The process list file consists of methods exposed as YAML templates stacked together
to form a **serially processed sequence of methods**. Each YAML template represents a standalone method or a loader which can be chained together with other templates to form a process list.

Please check how :ref:`howto_process_list`.
