.. _explanation_process_list:

What is a process list?
------------------------

A process list is a `YAML file <https://diamondlightsource.github.io/httomo/explanation/yaml.html>`_ which is required to execute processing of data in HTTomo. The process list file consists of methods exposed as YAML `templates <https://diamondlightsource.github.io/httomo/reference/templates.html>`_ stacked together 
to form a **serially processed sequence of methods**. Each YAML `template <https://diamondlightsource.github.io/httomo/explanation/templates.html>`_ represents a standalone method or a loader which can be chained together with other templates to form a process list. 

Please check `here <https://diamondlightsource.github.io/httomo/howto/process_list.html>`_ some typical processing pipelines for tomography that can be configured for HTTomo.
