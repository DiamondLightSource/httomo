
.. raw:: html

    <style> .blue {color:#2c7aa3} </style>

.. role:: blue

Frequently Asked Questions
---------------------------

.. dropdown:: How can I use a template?

    Template can be used to build a process list or a pipeline of methods. Normally the content of a template (a YAML file) is copied to build a process list file. Please see :ref:`howto_process_list`.

.. dropdown:: Can I create a template?

	You can create a template manually if you want to run a method from the external software. See more on :ref:`backends_list`.

.. dropdown:: How can I configure a multi-task pipeline?

	The multi-task pipeline is build from the available :ref:`reference_templates` by stacking them together. Please see :ref:`howto_process_list`.

.. dropdown:: How can I run HTTomo?

	Please see :ref:`howto_run`.

.. dropdown:: I have a Python method, can it be used with HTTomo?

    There is a high chance that it can be used. The method needs to be accessible in your Python environment and you will need a YAML template for it. See more on what kind of :ref:`backends_list` can be used with HTTomo. It is also recommended if you integrate your method in a library first. See :ref:`developers_content`.

.. dropdown:: How can I contribute to HTTomo?

	You can contribute by adding new methods to :ref:`backends_list` or by contributing to the source base of the `HTTomo project <https://github.com/DiamondLightSource/httomo>`_.

.. _faq_workstation:

Working from a workstation at Diamond Light Source
**************************************************

.. _`terminal`:

.. dropdown:: What is a terminal?

    A terminal could also be referred to as a console, shell, command
    prompt or command line.

    It is a program on your computer which can take in text based
    instructions and complete them. For example, navigating to a particular file
    or directory. It can also perform more complex tasks relating to
    software installation.

    It doesn't have a graphical interface, and it allows access to a wide
    range of commands quickly.

.. dropdown:: How can I use HTTomo inside the terminal?

    1. Using the module system, ``module load`` allows you to obtain an access to the installed HTTomo at Diamond computing systems.
    You can check which versions of HTTomo are installed with the command: :code:`module avail httomo`. You can either load a specific version
    with ``module load httomo/*httomo_version*`` or the default (recommended) version by executing:

    .. code-block:: console

        $ module load httomo

    This will add all of the related packages and files into your path, meaning
    that you will have an access to these packages from your loaded Python environment.

    2. Configure your pipeline using the templates as shown previously
    and run HTTomo.


.. dropdown:: What is ``module load`` doing?

    It is modifying the users environment, by including the path to certain
    environment modules. In case of HTTomo it enables a specific conda environment with Python.

    You can read more about how module works at `modules.readthedocs.io <https://modules.readthedocs.io>`_

.. dropdown:: What do I do if I have module loaded the wrong version of HTTomo?

    You can use repeat the module command, replacing ``load`` with ``unload``

    .. code-block:: console

        $ module unload httomo/*httomo_old_version* # unload old version first
        $ module load httomo/*httomo_version* # load the correct one
