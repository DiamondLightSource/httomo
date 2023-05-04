
.. raw:: html

    <style> .blue {color:#2c7aa3} </style>

.. role:: blue

Frequently Asked Questions
---------------------------

.. dropdown:: How can I use a template?

        TODO

.. dropdown:: Can I create a template?
	
	TODO

.. dropdown:: How can I configure the multi-task pipeline? 

	TODO

.. dropdown:: How can I run HTTomo?

	TODO

.. dropdown:: I have a Python method, can it be used with HTTomo?

	TODO
	
.. dropdown:: How can I contribute to HTTomo?

	TODO

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

    1. Using the module system, ``module load`` the version of HTTomo you are using

    .. code-block:: console
        
        $ module load httomo/*httomo_version*

    This will add all of the related packages and files into your path, meaning
    that your program will be able to access these packages when it is run.

    These packages are required for the various plugins to run correctly.

    2. Configure your pipeline using the templates as shown previously 
    and run HTTomo.


.. dropdown:: What is ``module load`` doing?

    It is modifying the users environment, by including the path to certain
    environment modules.
    
    You can read more about how module works at `modules.readthedocs.io <https://modules.readthedocs.io>`_

.. dropdown:: What do I do if I have module loaded the wrong version of HTTomo?

    You can use repeat the module command, replacing ``load`` with ``unload``

    .. code-block:: console
        
        $ module unload httomo/*httomo_old_version* # unload old version first
        $ module load httomo/*httomo_version* # load the correct one
