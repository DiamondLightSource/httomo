What is YAML?
-------------

.. code-block:: yaml

    # What does YAML mean?​
    YAML:​
        -Y: YAML​
        -A: Ain't​
        -M: Markup​
        -L: Language

.. note::

    Markup Language is language which annotates text using tags or keywords
    in order to define how content is displayed


* YAML documents are a collection of key-value pairs​
* Indentation is used to denote structure
* The length of the indentation should not matter, as long as you are consistent inside the same file

Format examples
===============

.. code-block:: yaml

    # A list of numbers using hyphens:​
    numbers:​
        - one​
        - two​
        - three​
    ​
    # The inline version:​
    numbers: [ one, two, three ]


* Indent with *spaces*, *not* tabs​.
* There must be spaces between elements​


.. code-block::

    # This is correct​
    name: tomo​

    # This will fail​
    name:tomo


.. code-block:: yaml

    # Strings don't require quotes:​
    title: Introduction to YAML​
    ​
    # But you can still use them if you prefer:​
    title-with-quotes: 'Introduction to YAML'​


Using a YAML file to specify a pipeline of functions for data processing is a common
and practical approach to creating a reproducible data analysis workflow. This approach
can help to ensure that your pipeline runs correctly and consistently over time,
regardless of the platform or environment in which it is executed. By using a YAML file
to define your pipeline, we are providing a user interface that is simple and intuitive for scientists
to use. This can be especially helpful for those who are less familiar with programming in general
or are new to the specific tools and libraries you are using.

We have a `YAML checker <https://diamondlightsource.github.io/httomo/utilities/yaml_checker.html>`_ that can help you validate your YAML file.
Before running your pipeline, we highly recommend that you validate your YAML file using this utility. 
The checker will help you to identify errors in your YAML file.
