{{ fullname | escape | underline}}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. currentmodule:: {{ fullname }}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. currentmodule:: {{ fullname }}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

.. automodule:: {{ fullname }}
