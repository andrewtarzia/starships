:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_


Overview
========

TODO.


Installation
============

To run the scripts or develop this code, you can clone the repo and use
`just <https://github.com/casey/just>`_ to setup the dev environment:

.. code-block:: bash

  just dev

Usage
=====


TODO.

I have written this template to build two cages, one porous organic cage and
one metal-organic cage. Both constructions use ``stk`` alchemical approach.
Upon using this template, you should see the shared functions and template
processes to take advantage of.

The template is set up to use project scripts (found in ``pyproject.toml``,
``src/cage_construct/scripts``), with shared functions in
``src/cage_construct/utilities``.

The output for the template is saved in ``examples/template_output`` using
``--working_path`` on all scripts.

The test and example suite is also exemplified based on the function
``cage_construct.foo``.
