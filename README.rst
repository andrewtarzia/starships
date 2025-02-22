:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_

Overview
========

Structure prediction of starship-like cages using toy-model ``cgx``-driven
graph enumeration.

This project is built on
`cgx <https://cgexplore.readthedocs.io/en/latest/>`_ and
`stk <https://stk.readthedocs.io/en/stable/>`_.

Installation
============

This code can be installed by cloning this repository and either using
`just <https://github.com/casey/just>`_:

.. code-block:: bash

  just dev


This will not install other dependencies available through conda:

.. code-block:: bash

  # for xtb
  mamba install xtb

  # for openmm and openmmtools
  mamba install openmm openmmtools

CREST must be downloaded from `crest <https://crest-lab.github.io/crest-docs/page/documentation/keywords.html>`_

Gulp (version: 6.1.2) must be downloaded from `gulp <https://gulp.curtin.edu.au/index.html>`_

or using ``conda``/``mamba`` (this will give the exact environment used in the paper):

.. code-block:: bash

  conda env create -f environment.yml
  mamba env create -f environment.yml

This will install the ``starships`` in a development state allowing
you to edit the source code, and will install the required dependencies
`openmm <https://openmm.org/>`_ and
`openmmtools <https://openmmtools.readthedocs.io/en/stable/gettingstarted.html>`_
using ``mamba``.


Important
---------

In each script, you will have to update the environment variables, namely
the paths where the data is written to and where software can be found.

Projects scripts
================

From ``cage_construct``, you get the following scripts available in the
command line.

Section 1: Atomistic modelling
------------------------------

``build_atomistic_models`` Builds atomistic cage models of the double-walled triangle, square and tetrahedron using GULP and xTB.

``atomistic_model_analysis`` Analyses provided crystal structures (availabe on Zenodo).

``crest_analysis`` Runs CREST over atomisitic ligands to generate a set of conformers.



Section 2: Structure prediction
-------------------------------

``run_cg_model`` Performs structure prediction using the ``Scrambler`` method (defined in ``utilities.py``).
Produces series of plots for this process.

``run_1d_scan`` Performs 1D scan over FF terms in the `known' cage topology.

``run_angle_scan`` Performs 2D scans over FF terms in the `known' cage topology.


Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006.
