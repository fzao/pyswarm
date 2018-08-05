=========================================================
Particle swarm optimization (PSO) with constraint support
=========================================================

The ``pyswarm`` package is a gradient-free, evolutionary optimization package 
for python that supports constraints.

This version is a fork of the orignal module from tisimst (see: `pyswarm <https://github.com/tisimst/pyswarm>`_)

What's New in this fork?
========================

- Class object approach
- Possibility to get all the results of the convergence phase
- Add a third option for parallelism in order to let the user to deal with it
- Hot start for the initial position of particles
- Python logger

Requirements
============

- NumPy
- Pathos

Installation and download
=========================

See the `package homepage`_ for helpful hints relating to downloading
and installing pyswarm.


Source Code
===========

The latest, bleeding-edge, but working, `code
<https://github.com/tisimst/pyDOE/tree/master/pyswarm>`_
and `documentation source
<https://github.com/tisimst/pyswarm/tree/master/doc/>`_ are
available `on GitHub <https://github.com/tisimst/pyswarm/>`_.

Contact
=======

Any feedback, questions, bug reports, or success stores should
be sent to the `author`_. I'd love to hear from you!

License
=======

This package is provided under two licenses:

1. The *BSD License*
2. Any other that the author approves (just ask!)

References
==========

- `Particle swarm optimization`_ on Wikipedia

.. _author: mailto:tisimst@gmail.com
.. _Particle swarm optimization: http://en.wikipedia.org/wiki/Particle_swarm_optimization
.. _package homepage: http://pythonhosted.org/pyswarm
