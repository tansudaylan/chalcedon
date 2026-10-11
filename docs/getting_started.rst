Getting started
===============

Installation
------------

Install Chalcedon from its repository root:

.. code-block:: console

   python -m pip install -e .

Set ``CHALCEDON_PATH`` to the repository root. The analytic examples write
their figures next to the example script or into ``visuals/``.

First run
---------

The caustic example evaluates a host lens, a subhalo, and external shear:

.. code-block:: console

   python examples/microlens_caustic.py --typefileplot png

The maps are deterministic predictions under the parameters encoded in the
example. The left panel shows the deflection field and the right panel shows
the resulting magnification and candidate contour locations.

Reusable functions
------------------

Use ``retr_defl`` to combine host, subhalo, and shear contributions; use
``retr_magn`` or ``retr_caustics`` to derive image-plane diagnostics. The same
package provides Einstein-radius and self-lensing-amplitude calculations used
by Ephesos and PCAT. The ``ttvfast`` integrator is not required for any of
these lensing calculations.