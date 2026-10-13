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

Truncated-halo mass
-------------------

The enclosed-mass kernel accepts a positive cutoff-to-scale-radius ratio
and returns the dimensionless mass ratio:

.. code-block:: python

   import numpy as np
   from chalcedon import retr_mcutfrommscl

   radius_ratio = np.logspace(-2, 2, 200)
   mass_ratio = retr_mcutfrommscl(radius_ratio)

At equal radii, the ratio is pi/4 - 1/2. Increasing the cutoff radius
increases the enclosed mass for the assumed truncated profile. Use
``retr_mcut(defs, asca, acut, adislens, mdencrit)`` to calculate the mass
from profile and lensing-distance inputs.

.. code-block:: console

   python examples/plot_truncation_mass.py --typefileplot png

Pass ``--typefileplot pdf`` for a vector figure. These calculations and
the example now live in Chalcedon; use Chalcedon imports in place of
the former Aspendos compatibility package.

Reusable functions
------------------

Use ``retr_defl`` to combine host, subhalo, and shear contributions; use
``retr_magn`` or ``retr_caustics`` to derive image-plane diagnostics. The same
package provides Einstein-radius and self-lensing-amplitude calculations used
by Ephesos and PCAT. The ``ttvfast`` integrator is not required for any of
these lensing calculations.