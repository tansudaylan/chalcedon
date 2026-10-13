API reference
=============

Native strong-lens models
-------------------------

.. automodule:: chalcedon.strong_lensing
   :members: LensModel, LensEquationSolver, LensCosmo, LensModelExtensions

Native image rendering
----------------------

.. automodule:: chalcedon.imaging
   :members: LightModel, Coordinates, PixelGrid, PSF, ImageModel, PointSource, magnitude2amplitude, magnitude2cps, cps2magnitude

Lensing and magnification
-------------------------

.. automodule:: chalcedon.main
   :members: retr_deflsie, retr_deflplum, retr_deflsubh, retr_deflextr, retr_defl, retr_convfromdefl, retr_invmfromdefl, retr_psecconv, retr_magn, retr_caustics

Self-lensing and orbital scales
-------------------------------

.. automodule:: chalcedon.main
   :members: retr_radieins_inft, retr_radieinsfromsmax, retr_radieinssbin, retr_amplslen, retr_magnpntslens, evaluate_self_lensing_model, retr_adislenssour, retr_mdencrit, retr_ratimassbeinsqrd
   :no-index: