# Chalcedon

## Purpose
I use Chalcedon to follow light through a specified lens configuration. Give it a host lens, a source, subhalos, and external shear, and calculate the resulting deflections, magnifications, critical curves, and caustics. The arcs look better when I can also inspect the lens equation that produced them.

## Scope
The calculations range from analytic deflection profiles to strong-lens image rendering. I keep them here so the modeling and inference packages can use the same lens physics.

Chalcedon provides the gravitational-lensing engines used by PCAT, Ephesos, and Pergamon:
- Singular isothermal ellipsoid (`retr_deflsie`) and Plummer-softened point-mass (`retr_deflplum`) deflections, truncated- and untruncated-NFW subhalo deflections (`retr_deflsubh`), external shear (`retr_deflextr`), and a combined host+subhalo+shear evaluator (`retr_defl`).
- Convergence, inverse magnification, and power-spectrum estimators from a deflection field (`retr_convfromdefl`, `retr_invmfromdefl`, `retr_psecconv`), plus critical curves and caustics (`retr_magn`, `retr_caustics`).
- Truncated-halo mass kernels (`retr_mcutfrommscl`, `retr_mcut`).
- Einstein-radius and self-lensing utilities (`retr_radieins_inft`, `retr_radieinsfromsmax`, `retr_radieinssbin`, `retr_amplslen`, `retr_magnpntslens`, `evaluate_self_lensing_model`) used by ephesos and PCAT.
- Cosmological lensing distances and critical density (`retr_adislenssour`, `retr_mdencrit`, `retr_ratimassbeinsqrd`).

## Native strong-lens models

`chalcedon.strong_lensing` provides composable `LensModel`, `LensEquationSolver`, `LensCosmo`, and `LensModelExtensions` interfaces. They calculate deflections, source-plane coordinates, numerical Jacobians, convergence, magnification, image positions, critical curves, and caustics. Supported mass profiles are SIS, SIE, EPL, point mass, shear, convergence, NFW, truncated NFW, and elliptical NFW/Hernquist. Coordinates, Einstein radii, and angular scale radii use arcseconds. Multiple lens planes use distance-weighted ray propagation with Astropy cosmology.

`chalcedon.imaging` supplies flux-normalized Sersic, Gaussian, Hernquist, interpolated-image, and Cartesian-shapelet light models, angular coordinate grids, point sources, and PSF convolution. The `ImageModel` evaluates the lens equation on supersampled coordinates, convolves the fine scene, and sums subpixel count rates into detector pixels. Mejiro can select these calculations with `backend="chalcedon"` without loading Lenstronomy. Mejiro also retains its separate Lenstronomy backend, which remains the default.

Elliptical halo profiles are integrated from their projected density, including inherited CSE parameter conventions. Their labels do not imply the old CSE approximation is used. Sersic half-light constants use the inverse incomplete gamma function. Finite-difference Jacobians and grid-seeded image solving require convergence checks for a user's accuracy requirements. Unsupported profiles, extinction/special rendering options, and JAX execution raise explicitly. Native NumPy/SciPy calculations do not need a separate lensing engine.

## Installation

```bash
cd /path/to/chalcedon
python -m pip install -e .
export CHALCEDON_PATH=/path/to/chalcedon
```

`CHALCEDON_PATH` identifies the repository root. Keep runtime inputs in `data/` and generated pipeline outputs in `visuals/`. Both directories are ignored by Git.

## Minimal usage

The runnable analytic example evaluates a host lens, one subhalo, and external shear on a normalized angular grid:

```bash
python examples/microlens_caustic.py --typefileplot png
```

The analytic example uses white backgrounds with black axes. Add
`--typeplotback dark` for black backgrounds with white axes, or pass
`typeplotback='dark'` to its `run_example()` function from Python.

![Chalcedon analytic deflection and magnification maps](examples/visuals/microlens_caustic.png)

The left panel exposes the intermediate deflection amplitude returned by `chalcedon.retr_caustics()`. The right panel shows the resulting magnification map and its contour candidates. Host and subhalo positions are marked explicitly. This is a deterministic lens-model prediction under the parameters encoded in the example, not an observed image.

## Truncated-halo mass

The enclosed-mass example evaluates the analytic kernel over positive cutoff-to-scale-radius ratios:

```bash
python examples/plot_truncation_mass.py --typefileplot png
python examples/plot_truncation_mass.py --typefileplot png --bokeh
```

The [`truncation_mass.ipynb` notebook](examples/truncation_mass.ipynb) calls the
same plotting function and shows the dimensionless mass ratios at selected
cutoff radii.

Add `--typeplotback dark` for black backgrounds with white axes in both the
static and HTML figures. The Python entry points also accept
`typeplotback='dark'` and inherit a surrounding TDpy theme when it is omitted.

![Dimensionless truncation mass across cutoff radius](examples/visuals/truncation_mass.png)

Install `python -m pip install -e '.[bokeh]'` to create the interactive companion at `examples/visuals/truncation_mass.html`.

`retr_mcutfrommscl(radius_ratio)` accepts scalars or NumPy arrays and returns the dimensionless mass ratio. At equal cutoff and scale radii, the ratio is $\pi/4 - 1/2$. The mass increases monotonically with cutoff radius for the assumed truncated profile. Use `retr_mcut` to calculate the corresponding halo mass from the profile normalization, angular radii, lens distance, and critical surface density. These are analytic model predictions, not measured halo masses.

## Inspecting the result
Look at the deflections and magnification alongside the rendered image. A convincing-looking arc is more useful when I can trace it back to the supplied mass and light profiles and check the numerical resolution.
