"""Render two synthetic source planes through one reference-normalized lens scene."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from astropy.cosmology import Planck18
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from chalcedon import ImageModel, LensModel, LightModel, PixelGrid, PSF
from chalcedon.imaging import make_grid_with_coordtransform
from chalcedon.paths import get_repository_path
from tdpy.cli import add_plot_arguments
from tdpy.plotting import save_bokeh_html, save_figure
from tdpy.verbosity import print


def build_scene():
    pixel_count = 101
    pixel_width = 0.03  # [arcsec]
    reference_redshift = 2.5
    lens_redshifts = [0.3, 0.7]
    source_redshifts = [0.5, 1.5]
    lens_parameters = [
        {'theta_E': 1.0},
        {'Rs': 0.12, 'alpha_Rs': 0.04, 'r_trunc': 0.5, 'center_x': 0.65, 'center_y': -0.2},
    ]
    source_parameters = [
        {'amp': 1.0, 'sigma': 0.045, 'center_x': 0.02, 'center_y': 0.01},
        {'amp': 0.7, 'sigma': 0.065, 'center_x': -0.04, 'center_y': 0.025},
    ]
    geometry = make_grid_with_coordtransform(pixel_count, pixel_width)
    grid = PixelGrid(pixel_count, pixel_count, geometry[2], geometry[3], geometry[6])
    lens = LensModel(['SIS', 'TNFW'], multi_plane=True, lens_redshift_list=lens_redshifts,
                     z_source=reference_redshift, cosmo=Planck18)
    psf = PSF('GAUSSIAN', fwhm=0.06)  # [arcsec]
    numerics = {'supersampling_factor': 3}
    images = []
    for redshift, parameters in zip(source_redshifts, source_parameters):
        renderer = ImageModel(grid, psf, lens, LightModel(['GAUSSIAN']), LightModel([]),
                              kwargs_numerics=numerics, source_redshift_list=[redshift])
        images.append(renderer.image(lens_parameters, [parameters], []))
    joint = ImageModel(grid, psf, lens, LightModel(['GAUSSIAN', 'GAUSSIAN']), LightModel([]),
                       kwargs_numerics=numerics, source_redshift_list=source_redshifts)
    images.append(joint.image(lens_parameters, source_parameters, []))
    half_width = 0.5 * pixel_count * pixel_width  # [arcsec]
    inputs = {
        'simulated': True, 'cosmology': 'Planck18', 'reference_source_redshift': reference_redshift,
        'lens_profiles': lens.lens_model_list, 'lens_redshifts': lens_redshifts,
        'source_redshifts': source_redshifts, 'lens_parameters': lens_parameters,
        'source_parameters': source_parameters, 'angular_unit': 'arcsec',
        'source_amplitude_unit': 'arbitrary integrated flux', 'psf_fwhm_arcsec': 0.06,
        'pixel_width_arcsec': pixel_width, 'supersampling_factor': 3,
    }
    return np.stack(images), (-half_width, half_width, -half_width, half_width), inputs


def compare_halo_redshifts():
    """Fit a lens-plane TNFW halo to a synthetic line-of-sight-halo image."""
    images, extent, inputs = build_scene()
    pixel_count = images.shape[-1]
    pixel_width = inputs['pixel_width_arcsec']  # [arcsec]
    geometry = make_grid_with_coordtransform(pixel_count, pixel_width)
    grid = PixelGrid(pixel_count, pixel_count, geometry[2], geometry[3], geometry[6])
    lens = LensModel(['SIS', 'TNFW'], multi_plane=True, lens_redshift_list=[0.3, 0.3],
                     z_source=inputs['reference_source_redshift'], cosmo=Planck18)
    renderers = [ImageModel(
        grid, PSF('GAUSSIAN', fwhm=inputs['psf_fwhm_arcsec']), lens,
        LightModel(['GAUSSIAN']), LightModel([]),
        kwargs_numerics={'supersampling_factor': inputs['supersampling_factor']},
        source_redshift_list=[redshift],
    ) for redshift in inputs['source_redshifts']]
    halo = inputs['lens_parameters'][1]
    initial = np.array([halo[name] for name in ('Rs', 'alpha_Rs', 'center_x', 'center_y')])

    def render(parameters, source_index):
        candidate = dict(halo, **dict(zip(('Rs', 'alpha_Rs', 'center_x', 'center_y'), parameters)))
        return renderers[source_index].image(
            [inputs['lens_parameters'][0], candidate], [inputs['source_parameters'][source_index]], [])

    scale = np.linalg.norm(images[1])
    fit = least_squares(lambda parameters: ((render(parameters, 1) - images[1]) / scale).ravel(),
                        initial, bounds=([pixel_width / 3, 0, extent[0], extent[2]],
                                         [extent[1], 1, extent[1], extent[3]]),
                        x_scale='jac', max_nfev=200)
    if not fit.success:
        raise RuntimeError(f'Halo-redshift comparison failed to converge: {fit.message}')
    before = np.stack([render(initial, source_index) for source_index in range(2)])
    fitted = np.stack([render(fit.x, source_index) for source_index in range(2)])
    truth = images[:2]
    fractions = lambda prediction: [float(np.linalg.norm(model - actual) / np.linalg.norm(actual))
                                    for model, actual in zip(prediction, truth)]
    metadata = dict(inputs, comparison_is_noiseless=True, fitted_halo_redshift=0.3,
                    fitted_source_redshift=1.5, held_out_source_redshift=0.5,
                    fitted_parameter_names=['Rs', 'alpha_Rs', 'center_x', 'center_y'],
                    fitted_parameters=fit.x.tolist(), initial_parameters=initial.tolist(),
                    fractional_l2_before=fractions(before), fractional_l2_after=fractions(fitted),
                    optimizer_evaluations=int(fit.nfev), source_light_and_macro_lens_fixed=True,
                    fit_active_bounds=fit.active_mask.tolist(),
                    optimizer='scipy.optimize.least_squares',
                    fit_lower_bounds=[pixel_width / 3, 0, extent[0], extent[2]],
                    fit_upper_bounds=[extent[1], 1, extent[1], extent[3]],
                    residual_definition='fitted lens-plane image minus line-of-sight image')
    return truth, before, fitted, extent, metadata


def run_halo_redshift_comparison(output_directory, typefileplot='png', typeplotback='white'):
    """Save measured image differences for a specified synthetic halo-redshift fit."""
    if typefileplot not in {'png', 'pdf'}:
        raise ValueError('typefileplot must be png or pdf')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    stem = Path(output_directory) / f'halo_redshift_comparison_{timestamp}'
    paths = {'figure': stem.with_suffix('.' + typefileplot), 'data': stem.with_suffix('.npz')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Refusing to overwrite an earlier halo-redshift comparison')
    print(f'Reading from {Path(__file__).name}...')
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    truth, before, fitted, extent, inputs = compare_halo_redshifts()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_digest:
        raise RuntimeError('Example source changed during the fit; outputs were left unwritten')
    inputs['source_sha256'] = source_digest
    inputs['fit_bounds_description'] = (
        'Angular scale from one third pixel to field half-width; nonnegative reduced '
        'normalization up to the 1 arcsec host Einstein angle; halo center inside the image field. '
        'Truncation radius stays fixed at 0.5 arcsec. These are illustrative fit bounds.')
    residual = fitted - truth
    maximum = float(max(truth.max(), fitted.max()))
    residual_scale = float(np.abs(residual).max())
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), sharex=True, sharey=True,
                                constrained_layout=True)
    for source_index, redshift in enumerate(inputs['source_redshifts']):
        for axis, image in zip(axes[source_index, :2], (truth[source_index], fitted[source_index])):
            brightness = axis.imshow(image, origin='lower', extent=extent, cmap='viridis',
                                     vmin=0, vmax=maximum)
        difference = axes[source_index, 2].imshow(
            residual[source_index], origin='lower', extent=extent, cmap='RdBu_r',
            vmin=-residual_scale, vmax=residual_scale)
        fraction = inputs['fractional_l2_after'][source_index]
        role = 'held out' if source_index == 0 else 'fitted'
        axes[source_index, 0].set_ylabel(f'Source z = {redshift} ({role})\nImage y [arcsec]')
        axes[source_index, 2].text(0.03, 0.96, f'Relative difference {fraction:.1%}',
                                   transform=axes[source_index, 2].transAxes, va='top')
    axes[0, 0].set_title('LOS halo at z = 0.7')
    axes[0, 1].set_title('Lens-plane fit at z = 0.3')
    axes[0, 2].set_title('Fit minus LOS image')
    for axis in axes.flat:
        axis.grid(False)
    for axis in axes[1]:
        axis.set_xlabel('Image x [arcsec]')
    figure.colorbar(brightness, ax=axes[:, :2].ravel().tolist(),
                    label='Flux per pixel [arbitrary units]', shrink=0.8)
    figure.colorbar(difference, ax=axes[:, 2].tolist(),
                    label='Flux difference [arbitrary units]', shrink=0.8)
    save_figure(figure, paths['figure'], typefileplot=typefileplot,
                typeplotback=typeplotback, close_figure=True)
    print(f"Writing to {paths['data']}...")
    np.savez_compressed(paths['data'], truth=truth, before=before, fitted=fitted,
                        residual=residual, extent=extent, inputs=json.dumps(inputs))
    return paths


def run_example(output_directory, typefileplot='png', typeplotback='white'):
    if typefileplot not in {'png', 'pdf'}:
        raise ValueError('typefileplot must be png or pdf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = Path(output_directory) / f'multiple_source_planes_{timestamp}'
    paths = {name: stem.with_suffix(suffix) for name, suffix in (
        ('figure', '.' + typefileplot), ('data', '.npz'), ('bokeh', '.html'),
    )}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Refusing to overwrite an earlier multi-source example')
    images, extent, inputs = build_scene()
    titles = ('Source at z = 0.5', 'Source at z = 1.5', 'Combined image')
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True,
                                constrained_layout=True)
    maximum = float(images.max())
    for axis, image, title in zip(axes, images, titles):
        artist = axis.imshow(image, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=maximum)
        axis.set(title=title, xlabel='Image x [arcsec]')
        axis.grid(False)
    axes[0].set_ylabel('Image y [arcsec]')
    figure.colorbar(artist, ax=axes, label='Flux per pixel [arbitrary units]', shrink=0.8)
    save_figure(figure, paths['figure'], typefileplot=typefileplot,
                typeplotback=typeplotback, close_figure=True)
    print(f"Writing to {paths['data']}...")
    np.savez_compressed(paths['data'], images=images, extent=extent, inputs=json.dumps(inputs))
    try:
        from bokeh.layouts import gridplot
        from bokeh.models import LinearColorMapper
        from bokeh.plotting import figure as bokeh_figure
    except ImportError:
        return paths
    mapper = LinearColorMapper(palette='Viridis256', low=0, high=maximum)
    plots = []
    for image, title in zip(images, titles):
        plot = bokeh_figure(title=title, width=340, height=340, match_aspect=True,
                            x_axis_label='Image x [arcsec]', y_axis_label='Image y [arcsec]')
        plot.image(image=[image], x=extent[0], y=extent[2], dw=extent[1]-extent[0],
                    dh=extent[3]-extent[2], color_mapper=mapper)
        plots.append(plot)
    save_bokeh_html(gridplot([plots]), paths['bokeh'], title='Chalcedon multiple source planes',
                    typeplotback=typeplotback)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    parser.add_argument('--output-directory', type=Path,
                        default=get_repository_path() / 'examples' / 'visuals')
    parser.add_argument('--compare-halos', action='store_true',
                        help='fit a lens-plane halo to a synthetic line-of-sight-halo image')
    arguments = parser.parse_args()
    runner = run_halo_redshift_comparison if arguments.compare_halos else run_example
    runner(arguments.output_directory, arguments.typefileplot, arguments.typeplotback)


if __name__ == '__main__':
    main()