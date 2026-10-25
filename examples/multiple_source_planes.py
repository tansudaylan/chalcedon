"""Render two synthetic source planes through one reference-normalized lens scene."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from astropy.cosmology import Planck18
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

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
    arguments = parser.parse_args()
    run_example(arguments.output_directory, arguments.typefileplot, arguments.typeplotback)


if __name__ == '__main__':
    main()