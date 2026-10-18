"""Native strong-lens surface-brightness rendering and angular pixel grids."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaincinv, eval_hermite, factorial
from tdpy.numerics import gaussian_sigma_from_fwhm

from .strong_lensing import ellipticity_to_axis_ratio, projected_halo_density, LensEquationSolver


def magnitude2cps(magnitude, magnitude_zero_point):
    return 10.0**(-0.4 * (np.asarray(magnitude) - magnitude_zero_point))


def cps2magnitude(cps, magnitude_zero_point):
    if np.any(np.asarray(cps) <= 0):
        raise ValueError('Count rate must be positive')
    return magnitude_zero_point - 2.5 * np.log10(cps)


def make_grid_with_coordtransform(numPix, deltapix, subgrid_res=1, left_lower=False, inverse=False):
    if subgrid_res != 1 or left_lower or inverse:
        raise NotImplementedError('Use a centred angular grid with subgrid_res=1')
    origin = -0.5 * (numPix - 1) * deltapix
    coordinate = origin + np.arange(numPix) * deltapix
    image_x, image_y = np.meshgrid(coordinate, coordinate)
    matrix = np.eye(2) * deltapix
    return (image_x.ravel(), image_y.ravel(), origin, origin,
            -origin / deltapix, -origin / deltapix, matrix, np.linalg.inv(matrix))


class Coordinates:
    def __init__(self, transform_pix2angle, ra_at_xy_0, dec_at_xy_0):
        self.transform_pix2angle = np.asarray(transform_pix2angle)
        self.transform_angle2pix = np.linalg.inv(self.transform_pix2angle)
        self.origin = np.array([ra_at_xy_0, dec_at_xy_0])

    def map_coord2pix(self, ra, dec):
        coordinates = np.broadcast_arrays(np.asarray(ra), np.asarray(dec))
        shape = coordinates[0].shape
        pixels = self.transform_angle2pix @ (np.stack(coordinates).reshape(2, -1) - self.origin[:, None])
        return pixels[0].reshape(shape), pixels[1].reshape(shape)

    def map_pix2coord(self, x, y):
        pixels = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        shape = pixels[0].shape
        coordinates = self.transform_pix2angle @ np.stack(pixels).reshape(2, -1) + self.origin[:, None]
        return coordinates[0].reshape(shape), coordinates[1].reshape(shape)


class PixelGrid(Coordinates):
    def __init__(self, nx, ny, ra_at_xy_0, dec_at_xy_0, transform_pix2angle):
        super().__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self.num_pixel_axes = (nx, ny)
        self.pixel_width = np.sqrt(abs(np.linalg.det(self.transform_pix2angle)))
        self.pixel_area = abs(np.linalg.det(self.transform_pix2angle))
        columns, rows = np.meshgrid(np.arange(nx), np.arange(ny))
        self.pixel_coordinates = self.map_pix2coord(columns.ravel(), rows.ravel())


class LightModel:
    """Evaluate and normalize explicit extended-source brightness profiles."""

    def __init__(self, light_model_list):
        self.light_model_list = list(light_model_list)
        supported = {'SERSIC', 'SERSIC_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE',
                     'INTERPOL', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'SHAPELETS'}
        if set(self.light_model_list) - supported:
            raise NotImplementedError('Unsupported Chalcedon light profiles: %s' % sorted(set(self.light_model_list) - supported))

    def _profile(self, x, y, model, parameters):
        offset_x = np.asarray(x) - parameters.get('center_x', 0.0)
        offset_y = np.asarray(y) - parameters.get('center_y', 0.0)
        angle, axis_ratio = ellipticity_to_axis_ratio(parameters.get('e1', 0.0), parameters.get('e2', 0.0))
        cosine, sine = np.cos(angle), np.sin(angle)
        major = cosine * offset_x + sine * offset_y
        minor = -sine * offset_x + cosine * offset_y
        radius = np.sqrt(axis_ratio * major**2 + minor**2 / axis_ratio)
        amplitude = parameters.get('amp', 1.0)
        if model.startswith('SERSIC'):
            index = parameters['n_sersic']
            scale = parameters['R_sersic']
            if index <= 0 or scale <= 0:
                raise ValueError('Sersic index and half-light radius must be positive')
            concentration = gammaincinv(2.0 * index, 0.5)
            return amplitude * np.exp(-concentration * ((radius / scale)**(1.0 / index) - 1.0))
        if model.startswith('GAUSSIAN'):
            width = parameters['sigma']
            return amplitude * np.exp(-0.5 * (radius / width)**2) / (2.0 * np.pi * width**2)
        if model.startswith('HERNQUIST'):
            return amplitude * projected_halo_density(np.maximum(radius, 1e-10), 'HERNQUIST',
                                                      {'Rs': parameters['Rs'], 'sigma0': 1.0})
        if model == 'INTERPOL':
            image = np.asarray(parameters['image'], dtype=float)
            scale = parameters['scale']
            rotation = parameters.get('phi_G', 0.0)
            image_x = (np.cos(rotation) * offset_x + np.sin(rotation) * offset_y) / scale + 0.5 * (image.shape[1] - 1)
            image_y = (-np.sin(rotation) * offset_x + np.cos(rotation) * offset_y) / scale + 0.5 * (image.shape[0] - 1)
            shape = np.broadcast_shapes(np.shape(image_x), np.shape(image_y))
            coordinates = np.vstack((np.broadcast_to(image_y, shape).ravel(), np.broadcast_to(image_x, shape).ravel()))
            return amplitude * map_coordinates(image, coordinates, order=1, mode='constant', cval=0.0).reshape(shape)
        scale = parameters['beta']
        amplitudes = np.atleast_1d(amplitude)
        result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
        position = 0
        for order in range(parameters['n_max'] + 1):
            for first_order in range(order, -1, -1):
                second_order = order - first_order
                first = eval_hermite(first_order, offset_x / scale) * np.exp(-0.5 * (offset_x / scale)**2)
                second = eval_hermite(second_order, offset_y / scale) * np.exp(-0.5 * (offset_y / scale)**2)
                normalization = np.sqrt(2.0**order * factorial(first_order) * factorial(second_order) * np.pi)
                result += amplitudes[position] * first * second / normalization
                position += 1
        return result

    def surface_brightness(self, x, y, kwargs_list, k=None):
        if len(kwargs_list) != len(self.light_model_list):
            raise ValueError('Provide one parameter dictionary per light profile')
        selected = range(len(kwargs_list)) if k is None else [k]
        result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
        for index in selected:
            result += self._profile(x, y, self.light_model_list[index], kwargs_list[index])
        return result

    def total_flux(self, kwargs_list, norm=False, k=None):
        fluxes = []
        selected = range(len(kwargs_list)) if k is None else [k]
        for index in selected:
            parameters = kwargs_list[index]
            model = self.light_model_list[index]
            amplitude = 1.0 if norm else parameters.get('amp', 1.0)
            if model.startswith('SERSIC'):
                order = parameters['n_sersic']
                concentration = gammaincinv(2.0 * order, 0.5)
                value = 2.0 * np.pi * parameters['R_sersic']**2 * order * np.exp(concentration) * gamma(2.0 * order) / concentration**(2.0 * order)
            elif model.startswith('GAUSSIAN'):
                value = 1.0
            elif model.startswith('HERNQUIST'):
                value = 2.0 * np.pi * parameters['Rs']**2
            elif model == 'INTERPOL':
                value = np.asarray(parameters['image']).sum() * parameters['scale']**2
            else:
                amplitudes = np.atleast_1d(parameters['amp'])
                value = 0.0
                position = 0
                for order in range(parameters['n_max'] + 1):
                    for first in range(order, -1, -1):
                        second = order - first
                        if first % 2 == 0 and second % 2 == 0:
                            integral = 2.0 * np.sqrt(np.pi) * np.sqrt(factorial(first) * factorial(second)) / (2.0**(order / 2.0) * factorial(first // 2) * factorial(second // 2))
                            value += amplitudes[position] * parameters['beta']**2 * integral
                        position += 1
                amplitude = 1.0
            fluxes.append(amplitude * value)
        return fluxes


def magnitude2amplitude(light_model_class, kwargs_light_mag, magnitude_zero_point):
    parameters = deepcopy(kwargs_light_mag)
    normalization = light_model_class.total_flux(parameters, norm=True)
    for item, flux in zip(parameters, normalization):
        if 'magnitude' not in item:
            continue
        if flux <= 0:
            raise ValueError('A magnitude requires a positive total light-profile flux')
        item['amp'] = magnitude2cps(item.pop('magnitude'), magnitude_zero_point) / flux
    return parameters


class PSF:
    def __init__(self, psf_type='NONE', **kwargs):
        if psf_type not in {'NONE', 'GAUSSIAN', 'PIXEL'}:
            raise NotImplementedError('Unsupported native PSF type %r' % psf_type)
        self.psf_type = psf_type
        self.kwargs = kwargs
        self.input_kernel = None
        self.kernel_point_source = None
        if psf_type == 'PIXEL':
            kernel = np.asarray(kwargs['kernel_point_source'], dtype=float)
            if kernel.ndim != 2 or not np.isfinite(kernel).all() or kernel.sum() <= 0:
                raise ValueError('PSF kernel must be a finite two-dimensional array with positive sum')
            factor = kwargs.get('point_source_supersampling_factor', 1)
            if not isinstance(factor, (int, np.integer)) or factor < 1:
                raise ValueError('PSF sampling factor must be a positive integer')
            self.input_kernel = (kernel / kernel.sum()
                                 if kwargs.get('kernel_point_source_normalisation', True) else kernel)
            if factor != 1:
                kernel = zoom(kernel, 1.0 / factor, order=1)
            self.kernel_point_source = kernel / kernel.sum() * self.input_kernel.sum()

    def convolve(self, image, pixel_scale, sampling_factor=1):
        if self.psf_type == 'PIXEL':
            factor = self.kwargs.get('point_source_supersampling_factor', 1)
            kernel = self.input_kernel
            if sampling_factor != factor:
                kernel = zoom(kernel, sampling_factor / factor, order=1)
                kernel = kernel / kernel.sum() * self.input_kernel.sum()
            return fftconvolve(image, kernel, mode='same')
        if self.psf_type == 'GAUSSIAN':
            sigma = gaussian_sigma_from_fwhm(self.kwargs['fwhm']) / pixel_scale
            return gaussian_filter(image, sigma, mode='constant')
        return image


class PointSource:
    def __init__(self, point_source_type_list, lens_model, fixed_magnification_list=None):
        self.models = list(point_source_type_list)
        self.lens_model = lens_model
        if set(self.models) - {'LENSED_POSITION', 'UNLENSED', 'SOURCE_POSITION'}:
            raise NotImplementedError('Unsupported point-source parameterization')


class ImageModel:
    """Ray-shoot extended light and integrate count rates over detector pixels."""

    def __init__(self, data_class, psf_class, lens_model_class, source_model_class,
                 lens_light_model_class, point_source_class=None, kwargs_numerics=None):
        self.grid = data_class
        self.psf = psf_class
        self.lens = lens_model_class
        self.source = source_model_class
        self.lens_light = lens_light_model_class
        self.point_source = point_source_class
        self.numerics = kwargs_numerics or {}

    def _render(self, light_model, parameters, kwargs_lens=None, unconvolved=False):
        columns, rows = self.grid.num_pixel_axes
        factor = int(self.numerics.get('supersampling_factor', 1))
        if factor < 1:
            raise ValueError('Supersampling factor must be positive')
        fine_columns, fine_rows = np.meshgrid((np.arange(columns * factor) + 0.5) / factor - 0.5,
                                              (np.arange(rows * factor) + 0.5) / factor - 0.5)
        image_x, image_y = self.grid.map_pix2coord(fine_columns.ravel(), fine_rows.ravel())
        singular = np.zeros(image_x.shape, dtype=bool)
        if kwargs_lens is not None:
            angular_x, angular_y = image_x, image_y
            for mass_profile, lens_parameters in zip(self.lens.lens_model_list, kwargs_lens):
                if mass_profile in {'SIS', 'SIE', 'EPL', 'POINT_MASS'}:
                    singular |= (np.isclose(angular_x, lens_parameters.get('center_x', 0.0), atol=1e-12, rtol=0.0)
                                 & np.isclose(angular_y, lens_parameters.get('center_y', 0.0), atol=1e-12, rtol=0.0))
            image_x, image_y = self.lens.ray_shooting(image_x, image_y, kwargs_lens)
        flux = light_model.surface_brightness(image_x, image_y, parameters)
        if np.any(singular):
            averaged = np.zeros(np.count_nonzero(singular))
            for offset_x, offset_y in ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)):
                offset = self.grid.transform_pix2angle @ np.array([offset_x, offset_y]) / factor
                source_x, source_y = self.lens.ray_shooting(angular_x[singular] + offset[0],
                                                          angular_y[singular] + offset[1], kwargs_lens)
                averaged += light_model.surface_brightness(source_x, source_y, parameters)
            flux[singular] = averaged / 4.0
        fine_image = (flux * self.grid.pixel_area / factor**2).reshape(rows * factor, columns * factor)
        if self.numerics.get('flux_evaluate_indexes') is not None:
            mask = np.asarray(self.numerics['flux_evaluate_indexes']).reshape(rows, columns)
            fine_image *= np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
        fine_convolution = self.numerics.get('supersampling_convolution', True)
        if not unconvolved and fine_convolution:
            fine_image = self.psf.convolve(fine_image, self.grid.pixel_width / factor, factor)
        image = fine_image.reshape(rows, factor, columns, factor).sum(axis=(1, 3))
        return image if unconvolved or fine_convolution else self.psf.convolve(image, self.grid.pixel_width)

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False):
        return self._render(self.lens_light, kwargs_lens_light, unconvolved=unconvolved)

    def source_surface_brightness(self, kwargs_source, kwargs_lens, kwargs_extinction=None,
                                  kwargs_special=None, unconvolved=False):
        if kwargs_extinction or kwargs_special:
            raise NotImplementedError('Extinction and special rendering options require explicit native support')
        return self._render(self.source, kwargs_source, kwargs_lens, unconvolved)

    def image(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps=None,
              kwargs_extinction=None, kwargs_special=None, unconvolved=False,
              source_add=True, lens_light_add=True, point_source_add=True):
        columns, rows = self.grid.num_pixel_axes
        result = np.zeros((rows, columns))
        if lens_light_add:
            result += self.lens_surface_brightness(kwargs_lens_light, unconvolved)
        if source_add:
            result += self.source_surface_brightness(kwargs_source, kwargs_lens, kwargs_extinction, kwargs_special, unconvolved)
        if point_source_add and kwargs_ps:
            points = np.zeros_like(result)
            if self.point_source is None:
                raise ValueError('Point-source parameters require a point-source model')
            for model, parameters in zip(self.point_source.models, kwargs_ps):
                if model == 'SOURCE_POSITION':
                    image_x, image_y = LensEquationSolver(self.lens).image_position_from_source(parameters['ra_source'], parameters['dec_source'], kwargs_lens)
                    amplitudes = parameters['source_amp'] * np.abs(self.lens.magnification(image_x, image_y, kwargs_lens))
                else:
                    image_x, image_y = parameters['ra_image'], parameters['dec_image']
                    amplitudes = parameters['point_amp']
                pixel_x, pixel_y = self.grid.map_coord2pix(image_x, image_y)
                for position_x, position_y, amplitude in zip(np.atleast_1d(pixel_x), np.atleast_1d(pixel_y), np.broadcast_to(amplitudes, np.shape(pixel_x))):
                    column = int(np.floor(position_x))
                    row = int(np.floor(position_y))
                    for shift_y, shift_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        if 0 <= row + shift_y < rows and 0 <= column + shift_x < columns:
                            weight_x = position_x - column if shift_x else 1.0 - position_x + column
                            weight_y = position_y - row if shift_y else 1.0 - position_y + row
                            points[row + shift_y, column + shift_x] += amplitude * weight_x * weight_y
            result += points if unconvolved else self.psf.convolve(points, self.grid.pixel_width)
        return result


data_util = SimpleNamespace(magnitude2cps=magnitude2cps, cps2magnitude=cps2magnitude,
                            magnitude2amplitude=magnitude2amplitude)
util = SimpleNamespace(make_grid_with_coordtransform=make_grid_with_coordtransform)