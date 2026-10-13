"""Composable strong-lensing models with angular coordinates in arcseconds."""

import numpy as np
from scipy.special import hyp2f1
from functools import lru_cache


def _projected_radial_function(radius):
    radius = np.maximum(np.asarray(radius, dtype=float), 1e-12)
    values = np.ones_like(radius)
    inner = radius < 1.0
    outer = radius > 1.0
    values[inner] = np.arccosh(1.0 / radius[inner]) / np.sqrt(1.0 - radius[inner]**2)
    values[outer] = np.arccos(1.0 / radius[outer]) / np.sqrt(radius[outer]**2 - 1.0)
    return values


def projected_halo_density(radius, profile, parameters):
    """Return the projected NFW or Hernquist convergence at circular radius."""

    scaled = np.maximum(np.asarray(radius, dtype=float) / parameters['Rs'], 1e-12)
    radial = _projected_radial_function(scaled)
    near_scale = np.abs(scaled - 1.0) < 1e-5
    if profile.startswith('NFW'):
        normalization = parameters['alpha_Rs'] / (2.0 * parameters['Rs'] * (1.0 - np.log(2.0)))
        ratio = np.divide(1.0 - radial, scaled**2 - 1.0,
                          out=np.full_like(scaled, 1.0 / 3.0), where=~near_scale)
        return normalization * ratio
    ratio = np.divide((2.0 + scaled**2) * radial - 3.0, (scaled**2 - 1.0)**2,
                      out=np.full_like(scaled, 4.0 / 15.0), where=~near_scale)
    return parameters['sigma0'] * ratio


@lru_cache(maxsize=1)
def _halo_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = 0.5 * (nodes + 1.0)
    return nodes**2, weights * nodes


def _elliptical_halo_deflection(x, y, profile, parameters):
    angle, axis_ratio = ellipticity_to_axis_ratio(parameters.get('e1', 0.0), parameters.get('e2', 0.0))
    if profile == 'HERNQUIST_ELLIPSE_CSE':
        parameters = {**parameters, 'Rs': parameters['Rs'] * np.sqrt(axis_ratio),
                      'sigma0': parameters['sigma0'] / axis_ratio}
    cosine, sine = np.cos(angle), np.sin(angle)
    major = cosine * x + sine * y
    minor = -sine * x + cosine * y
    integral_x = np.zeros_like(major)
    integral_y = np.zeros_like(minor)
    nodes, weights = _halo_quadrature()
    for node, weight in zip(nodes, weights):
        denominator = 1.0 - (1.0 - axis_ratio**2) * node
        radius = np.sqrt(axis_ratio * node * (major**2 + minor**2 / denominator))
        density = projected_halo_density(radius, profile, parameters)
        integral_x += weight * density / np.sqrt(denominator)
        integral_y += weight * density / denominator**1.5
    aligned_x = axis_ratio * major * integral_x
    aligned_y = axis_ratio * minor * integral_y
    return cosine * aligned_x - sine * aligned_y, sine * aligned_x + cosine * aligned_y


def ellipticity_to_axis_ratio(e1=0.0, e2=0.0):
    ellipticity = np.hypot(e1, e2)
    if not np.isfinite(ellipticity) or ellipticity >= 1.0:
        raise ValueError('Ellipticity magnitude must be finite and below one')
    return 0.5 * np.arctan2(e2, e1), (1.0 - ellipticity) / (1.0 + ellipticity)


def _power_law_deflection(x, y, parameters):
    slope = parameters.get('gamma', 2.0) - 1.0
    if not 0.0 < slope < 2.0 or parameters['theta_E'] <= 0:
        raise ValueError('EPL requires theta_E > 0 and 1 < gamma < 3')
    angle, axis_ratio = ellipticity_to_axis_ratio(parameters.get('e1', 0.0), parameters.get('e2', 0.0))
    cosine, sine = np.cos(angle), np.sin(angle)
    major = cosine * x + sine * y
    minor = -sine * x + cosine * y
    radius = np.hypot(axis_ratio * major, minor)
    safe_radius = np.maximum(radius, 1e-12)
    scale = parameters['theta_E'] * np.sqrt(axis_ratio)
    phase = (axis_ratio * major + 1j * minor) / safe_radius
    ellipticity = (1.0 - axis_ratio) / (1.0 + axis_ratio)
    deflection = (2.0 * scale / (1.0 + axis_ratio) * (scale / safe_radius)**(slope - 1.0)
                  * phase * hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -ellipticity * phase**2))
    return (cosine * deflection.real - sine * deflection.imag,
            sine * deflection.real + cosine * deflection.imag)


class LensModel:
    """Evaluate a sum of explicitly supported lens-plane mass profiles."""

    def __init__(self, lens_model_list, *, use_jax=None, multi_plane=False,
                 z_lens=None, z_source=None, lens_redshift_list=None, cosmo=None):
        self.lens_model_list = list(lens_model_list)
        supported = {'SIS', 'SIE', 'EPL', 'SHEAR', 'SHEAR_GAMMA_PSI', 'CONVERGENCE', 'POINT_MASS', 'TNFW',
                 'NFW', 'NFW_ELLIPSE_CSE', 'HERNQUIST', 'HERNQUIST_ELLIPSE_CSE'}
        unsupported = set(self.lens_model_list) - supported
        if unsupported:
            raise NotImplementedError('Unsupported Chalcedon mass profiles: %s' % sorted(unsupported))
        self.multi_plane = multi_plane
        self.z_source = z_source
        self.cosmo = cosmo
        self.lens_redshift_list = lens_redshift_list
        if multi_plane:
            if (cosmo is None or z_source is None or lens_redshift_list is None
                    or len(lens_redshift_list) != len(self.lens_model_list)
                    or any(not 0 < redshift < z_source for redshift in lens_redshift_list)):
                raise ValueError('Multiple lens planes require valid redshifts and an Astropy cosmology')
        if use_jax is True or (isinstance(use_jax, list) and any(use_jax)):
            raise NotImplementedError('Chalcedon uses its native NumPy backend')

    def alpha(self, x, y, kwargs_lens):
        if self.multi_plane:
            source_x, source_y = self.ray_shooting(x, y, kwargs_lens)
            return np.asarray(x) - source_x, np.asarray(y) - source_y
        if len(kwargs_lens) != len(self.lens_model_list):
            raise ValueError('Provide one parameter dictionary per mass profile')
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        alpha_x = np.zeros_like(x)
        alpha_y = np.zeros_like(y)
        for model, parameters in zip(self.lens_model_list, kwargs_lens):
            offset_x = x - parameters.get('center_x', parameters.get('ra_0', 0.0))
            offset_y = y - parameters.get('center_y', parameters.get('dec_0', 0.0))
            radius = np.hypot(offset_x, offset_y)
            if model == 'SIS':
                scale = np.divide(parameters['theta_E'], radius,
                                  out=np.zeros_like(radius), where=radius > 0)
                alpha_x += scale * offset_x
                alpha_y += scale * offset_y
            elif model in {'SIE', 'EPL'}:
                profile_parameters = dict(parameters)
                if model == 'SIE':
                    profile_parameters['gamma'] = 2.0
                profile_x, profile_y = _power_law_deflection(offset_x, offset_y, profile_parameters)
                alpha_x += profile_x
                alpha_y += profile_y
            elif model.startswith('NFW') or model.startswith('HERNQUIST'):
                profile_x, profile_y = _elliptical_halo_deflection(offset_x, offset_y, model, parameters)
                alpha_x += profile_x
                alpha_y += profile_y
            elif model == 'TNFW':
                from .main import retr_deflcutf

                safe_radius = np.maximum(radius, 1e-8 * parameters['Rs'])
                amplitude = parameters['alpha_Rs'] / (1.0 - np.log(2.0))
                radial = retr_deflcutf(safe_radius, amplitude, parameters['Rs'], parameters['r_trunc'])
                alpha_x += radial * offset_x / safe_radius
                alpha_y += radial * offset_y / safe_radius
            elif model == 'POINT_MASS':
                scale = np.divide(parameters['theta_E']**2, radius**2,
                                  out=np.zeros_like(radius), where=radius > 0)
                alpha_x += scale * offset_x
                alpha_y += scale * offset_y
            elif model in {'SHEAR', 'SHEAR_GAMMA_PSI'}:
                if model == 'SHEAR_GAMMA_PSI':
                    gamma1 = parameters['gamma_ext'] * np.cos(2.0 * parameters['psi_ext'])
                    gamma2 = parameters['gamma_ext'] * np.sin(2.0 * parameters['psi_ext'])
                else:
                    gamma1 = parameters.get('gamma1', 0.0)
                    gamma2 = parameters.get('gamma2', 0.0)
                alpha_x += gamma1 * offset_x + gamma2 * offset_y
                alpha_y += gamma2 * offset_x - gamma1 * offset_y
            elif model == 'CONVERGENCE':
                alpha_x += parameters['kappa'] * offset_x
                alpha_y += parameters['kappa'] * offset_y
        return alpha_x, alpha_y

    def ray_shooting(self, x, y, kwargs_lens):
        if self.multi_plane:
            if len(kwargs_lens) != len(self.lens_model_list):
                raise ValueError('Provide one parameter dictionary per mass profile')
            image_x, image_y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            deflections = []
            redshifts = sorted(set(self.lens_redshift_list))
            source_distance = self.cosmo.angular_diameter_distance(self.z_source).value
            for redshift in redshifts:
                plane_x, plane_y = image_x.copy(), image_y.copy()
                plane_distance = self.cosmo.angular_diameter_distance(redshift).value
                for earlier_redshift, earlier_x, earlier_y in deflections:
                    between = self.cosmo.angular_diameter_distance_z1z2(earlier_redshift, redshift).value
                    to_source = self.cosmo.angular_diameter_distance_z1z2(earlier_redshift, self.z_source).value
                    propagation = between * source_distance / (plane_distance * to_source)
                    plane_x -= propagation * earlier_x
                    plane_y -= propagation * earlier_y
                selected = [index for index, value in enumerate(self.lens_redshift_list) if value == redshift]
                plane_model = LensModel([self.lens_model_list[index] for index in selected])
                plane_alpha = plane_model.alpha(plane_x, plane_y, [kwargs_lens[index] for index in selected])
                deflections.append((redshift, *plane_alpha))
            return (image_x - sum(item[1] for item in deflections),
                    image_y - sum(item[2] for item in deflections))
        alpha_x, alpha_y = self.alpha(x, y, kwargs_lens)
        return np.asarray(x) - alpha_x, np.asarray(y) - alpha_y

    def hessian(self, x, y, kwargs_lens, diff=1e-5):
        plus_x = self.alpha(np.asarray(x) + diff, y, kwargs_lens)
        minus_x = self.alpha(np.asarray(x) - diff, y, kwargs_lens)
        plus_y = self.alpha(x, np.asarray(y) + diff, kwargs_lens)
        minus_y = self.alpha(x, np.asarray(y) - diff, kwargs_lens)
        return ((plus_x[0] - minus_x[0]) / (2.0 * diff),
                (plus_y[0] - minus_y[0]) / (2.0 * diff),
                (plus_x[1] - minus_x[1]) / (2.0 * diff),
                (plus_y[1] - minus_y[1]) / (2.0 * diff))

    def kappa(self, x, y, kwargs_lens):
        hessian_xx, _, _, hessian_yy = self.hessian(x, y, kwargs_lens)
        return 0.5 * (hessian_xx + hessian_yy)

    def magnification(self, x, y, kwargs_lens):
        hessian_xx, hessian_xy, hessian_yx, hessian_yy = self.hessian(x, y, kwargs_lens)
        determinant = (1.0 - hessian_xx) * (1.0 - hessian_yy) - hessian_xy * hessian_yx
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / determinant


class LensEquationSolver:
    """Find distinct point-source images with native lens-equation root solving."""

    def __init__(self, lens_model):
        self.lens_model = lens_model

    def image_position_from_source(self, source_x, source_y, kwargs_lens,
                                   search_window=6.0, min_distance=0.01, solver=None):
        from scipy.optimize import root

        if search_window <= 0 or min_distance <= 0:
            raise ValueError('Search window and minimum separation must be positive')
        centers = np.asarray([[item.get('center_x', 0.0), item.get('center_y', 0.0)]
                              for item in kwargs_lens])
        center = centers.mean(axis=0) if centers.size else np.zeros(2)
        grid = np.linspace(-0.5 * search_window, 0.5 * search_window, 25)
        solutions = []

        def equation(position):
            source = self.lens_model.ray_shooting(position[0], position[1], kwargs_lens)
            return np.asarray(source) - [source_x, source_y]

        for offset_x in grid:
            for offset_y in grid:
                candidate = root(equation, center + [offset_x, offset_y])
                if (not candidate.success or np.linalg.norm(equation(candidate.x)) > 1e-7
                        or np.any(np.abs(candidate.x - center) > 0.5 * search_window)):
                    continue
                if not any(np.linalg.norm(candidate.x - existing) < min_distance for existing in solutions):
                    solutions.append(candidate.x)
        if not solutions:
            return np.array([]), np.array([])
        positions = np.asarray(solutions)
        order = np.argsort(-np.abs(self.lens_model.magnification(positions[:, 0], positions[:, 1], kwargs_lens)))
        return positions[order, 0], positions[order, 1]


class LensCosmo:
    """Astropy cosmological distances and physical SIS lens conversions."""

    def __init__(self, z_lens, z_source, cosmo):
        from astropy.constants import G, c
        import astropy.units as u

        if not 0 < z_lens < z_source:
            raise ValueError('Require 0 < lens redshift < source redshift')
        self.dd = cosmo.angular_diameter_distance(z_lens).to_value(u.Mpc)
        self.ds = cosmo.angular_diameter_distance(z_source).to_value(u.Mpc)
        self.dds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to_value(u.Mpc)
        self.sigma_crit = (c**2 / (4.0 * np.pi * G) * (self.ds / (self.dd * self.dds)) / u.Mpc).to_value(u.M_sun / u.Mpc**2)

    def sis_sigma_v2theta_E(self, sigma_v):
        from astropy.constants import c
        import astropy.units as u

        return (4.0 * np.pi * (np.asarray(sigma_v) / c.to_value(u.km / u.s))**2
                * self.dds / self.ds * u.rad).to_value(u.arcsec)

    def sis_theta_E2sigma_v(self, theta_E):
        from astropy.constants import c
        import astropy.units as u

        angle = (np.asarray(theta_E) * u.arcsec).to_value(u.rad)
        return c.to_value(u.km / u.s) * np.sqrt(angle * self.ds / (4.0 * np.pi * self.dds))


class LensModelExtensions:
    """Critical curves and source-plane caustics from native model Jacobians."""

    def __init__(self, lens_model):
        self.lens_model = lens_model

    def critical_curve_caustics(self, kwargs_lens, compute_window=5.0, grid_scale=0.02,
                                center_x=0.0, center_y=0.0):
        from skimage.measure import find_contours

        side = max(3, int(np.ceil(compute_window / grid_scale)) + 1)
        coordinate = np.linspace(-0.5 * compute_window, 0.5 * compute_window, side)
        grid_x, grid_y = np.meshgrid(coordinate + center_x, coordinate + center_y)
        hessian_xx, hessian_xy, hessian_yx, hessian_yy = self.lens_model.hessian(grid_x, grid_y, kwargs_lens)
        determinant = (1.0 - hessian_xx) * (1.0 - hessian_yy) - hessian_xy * hessian_yx
        critical_x, critical_y, caustic_x, caustic_y = [], [], [], []
        for contour in find_contours(determinant, 0.0):
            image_x = np.interp(contour[:, 1], np.arange(side), coordinate) + center_x
            image_y = np.interp(contour[:, 0], np.arange(side), coordinate) + center_y
            source_x, source_y = self.lens_model.ray_shooting(image_x, image_y, kwargs_lens)
            critical_x.append(image_x)
            critical_y.append(image_y)
            caustic_x.append(source_x)
            caustic_y.append(source_y)
        return critical_x, critical_y, caustic_x, caustic_y