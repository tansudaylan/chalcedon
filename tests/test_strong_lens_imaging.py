import numpy as np
import pytest

from chalcedon import LensModel, LightModel, PixelGrid, PSF, ImageModel
from chalcedon.imaging import magnitude2amplitude, make_grid_with_coordtransform


def test_native_light_profiles_have_correct_total_flux_normalization():
    light = LightModel(['SERSIC_ELLIPSE'])
    parameters = [{'R_sersic': 0.3, 'n_sersic': 1.0, 'e1': 0.2, 'e2': 0.1, 'magnitude': 23.0}]
    normalized = magnitude2amplitude(light, parameters, 25.0)

    assert light.total_flux(normalized)[0] == pytest.approx(10.0**0.8)
    assert 'magnitude' in parameters[0]


def test_native_image_ray_shooting_and_pieces_produce_an_einstein_ring():
    geometry = make_grid_with_coordtransform(101, 0.04)
    grid = PixelGrid(101, 101, geometry[2], geometry[3], geometry[6])
    model = ImageModel(grid, PSF(), LensModel(['SIS']), LightModel(['GAUSSIAN']), LightModel(['GAUSSIAN']),
                       kwargs_numerics={'supersampling_factor': 3})
    mass = [{'theta_E': 1.0}]
    source = [{'amp': 1.0, 'sigma': 0.05}]
    lens_light = [{'amp': 0.1, 'sigma': 0.3}]
    image = model.image(mass, source, lens_light)
    source_piece = model.source_surface_brightness(source, mass)
    lens_piece = model.lens_surface_brightness(lens_light)

    np.testing.assert_allclose(image, source_piece + lens_piece)
    coordinates = geometry[0].reshape(101, 101), geometry[1].reshape(101, 101)
    radii = np.hypot(*coordinates)
    assert radii.ravel()[source_piece.argmax()] == pytest.approx(1.0, abs=0.05)
    assert source_piece.sum() > 1.0
    assert source_piece[50, 50] < 1e-20


def test_multiple_source_planes_render_a_shared_physical_mass_distribution():
    from astropy.cosmology import Planck18

    geometry = make_grid_with_coordtransform(31, 0.05)
    grid = PixelGrid(31, 31, geometry[2], geometry[3], geometry[6])
    reference_redshift = 2.0
    lens = LensModel(['CONVERGENCE', 'CONVERGENCE'], multi_plane=True,
                     lens_redshift_list=[0.3, 0.8], z_source=reference_redshift, cosmo=Planck18)
    parameters = [{'kappa': 0.1}, {'kappa': 0.2}]
    sources = [{'amp': 1.0, 'sigma': 0.1, 'center_x': 0.08},
               {'amp': 0.6, 'sigma': 0.16, 'center_y': -0.1}]
    source_redshifts = [0.5, 1.5]
    renderer = ImageModel(grid, PSF(), lens, LightModel(['GAUSSIAN', 'GAUSSIAN']),
                          LightModel([]), source_redshift_list=source_redshifts)
    actual = renderer.image(parameters, sources, [])
    distance = Planck18.comoving_distance
    reference = distance(reference_redshift).value
    lens_distances = [distance(redshift).value for redshift in (0.3, 0.8)]
    physical = [value['kappa'] * reference / (reference - position)
                for value, position in zip(parameters, lens_distances)]
    second_position = 1.0 - physical[0] * (lens_distances[1] - lens_distances[0]) / lens_distances[1]
    expected = np.zeros(31 * 31)
    for redshift, source in zip(source_redshifts, sources):
        source_distance = distance(redshift).value
        factor = 1.0
        for position, strength, ray_position in zip(lens_distances, physical, [1.0, second_position]):
            if position < source_distance:
                factor -= strength * ray_position * (source_distance - position) / source_distance
        expected += LightModel(['GAUSSIAN']).surface_brightness(
            geometry[0] * factor, geometry[1] * factor, [source],
        ) * grid.pixel_area
    np.testing.assert_allclose(actual, expected.reshape(31, 31), rtol=1e-13, atol=1e-15)


def test_multiple_source_rendering_preserves_psf_additivity_and_lens_light():
    from astropy.cosmology import Planck18

    geometry = make_grid_with_coordtransform(31, 0.05)
    grid = PixelGrid(31, 31, geometry[2], geometry[3], geometry[6])
    lens = LensModel(['SIS'], multi_plane=True, lens_redshift_list=[0.3], z_source=2.0, cosmo=Planck18)
    psf = PSF('GAUSSIAN', fwhm=0.1)
    light = LightModel(['GAUSSIAN'])
    sources = [{'amp': 1.0, 'sigma': 0.08}, {'amp': 0.6, 'sigma': 0.12}]
    source_redshifts = [0.2, 1.5]
    numerics = {'supersampling_factor': 3}
    joint = ImageModel(grid, psf, lens, LightModel(['GAUSSIAN', 'GAUSSIAN']), light,
                       kwargs_numerics=numerics, source_redshift_list=source_redshifts)
    parameters = [{'theta_E': 0.2}]
    lens_light = [{'amp': 0.3, 'sigma': 0.15}]
    image = joint.image(parameters, sources, lens_light)
    expected = joint.lens_surface_brightness(lens_light)
    for redshift, source in zip(source_redshifts, sources):
        renderer = ImageModel(grid, psf, lens, light, LightModel([]), kwargs_numerics=numerics,
                              source_redshift_list=[redshift])
        expected += renderer.image(parameters, [source], [])
    np.testing.assert_allclose(image, expected, rtol=1e-14, atol=1e-15)
    foreground = ImageModel(grid, psf, lens, light, LightModel([]), kwargs_numerics=numerics,
                            source_redshift_list=[0.2]).image(parameters, sources[:1], [])
    unlensed = ImageModel(grid, psf, LensModel([]), light, LightModel([]),
                          kwargs_numerics=numerics).image([], sources[:1], [])
    np.testing.assert_array_equal(foreground, unlensed)


@pytest.mark.parametrize('redshifts', [[], [0.0], [np.nan], [np.inf], [-1.0]])
def test_multiple_source_renderer_validates_redshifts(redshifts):
    from astropy.cosmology import Planck18

    geometry = make_grid_with_coordtransform(11, 0.1)
    grid = PixelGrid(11, 11, geometry[2], geometry[3], geometry[6])
    lens = LensModel([], multi_plane=True, lens_redshift_list=[], z_source=2.0, cosmo=Planck18)
    with pytest.raises(ValueError, match='source redshift|Source redshifts'):
        ImageModel(grid, PSF(), lens, LightModel(['GAUSSIAN']), LightModel([]),
                   source_redshift_list=redshifts)


def test_multiple_source_example_uses_native_scene_and_saves_reproducible_outputs(tmp_path):
    import json
    from pathlib import Path
    import runpy

    example = runpy.run_path(Path(__file__).parents[1] / 'examples' / 'multiple_source_planes.py')
    images, extent, inputs = example['build_scene']()
    np.testing.assert_allclose(images[2], images[0] + images[1], rtol=1e-14, atol=1e-15)
    assert np.isfinite(images).all() and np.all(images >= 0.0)
    assert all(image.max() > 0.0 for image in images)
    assert inputs['simulated'] and inputs['source_redshifts'] == [0.5, 1.5]
    outputs = example['run_example'](tmp_path)
    assert outputs['figure'].is_file()
    with np.load(outputs['data'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['images'], images)
        np.testing.assert_array_equal(saved['extent'], extent)
        assert json.loads(str(saved['inputs'])) == inputs
    try:
        import bokeh
    except ImportError:
        pass
    else:
        assert 'Combined image' in outputs['bokeh'].read_text(encoding='utf-8')


def test_native_pixel_coordinates_roundtrip_and_psf_conserves_flux():
    geometry = make_grid_with_coordtransform(51, 0.1)
    grid = PixelGrid(51, 51, geometry[2], geometry[3], geometry[6])
    angles = grid.map_pix2coord([4.0, 20.0], [10.0, 30.0])
    np.testing.assert_allclose(grid.map_coord2pix(*angles), [[4.0, 20.0], [10.0, 30.0]])
    image = np.zeros((51, 51))
    image[25, 25] = 3.0
    convolved = PSF('PIXEL', kernel_point_source=np.ones((3, 3))).convolve(image, 0.1)
    assert convolved.sum() == pytest.approx(3.0)


def test_native_point_source_and_supersampled_flux_are_normalized():
    from chalcedon import PointSource

    geometry = make_grid_with_coordtransform(51, 0.1)
    grid = PixelGrid(51, 51, geometry[2], geometry[3], geometry[6])
    mass = LensModel([])
    renderer = ImageModel(grid, PSF('PIXEL', kernel_point_source=np.ones((3, 3))),
                          mass, LightModel([]), LightModel([]),
                          point_source_class=PointSource(['UNLENSED'], mass))
    image = renderer.image([], [], [], kwargs_ps=[{'ra_image': [0.03], 'dec_image': [-0.02], 'point_amp': [7.0]}])

    assert image.sum() == pytest.approx(7.0)


def test_native_sis_critical_curve_and_caustic_have_expected_geometry():
    from chalcedon import LensModelExtensions

    critical_x, critical_y, source_x, source_y = LensModelExtensions(LensModel(['SIS'])).critical_curve_caustics(
        [{'theta_E': 1.0}], compute_window=4.0, grid_scale=0.04,
    )

    main_curve = np.argmax([len(values) for values in critical_x])
    np.testing.assert_allclose(np.hypot(critical_x[main_curve], critical_y[main_curve]), 1.0, atol=0.002)
    np.testing.assert_allclose(np.hypot(source_x[main_curve], source_y[main_curve]), 0.0, atol=0.002)


@pytest.mark.parametrize("fwhm, pixel_scale", [(0.2, 0.05), (0.27, 0.1)])
def test_gaussian_psf_preserves_original_angular_to_pixel_width(fwhm, pixel_scale):
    from scipy.ndimage import gaussian_filter

    image = np.arange(63, dtype=float).reshape(7, 9)
    sigma = fwhm / (np.sqrt(8.0 * np.log(2.0)) * pixel_scale)  # [pixel]
    expected = gaussian_filter(image, sigma, mode='constant')
    actual = PSF('GAUSSIAN', fwhm=fwhm).convolve(image, pixel_scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_native_oversampling_preserves_gaussian_count_rate_and_psf_throughput():
    light = LightModel(['GAUSSIAN'])
    parameters = [{'amp': 3.0, 'sigma': 0.2}]
    fluxes = []
    for factor in (1, 3, 5):
        geometry = make_grid_with_coordtransform(51, 0.1)
        grid = PixelGrid(51, 51, geometry[2], geometry[3], geometry[6])
        image = ImageModel(grid, PSF('GAUSSIAN', fwhm=0.2), LensModel([]), light, LightModel([]),
                           kwargs_numerics={'supersampling_factor': factor}).image([], parameters, [], lens_light_add=False)
        fluxes.append(image.sum())
    np.testing.assert_allclose(fluxes, 3.0, rtol=1e-8)
    impulse = np.zeros((51, 51))
    impulse[25, 25] = 1.0
    psf = PSF('PIXEL', kernel_point_source=np.full((3, 3), 0.8 / 9.0),
              kernel_point_source_normalisation=False)
    assert psf.convolve(impulse, 0.1).sum() == pytest.approx(0.8)


def test_native_shapelet_coefficients_preserve_serialized_order_and_normalization():
    x = np.array([0.2, 0.4, 0.8, 1.2])  # [arcsec]
    y = np.array([0.1, -0.2, 0.3, 0.8])  # [arcsec]
    parameters = {'beta': 0.3, 'n_max': 2, 'amp': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    actual = LightModel(['SHAPELETS']).surface_brightness(x, y, [parameters])
    expected = [0.40291315419633306, 0.35021066999532496, 0.03686407957914564, 0.00003058341935266707]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)