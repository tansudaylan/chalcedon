import numpy as np
import pytest

from chalcedon.strong_lensing import LensModel, LensEquationSolver, LensCosmo


def test_native_sis_ray_shooting_and_convergence_have_analytic_values():
    model = LensModel(['SIS'])
    parameters = [{'theta_E': 1.2, 'center_x': 0.1, 'center_y': -0.2}]
    image_x = np.array([1.6, -1.4])  # [arcsec]
    image_y = np.array([-0.2, -0.2])  # [arcsec]

    source_x, source_y = model.ray_shooting(image_x, image_y, parameters)

    np.testing.assert_allclose(source_x, [0.4, -0.2])
    np.testing.assert_allclose(source_y, image_y)
    np.testing.assert_allclose(model.kappa(image_x, image_y, parameters), 0.4, atol=1e-8)


def test_native_shear_and_mass_sheet_are_composable():
    model = LensModel(['SHEAR', 'CONVERGENCE'])
    parameters = [{'gamma1': 0.1, 'gamma2': 0.2}, {'kappa': 0.3}]

    alpha_x, alpha_y = model.alpha(np.array([1.0]), np.array([2.0]), parameters)

    np.testing.assert_allclose(alpha_x, [0.8])
    np.testing.assert_allclose(alpha_y, [0.6])
    np.testing.assert_allclose(model.kappa([1.0], [2.0], parameters), 0.3, atol=1e-8)


def test_unsupported_mass_profile_does_not_fall_back_to_another_package():
    with pytest.raises(NotImplementedError):
        LensModel(['UNKNOWN'])


def test_round_sie_and_isothermal_epl_reduce_to_sis():
    x = np.array([0.2, 0.5, -1.0])  # [arcsec]
    y = np.array([0.3, -0.4, 0.8])  # [arcsec]
    parameters = [{'theta_E': 1.2, 'e1': 0.0, 'e2': 0.0, 'gamma': 2.0}]
    expected = LensModel(['SIS']).alpha(x, y, parameters)

    for name in ('SIE', 'EPL'):
        actual = LensModel([name]).alpha(x, y, parameters)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_native_sis_solver_recovers_two_analytic_images():
    positions = LensEquationSolver(LensModel(['SIS'])).image_position_from_source(
        0.2, 0.0, [{'theta_E': 1.0}], search_window=4.0,
    )

    np.testing.assert_allclose(np.sort(positions[0]), [-0.8, 1.2], atol=1e-7)
    np.testing.assert_allclose(positions[1], 0.0, atol=1e-7)


def test_native_cosmology_roundtrips_velocity_dispersion():
    from astropy.cosmology import Planck18

    cosmology = LensCosmo(0.5, 1.5, Planck18)
    angle = cosmology.sis_sigma_v2theta_E(220.0)  # [arcsec]

    assert angle > 0
    assert cosmology.sigma_crit > 0
    assert cosmology.sis_theta_E2sigma_v(angle) == pytest.approx(220.0)


def test_native_nfw_scale_deflection_and_density_have_known_values():
    model = LensModel(['NFW'])
    parameters = [{'Rs': 2.0, 'alpha_Rs': 0.3}]
    alpha_x, alpha_y = model.alpha(np.array([2.0]), np.array([0.0]), parameters)

    np.testing.assert_allclose(alpha_x, [0.3], rtol=2e-5)
    np.testing.assert_allclose(alpha_y, [0.0], atol=1e-12)


def test_native_elliptical_halo_is_rotationally_covariant():
    x = np.array([0.3, 1.0, -0.4])  # [arcsec]
    y = np.array([0.2, -0.5, 0.7])  # [arcsec]
    angle = 0.4  # [rad]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_x, rotated_y = rotation @ np.vstack((x, y))
    base = {'Rs': 1.0, 'alpha_Rs': 0.2, 'e1': 0.2, 'e2': 0.0}
    rotated = {**base, 'e1': 0.2 * np.cos(2.0 * angle), 'e2': 0.2 * np.sin(2.0 * angle)}
    model = LensModel(['NFW_ELLIPSE_CSE'])

    expected = rotation @ np.asarray(model.alpha(x, y, [base]))
    actual = model.alpha(rotated_x, rotated_y, [rotated])

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_native_two_plane_mass_sheets_follow_distance_weighted_ray_propagation():
    from astropy.cosmology import Planck18

    model = LensModel(['CONVERGENCE', 'CONVERGENCE'], lens_redshift_list=[0.3, 0.6],
                      z_source=1.5, cosmo=Planck18, multi_plane=True)
    parameters = [{'kappa': 0.1}, {'kappa': 0.2}]
    propagation = (Planck18.angular_diameter_distance_z1z2(0.3, 0.6).value
                   * Planck18.angular_diameter_distance(1.5).value
                   / (Planck18.angular_diameter_distance(0.6).value
                      * Planck18.angular_diameter_distance_z1z2(0.3, 1.5).value))
    factor = 1.0 - 0.1 - 0.2 * (1.0 - propagation * 0.1)

    source = model.ray_shooting([1.0], [2.0], parameters)

    np.testing.assert_allclose(source, [[factor], [2.0 * factor]])


def test_native_truncated_nfw_uses_existing_chalcedon_kernel():
    from chalcedon.main import retr_deflcutf

    radii = np.array([0.1, 0.5, 1.0])  # [arcsec]
    parameters = [{'Rs': 0.3, 'alpha_Rs': 0.02, 'r_trunc': 2.0}]
    expected = retr_deflcutf(radii, 0.02 / (1.0 - np.log(2.0)), 0.3, 2.0)
    actual = LensModel(['TNFW']).alpha(radii, np.zeros_like(radii), parameters)

    np.testing.assert_allclose(actual[0], expected)
    np.testing.assert_allclose(actual[1], 0.0)


def test_native_hernquist_ellipse_preserves_serialized_radius_normalization():
    x = np.array([0.2, 0.4, 0.8, 1.2])  # [arcsec]
    y = np.array([0.1, -0.2, 0.3, 0.8])  # [arcsec]
    actual = LensModel(['HERNQUIST_ELLIPSE_CSE']).alpha(x, y, [{'Rs': 1.0, 'sigma0': 0.4, 'e1': 0.2, 'e2': 0.0}])
    expected = [[0.23456980693596727, 0.2712327004506171, 0.28093790934449847, 0.2080919630450257],
                [0.16401342622697448, -0.18401918506626605, 0.1379225820346579, 0.1727034868683604]]
    np.testing.assert_allclose(actual, expected, rtol=5e-6)