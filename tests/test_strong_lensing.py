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
    assert cosmology.z_lens == 0.5
    assert cosmology.z_source == 1.5
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
    first_distance, second_distance, source_distance = (
        Planck18.comoving_distance(redshift).value for redshift in (0.3, 0.6, 1.5)
    )  # [Mpc]
    propagation = ((second_distance-first_distance) * source_distance
                   / (second_distance * (source_distance-first_distance)))
    factor = 1.0 - 0.1 - 0.2 * (1.0 - propagation * 0.1)

    source = model.ray_shooting([1.0], [2.0], parameters)

    np.testing.assert_allclose(source, [[factor], [2.0 * factor]])


@pytest.mark.parametrize('source_redshift', [0.2, 0.5, 0.6, 1.0, 2.0, 3.0])
def test_source_planes_share_reference_mass_normalization_and_causal_lens_order(source_redshift):
    from astropy.cosmology import Planck18

    lens_redshifts = [0.3, 0.6]
    reference_redshift = 2.0
    convergence = [0.1, 0.2]
    model = LensModel(['CONVERGENCE', 'CONVERGENCE'], multi_plane=True,
                      lens_redshift_list=lens_redshifts, z_source=reference_redshift, cosmo=Planck18)
    parameters = [{'kappa': value} for value in convergence]
    physical_strengths = [
        value * Planck18.angular_diameter_distance(reference_redshift).value
        / ((Planck18.comoving_distance(reference_redshift) - Planck18.comoving_distance(redshift)).value
           / (1.0 + reference_redshift))
        for redshift, value in zip(lens_redshifts, convergence)
    ]
    second_plane_position = (
        1.0 - physical_strengths[0]
        * ((Planck18.comoving_distance(0.6) - Planck18.comoving_distance(0.3)).value / 1.6)
        / Planck18.angular_diameter_distance(0.6).value
    )
    expected_factor = 1.0
    for redshift, strength, position in zip(lens_redshifts, physical_strengths, [1.0, second_plane_position]):
        if redshift < source_redshift:
            expected_factor -= (
                     strength * position
                     * ((Planck18.comoving_distance(source_redshift) - Planck18.comoving_distance(redshift)).value
                         / (1.0 + source_redshift))
                / Planck18.angular_diameter_distance(source_redshift).value
            )
    actual = model.ray_shooting([1.0], [2.0], parameters, z_source=source_redshift)
    np.testing.assert_allclose(actual, [[expected_factor], [2.0 * expected_factor]], rtol=1e-14)
    if source_redshift == reference_redshift:
        np.testing.assert_array_equal(actual, model.ray_shooting([1.0], [2.0], parameters))


@pytest.mark.parametrize('source_redshift', [0.0, -0.5, np.nan, np.inf])
def test_source_plane_rejects_invalid_redshifts(source_redshift):
    from astropy.cosmology import Planck18

    model = LensModel(['CONVERGENCE'], multi_plane=True,
                      lens_redshift_list=[0.3], z_source=2.0, cosmo=Planck18)
    with pytest.raises(ValueError, match='finite and positive'):
        model.ray_shooting([1.0], [2.0], [{'kappa': 0.1}], z_source=source_redshift)


def test_source_redshift_requires_explicit_reference_geometry():
    with pytest.raises(ValueError, match='reference z_source'):
        LensModel(['SIS']).ray_shooting([1.0], [2.0], [{'theta_E': 1.0}], z_source=1.0)


@pytest.mark.parametrize('source_redshift', [0.5, 1.5, 4.0])
@pytest.mark.parametrize('curved', [False, True])
def test_source_mapping_is_invariant_under_reference_redshift_conversion(source_redshift, curved):
    from astropy.cosmology import LambdaCDM, Planck18
    from chalcedon.strong_lensing import _distance_between

    cosmology = LambdaCDM(H0=70, Om0=0.3, Ode0=0.6) if curved else Planck18
    redshifts = [0.3, 0.7]
    parameters = [{'theta_E': 0.8}, {'theta_E': 0.1, 'center_x': 0.4, 'center_y': -0.1}]
    reference = LensModel(['SIS', 'POINT_MASS'], multi_plane=True,
                          lens_redshift_list=redshifts, z_source=2.0, cosmo=cosmology)
    converted = LensModel(['SIS', 'POINT_MASS'], multi_plane=True,
                          lens_redshift_list=redshifts, z_source=3.0, cosmo=cosmology)
    converted_parameters = []
    for index, (redshift, profile) in enumerate(zip(redshifts, parameters)):
        ratio = (
            _distance_between(cosmology, redshift, 3.0).value
            / cosmology.angular_diameter_distance(3.0).value
            / (_distance_between(cosmology, redshift, 2.0).value
               / cosmology.angular_diameter_distance(2.0).value)
        )
        converted_parameters.append(dict(profile, theta_E=profile['theta_E'] * ratio**(1.0 if index == 0 else 0.5)))
    image_x = np.array([-0.8, 0.2, 0.7, 1.1])  # [arcsec]
    image_y = np.array([0.3, -0.6, 0.9, -0.2])  # [arcsec]
    expected = reference.ray_shooting(image_x, image_y, parameters, z_source=source_redshift)
    actual = converted.ray_shooting(image_x, image_y, converted_parameters, z_source=source_redshift)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_angular_distance_helper_supports_older_cosmology_signature():
    from chalcedon.strong_lensing import _distance_between

    class LegacyCosmology:
        def angular_diameter_distance(self, redshift):
            return redshift

        def angular_diameter_distance_z1z2(self, first, second):
            return second - first

    assert _distance_between(LegacyCosmology(), 0.3, 1.0) == pytest.approx(0.7)


@pytest.mark.parametrize('source_redshift', [0.2, 0.6, 1.5, 2.0, 3.0])
def test_source_plane_sis_positions_and_magnification_match_analytic_solution(source_redshift):
    from astropy.cosmology import Planck18
    from chalcedon.strong_lensing import _distance_between

    lens = LensModel(['SIS'], multi_plane=True, lens_redshift_list=[0.3], z_source=2.0, cosmo=Planck18)
    parameters = [{'theta_E': 0.8}]
    position_x, position_y = LensEquationSolver(lens).image_position_from_source(
        0.05, 0.0, parameters, search_window=4.0, z_source=source_redshift,
    )
    if source_redshift < 0.3:
        np.testing.assert_allclose(position_x, [0.05], atol=1e-7)
        np.testing.assert_allclose(lens.magnification(position_x, position_y, parameters,
                                                    z_source=source_redshift), [1.0], atol=1e-10)
    else:
        einstein_radius = (
            0.8 * _distance_between(Planck18, 0.3, source_redshift).value
            / Planck18.angular_diameter_distance(source_redshift).value
            / (_distance_between(Planck18, 0.3, 2.0).value / Planck18.angular_diameter_distance(2.0).value)
        )  # [arcsec]
        np.testing.assert_allclose(np.sort(position_x), [0.05-einstein_radius, 0.05+einstein_radius], atol=1e-7)
        expected = 1.0 / (1.0-einstein_radius/np.abs(position_x))
        np.testing.assert_allclose(lens.magnification(position_x, position_y, parameters,
                                                    z_source=source_redshift), expected, rtol=1e-6)
    np.testing.assert_allclose(position_y, 0.0, atol=1e-7)


def test_native_truncated_nfw_uses_existing_chalcedon_kernel():
    from chalcedon.main import retr_deflcutf

    radii = np.array([0.1, 0.5, 1.0])  # [arcsec]
    parameters = [{'Rs': 0.3, 'alpha_Rs': 0.02, 'r_trunc': 2.0}]
    expected = retr_deflcutf(radii, 0.02 / (1.0 - np.log(2.0)), 0.3, 2.0)
    actual = LensModel(['TNFW']).alpha(radii, np.zeros_like(radii), parameters)

    np.testing.assert_allclose(actual[0], expected)
    np.testing.assert_allclose(actual[1], 0.0)


@pytest.mark.parametrize(
    ('profile', 'parameters', 'expected_x', 'expected_y'),
    [
        ('TNFWC', {'Rs': 1.1, 'alpha_Rs': 0.15, 'r_core': 0.08, 'r_trunc': 3.4,
                   'center_x': 0.1, 'center_y': -0.2},
         [0.05247255718934373, 0.1434306605183715, -0.08506484567580612],
         [0.10494511437868748, 0.017928832564796437, 0.11909078394612856]),
        ('PSEUDO_DPL', {'Rs': 1.2, 'alpha_Rs': 0.2, 'gamma_inner': 1.2,
                        'gamma_outer': 4.0, 'center_x': 0.1, 'center_y': -0.2},
         [0.08246705230082416, 0.2038559567188657, -0.11961929122517725],
         [0.16493410460164834, 0.025481994589858212, 0.16746700771524814]),
        ('CNFW', {'Rs': 1.2, 'alpha_Rs': 0.2, 'r_core': 0.15,
              'center_x': 0.1, 'center_y': -0.2},
         [0.06225423964166948, 0.17886723838705904, -0.10698952352879361],
         [0.12450847928333897, 0.02235840479838238, 0.14978533294031104]),
        ('ULDM', {'kappa_0': 0.3, 'theta_c': 0.2, 'slope': 8,
              'center_x': 0.1, 'center_y': -0.2},
         [0.018595758106226823, 0.025034714115894592, -0.013759228273277592],
         [0.03719151621245365, 0.003129339264486824, 0.019262919582588627]),
        ('GAUSSIAN', {'amp': 0.5, 'sigma': 0.3, 'center_x': 0.1, 'center_y': -0.2},
         [0.10676225185107714, 0.19058989428121903, -0.1057746382372381],
         [0.2135245037021543, 0.02382373678515238, 0.14808449353213332]),
    ],
)
def test_native_pyhalo_mass_profiles_match_reference_deflections(
    profile, parameters, expected_x, expected_y
):
    image_x = np.array([0.3, 0.9, -0.4])
    image_y = np.array([0.2, -0.1, 0.5])

    alpha_x, alpha_y = LensModel([profile]).alpha(image_x, image_y, [parameters])

    np.testing.assert_allclose(alpha_x, expected_x, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(alpha_y, expected_y, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize(
    ('profile', 'parameters'),
    [
        ('TNFWC', {'Rs': 1.0, 'alpha_Rs': 0.2, 'r_core': 0.1, 'r_trunc': 3.0}),
        ('PSEUDO_DPL', {'Rs': 1.0, 'alpha_Rs': 0.2, 'gamma_inner': 1.2,
                        'gamma_outer': 4.0}),
    ],
)
def test_native_radial_profile_normalization_is_alpha_at_scale_radius(profile, parameters):
    alpha_x, alpha_y = LensModel([profile]).alpha(
        parameters['Rs'], 0.0, [parameters]
    )

    assert np.hypot(alpha_x, alpha_y) == pytest.approx(parameters['alpha_Rs'])


@pytest.mark.parametrize(
    ('profile', 'parameters'),
    [
        ('TNFWC', {'Rs': 1.0, 'alpha_Rs': 0.2, 'r_core': 0.0, 'r_trunc': 3.0}),
        ('PSEUDO_DPL', {'Rs': 1.0, 'alpha_Rs': 0.2, 'gamma_inner': 3.0,
                        'gamma_outer': 4.0}),
        ('CNFW', {'Rs': 1.0, 'alpha_Rs': 0.2, 'r_core': 0.0}),
        ('ULDM', {'kappa_0': 0.3, 'theta_c': 0.0, 'slope': 8}),
        ('GAUSSIAN', {'amp': 0.5, 'sigma': 0.0}),
    ],
)
def test_native_pyhalo_profiles_reject_invalid_parameters(profile, parameters):
    with pytest.raises(ValueError):
        LensModel([profile]).alpha([0.5], [0.0], [parameters])


def test_native_hernquist_ellipse_preserves_serialized_radius_normalization():
    x = np.array([0.2, 0.4, 0.8, 1.2])  # [arcsec]
    y = np.array([0.1, -0.2, 0.3, 0.8])  # [arcsec]
    actual = LensModel(['HERNQUIST_ELLIPSE_CSE']).alpha(x, y, [{'Rs': 1.0, 'sigma0': 0.4, 'e1': 0.2, 'e2': 0.0}])
    expected = [[0.23456980693596727, 0.2712327004506171, 0.28093790934449847, 0.2080919630450257],
                [0.16401342622697448, -0.18401918506626605, 0.1379225820346579, 0.1727034868683604]]
    np.testing.assert_allclose(actual, expected, rtol=5e-6)