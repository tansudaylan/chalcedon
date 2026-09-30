import numpy as np
import pytest

import chalcedon


def test_singular_isothermal_sphere_deflection_has_constant_magnitude():
    xposgrid = np.array([0.3, -1.2, 2.5, 0.0])
    yposgrid = np.array([0.4, 0.5, -0.1, -3.0])

    defl = chalcedon.retr_deflsie(xposgrid, yposgrid, 0., 0., 1.2)

    np.testing.assert_allclose(np.linalg.norm(defl, axis=1), 1.2)
    np.testing.assert_allclose(defl[:, 0] / defl[:, 1], xposgrid / yposgrid)


def test_nearly_round_ellipsoid_approaches_the_sphere_and_rotates_with_the_position_angle():
    xposgrid = np.array([0.7, -0.4, 1.3])
    yposgrid = np.array([0.2, 0.9, -1.1])

    deflsphe = chalcedon.retr_deflsie(xposgrid, yposgrid, 0., 0., 1., 0.)
    defliell = chalcedon.retr_deflsie(xposgrid, yposgrid, 0., 0., 1., 1e-4, 0.8)
    np.testing.assert_allclose(defliell, deflsphe, atol=1e-3)

    # rotating the lens and the grid together rotates the deflection vectors
    angl = 0.6  # [rad]
    rota = np.array([[np.cos(angl), -np.sin(angl)], [np.sin(angl), np.cos(angl)]])
    xposrota, yposrota = rota @ np.vstack((xposgrid, yposgrid))
    deflbase = chalcedon.retr_deflsie(xposgrid, yposgrid, 0., 0., 1., 0.3, 0.)
    deflrota = chalcedon.retr_deflsie(xposrota, yposrota, 0., 0., 1., 0.3, -angl)
    np.testing.assert_allclose(deflrota, (rota @ deflbase.T).T, atol=1e-12)


def test_isothermal_convergence_equals_half_the_einstein_radius_over_radius():
    numbside = 201
    sizepixl = 0.02  # [arcsec]
    coordinate = (np.arange(numbside) - numbside // 2) * sizepixl
    xposgrid, yposgrid = np.meshgrid(coordinate, coordinate, indexing='ij')
    defl = chalcedon.retr_deflsie(xposgrid.ravel(), yposgrid.ravel(), 0., 0., 1.).reshape(numbside, numbside, 2)

    conv = chalcedon.retr_convfromdefl(defl, sizepixl)
    radi = np.sqrt(xposgrid**2 + yposgrid**2)
    indxring = (radi > 0.5) & (radi < 1.5)

    np.testing.assert_allclose(conv[indxring], 0.5 / radi[indxring], rtol=0.02)


def test_truncated_nfw_approaches_the_untruncated_profile_for_distant_cutoffs():
    angl = np.array([0.1, 0.5, 1., 2., 4.])

    defltrun = chalcedon.retr_deflcutf(angl, 1., 1., 1e4)
    deflasym = chalcedon.retr_deflcutf(angl, 1., 1., None, asym=True)

    np.testing.assert_allclose(defltrun, deflasym, rtol=1e-2)


def test_subhalo_deflection_points_radially_with_the_profile_magnitude():
    xposgrid = np.array([1.1, 0.2, -0.5])
    yposgrid = np.array([0.3, -0.7, 0.4])

    defl = chalcedon.retr_deflsubh(xposgrid, yposgrid, 0.1, 0.1, 0.02, 0.5, 5.)
    radi = np.hypot(xposgrid - 0.1, yposgrid - 0.1)

    np.testing.assert_allclose(np.linalg.norm(defl, axis=1), chalcedon.retr_deflcutf(radi, 0.02, 0.5, 5.))
    offs = np.vstack((xposgrid - 0.1, yposgrid - 0.1)).T
    np.testing.assert_allclose(defl[:, 0] * offs[:, 1] - defl[:, 1] * offs[:, 0], 0., atol=1e-15)


def test_dictionary_deflection_sums_host_subhalos_and_shear():
    xposgrid = np.linspace(-1., 1., 7)
    yposgrid = np.linspace(1., -1., 7)
    indxpixl = np.arange(7)
    dictchalinpt = {
        'xposhost': 0.1, 'yposhost': -0.1, 'beinhost': 1., 'ellphost': 0.2, 'anglhost': 0.4,
        'xpossubh': np.array([0.5]), 'ypossubh': np.array([0.2]), 'defssubh': np.array([0.01]),
        'ascasubh': np.array([0.3]), 'acutsubh': np.array([3.]), 'sherextr': 0.05, 'sangextr': 0.3,
    }

    dictchaloutp = chalcedon.retr_defl(xposgrid, yposgrid, indxpixl, dictchalinpt)

    expected = chalcedon.retr_deflsie(xposgrid, yposgrid, 0.1, -0.1, 1., 0.2, 0.4) + \
        chalcedon.retr_deflsubh(xposgrid, yposgrid, 0.5, 0.2, 0.01, 0.3, 3.) + \
        chalcedon.retr_deflextr(xposgrid, yposgrid, 0.05, 0.3)
    np.testing.assert_allclose(dictchaloutp['defltotl'], expected)
    np.testing.assert_allclose(
        chalcedon.retr_defl(xposgrid, yposgrid, indxpixl, 0.5, 0.2, 0.01, asca=0.3, acut=3.),
        dictchaloutp['deflsubh'],
    )


def test_point_and_plummer_lenses():
    assert chalcedon.retr_magnpntslens(1.) == pytest.approx(3. / np.sqrt(5.))

    defl = chalcedon.retr_deflplum(np.array([100.]), np.array([0.]), 0., 0., 1., 0.03)
    np.testing.assert_allclose(defl[0], [1. / 100., 0.], rtol=1e-6)


def test_einstein_radius_and_truncated_mass_scalings():
    # one solar mass at one AU has an Einstein radius of about 0.043 solar radii
    assert chalcedon.retr_radieinsfromsmax(1., 1.) == pytest.approx(0.0427, rel=1e-2)
    assert chalcedon.retr_radieinssbin(365.25, 1., 0.) == pytest.approx(chalcedon.retr_radieinsfromsmax(1., 1.))

    assert chalcedon.retr_mcut(2., 0.5, 0.5 * 3., 1e3, 4.) == pytest.approx(
        2. * np.pi * 1e3**2 * 4. * 0.5 * chalcedon.retr_mcutfrommscl(3.))
    with pytest.raises(ValueError):
        chalcedon.retr_mcutfrommscl(0.)

    assert chalcedon.retr_adislenssour(1., 2., 0.5, 1.) == pytest.approx(2. - 1.5 / 2.)


def test_self_lensing_model_produces_symmetric_brightening():
    time_days = np.linspace(-0.5, 0.5, 101)  # [day]

    relative_flux = chalcedon.evaluate_self_lensing_model(
        time_days, period_days=30.0, source_radius_solar=1.0, source_mass_solar=1.0,
        lens_mass_solar=0.6, impact_parameter=0.2, grid_size=301,
    )

    assert np.argmax(relative_flux) == time_days.size // 2
    np.testing.assert_allclose(relative_flux, relative_flux[::-1], atol=1e-12)
    assert np.max(relative_flux) > 1.0005
