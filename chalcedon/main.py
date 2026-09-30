"""Gravitational lensing: deflection fields, lens mass scales, magnification, and self-lensing light curves."""

import numpy as np
import scipy.fftpack
import tdpy

from tdpy import summgene


def retr_radieins_inft( \
                       # velocity dispersion [km/s]
                       dispvelo, \
                      ):
    '''
    Calculate the Einstein radius for a source position at infinity
    '''
    
    radieins = 4 * np.pi * (dispvelo / 3e5)**2 * (180. / np.pi * 3600.) # [arcsec]
    
    return radieins


def retr_dflxslensing(time, epocslen, amplslen, duratrantotl):
    '''
    Return the self-lensing signature
    '''    
    
    timediff = time - epocslen
    
    dflxslensing = 1e-3 * amplslen * np.heaviside(duratrantotl / 48. + timediff, 0.5) * np.heaviside(duratrantotl / 48. - timediff, 0.5)
    
    return dflxslensing


def retr_angleinscosm(masslens, distlenssour, distlens, distsour):
    '''
    Return Einstein radius for a cosmological source and lens.
    '''
    
    angleins = np.sqrt(masslens / 10**(11.09) * distlenssour / distlens / distsour)
    
    return angleins


def retr_amplslen( \
                  # orbital period [days]
                  peri, \
                  # radistar: radius of the star [Solar radius]
                  radistar, \
                  # mass of the companion [Solar mass]
                  masscomp, \
                  # mass of the star [Solar mass]
                  massstar, \
                 ):
    '''
    Calculate the self-lensing amplitude in ppt.
    '''
    
    # Equation 5 in Masuda Hotokezaka 2019
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.) * 1e3 # [ppt]

    return amplslen


def retr_radieinsfromsmax( \
                          # mass of the lens [Solar mass]
                          masslens, \
                          # separation between the lens and the source [AU]
                          smax, \
                         ):
    '''
    Return the Einstein radius [Solar radius] of a stellar lens at a given separation from its source, sqrt(4 G M a / c^2).
    '''
    
    radieins = 0.04273 * np.sqrt(masslens * smax) # [R_S]
    
    return radieins


def retr_radieinssbin( \
                  # orbital period [days]
                  peri, \
                  # mass of the companion [Solar mass]
                  masscomp, \
                  # mass of the star [Solar mass]
                  massstar, \
                 ):
    '''
    Return Einstein radius for a stellar lens and source in proximity.
    '''
    
    # Equation 6 in Masuda Hotokezaka 2019, with Kepler's third law for the separation
    smax = (peri / 365.25)**(2. / 3.) * (masscomp + massstar)**(1. / 3.) # [AU]
    radieins = retr_radieinsfromsmax(masscomp, smax) # [R_S]
    
    return radieins


def retr_magnpntslens(distnorm):
    '''
    Return the magnification of a point lens at a source-lens separation in units of the Einstein radius.
    '''
    
    magn = (distnorm**2 + 2.) / (distnorm * np.sqrt(distnorm**2 + 4.))
    
    return magn


def evaluate_self_lensing_model(
    time_days,
    *,
    period_days,
    source_radius_solar,
    source_mass_solar,
    lens_mass_solar,
    impact_parameter=0.0,
    limb_darkening_coefficients=(0.4, 0.25),
    grid_size=301,
):
    """Integrate point-lens magnification over a limb-darkened stellar disk and return the relative flux."""

    from astropy.constants import G, M_sun, R_sun, c
    from scipy.ndimage import map_coordinates
    from scipy.signal import fftconvolve
    from tdpy.exoplanet import quadratic_limb_darkened_stellar_grid

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < 2 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if min(period_days, source_radius_solar, source_mass_solar, lens_mass_solar) <= 0.0:
        raise ValueError("period, radii, and masses must be positive")
    if impact_parameter < 0.0:
        raise ValueError("impact_parameter must be nonnegative")
    if grid_size < 101 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 101")

    period_seconds = period_days * 86400.0  # [s]
    total_mass = (source_mass_solar + lens_mass_solar) * M_sun
    semimajor_axis = (G * total_mass * period_seconds**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    source_radius = source_radius_solar * R_sun
    semimajor_axis_source_radii = (semimajor_axis / source_radius).decompose().value
    einstein_radius = np.sqrt(4.0 * G * lens_mass_solar * M_sun * semimajor_axis / c**2)
    einstein_radius_ratio = (einstein_radius / source_radius).decompose().value

    image_x, image_y, radial_distance, stellar_brightness = quadratic_limb_darkened_stellar_grid(
        grid_size, limb_darkening_coefficients
    )
    unocculted_flux = stellar_brightness.sum()

    pixel_size = 2.0 / (grid_size - 1)
    normalized_separation = np.maximum(radial_distance, 0.5 * pixel_size) / einstein_radius_ratio
    excess_flux_grid = fftconvolve(stellar_brightness, retr_magnpntslens(normalized_separation) - 1.0, mode="full")

    orbital_phase = 2.0 * np.pi * time_days / period_days
    lens_x = semimajor_axis_source_radii * np.sin(orbital_phase)
    lens_y = impact_parameter * np.cos(orbital_phase)
    pixels_per_stellar_radius = 0.5 * (grid_size - 1)
    sample_coordinates = np.vstack(
        (
            lens_y * pixels_per_stellar_radius + grid_size - 1,
            lens_x * pixels_per_stellar_radius + grid_size - 1,
        )
    )
    excess_flux = map_coordinates(excess_flux_grid, sample_coordinates, order=1, mode="constant", cval=0.0)
    return 1.0 + excess_flux / unocculted_flux


def retr_adislenssour(adislens, adissour, redslens, redssour):
    '''
    Return the angular diameter distance between the lens and the source in a flat universe.
    '''
    
    adislenssour = adissour - (1. + redslens) / (1. + redssour) * adislens
    
    return adislenssour


def retr_mcutfrommscl(fracacutasca):
    """Return the truncated NFW mass in units of the scale mass for a positive cutoff-to-scale-radius ratio."""

    radius_ratio = np.asarray(fracacutasca, dtype=float)
    if np.any(radius_ratio <= 0):
        raise ValueError("The cutoff-to-scale-radius ratio must be positive.")

    mcut = radius_ratio**2 / (radius_ratio**2 + 1.)**2 * (
        (radius_ratio**2 - 1.) * np.log(radius_ratio)
        + radius_ratio * np.pi
        - (radius_ratio**2 + 1.)
    )

    return mcut


def retr_mcut(defs, asca, acut, adislens, mdencrit):
    """Return the truncated NFW subhalo mass from its deflection scale, scale radius, and cutoff radius."""

    mscl = defs * np.pi * adislens**2 * mdencrit * asca
    mcut = mscl * retr_mcutfrommscl(acut / asca)
    
    return mcut


def retr_factmcutfromdefs(adissour, adislens, adislenssour, asca, acut):
    
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    fracacutasca = acut / asca
    
    factmcutfromdefs = np.pi * adislens**2 * mdencrit * asca * retr_mcutfrommscl(fracacutasca)

    return factmcutfromdefs


def retr_mdencrit(adissour, adislens, adislenssour):
    '''
    Calculate the critical mass density at a given angular diameter distance to the source, to the lens, and between the lens and the source.
    '''
    
    dictfact = tdpy.retr_factconv()
    
    mdencrit = dictfact['factnewtlght'] / 4. / np.pi * adissour / adislenssour / adislens
        
    return mdencrit


def retr_ratimassbeinsqrd(adissour, adislens, adislenssour):
    '''
    Calculate the ratio between the mass of the lens and square of its Einstein radius
    '''
    
    # calculate the critical mass density
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    # ratio between the mass of the lens and square of its Einstein radius
    ratimassbeinsqrd = np.pi * adislens**2 * mdencrit

    return ratimassbeinsqrd


def retr_deflextr(xposgrid, yposgrid, sher, sang):
    '''
    Return deflection field due to large-scale structure
    '''    
    
    factcosi = sher * np.cos(2. * sang)
    factsine = sher * np.sin(2. * sang)
    deflxpos = factcosi * xposgrid + factsine * yposgrid
    deflypos = factsine * xposgrid - factcosi * yposgrid
    
    deflextr = np.vstack((deflxpos, deflypos)).T

    return deflextr


def retr_deflcutf(angl, defs, asca, acut, asym=False):
    '''
    Return the radial deflection of a truncated NFW (asym=False) or untruncated NFW (asym=True) subhalo at radii angl.
    '''

    fracanglasca = np.asarray(angl, dtype=float) / asca
    
    deflcutf = defs / fracanglasca
    
    # second term in the NFW deflection profile; it equals unity at the scale radius
    fact = np.ones_like(fracanglasca)
    indxlowr = fracanglasca < 1.
    indxuppr = fracanglasca > 1.
    fact[indxlowr] = np.arccosh(1. / fracanglasca[indxlowr]) / np.sqrt(1. - fracanglasca[indxlowr]**2)
    fact[indxuppr] = np.arccos(1. / fracanglasca[indxuppr]) / np.sqrt(fracanglasca[indxuppr]**2 - 1.)
    
    if asym:
        deflcutf *= np.log(fracanglasca / 2.) + fact
    else:
        fracacutasca = acut / asca
        factcutf = fracacutasca**2 / (fracacutasca**2 + 1)**2 * ((fracacutasca**2 + 1. + 2. * (fracanglasca**2 - 1.)) * fact + \
                np.pi * fracacutasca + (fracacutasca**2 - 1.) * np.log(fracacutasca) + np.sqrt(fracanglasca**2 + fracacutasca**2) * (-np.pi + (fracacutasca**2 - 1.) / fracacutasca * \
                np.log(fracanglasca / (np.sqrt(fracanglasca**2 + fracacutasca**2) + fracacutasca))))
        deflcutf *= factcutf
       
    return deflcutf


def retr_deflsubh(xposgrid, yposgrid, xpos, ypos, defs, asca, acut=None):
    '''
    Return the (N, 2) deflection field of a truncated NFW subhalo, or of an untruncated one if acut is None.
    '''

    xposgridtran = np.asarray(xposgrid, dtype=float) - xpos
    yposgridtran = np.asarray(yposgrid, dtype=float) - ypos
    # avoid the coordinate singularity at the subhalo center
    anglgrid = np.maximum(np.sqrt(xposgridtran**2 + yposgridtran**2), 1e-12 * asca)
    
    defl = retr_deflcutf(anglgrid, defs, asca, acut, asym=acut is None)
    deflsubh = np.vstack((xposgridtran / anglgrid * defl, yposgridtran / anglgrid * defl)).T
    
    return deflsubh


def retr_deflsie(xposgrid, yposgrid, xpos, ypos, bein, ellp=0., angl=0.):
    '''
    Return the (N, 2) deflection field of a singular isothermal ellipsoid with ellipticity ellp and position angle angl [rad].
    '''
    
    if ellp < 0. or ellp >= 1.:
        raise ValueError('ellp must be in [0, 1).')
    
    # translate and rotate the grid into the frame of the ellipsoid
    xposgridtran = np.asarray(xposgrid, dtype=float) - xpos
    yposgridtran = np.asarray(yposgrid, dtype=float) - ypos
    xposgridrttr = np.cos(angl) * xposgridtran - np.sin(angl) * yposgridtran
    yposgridrttr = np.sin(angl) * xposgridtran + np.cos(angl) * yposgridtran
    
    axisrati = 1. - ellp
    facteccc = np.sqrt(1. - axisrati**2)
    factrcor = np.maximum(np.sqrt(axisrati**2 * xposgridrttr**2 + yposgridrttr**2), 1e-30)
    if facteccc < 1e-6:
        # singular isothermal sphere limit
        deflxposrttr = bein * xposgridrttr / factrcor
        deflyposrttr = bein * yposgridrttr / factrcor
    else:
        deflxposrttr = bein * axisrati / facteccc * np.arctan(facteccc * xposgridrttr / factrcor)
        deflyposrttr = bein * axisrati / facteccc * np.arctanh(np.clip(facteccc * yposgridrttr / factrcor, -1. + 1e-15, 1. - 1e-15))
    
    # rotate the vector back to the original basis
    deflxpos = np.cos(angl) * deflxposrttr + np.sin(angl) * deflyposrttr
    deflypos = -np.sin(angl) * deflxposrttr + np.cos(angl) * deflyposrttr
    deflsie = np.vstack((deflxpos, deflypos)).T
    
    return deflsie


def retr_deflplum(xposgrid, yposgrid, xpos, ypos, bein, rcor):
    '''
    Return the deflection of a point mass with Einstein radius bein softened by a core radius rcor (Plummer lens), with a trailing axis of size 2.
    '''
    
    xposgridtran = np.asarray(xposgrid, dtype=float) - xpos
    yposgridtran = np.asarray(yposgrid, dtype=float) - ypos
    radisqrd = xposgridtran**2 + yposgridtran**2 + rcor**2
    deflplum = np.stack((bein**2 * xposgridtran / radisqrd, bein**2 * yposgridtran / radisqrd), axis=-1)
    
    return deflplum


def retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt, *args, **kwargs):
    '''
    Return the deflection due to a singular isothermal ellipsoidal host, truncated NFW subhalos, and external shear.

    Dictionary input (keys xposhost, yposhost, beinhost, ellphost, anglhost, optionally arrays xpossubh, ypossubh,
    defssubh, ascasubh, acutsubh, and optionally external shear sherextr, sangextr) returns a dictionary with
    deflhost, deflsubh, deflextr (if sheared), and defltotl of shape (N, 2).
    Positional input retr_defl(xposgrid, yposgrid, indxpixlelem, xpos, ypos, bein_or_defs, ellp=, angl=, asca=, acut=)
    returns the (N, 2) deflection of one host (if asca is None) or one subhalo.
    '''

    xposgrid = np.asarray(xposgrid)[indxpixlelem]
    yposgrid = np.asarray(yposgrid)[indxpixlelem]

    if not isinstance(dictchalinpt, dict):
        if len(args) < 2:
            raise TypeError('Positional retr_defl() calls require xpos, ypos, and a deflection scale.')
        xpos, ypos, scal = dictchalinpt, args[0], args[1]
        if kwargs.get('asca') is None:
            ellp = kwargs.get('ellp')
            angl = kwargs.get('angl')
            return retr_deflsie(xposgrid, yposgrid, xpos, ypos, scal, 0. if ellp is None else ellp, 0. if angl is None else angl)
        return retr_deflsubh(xposgrid, yposgrid, xpos, ypos, scal, kwargs['asca'], kwargs.get('acut'))
    
    dictchaloutp = dict()
    dictchaloutp['deflhost'] = retr_deflsie(xposgrid, yposgrid, dictchalinpt['xposhost'], dictchalinpt['yposhost'], \
                                            dictchalinpt['beinhost'], dictchalinpt.get('ellphost', 0.), dictchalinpt.get('anglhost', 0.))
    dictchaloutp['deflsubh'] = np.zeros_like(dictchaloutp['deflhost'])
    if 'xpossubh' in dictchalinpt:
        acutsubh = dictchalinpt.get('acutsubh')
        for k in range(np.size(dictchalinpt['xpossubh'])):
            dictchaloutp['deflsubh'] += retr_deflsubh(xposgrid, yposgrid, dictchalinpt['xpossubh'][k], dictchalinpt['ypossubh'][k], \
                                                      dictchalinpt['defssubh'][k], dictchalinpt['ascasubh'][k], \
                                                      None if acutsubh is None else acutsubh[k])
    dictchaloutp['defltotl'] = dictchaloutp['deflhost'] + dictchaloutp['deflsubh']
    if 'sherextr' in dictchalinpt:
        dictchaloutp['deflextr'] = retr_deflextr(xposgrid, yposgrid, dictchalinpt['sherextr'], dictchalinpt['sangextr'])
        dictchaloutp['defltotl'] += dictchaloutp['deflextr']

    return dictchaloutp


def retr_convfromdefl(defl, sizepixl):
    '''
    Return the convergence map from a deflection map of shape (numbside, numbside, 2) on a grid with pixel size sizepixl.
    '''
    
    conv = np.abs(np.gradient(defl[:, :, 0], sizepixl, axis=0) + np.gradient(defl[:, :, 1], sizepixl, axis=1)) / 2.
    
    return conv


def retr_invmfromdefl(defl, sizepixl):
    '''
    Return the inverse magnification (Jacobian determinant of the lens equation) from a deflection map of shape (numbside, numbside, 2).
    '''
    
    invm = (1. - np.gradient(defl[:, :, 0], sizepixl, axis=0)) * (1. - np.gradient(defl[:, :, 1], sizepixl, axis=1)) - \
                                np.gradient(defl[:, :, 0], sizepixl, axis=1) * np.gradient(defl[:, :, 1], sizepixl, axis=0)
    
    return invm


def retr_psecconv(conv):
    '''
    Return the lowest-frequency quadrant of the two-dimensional power spectrum of a square convergence map (arbitrary normalization of 1e-3).
    '''
    
    numbsidehalf = conv.shape[0] // 2
    psec = (np.abs(scipy.fftpack.fft2(conv))**2)[:numbsidehalf, :numbsidehalf] * 1e-3
    
    return psec


def retr_magn(xposgrid, yposgrid, deflfield):
    """Return the magnification map of an (N, 2) deflection field sampled on a structured grid."""

    deflfield = np.asarray(deflfield)
    if deflfield.shape[-1] != 2:
        raise ValueError('deflfield must have shape (N, 2) for the x/y deflection components.')

    xuniq = np.unique(np.asarray(xposgrid))
    yuniq = np.unique(np.asarray(yposgrid))
    if xuniq.size < 2 or yuniq.size < 2:
        return np.ones_like(np.asarray(xposgrid), dtype=float)
    if xuniq.size != yuniq.size or not np.isclose(xuniq[1] - xuniq[0], yuniq[1] - yuniq[0]):
        raise ValueError('retr_magn() requires a square grid with equal pixel sizes along both axes.')

    invm = retr_invmfromdefl(deflfield.reshape(xuniq.size, yuniq.size, 2), xuniq[1] - xuniq[0])
    magn = 1. / np.maximum(np.abs(invm), 1e-12)
    return magn


def retr_caustics(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
    """Compute the deflection field and return a magnification map plus contour candidates."""

    from skimage import measure

    dictchaloutp = retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt)
    magn = retr_magn(xposgrid, yposgrid, dictchaloutp['defltotl'])

    cont = measure.find_contours(magn, 0.8)
    dictchaloutp['magn'] = magn
    dictchaloutp['contours'] = cont
    return dictchaloutp
