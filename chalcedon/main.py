import numpy as np
import scipy
import skimage

import tdpy
import aspendos

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
    
    # Equation 6 in Masuda Hotokezaka 2019
    radieins = 0.04273 * (peri / 365.25)** (1. / 3.) * masscomp**(1. / 2.) * (masscomp + massstar)**(1. / 6.) # [R_S]
    
    return radieins


def retr_factmcutfromdefs(adissour, adislens, adislenssour, asca, acut):
    
    mdencrit = retr_mdencrit(adissour, adislens, adislenssour)
    
    fracacutasca = acut / asca
    
    factmcutfromdefs = np.pi * adislens**2 * mdencrit * asca * aspendos.retr_mcutfrommscl(fracacutasca)

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
    factsine = sher * np.cos(2. * sang)
    deflxpos = factcosi * xposgrid + factsine * yposgrid
    deflypos = factsine * xposgrid - factcosi * yposgrid
    
    deflextr = np.vstack((deflxpos, deflypos)).T

    return deflextr


def retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt, *args, **kwargs):
    '''
    Return deflection due to a main halo without a cutoff radius and subhalos with cutoff radii
    '''

    if not isinstance(dictchalinpt, dict):
        xposlens = dictchalinpt
        if len(args) < 2:
            raise TypeError('Legacy retr_defl() call requires xpos, ypos, and deflection scale inputs.')
        yposlens = args[0]
        defllens = args[1]
        asca = kwargs.get('asca', None)
        ellphost = kwargs.get('ellp', 0.)

        if asca is None:
            dictchalinpt = {
                'xposhost': xposlens,
                'yposhost': yposlens,
                'beinhost': defllens,
                'ellphost': ellphost,
            }
        else:
            dictchalinpt = {
                'xposhost': 0.,
                'yposhost': 0.,
                'beinhost': 0.,
                'ellphost': 0.,
                'xpossubh': np.atleast_1d(xposlens),
                'ypossubh': np.atleast_1d(yposlens),
                'ascasubh': np.atleast_1d(asca),
            }
    
    dictchaloutp = dict()
    
    if not 'ellphost' in dictchalinpt:
        dictchalinpt['ellphost'] = 0.

    # check inputs
    if dictchalinpt['ellphost'] is not None and (dictchalinpt['ellphost'] < 0. or dictchalinpt['ellphost'] > 1.):
        raise Exception('')
    
    numbiter = 1
    if 'xpossubh' in dictchalinpt:
        numbsubh = dictchalinpt['xpossubh'].size
        numbiter += numbsubh
    
    dictchalinpt['boolasym'] = True
    
    indxiter = np.arange(numbiter)
    for u in indxiter:
        
        if u == 0:
            strgcomp = 'host'
        
            xposlens = dictchalinpt['xposhost']
            yposlens = dictchalinpt['yposhost']
            defllens = dictchalinpt['yposhost']
        else:
            k = u - 1
            strgcomp = 'subh%08d' % k
            
            xposlens = dictchalinpt['xpossubh'][k]
            yposlens = dictchalinpt['ypossubh'][k]
            defllens = dictchalinpt['ypossubh'][k]
            
            print('xposlens')
            summgene(xposlens)
            print('yposlens')
            summgene(yposlens)
            print('defllens')
            summgene(defllens)

        # dictionary of parameter names for the component
        dictstrg = dict()
        for name in ['xpos', 'ypos', 'ellp', 'bein', 'deflxpos', 'deflypos']:
            dictstrg[name] = name + strgcomp
        
        # translate the grid
        ## horizontal distance to the component [arcsec]
        xposgridtran = xposgrid[indxpixlelem] - xposlens
        ## vertical distance to the component [arcsec]
        yposgridtran = yposgrid[indxpixlelem] - yposlens
        
        anglgrid = np.sqrt(xposgridtran**2 + yposgridtran**2)
        
        # rotate the grid
        xposgridrttr = np.cos(anglgrid) * xposgridtran - np.sin(anglgrid) * yposgridtran
        yposgridrttr = np.sin(anglgrid) * xposgridtran + np.cos(anglgrid) * yposgridtran
        
        # main halo
        if u == 0:
            axisrati = 1. - dictchalinpt[dictstrg['ellp']]
            axisrati = np.clip(axisrati, 1e-6, 1. - 1e-6)
            facteccc = np.sqrt(max(1e-12, 1. - axisrati**2))
            factrcor = np.sqrt(axisrati**2 * xposgridrttr**2 + yposgridrttr**2)
            factrcor = np.where(factrcor == 0., 1e-30, factrcor)
            argatanh = facteccc * yposgridrttr / factrcor
            argatanh = np.clip(argatanh, -1. + 1e-12, 1. - 1e-12)
            
            deflxposrttr = dictchalinpt[dictstrg['bein']] * axisrati / facteccc *  np.arctan(facteccc * xposgridrttr / factrcor)
            deflyposrttr = dictchalinpt[dictstrg['bein']] * axisrati / facteccc * np.arctanh(argatanh)
        
        # subhalos
        else:
            # component-centric radius [arcsec]
            radigrid = np.sqrt(xposgridtran**2 + yposgridtran**2)
        
            fracanglasca = radigrid / dictchalinpt['ascasubh'][k]
            
            # second term in the NFW deflection profile
            fact = np.ones_like(fracanglasca)
            indxlowr = np.where(fracanglasca < 1.)[0]
            indxuppr = np.where(fracanglasca > 1.)[0]
            fact[indxlowr] = np.arccosh(1. / fracanglasca[indxlowr]) / np.sqrt(1. - fracanglasca[indxlowr]**2)
            fact[indxuppr] = np.arccos(1. / fracanglasca[indxuppr]) / np.sqrt(fracanglasca[indxuppr]**2 - 1.)
            
            if dictchalinpt['boolasym']:
                factcutf = np.log(fracanglasca / 2.) + fact
            else:
                fracacutasca = acut / asca
                factcutf = fracacutasca**2 / (fracacutasca**2 + 1)**2 * ((fracacutasca**2 + 1. + 2. * (fracanglasca**2 - 1.)) * fact + \
                        np.pi * fracacutasca + (fracacutasca**2 - 1.) * np.log(fracacutasca) + \
                        np.sqrt(fracanglasca**2 + fracacutasca**2) * (-np.pi + (fracacutasca**2 - 1.) / fracacutasca * \
                        np.log(fracanglasca / (np.sqrt(fracanglasca**2 + fracacutasca**2) + fracacutasca))))
            
            deflxposrttr = factcutf * fact * xposgridtran
            deflyposrttr = factcutf * fact * yposgridtran

        # rotate back vector to original basis
        dictchaloutp['deflxposhost'] = np.cos(anglgrid) * deflxposrttr + np.sin(anglgrid) * deflyposrttr
        dictchaloutp['deflyposhost'] = -np.sin(anglgrid) * deflxposrttr + np.cos(anglgrid) * deflyposrttr
   
        dictchaloutp['deflhost'] = np.vstack((dictchaloutp['deflxposhost'], dictchaloutp['deflyposhost'])).T
        
        if u == 0:
            dictchaloutp['defltotl'] = np.copy(dictchaloutp['deflhost'])
        else:
            dictchaloutp['defltotl'] += dictchaloutp['deflhost']
        

    return dictchaloutp


def retr_caustics(xposgrid, yposgrid, indxpixlelem, dictchalinpt):
   
    from skimage import measure
    
    dictchaloutp = retr_defl(xposgrid, yposgrid, indxpixlelem, dictchalinpt)
    magn = retr_magn()
    
    cont = measure.find_contours(magn, 0.8)
    print('cont')
    print(cont)
