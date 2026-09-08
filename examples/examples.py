import sys

import numpy as np

import chalcedon


def cnfg_microlens():
    numbsidecart = 100
    maxmfovw = 2.
    xposside = np.linspace(-maxmfovw, maxmfovw, 100)
    yposside = xposside
    xposgrid, yposgrid = np.meshgrid(xposside, yposside, indexing='ij')
    xposgridflat = xposgrid.flatten()
    yposgridflat = yposgrid.flatten()
    
    numbpixl = numbsidecart**2
    indxpixl = np.arange(numbpixl)
    
    dictchalinpt = dict()
    dictchalinpt['sherextr'] = 0.3
    dictchalinpt['sangextr'] = 0.3
    
    dictchalinpt['xposhost'] = 0.3
    dictchalinpt['yposhost'] = 0.3
    dictchalinpt['beinhost'] = 0.3
    dictchalinpt['ellphost'] = 0.3
    
    dictchalinpt['xpossubh'] = np.array([1.])
    dictchalinpt['ypossubh'] = np.array([1.])
    dictchalinpt['defssubh'] = np.array([1.])
    dictchalinpt['ascasubh'] = np.array([1.])
    dictchalinpt['acutsubh'] = np.array([1.])
    
    indxpixlelem = indxpixl

    dictchaloutp = chalcedon.retr_caustics( \
                                           xposgrid=xposgridflat, \
                                           yposgrid=yposgridflat, \
                                           indxpixlelem=indxpixlelem, \
                                           dictchalinpt=dictchalinpt, \
                                          )
    return dictchaloutp


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python examples.py <config_name> [args...]')

    func = globals().get(sys.argv[1])
    if func is None:
        raise SystemExit(f'Unknown example configuration: {sys.argv[1]}')

    func(*sys.argv[2:])

