from __future__ import print_function, division

import numpy as np
from transit import Central, System, Body

def dilution_samples(s, which='A', band='Kepler'):
    """
    Returns dilution samples in band, according to StarModel s

    :param s:
        :class:`BinaryStarModel` Object, (defined in ``isochrones`` module)

    :param which:
        'A' for primary, 'B' for secondary.

    :param band:
        Photometric bandpass of interest.
    """

    mA = s.samples['{}_mag_A'.format(band,which)]
    mB = s.samples['{}_mag_B'.format(band,which)]
    
    fA = 10**(-0.4*mA)
    fB = 10**(-0.4*mB)
    if which=='A':
        return fB/(fB+fA)
    elif which=='B':
        return fA/(fB+fA)
    else:
        raise ValueError('Invalid choice: {}'.format(which))

def density_samples(s, which='A'):
    """
    Returns density samples according to StarModel

    :param s:
        :class:`StarModel` object.

    :param which:
        'A' for primary, 'B' for secondary.
    """
    if which=='A':
        if 'mass_A' in s.samples:
            m = s.samples['mass_A'.format(which)]
        else:
            m = s.samples['mass'.format(which)]
        r = s.samples['radius'.format(which)]
    elif which=='B':
        m = s.samples['mass_B']
        r = s.samples['radius_B']
    else:
        raise ValueError('Invalid choice: {}'.format(which))
    return 0.75*m*M_sun / (np.pi * (r*R_sun)**3)




def t_folded(t, per, ep):
    return (t + per/2 - ep) % per - (per/2)

def lc_eval(p, t, texp=None):
    """
    Returns flux at given times, given parameters.

    :param p:
        Parameter vector, of length 4 + 6*Nplanets
        p[0:4] = [rhostar, q1, q2, dilution]
        p[4+i*6:10+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

    :param t:
        Times at which to evaluate model.

    :param texp:
        Exposure time.  If not provided, assumed to be median t[1:]-t[:-1]

    """
    if texp is None:
        texp = np.median(t[1:] - t[:-1])
        
    n_planets = (len(p) - 4)//6
    
    rhostar, q1, q2, dilution = p[:4]

    central = Central(q1=q1, q2=q2)
    central.density = rhostar
    s = System(central, dilution=dilution)

    tot = 0
    close_to_transit = np.zeros_like(t).astype(bool)

    for i in range(n_planets):
        period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
        r = central.radius * rprs
        #body = Body(flux=0, radius=r, mass=0, period=period, t0=epoch,
        #           e=e, omega=w, b=b)
        body = Body(radius=r, mass=0, period=period, t0=epoch,
                   e=e, omega=w, b=b)
        s.add_body(body)

        tfold = t_folded(t, period, epoch)

    return s.light_curve(t, texp=texp)
        
