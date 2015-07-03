from __future__ import print_function, division

import numpy as np
from transit import Central, System, Body

def t_folded(t, per, ep):
    return (t + per/2 - ep) % per - (per/2)

def lc_eval(p, t, edge=2, texp=0.01):
    """
    Returns flux at given times, given parameters.

    :param p:
        Parameter vector, of length 4 + 6*Nplanets
        p[0:4] = [rhostar, q1, q2, dilution]
        p[4+i*6:10+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

    :param t:
        Times at which to evaluate model.

    :param edge:
        How many "durations" (approximately calculated) from transit center
        to bother calculating transit model.  If the eccentricity is significant,
        you may need to use a larger edge (default = 2).

    """
    n_planets = (len(p) - 4)//6
    
    rhostar, q1, q2, dilution = p[:4]

    central = Central(q1=q1, q2=q2)
    central.density = rhostar
    s = System(central, dilution=dilution)

    tot = 0
    close_to_transit = np.zeros_like(t).astype(bool)

    duration_guess = 1
    for i in range(n_planets):
        period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
        r = central.radius * rprs
        body = Body(flux=0, r=r, mass=0, period=period, t0=epoch,
                   e=e, omega=w, b=b)
        s.add_body(body)

        tfold = t_folded(t, period, epoch)

        #because duration hack sometimes fails...
        try:
            duration = body.duration_approx
            duration_guess = duration
        except:
            duration = duration_guess

        close_to_transit += np.absolute(tfold) < edge*(duration)

    f = np.ones_like(t)
    f[close_to_transit] = s.light_curve(t[close_to_transit], texp=texp)
    return f
    
        
