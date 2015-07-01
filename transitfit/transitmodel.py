from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transit import Central, System, Body

from .utils import t_folded

class TransitModel(object):
    """Object holding time/flux data allowing for simple transit fitting

    :param time,flux,flux_err:
        Time series data.

    :param period,epoch:
        Either single values, or lists of values if there are multiple
        planets in the system.

    :param texp:
        Exposure time.  If not provided, will be assumed to be median
        of delta-t.
        
    """
    def __init__(self, time, flux, flux_err=0.0001,
                period=None, epoch=None, duration=None,
                texp=None):
        
        if texp is None:
            texp = np.median(time[1:]-time[:-1])
        self.texp = texp
        
        if period is None:
            period = []
        if epoch is None:
            epoch = []
        if duration is None:
            duration = []
        
        if type(period)==float:
            period = [period]
        if type(epoch)==float:
            epoch = [epoch]
        if type(duration)==float:
            duration = [duration]
        
        assert len(period)==len(epoch)==len(duration)

        self.period = period
        self.epoch = epoch
        self.duration = duration
        
        self._time = np.array(time)
        self._flux = np.array(flux)
        self._flux_err = np.array(flux_err)

        
    @property
    def time(self):
        return self._time.ravel()
    
    @property
    def flux(self):
        return self._flux.ravel()
        
    @property
    def flux_err(self):
        return self._flux_err.ravel()
        
    @property
    def n_planets(self):
        return len(self.period)
        
    def add_planet(self, period, epoch):
        self.period.append(period)
        self.epoch.append(epoch)

    def tfold(self, i=0):
        """Times folded on the period and epoch of planet i
        """
        return t_folded(self.time, self.period[i], self.epoch[i])

    def close(self, i=0, width=2):
        """Boolean array with True everywhere within width*duration of planet i 
        """
        tfold = self.tfold(i)
        return np.absolute(tfold) < width*self.duration[i]

    @property
    def anyclose(self):
        close = np.ones_like(self.time).astype(bool)
        
        
    def continuum(self, p, t):
        """out-of-transit light curve--- can in principle be variational model
        """
        return np.ones_like(t)
        
    def light_curve(self, p, t, edge=2):
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
        if self.n_planets==0:
            raise ValueError('Must add planet before evaluating light curve')

        self._pars = p
        self._duration_approx = list(self.duration) #making a copy
    
        rhostar, q1, q2, dilution = p[:4]
        
        central = Central(q1=q1, q2=q2)
        central.density = rhostar
        s = System(central, dilution=dilution)
        
        tot = 0
        close_to_transit = np.zeros_like(t).astype(bool)
        for i in range(self.n_planets):
            period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
            r = central.radius * rprs
            body = Body(flux=0, r=r, mass=0, period=period, t0=epoch,
                       e=e, omega=w, b=b)
            s.add_body(body)
            
            tfold = t_folded(t, period, epoch)
            
            #because duration hack sometimes fails...
            try:
                duration = body.duration_approx
                self._duration_approx[i] = duration
            except:
                duration = self._duration_approx[i]
        
            close_to_transit += np.absolute(tfold) < edge*(duration)
        
        f = self.continuum(p, t)
        f[close_to_transit] = s.light_curve(t[close_to_transit], texp=self.texp)
        return f

    def lnpost(self, p):
        prior = self.lnprior(p)
        if np.isfinite(prior):
            like = self.lnlike(p)
        else:
            return prior
        return prior + like
                    
    def lnlike(self, p):
        flux_model = self.light_curve(p, self.time)
        return (-0.5 * (flux_model - self.flux)**2 / self.flux_err**2).sum()
        
    def lnprior(self, p):
        rhostar, q1, q2, dilution = p[:4]
        if not (0 <= q1 <=1 and 0 <= q2 <= 1):
            return -np.inf
        if rhostar < 0:
            return -np.inf
        if not (0 <= dilution < 1):
            return -np.inf
        
        for i in range(self.n_planets):
            period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
            if period <= 0:
                return -np.inf
            if not 0 <= e <= 1:
                return -np.inf
            if not 0 <= b < 1+rprs:
                return -np.inf
            if rprs < 0:
                return -np.inf
            
        return 0
