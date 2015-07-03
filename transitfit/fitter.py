from __future__ import print_function, division

import numpy as np
from scipy.optimize import minimize

from .utils import lc_eval

class TransitModel(object):
    def __init__(self, lc, edge=2):
        self.lc = lc
        self.edge = edge

        self._bestfit = None
        
    def evaluate(self, p):
        """Evaluates light curve model at light curve times

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
        return lc_eval(p, self.lc.t, edge=self.edge,
                       texp=self.lc.texp)

    def fit_leastsq(self, p0, method='Powell', **kwargs):
        fit = minimize(self.cost, p0, method=method, **kwargs)
        self._bestfit = fit.x
        return fit

        
    def __call__(self, p):
        return self.lnpost(p)

    def cost(self, p):
        return -self.lnpost(p)
    
    def lnpost(self, p):
        prior = self.lnprior(p)
        if np.isfinite(prior):
            like = self.lnlike(p)
        else:
            return prior
        return prior + like
                    
    def lnlike(self, p):
        flux_model = self.evaluate(p)
        return (-0.5 * (flux_model - self.lc.flux)**2 / self.lc.flux_err**2).sum()
        
    def lnprior(self, p):
        rhostar, q1, q2, dilution = p[:4]
        if not (0 <= q1 <=1 and 0 <= q2 <= 1):
            return -np.inf
        if rhostar < 0:
            return -np.inf
        if not (0 <= dilution < 1):
            return -np.inf
        
        for i in range(self.lc.n_planets):
            period, epoch, b, rprs, e, w = p[4+i*6:10+i*6]
            if period <= 0:
                return -np.inf
            if not 0 <= e < 1:
                return -np.inf
            if not 0 <= b < 1+rprs:
                return -np.inf
            if rprs < 0:
                return -np.inf
            
        return 0

    def plot_planets(self, params, width=2, color='r',
                     marker='o', ls='none', ms=0.5, **kwargs):
        fig = self.lc.plot_planets(width=width, **kwargs)

        # Scale widths for each plot by duration.
        maxdur = max([p.duration for p in self.lc.planets])
        widths = [width / (p.duration/maxdur) for p in self.lc.planets]

        depth = (1 - self.evaluate(params))*1e6
        
        for i,ax in enumerate(fig.axes):
            tfold = self.lc.t_folded(i) * 24
            close = self.lc.close(i, width=widths[i], only=True)
            ax.plot(tfold[close], depth[close], color=color, mec=color,
                    marker=marker, ls=ls, ms=ms, **kwargs)

        return fig
