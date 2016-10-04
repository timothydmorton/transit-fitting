from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os, os.path

from scipy.special import beta

from astropy import constants as const
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
R_sun = const.R_sun.cgs.value
DAY = 86400


import emcee

import corner
    
from .utils import lc_eval

class TransitModel(object):
    """Model of one or more transiting planets around a particular star
    
    :param lc:
        LightCurve object

    :param width:
        The number of durations around a transit calculated to evaluate
        model.  Model is evaluated between ``tc - width*duration`` and
        ``tc + width*duration``, where ``tc`` is the transit center time.

    """
    def __init__(self, lc, width=2, continuum_method='constant',
                 fix_zp=False, fix_circular=False, no_dilution=False):

        self.lc = lc
        self.width = width
        self.continuum_method = continuum_method

        self.fix_zp = fix_zp
        self.fix_circular = fix_circular
        self.no_dilution = no_dilution

        self._bestfit = None
        self._samples = None


    def continuum(self, p, t):
        """ Out-of-transit 'continuum' model.

        :param p:
            List of parameters.  For now all that is implemented
            is a constant.

        param t:
            Times at which to evaluate model.
        
        """
        p = np.atleast_1d(p)
        if self.fix_zp:
            return np.ones_like(t)
        return p[0]*np.ones_like(t)
        



    def evaluate(self, p):
        """Evaluates light curve model at light curve times

        :param p:
            Parameter vector, of length 5 + 6*Nplanets
            p[0] = flux zero-point
            p[1:5] = [rhostar, q1, q2, dilution]
            p[5+i*6:11+i*6] = [period, epoch, b, rprs, e, w] for i-th planet

        :param t:
            Times at which to evaluate model.
            

        """
        f = self.continuum(p[0], self.lc.t)

        # Identify points near any transit
        close = np.zeros_like(self.lc.t).astype(bool)
        for i in range(self.lc.n_planets):
            close += self.lc.close(i, width=self.width)

        f[close] = lc_eval(p[1:], self.lc.t[close], texp=self.lc.texp)
        return f

    def fit_leastsq(self, p0, method='Powell', **kwargs):
        fit = minimize(self.cost, p0, method=method, **kwargs)
        self._bestfit = fit.x
        return fit

    def fit_emcee(self, p0=None, nwalkers=200, threads=1,
                  nburn=10, niter=100, **kwargs):
        if p0 is None:
            p0 = self.lc.default_params

        ndim = len(p0)

        # TODO: improve walker initialization!

        nw = nwalkers
        p0 = np.ones((nw,ndim)) * np.array(p0)[None,:]

        p0[:, 0] += np.random.normal(size=nw)*0.0001 #flux zp
        p0[:, 1] += np.random.normal(size=nw) #rhostar
        p0[:, 2] += np.random.normal(size=nw)*0.1 #q1
        p0[:, 3] += np.random.normal(size=nw)*0.1 #q2
        p0[:, 4] += np.random.normal(size=nw)*0.01 #dilution

        for i in range(self.lc.n_planets):
            per, per_unc = self.lc.planets[i]._period
            ep, ep_unc = self.lc.planets[i]._epoch
            p0[:, 5 + 6*i] += np.random.normal(size=nw)*per_unc*0.01 #period
            p0[:, 6 + 6*i] += np.random.normal(size=nw)*ep_unc*0.01 #epoch
            p0[:, 7 + 6*i] = np.random.random(size=nw)*0.8 # impact param
            p0[:, 8 + 6*i] *= (1 + np.random.normal(size=nw)*0.01) #rprs
            p0[:, 9 + 6*i] = np.random.random(size=nw)*0.5 # eccentricity
            p0[:, 10 + 6*i] = np.random.random(size=nw)*2*np.pi

        p0 = np.absolute(p0) # no negatives allowed 
                                                
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        pos,prob,state = sampler.run_mcmc(p0, nburn)
        sampler.reset()

        sampler.run_mcmc(pos, niter)

        self.sampler = sampler
        return sampler
        
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
        if self.no_dilution:
            p[4] = 0.
        try:
            flux_model = self.evaluate(p)
        except ValueError:
            return -np.inf
        
        return (-0.5 * (flux_model - self.lc.flux)**2 / self.lc.flux_err**2).sum()
        
    def lnprior(self, p):
        flux_zp, rhostar, q1, q2, dilution = p[:5]

        if not (0 <= q1 <=1 and 0 <= q2 <= 1):
            return -np.inf
        if rhostar < 0:
            return -np.inf
        if not (0 <= dilution < 1):
            return -np.inf

        tot = 0


        # Apply stellar density prior if relevant.
        if self.lc.rhostar is not None:
            tot += np.log(self.lc.rhostar_pdf(rhostar))
            
        if not self.no_dilution:
            if self.lc.dilution is not None:
                tot += np.log(self.lc.dilution_pdf(dilution))

        for i in xrange(self.lc.n_planets):
            period, epoch, b, rprs, e, w = p[5+i*6:11+i*6]

            if not 0 < e < 1:
                return -np.inf
            if period <= 0:
                return -np.inf
            if rprs <= 0:
                return -np.inf
            if b < 0 or b > (1 + rprs):
                return -np.inf

            # If fixing circular fit, use this prior on eccentricity
            if self.fix_circular:
                tot += -0.5*e**2 / 0.0001**2

            factor = 1.0
            if e > 0:
                factor = (1 + e * np.sin(w)) / (1 - e * e)

            aR = (rhostar * G * (period*DAY)**2 / (3*np.pi))**(1./3)
                
            arg = b * factor/aR
            if arg > 1.0:
                return -np.inf
                
            
            # Priors on period, epoch based on discovery measurements
            prior_p, prior_p_err = self.lc.planets[i]._period
            tot += -0.5*(period - prior_p)/prior_p_err**2

            prior_ep, prior_ep_err = self.lc.planets[i]._epoch
            tot += -0.5*(epoch - prior_ep)/prior_ep_err**2

            # log-flat prior on rprs
            tot += np.log(1 / rprs)

            # Beta prior on eccentricity
            a,b = (0.4497, 1.7938)
            eccprior = 1/beta(a,b) * e**(a-1) * (1 - e)**(b-1)
            tot += np.log(eccprior)
            
        return tot

    def plot_planets(self, params, width=2, color='r', fig=None,
                     marker='o', ls='none', ms=0.5, **kwargs):
        
        if fig is None:
            fig = self.lc.plot_planets(width=width, **kwargs)

        # Scale widths for each plot by duration.
        maxdur = max([p.duration for p in self.lc.planets])
        widths = [width / (p.duration/maxdur) for p in self.lc.planets]

        depth = (1 - self.evaluate(params))*1e6
        
        try:
            len(fig.axes)
            axes = fig.axes
        except TypeError:
            axes = [fig.axes]

        for i,ax in enumerate(fig.axes):
            tfold = self.lc.t_folded(i) * 24
            close = self.lc.close(i, width=widths[i], only=True)
            ax.plot(tfold[close], depth[close], color=color, mec=color,
                    marker=marker, ls=ls, ms=ms, **kwargs)

        return fig

    @property
    def samples(self):
        if not hasattr(self,'sampler') and self._samples is None:
            raise AttributeError('Must run MCMC (or load from file) '+
                                 'before accessing samples')
        
        if self._samples is not None:
            df = self._samples
        else:
            self._make_samples()
            df = self._samples

        return df
        
    def _make_samples(self):
        ok = np.isfinite(self.sampler.flatlnprobability)

        flux_zp = self.sampler.flatchain[ok,0]
        rho = self.sampler.flatchain[ok,1]
        q1 = self.sampler.flatchain[ok,2]
        q2 = self.sampler.flatchain[ok,3]
        dilution = self.sampler.flatchain[ok,4]

        df = pd.DataFrame(dict(flux_zp=flux_zp,
                               rho=rho, q1=q1, q2=q2,
                               dilution=dilution))
        df = pd.DataFrame()
        #Keep things in the right order
        for par in ['flux_zp', 'rho', 'q1', 'q2', 'dilution']:
            df[par] = eval(par)

        for i in range(self.lc.n_planets):
            for j, par in enumerate(['period', 'epoch', 'b', 'rprs',
                                     'ecc', 'omega']):
                column = self.sampler.flatchain[ok, 5+j+i*6]
                
                if par=='omega':
                    column = column % (2*np.pi)
                    
                df['{}_{}'.format(par,i+1)] = column

        self._samples = df
        self._lnprob = self.sampler.flatlnprobability[ok]

    def corner(self, params=None, i=0, query=None, extent=0.999,
                 planet_only=False,
                 **kwargs):
        """
        Makes a nifty corner plot for planet i

        Uses :func:`triangle.corner`.

        :param params: (optional)
            Names of columns to plot.  Set planet_only to be ``True``
            to leave out star params.

        :param i:
            Planet number (starting from 0)

        :param query: (optional)
            Optional query on samples.

        :param extent: (optional)
            Will be appropriately passed to :func:`triangle.corner`.

        :param **kwargs:
            Additional keyword arguments passed to :func:`triangle.corner`.

        :return:
            Figure oject containing corner plot.
            
        """
        if params is None:
            if planet_only:
                params = []
            else:
                params = ['dilution', 'rho', 'q1', 'q2']
            for par in ['period', 'epoch', 'b', 'rprs',
                        'ecc', 'omega']:
                params.append('{}_{}'.format(par, i+1))


        df = self.samples

        if query is not None:
            df = df.query(query)

        #convert extent to ranges, but making sure
        # that truths are in range.
        extents = []
        remove = []
        for i,par in enumerate(params):
            values = df[par]
            qs = np.array([0.5 - 0.5*extent, 0.5 + 0.5*extent])
            minval, maxval = values.quantile(qs)
            if 'truths' in kwargs:
                datarange = maxval - minval
                if kwargs['truths'][i] < minval:
                    minval = kwargs['truths'][i] - 0.05*datarange
                if kwargs['truths'][i] > maxval:
                    maxval = kwargs['truths'][i] + 0.05*datarange
            extents.append((minval,maxval))
            
        return corner.corner(df[params], labels=params, 
                               range=extents, **kwargs)

    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)

        Samples are saved to /samples location under given path,
        and object properties are also attached, so suitable for
        re-loading via :func:`TransitModel.load_hdf`.
        
        :param filename:
            Name of file to save to.  Should be .h5 file.

        :param path: (optional)
            Path within HDF file structure to save to.

        :param overwrite: (optional)
            If ``True``, delete any existing file by the same name
            before writing.

        :param append: (optional)
            If ``True``, then if a file exists, then just the path
            within the file will be updated.
        """
        
        if os.path.exists(filename):
            store = pd.HDFStore(filename)
            if path in store:
                store.close()
                if overwrite:
                    os.remove(filename)
                elif not append:
                    raise IOError('{} in {} exists.  Set either overwrite or append option.'.format(path,filename))
            else:
                store.close()

                
        self.samples.to_hdf(filename, '{}/samples'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/samples'.format(path)).attrs
        attrs.width = self.width
        attrs.continuum_method = self.continuum_method
        attrs.lc_type = type(self.lc)
        
        store.close()

        self.lc.save_hdf(filename, path=path, append=True)
        
    @classmethod
    def load_hdf(cls, filename, path=''):
        """
        A class method to load a saved StarModel from an HDF5 file.

        File must have been created by a call to :func:`StarModel.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`StarModel` object.
        """
        store = pd.HDFStore(filename)
        try:
            samples = store['{}/samples'.format(path)]
            attrs = store.get_storer('{}/samples'.format(path)).attrs        
        except:
            store.close()
            raise
        width = attrs.width
        continuum_method = attrs.continuum_method
        lc_type = attrs.lc_type
        store.close()

        lc = lc_type.load_hdf(filename, path=path)

        mod = cls(lc, width=width, continuum_method=continuum_method)
        mod._samples = samples
        
        return mod
    

class BinaryTransitModel(TransitModel):
    """
    TransitModel of two stars

    :param lc:
        BinaryLightCurve object.

    :param which:
        e.g., for three-planet system: ['A', 'A', 'B']

    """
    def __init__(self, lc, which=None):
        
        if which is None:
            which = ['A'] * lc.n_planets

        def evaluate(self, p):
            """
            Evaluates light curve model at light curve times.

            Difference with TransitModel is that there are now two stars.

            

            :param p:
                Parameter vector, of length 1 + 4*2 + 6*Nplanets
                p[0] = flux zero-point
                p[1:5] = [rhostar_A, q1_A, q2_A, dilution_A]
                p[5:9] = [rhostar_B, q1_B, q2_B, dilution_B]
                p[9+i*6 : 15+i*6] = [per, ep, b, rprs, e, w] for i-th planet  
            """
            
            # So as to be careful to not pass slices of p around...
            p = list(p)

            f = self.continuum(p[0], self.lc, t)

            # Must use self.which to determine which star parameters
            #  get passed to lc_eval for each planet.

            close_A = np.zeros_like(self.lc.t).astype(bool)
            close_B = np.zeros_like(self.lc.t).astype(bool)

            # Build close_A and close_B properly, depending on self.which
            for i in range(self.lc.n_planets):
                pass


            #f[close_A] = 
            #f[close_B] = 
            
            return f

        def lnprior(self, p):
            pass
            
        def _make_samples(self):
            pass

        def corner(self, params=None, **kwargs):
            # Set default params appropriately
            
            super(BinaryTransitModel, self).corner(params, **kwargs)
