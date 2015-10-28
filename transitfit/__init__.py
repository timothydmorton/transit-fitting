__version__ = '0.1'

try:
    __TRANSITFIT_SETUP__
except NameError:
    __TRANSITFIT_SETUP__ = False

if not __TRANSITFIT_SETUP__:
    from .lightcurve import LightCurve, Planet
    from .kepler import KeplerLightCurve
    from .fitter import TransitModel
