from ..base import Option as _Option
from ..vanillaoptions import GBSOption as _GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings
import numdifftools as _nd


class HestonNandiOption(_Option):
    """
    Option class for the Heston-Nandi Garch Option Model.

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in days. i.e. 1/252 for 1 business day, 5/252 for 1 week etc...
    r : float
        Daily rate of interest. i.e. 0.25/252 means about 0.001% per day.
    lamb : float

    omega : float
        The GARCH model parameter which specifies the constant coefficient of the variance
        equation.
    alpha : float
        The GARCH model parameter which specifies the autoregressive coefficient
    beta : float
        The GARCH model parameter which specifies the variance coefficient.
    gamma : float
        The GARCH model parameter which specifies the asymmetry coefficient.

    Notes
    -----


    Returns
    -------
    HestonNandiOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.heston_nandi_options.HestonNandiOption(S=80, K=82, t=1/3, td=1/4, r=0.06, D=4, sigma=0.30)
    >>> opt.call()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "HestonNandiOption"
    __title__ = "The Heston-Nandi Garch Option Model"

    def __init__(
        self,
        S: float,
        K: float,
        t: float,
        r: float,
        lamb: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
    ):

        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._lamb = lamb
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def call(self):
        """
        # Integrate:
        call1 = integrate(.fstarHN, 0, Inf, const = 1, model = model,
            S = S, X = X, Time.inDays = Time.inDays, r.daily = r.daily)
        # For SPlus Compatibility:
        if (is.null(call1$value)) call1$value = call1$integral
        call2 = integrate(.fstarHN, 0, Inf, const = 0, model = model,
            S = S, X = X, Time.inDays = Time.inDays, r.daily = r.daily)
        # For SPlus Compatibility:
        if (is.null(call2$value)) call2$value = call2$integral

        # Compute Call Price:
        call.price = S/2 + exp(-r.daily*Time.inDays) * call1$value -
            X * exp(-r.daily*Time.inDays) * ( 1/2 + call2$value )

        """
        pass

    def put(self):
        return self.call() + self._K * _np.exp(-self._r * self._t) - self._S
