from ..base import GreeksFDM, Option as _Option
from ..utils import docstring_from
from ..vanillaoptions import GBSOption as _GBSOption

from .mc_innovations import *
from .mc_paths import *
from .mc_payoffs import *

import numpy as _np


class MonteCarloOption:  # _Option
    """
    Class for the valuation of options using Monte Carlo methods.

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b : float
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.

    Notes
    -----
    Including a cost of carry term b, the model can
    used to price European Options on:

    b = r       stocks (Black and Scholes’ stock option model)
    b = r - q   stocks and stock indexes paying a continuous dividend yield q (Merton’s stock option model)
    b = 0       futures (Black’s futures option model)
    b = r - rf  currency options with foreign interst rate rf (Garman and Kohlhagen’s currency option model)

    Returns
    -------
    GBSOption object.

    Example
    -------
    >>> import energyderivatives as ed
    >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas

    """

    __name__ = "MCOption"
    __title__ = "Monte Carlo Simulation Option"

    def __init__(
        self,
        mc_loops: int,
        path_length: int,
        mc_samples: int,
        dt: float,
        S: float,
        K: float,
        t: float,
        r: float,
        b: float,
        sigma: float = None,
        Innovation=None,
        Path=None,
        Payoff=None,
        trace=True,
        antithetic=True,
        standardization=False,
        eps=None,
        **kwargs,
    ):

        self._mc_loops = mc_loops
        self._path_length = path_length
        self._mc_samples = mc_samples
        self._dt = dt
        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma
        self._Innovation = Innovation(mc_samples, path_length, eps)
        self._Path = Path
        self._Payoff = Payoff
        self._trace = trace
        self._antithetic = antithetic
        self._standardization = standardization
        self._eps = eps
        self._kwargs = kwargs

    def call(self):
        """
        Returns an array of the average option call price per Monte Carlo Loop. Final option value
        is the average of the returned array.

        Returns
        -------
        numpy array

        Example
        -------
        >>> import energyderivatives as ed
        >>> inno = ed.monte_carlo_options.NormalSobolInnovations
        >>> path = ed.monte_carlo_options.WienerPath
        >>> payoff = ed.monte_carlo_options.PlainVanillaPayoff
        >>> opt = ed.monte_carlo_options.MonteCarloOption(50, 30, 5000,
                1/360, 100, 100, 1/12, 0.1, 0.1, 0.4, inno, path, payoff)
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        return self._sim_mc(call=True)

    def put(self):
        """
        Returns an array of the average option out price per Monte Carlo Loop. Final option value
        is the average of the returned array.

        Returns
        -------
        numpy array

        Example
        -------
        >>> import energyderivatives as ed
        >>> inno = ed.monte_carlo_options.NormalSobolInnovations
        >>> path = ed.monte_carlo_options.WienerPath
        >>> payoff = ed.monte_carlo_options.PlainVanillaPayoff
        >>> opt = ed.monte_carlo_options.MonteCarloOption(50, 30, 5000,
                1/360, 100, 100, 1/12, 0.1, 0.1, 0.4, inno, path, payoff)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        return self._sim_mc(call=False)

    def _sim_mc(self, call=True):

        dt = self._dt
        trace = self._trace

        if trace:
            print("\nMonte Carlo Simulation Path:\n\n")
            print("\nLoop:\t", "No\t")

        iteration = _np.zeros(self._mc_loops)

        # MC Iteration Loop:

        for i in range(self._mc_loops):
            #     # if ( i > 1) init = FALSE
            # Generate Innovations:
            eps = self._Innovation.sample_innovation()

            # Use Antithetic Variates if requested:
            if self._antithetic:
                eps = _np.concatenate((eps, -eps))
            #     # Standardize Variates if requested:
            #     if self._standardization:
            #         pass
            #         # eps = (eps-mean(eps))/sqrt(var(as.vector(eps)))

            # Calculate for each path the option price:
            path = self._Path(eps, self._sigma, self._dt, self._b)

            # so I think the original fOptions function has an error. It calcs
            # the payoff along the wrong dimensions such that it only calcs
            # along path_length number of samples vs mc_samples. I think the t()
            # is the problem.

            if call == True:
                payoff = self._Payoff(
                    path, self._S, self._K, self._t, self._r, self._b, self._sigma
                ).call()
            else:
                payoff = self._Payoff(
                    path, self._S, self._K, self._t, self._r, self._b, self._sigma
                ).put()

            tmp = _np.mean(payoff)

            if tmp == _np.inf:
                import warnings

                warnings.warn(f"Warning: mc_loop {i} returned Inf.")
                return (eps, path, payoff)

            iteration[i] = tmp

            if trace:
                print(
                    "\nLoop:\t",
                    i,
                    "\t:",
                    iteration[i],
                    _np.sum(iteration) / (i + 1),
                    end="",
                )
        print("\n")

        return iteration
