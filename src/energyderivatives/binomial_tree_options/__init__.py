from ..base import Option as _Option
from ..vanillaoptions import GBSOption as _GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings
import numdifftools as _nd


class BiTreeOption:
    def plot(self):
        pass


class CRRBinomialTreeOption(_Option, BiTreeOption):
    def __init__(self, S: float, K: float, t: float, r: float, b: float, sigma: float):
        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma

    def call(self, n: int = 5, type: str = "european"):
        """
        Returns the calculated price of a call option according to the
        Cox-Ross-Rubinstein Binomial Tree Option Model option price model.

        Parameters
        ----------
        n : int
            Number of time steps, by default 5.
        type : str
            'european' for european option, 'american' for american option. By default "european"

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5)
        >>> opt.call(n=5)
        >>> opt.call(n=5, type='american)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        z = 1

        return self._calc_price(z, n, type)

    def put(self, n: int = 5, type: str = "european"):
        """
        Returns the calculated price of a put option according to the
        Cox-Ross-Rubinstein Binomial Tree Option Model option price model.

        Parameters
        ----------
        n : int
            Number of time steps, by default 5.
        type : str
            'european' for european option, 'american' for american option. By default "european"

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4)
        >>> opt.put(n=5)
        >>> opt.put(n=5, type='american)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        z = -1
        return self._calc_price(z, n, type)

    def _calc_price(self, z, n, type):
        dt = self._t / n
        u = _np.exp(self._sigma * _np.sqrt(dt))
        d = 1 / u
        p = (_np.exp(self._b * dt) - d) / (u - d)
        Df = _np.exp(-self._r * dt)

        OptionValue = z * (
            self._S * u ** _np.arange(0, n + 1) * d ** _np.arange(n, -1, -1) - self._K
        )
        OptionValue = (_np.abs(OptionValue) + OptionValue) / 2

        if type == "european":
            return self._euro(OptionValue, n, Df, p)[0]
        elif type == "american":
            return self._amer(
                OptionValue,
                n,
                Df,
                p,
                self._K,
                d,
                self._S,
                u,
                z,
            )[0]

    def _euro(self, OptionValue, n, Df, p):
        for j in _np.arange(0, n)[::-1]:
            for i in _np.arange(0, j + 1):
                OptionValue[i] = (
                    p * OptionValue[i + 1] + (1 - p) * OptionValue[i]
                ) * Df

        return OptionValue

    def _amer(self, OptionValue, n, Df, p, K, d, S, u, z):
        for j in _np.arange(0, n)[::-1]:
            for i in _np.arange(0, j + 1):
                OptionValue[i] = max(
                    (z * (S * u ** i * d ** (abs(i - j)) - K)),
                    (p * OptionValue[i + 1] + (1 - p) * OptionValue[i]) * Df,
                )
        return OptionValue


class JRBinomialTreeOption(_Option, BiTreeOption):
    pass


class TIANBinomialTreeOption(_Option, BiTreeOption):
    pass


class BinomialTreeOption(_Option, BiTreeOption):
    pass
