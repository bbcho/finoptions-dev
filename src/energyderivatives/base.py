from abc import ABC, abstractmethod
import numpy as _np
from math import atan as _atan


class _Base(ABC):
    def __init__(self):
        pass

    def _NDF(self, x):
        """
        Calculate the normal distribution function of x
        
        Parameters
        ----------

        x : float
            Value to calculate the normal distribution function for x.

        Returns
        -------
        float

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas 
        """

        return _np.exp(-x * x / 2) / _np.sqrt(8 * _atan(1))

    def _CND(self, x):
        """
        Calculate the cumulated normal distribution function of x
        
        Parameters
        ----------

        x : float
            Value to calculate the cumulated normal distribution function for x.

        Returns
        -------
        float

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        k = 1 / (1 + 0.2316419 * abs(x))
        a1 = 0.319381530
        a2 = -0.356563782
        a3 = 1.781477937
        a4 = -1.821255978
        a5 = 1.330274429
        result = (
            self._NDF(x)
            * (a1 * k + a2 * k ** 2 + a3 * k ** 3 + a4 * k ** 4 + a5 * k ** 5)
            - 0.5
        )
        result = 0.5 - result * _np.sign(x)

        return result


class Derivative(_Base):
    pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


class Option(Derivative):
    """
    Base class for options
    
    put : bool
        If put is True, initialize a put option object. If put is False, initialize a call option object.
    """

    def __init__(self, S: float, K: float, t: float, r: float, b: float, sigma: float):
        self._S = S
        self._K = K
        self._r = r
        self._t = t
        self._b = b
        self._sigma = sigma

    def simulate(self):
        print("sim run")

    def get_params(self):
        return {
            "level S": self._S,
            "strike K": self._K,
            "risk-free-rate e": self._r,
            "time-to-maturity t": self._t,
            "annualized-cost-to-carry": self._b,
            "annualized-volatility sigma": self._sigma,
        }

    def _check_sigma(self, func):
        if self._sigma is None:
            raise ValueError(f"sigma not defined. Required for {func}() method")

    def put():
        pass

    def call():
        pass

    def greeks():
        pass

