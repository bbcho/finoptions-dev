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
            "level": self._S,
            "strike": self._K,
            "risk-free-rate": self._r,
            "time-to-maturity": self._t,
            "b": self._b,
            "annualized-volatility": self._sigma,
        }

    def put():
        pass

    def call():
        pass

    def greeks():
        pass


class GBSOption(Option):
    """
    Calculate the Generalized Black-Scholes option price either for a call or put option

    Parameters
    ----------

    S : float
        Level or index price
    K : float
        Strike price
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%)
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    b : float
    sigma : float
        Annualized volatility of the underlying asset
    
    Returns
    -------
    GBSOption object

    Example
    -------
    >>> import energyderivatices as ed
    >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.put()
    >>> opt.call()

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    def __init__(self, S: float, K: float, t: float, r: float, b: float, sigma: float):

        # call the init method of the Option class
        super().__init__(S, K, t, r, b, sigma)

        # additional init for d1 and d2
        # fmt: off
        self._d1 = (
            _np.log(self._S / self._K)
            + (self._b + self._sigma * self._sigma / 2) * self._t) / (self._sigma * _np.sqrt(self._t)
            )

        self._d2 = self._d1 - self._sigma * _np.sqrt(self._t)
        # fmt: on

    def call(self):
        """
        Returns the calculated price of a call option according to the
        Generalized Black-Scholes option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatices as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        # fmt: off
        result = (
                    self._S*_np.exp((self._b-self._r)*self._t)*self._CND(self._d1) 
                    - self._K*_np.exp(-self._r*self._t)*self._CND(self._d2)
                )
        # fmt: on

        return result

    def put(self):
        """
        Returns the calculated price of a put option according to the
        Generalized Black-Scholes option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatices as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        # fmt: off
        result = (
                    self._K * _np.exp(-self._r * self._t) * self._CND(-self._d2) 
                    - self._S * _np.exp((self._b - self._r) * self._t) * self._CND(-self._d1)
                )
        # fmt: on

        return result


if __name__ == "__main__":

    opt = GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    # print("{:.6f}".format(opt.call()))
    print(opt.call())
    print(opt.put())

