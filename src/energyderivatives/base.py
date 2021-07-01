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
    >>> import energyderivatives as ed
    >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

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
        >>> import energyderivatives as ed
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
        >>> import energyderivatives as ed
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

    def delta(self, call: bool):
        """
        Method to return delta greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns delta greek for call option if True, else returns delta greek for put options. By default True.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.delta(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        if call == True:
            return _np.exp((self._b - self._r) * self._t) * self._CND(self._d1)
        else:
            return _np.exp((self._b - self._r) * self._t) * (self._CND(self._d1) - 1)

    def theta(self, call: bool):
        """
        Method to return theta greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns theta greek for call option if True, else returns theta greek for put options. By default True.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.theta(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        Theta1 = -(
            self._S
            * _np.exp((self._b - self._r) * self._t)
            * self._NDF(self._d1)
            * self._sigma
        ) / (2 * _np.sqrt(self._t))

        if call == True:
            return (
                Theta1
                - (self._b - self._r)
                * self._S
                * _np.exp((self._b - self._r) * self._t)
                * self._CND(+self._d1)
                - self._r * self._K * _np.exp(-self._r * self._t) * self._CND(+self._d2)
            )
        else:
            return (
                Theta1
                + (self._b - self._r)
                * self._S
                * _np.exp((self._b - self._r) * self._t)
                * self._CND(-self._d1)
                + self._r * self._K * _np.exp(-self._r * self._t) * self._CND(-self._d2)
            )

    def vega(self):
        """
        Method to return vega greek for either call or put options.

        Parameters
        ----------
        None

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.vega(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        # for both call and put options
        return (
            self._S
            * _np.exp((self._b - self._r) * self._t)
            * self._NDF(self._d1)
            * _np.sqrt(self._t)
        )

    def rho(self, call: bool):
        """
        Method to return rho greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns rho greek for call option if True, else returns rho greek for put options. By default True.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.rho(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        if call == True:
            price = self.call()
            if self._b != 0:
                result = (
                    self._t
                    * self._K
                    * _np.exp(-self._r * self._t)
                    * self._CND(self._d2)
                )
            else:
                result = -self._t * price
        else:
            price = self.put()
            if self._b != 0:
                result = (
                    -self._t
                    * self._K
                    * _np.exp(-self._r * self._t)
                    * self._CND(-self._d2)
                )
            else:
                result = -self._t * price

        return result

    def lamb(self, call: bool):
        """
        Method to return lambda greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns lambda greek for call option if True, else returns lambda greek for put options. By default True.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.lambda(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        if call == True:
            price = self.call()
            result = (
                _np.exp((self._b - self._r) * self._t)
                * self._CND(self._d1)
                * self._S
                / price
            )

        else:
            price = self.put()
            result = (
                _np.exp((self._b - self._r) * self._t)
                * (self._CND(self._d1) - 1)
                * self._S
                / price
            )

        return result

    def gamma(self):
        """
        Method to return gamma greek for either call or put options.

        Parameters
        ----------
        None

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.gamma(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        # for both call and put options
        return (
            _np.exp((self._b - self._r) * self._t)
            * self._NDF(self._d1)
            / (self._S * self._sigma * _np.sqrt(self._t))
        )

    def c_of_c(self, call: bool):
        """
        Method to return C of C greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns C of C greek for call option if True, else returns C of C greek for put options. By default True.

        Returns
        -------
        float

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.c_of_c(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        if call == True:
            return (
                self._t
                * self._S
                * _np.exp((self._b - self._r) * self._t)
                * self._CND(self._d1)
            )
        else:
            return (
                -self._t
                * self._S
                * _np.exp((self._b - self._r) * self._t)
                * self._CND(-self._d1)
            )

    def greeks(self, call: bool):
        """
        Method to return greeks delta, theta, vegam rho, lambda, gamma and C of C for either
        call or put options.

        Parameters
        ----------
        call : bool
            Returns greeks for call option if True, else returns greeks for put options. By default True.

        Returns
        -------
        dictionary of greeks

        Example
        -------
        >>> import energyderivatives as ed
        >>> opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.greeks(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
            "CofC": self.c_of_c(call),
        }

        return gk


if __name__ == "__main__":

    opt = GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    # print("{:.6f}".format(opt.call()))
    print(opt.call())
    print(opt.put())
    print(opt.delta(call=True))
    print(opt.delta(call=False))

    print(opt.greeks(call=True))

