from .base import Option
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings


class GBSOption(Option):
    """
    Calculate the Generalized Black-Scholes option price for either a call or put option.

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
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility. 
        Required otherwise. By default None.
    
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

    def __init__(
        self, S: float, K: float, t: float, r: float, b: float, sigma: float = None
    ):

        # call the init method of the Option class
        super().__init__(S, K, t, r, b, sigma)

        # additional init for d1 and d2
        # fmt: off
        if sigma is not None:
            self._d1 = (
                _np.log(self._S / self._K)
                + (self._b + self._sigma * self._sigma / 2) * self._t) / (self._sigma * _np.sqrt(self._t)
                )

            self._d2 = self._d1 - self._sigma * _np.sqrt(self._t)
        # fmt: on

    def _check_sigma(self, func):
        if self._sigma is None:
            raise ValueError(f"sigma not defined. Required for {func}() method")

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
        self._check_sigma("call")
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
        self._check_sigma("put")
        # fmt: off
        result = (
                    self._K * _np.exp(-self._r * self._t) * self._CND(-self._d2) 
                    - self._S * _np.exp((self._b - self._r) * self._t) * self._CND(-self._d1)
                )
        # fmt: on

        return result

    def delta(self, call: bool = True):
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
        self._check_sigma("delta")
        if call == True:
            return _np.exp((self._b - self._r) * self._t) * self._CND(self._d1)
        else:
            return _np.exp((self._b - self._r) * self._t) * (self._CND(self._d1) - 1)

    def theta(self, call: bool = True):
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
        self._check_sigma("theta")
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
        self._check_sigma("vega")
        # for both call and put options
        return (
            self._S
            * _np.exp((self._b - self._r) * self._t)
            * self._NDF(self._d1)
            * _np.sqrt(self._t)
        )

    def rho(self, call: bool = True):
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
        self._check_sigma("rho")
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

    def lamb(self, call: bool = True):
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
        self._check_sigma("lamb")
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
        self._check_sigma("gamma")
        # for both call and put options
        return (
            _np.exp((self._b - self._r) * self._t)
            * self._NDF(self._d1)
            / (self._S * self._sigma * _np.sqrt(self._t))
        )

    def c_of_c(self, call: bool = True):
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
        self._check_sigma("c_of_c")
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

    def greeks(self, call: bool = True):
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
        self._check_sigma("greeks")
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

    def volatility(
        self,
        price: float,
        call: bool = True,
        tol=_sys.float_info.epsilon,
        maxiter=10000,
    ):
        """
        Compute the implied volatility of the GBSOption.
        """
        if self._sigma is not None:
            _warnings.warn("sigma is not None but calculating implied volatility.")

        def _func(sigma):
            temp = GBSOption(
                S=self._S, K=self._K, t=self._t, r=self._r, b=self._b, sigma=sigma
            )
            if call == True:
                return price - temp.call()
            else:
                return price - temp.put()

        sol = _root_scalar(_func, bracket=[-10, 10], xtol=tol, maxiter=maxiter)

        return sol

    def summary(self, printer=True):
        out = f"""
        Title: Black Scholes Option Valuation

        Parameters:
            S = {self._S}
            K = {self._K}
            t = {self._t}
            r = {self._r}
            b = {self._b}
            sigma = {self._sigma}
        """

        if self._sigma is not None:
            price = f"""
        Option Price:
            call: {round(self.call(),6)}
            put: {round(self.put(),6)}
        """
            out += price

        if printer == True:
            print(out)
        else:
            return out

    def __str__(self):
        return self.summary(printer=False)

    def __repr__(self):
        out = f"GBSOption({self._S}, {self._K}, {self._t}, {self._r}, {self._b}, {self._sigma})"
        return out
