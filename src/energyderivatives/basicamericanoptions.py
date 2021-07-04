from .base import Option
from .vanillaoptions import GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings


class RollGeskeWhaleyOption(Option):
    """
    Roll-Geske-Whaley Calls on Dividend Paying Stocks

    Calculates the option price of an American call on a stock
    paying a single dividend with specified time to divident
    payout. The option valuation formula derived by Roll, Geske 
    and Whaley is used.
    
    Parameters
    ----------

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    td : float
        Time to dividend payout in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    D : float
        A single dividend with time to dividend payout td.
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility. 
        Required otherwise. By default None.
    
    Notes
    -----

    put price does not exist.

    Returns
    -------
    RollGeskeWhaleyOption object.

    Example
    -------
    >>> import energyderivatives as ed
    >>> opt = ed.RollGeskeWhaleyOption(S=80, K=82, t=1/3, td=1/4, r=0.06, D=4, sigma=0.30)
    >>> opt.call()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "RollGeskeWhaleyOption"
    __title__ = "Roll-Geske-Whaley Calls on Dividend Paying Stocks"

    def __init__(
        self, S: float, K: float, t: float, td: float, r: float, D: float, sigma: float
    ):

        self._S = S
        self._K = K
        self._t = t
        self._td = td
        self._r = r
        self._D = D
        self._sigma = sigma

        # Settings:
        self._big = 100000000
        self._eps = 1.0e-5

    def is_call_optimal(self):
        """
        Method to determine if it is currently optimal to exercise the option.

        Returns
        -------
        True if it is optimal to exercise the option.
        False if it is NOT optimal to exercise the option.
        """
        if self._D <= self._K * (1 - _np.exp(-self._r * (self._t - self._td))):
            return False
        else:
            return True

    def call(self):
        # Compute:
        Sx = self._S - self._D * _np.exp(-self._r * self._td)

        if self._D <= self._K * (1 - _np.exp(-self._r * (self._t - self._td))):
            result = GBSOption(
                Sx, self._K, self._t, self._r, b=self._r, sigma=self._sigma
            ).call()
            # print("\nWarning: Not optimal to exercise\n")
            return result

        ci = GBSOption(
            self._S, self._K, self._t - self._td, self._r, b=self._r, sigma=self._sigma
        ).call()

        HighS = self._S

        while (ci - HighS - self._D + self._K > 0) & (HighS < self._big):
            HighS = HighS * 2
            ci = GBSOption(
                HighS,
                self._K,
                self._t - self._td,
                self._r,
                b=self._r,
                sigma=self._sigma,
            ).call()

        if HighS > self._big:
            result = GBSOption(
                Sx, self._K, self._t, self._r, b=self._r, sigma=self._sigma
            ).call()
            raise ValueError("HighS > big setting")

        LowS = 0
        I = HighS * 0.5
        ci = GBSOption(
            I, self._K, self._t - self._td, self._r, b=self._r, sigma=self._sigma
        ).call()

        # Search algorithm to find the critical stock price I
        while (abs(ci - I - self._D + self._K) > self._eps) & (
            (HighS - LowS) > self._eps
        ):
            if ci - I - self._D + self._K < 0:
                HighS = I
            else:
                LowS = I
            I = (HighS + LowS) / 2
            ci = GBSOption(
                I, self._K, self._t - self._td, self._r, b=self._r, sigma=self._sigma
            ).call()

        a1 = (_np.log(Sx / self._K) + (self._r + self._sigma ** 2 / 2) * self._t) / (
            self._sigma * _np.sqrt(self._t)
        )
        a2 = a1 - self._sigma * _np.sqrt(self._t)
        b1 = (_np.log(Sx / I) + (self._r + self._sigma ** 2 / 2) * self._td) / (
            self._sigma * _np.sqrt(self._td)
        )
        b2 = b1 - self._sigma * _np.sqrt(self._td)

        result = (
            Sx * self._CND(b1)
            + Sx * self._CBND(a1, -b1, -_np.sqrt(self._td / self._t))
            - self._K
            * _np.exp(-self._r * self._t)
            * self._CBND(a2, -b2, -_np.sqrt(self._td / self._t))
            - (self._K - self._D) * _np.exp(-self._r * self._td) * self._CND(b2)
        )

        return result

    def put(self):
        print("put option price not defined for RollGeskeWhaleyOption")

    def get_params(self):
        return {
            "S": self._S,
            "K": self._K,
            "t": self._t,
            "td": self._td,
            "r": self._r,
            "D": self._D,
            "sigma": self._sigma,
        }

    def summary(self, printer=True):
        """
        Print summary report of option
        
        Parameters
        ----------
        printer : bool
            True to print summary. False to return a string.
        """
        out = f"Title: {self.__title__}\n\nParameters:\n\n"

        params = self.get_params()

        for p in params:
            out += f"  {p} = {params[p]}\n"

        try:
            # if self._sigma or its variations are not None add call and put prices
            price = f"\nOption Price:\n\n  call: {round(self.call(),6)}\n"
            out += price
        except:
            pass

        out += f"  Optimal to Exercise Call Option: {self.is_call_optimal()}"

        if printer == True:
            print(out)
        else:
            return out


class BAWAmericanApproxOption(Option):
    """
    Barone-Adesi and Whaley Approximation, calculates the option price of an 
    American call or put option on an underlying asset for a given cost-of-carry 
    rate. The quadratic approximation method by Barone-Adesi and Whaley is used.

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

    Note
    ----
    that setting: 
    b = r we get Black and Scholes’ stock option model
    b = r-q we get Merton’s stock option model with continuous dividend yield q
    b = 0 we get Black’s futures option model
    b = r-rf we get Garman and Kohlhagen’s currency option model with foreign 
    interest rate rf
    
    Returns
    -------
    GBSOption object.

    Example
    -------
    >>> import energyderivatives as ed
    >>> opt = ed.BAWAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "BAWAmericanApproxOption"
    __title__ = "Barone-Adesi and Whaley Approximation"

    def _bawKc(self):
        # Newton Raphson algorithm to solve for the critical commodity
        # price for a Call.
        # Calculation of seed value, Si
        n = 2 * self._b / self._sigma ** 2
        m = 2 * self._r / self._sigma ** 2
        q2u = (-(n - 1) + _np.sqrt((n - 1) ** 2 + 4 * m)) / 2
        Su = self._K / (1 - 1 / q2u)
        h2 = (
            -(self._b * self._t + 2 * self._sigma * _np.sqrt(self._t))
            * self._K
            / (Su - self._K)
        )
        Si = self._K + (Su - self._K) * (1 - _np.exp(h2))
        K = 2 * self._r / (self._sigma ** 2 * (1 - _np.exp(-self._r * self._t)))
        d1 = (_np.log(Si / K) + (self._b + self._sigma ** 2 / 2) * self._t) / (
            self._sigma * _np.sqrt(self._t)
        )
        Q2 = (-(n - 1) + _np.sqrt((n - 1) ** 2 + 4 * K)) / 2
        LHS = Si - K
        RHS = (
            GBSOption(Si, K, self._t, self._r, self._b, self._sigma).call()
            + (1 - _np.exp((self._b - self._r) * self._t) * self._CND(d1)) * Si / Q2
        )
        bi = (
            _np.exp((self._b - self._r) * self._t) * self._CND(d1) * (1 - 1 / Q2)
            + (
                1
                - _np.exp((self._b - self._r) * self._t)
                * self._CND(d1)
                / (self._sigma * _np.sqrt(self._t))
            )
            / Q2
        )
        E = 0.000001

        # Newton Raphson algorithm for finding critical price Si
        while abs(LHS - RHS) / K > E:
            Si = (K + RHS - bi * Si) / (1 - bi)
            d1 = (_np.log(Si / K) + (self._b + self._sigma ** 2 / 2) * self._t) / (
                self._sigma * _np.sqrt(self._t)
            )
            LHS = Si - K
            RHS = (
                GBSOption(Si, K, self._t, self._r, self._b, self._sigma).call()
                + (1 - _np.exp((self._b - self._r) * self._t) * self._CND(d1)) * Si / Q2
            )
            bi = (
                _np.exp((self._b - self._r) * self._t) * self._CND(d1) * (1 - 1 / Q2)
                + (
                    1
                    - _np.exp((self._b - self._r) * self._t)
                    * self._CND(d1)
                    / (self._sigma * _np.sqrt(self._t))
                )
                / Q2
            )

        # Return Value:
        return Si

