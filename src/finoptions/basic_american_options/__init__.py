from ..base import GreeksFDM, Option as _Option
from ..vanillaoptions import GBSOption as _GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings
import numdifftools as _nd
from ..utils import docstring_from


class RollGeskeWhaleyOption(_Option):
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
    >>> import finoptions as fo
    >>> opt = fo.RollGeskeWhaleyOption(S=80, K=82, t=1/3, td=1/4, r=0.06, D=4, sigma=0.30)
    >>> opt.call()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "RollGeskeWhaleyOption"
    __title__ = "Roll-Geske-Whaley Calls on Dividend Paying Stocks"

    def __init__(
        self,
        S: float,
        K: float,
        t: float,
        td: float,
        r: float,
        D: float,
        sigma: float = None,
    ):
        if self._check_array(S, K, t, td, r, D, sigma) == True:
            raise TypeError("Arrays not supported as arguments for this option class")

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

        self._greeks = GreeksFDM(self)

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
            result = _GBSOption(
                Sx, self._K, self._t, self._r, b=self._r, sigma=self._sigma
            ).call()
            # print("\nWarning: Not optimal to exercise\n")
            return result

        ci = _GBSOption(
            self._S, self._K, self._t - self._td, self._r, b=self._r, sigma=self._sigma
        ).call()

        HighS = self._S

        while (ci - HighS - self._D + self._K > 0) & (HighS < self._big):
            HighS = HighS * 2
            ci = _GBSOption(
                HighS,
                self._K,
                self._t - self._td,
                self._r,
                b=self._r,
                sigma=self._sigma,
            ).call()

        if HighS > self._big:
            result = _GBSOption(
                Sx, self._K, self._t, self._r, b=self._r, sigma=self._sigma
            ).call()
            raise ValueError("HighS > big setting")

        LowS = 0
        I = HighS * 0.5
        ci = _GBSOption(
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
            ci = _GBSOption(
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
        out = f"Title: {self.__title__} Valuation\n\nParameters:\n\n"

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

    def delta(self):
        """
        Method to return delta greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.delta()

    def theta(self):
        """
        Method to return theta greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.theta()

    def vega(self):
        """
        Method to return vega greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.vega()

    def rho(self):
        """
        Method to return rho greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.rho()

    def lamb(self):
        """
        Method to return lamb greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.lamb()

    def gamma(self):
        """
        Method to return delta greek for a call options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.gamma()

    def greeks(self):
        """
        Method to return greeks as a dictiontary for a call option using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self._greeks.greeks()

    def volatility(
        self,
        price: float,
        tol=_sys.float_info.epsilon,
        maxiter=10000,
        verbose=False,
    ):
        """
        Compute the implied volatility of the RollGeskeWhaleyOption.

        Parameters
        ----------
        price : float
            Current price of the option
        tol : float
            max tolerance to fit the price to. By default system tolerance.
        maxiter : int
            number of iterations to run to fit price.
        verbose : bool
            True to return full optimization details from root finder function. False to just return the implied volatility numbers.

        Returns
        -------
        float

        Example
        -------
        """
        sol = self._volatility(price, True, tol, maxiter, verbose)

        return sol


class BAWAmericanApproxOption(_Option):
    """
    Barone-Adesi and Whaley Approximation. Calculates the option price of an
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

    Returns
    -------
    Option object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.BAWAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "BAWAmericanApproxOption"
    __title__ = "Barone-Adesi and Whaley Approximation"

    def __init__(
        self, S: float, K: float, t: float, r: float, b: float, sigma: float = None
    ):
        # only being used for check_array. Remove __init__ once arrays work.
        if self._check_array(S, K, t, r, b, sigma) == True:
            raise TypeError("Arrays not supported as arguments for this option class")

        super().__init__(S, K, t, r, b, sigma)

        self._greeks = GreeksFDM(self)

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
        X = 2 * self._r / (self._sigma ** 2 * (1 - _np.exp(-self._r * self._t)))
        d1 = (_np.log(Si / self._K) + (self._b + self._sigma ** 2 / 2) * self._t) / (
            self._sigma * _np.sqrt(self._t)
        )
        Q2 = (-(n - 1) + _np.sqrt((n - 1) ** 2 + 4 * X)) / 2
        LHS = Si - self._K
        RHS = (
            _GBSOption(Si, self._K, self._t, self._r, self._b, self._sigma).call()
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
        while abs(LHS - RHS) / self._K > E:
            Si = (self._K + RHS - bi * Si) / (1 - bi)
            d1 = (
                _np.log(Si / self._K) + (self._b + self._sigma ** 2 / 2) * self._t
            ) / (self._sigma * _np.sqrt(self._t))
            LHS = Si - self._K
            RHS = (
                _GBSOption(Si, self._K, self._t, self._r, self._b, self._sigma).call()
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

    def _bawKp(self):
        # Newton Raphson algorithm to solve for the critical commodity
        # price for a Put.
        # Calculation of seed value, Si
        n = 2 * self._b / self._sigma ** 2
        m = 2 * self._r / self._sigma ** 2
        q1u = (-(n - 1) - _np.sqrt((n - 1) ** 2 + 4 * m)) / 2
        Su = self._K / (1 - 1 / q1u)
        h1 = (
            (self._b * self._t - 2 * self._sigma * _np.sqrt(self._t))
            * self._K
            / (self._K - Su)
        )
        Si = Su + (self._K - Su) * _np.exp(h1)
        X = 2 * self._r / (self._sigma ** 2 * (1 - _np.exp(-self._r * self._t)))
        d1 = (_np.log(Si / self._K) + (self._b + self._sigma ** 2 / 2) * self._t) / (
            self._sigma * _np.sqrt(self._t)
        )
        Q1 = (-(n - 1) - _np.sqrt((n - 1) ** 2 + 4 * X)) / 2
        LHS = self._K - Si
        RHS = (
            _GBSOption(Si, self._K, self._t, self._r, self._b, self._sigma).put()
            - (1 - _np.exp((self._b - self._r) * self._t) * self._CND(-d1)) * Si / Q1
        )
        bi = (
            -_np.exp((self._b - self._r) * self._t) * self._CND(-d1) * (1 - 1 / Q1)
            - (
                1
                + _np.exp((self._b - self._r) * self._t)
                * self._CND(-d1)
                / (self._sigma * _np.sqrt(self._t))
            )
            / Q1
        )
        E = 0.000001

        # Newton Raphson algorithm for finding critical price Si
        while abs(LHS - RHS) / self._K > E:
            Si = (self._K - RHS + bi * Si) / (1 + bi)
            d1 = (
                _np.log(Si / self._K) + (self._b + self._sigma ** 2 / 2) * self._t
            ) / (self._sigma * _np.sqrt(self._t))
            LHS = self._K - Si
            RHS = (
                _GBSOption(Si, self._K, self._t, self._r, self._b, self._sigma).put()
                - (1 - _np.exp((self._b - self._r) * self._t) * self._CND(-d1))
                * Si
                / Q1
            )
            bi = (
                -_np.exp((self._b - self._r) * self._t) * self._CND(-d1) * (1 - 1 / Q1)
                - (
                    1
                    + _np.exp((self._b - self._r) * self._t)
                    * self._CND(-d1)
                    / (self._sigma * _np.sqrt(self._t))
                )
                / Q1
            )

        # Return Value:
        return Si

    def call(self):
        """
        Returns the calculated price of a call option according to the
        Barone-Adesi and Whaley Approximation option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.BAWAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        if self._b >= self._r:
            result = _GBSOption(
                self._S, self._K, self._t, self._r, self._b, self._sigma
            ).call()
        else:
            Sk = self._bawKc()
            n = 2 * self._b / self._sigma ** 2
            X = 2 * self._r / (self._sigma ** 2 * (1 - _np.exp(-self._r * self._t)))
            d1 = (
                _np.log(Sk / self._K) + (self._b + self._sigma ** 2 / 2) * self._t
            ) / (self._sigma * _np.sqrt(self._t))
            Q2 = (-(n - 1) + _np.sqrt((n - 1) ** 2 + 4 * X)) / 2
            a2 = (Sk / Q2) * (
                1 - _np.exp((self._b - self._r) * self._t) * self._CND(d1)
            )
            if self._S < Sk:
                result = (
                    _GBSOption(
                        self._S, self._K, self._t, self._r, self._b, self._sigma
                    ).call()
                    + a2 * (self._S / Sk) ** Q2
                )
            else:
                result = self._S - self._K

        # Return Value:
        return result

    def put(self):
        """
        Returns the calculated price of a call option according to the
        Barone-Adesi and Whaley Approximation option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.BAWAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        Sk = self._bawKp()
        n = 2 * self._b / self._sigma ** 2
        X = 2 * self._r / (self._sigma ** 2 * (1 - _np.exp(-self._r * self._t)))
        d1 = (_np.log(Sk / self._K) + (self._b + self._sigma ** 2 / 2) * self._t) / (
            self._sigma * _np.sqrt(self._t)
        )
        Q1 = (-(n - 1) - _np.sqrt((n - 1) ** 2 + 4 * X)) / 2
        a1 = -(Sk / Q1) * (1 - _np.exp((self._b - self._r) * self._t) * self._CND(-d1))
        if self._S > Sk:
            result = (
                _GBSOption(
                    self._S, self._K, self._t, self._r, self._b, self._sigma
                ).put()
                + a1 * (self._S / Sk) ** Q1
            )
        else:
            result = self._K - self._S

        return result

    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        return self._greeks.delta(call=call)

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.rho)
    def rho(self, call: bool = True):
        return self._greeks.rho(call=call)

    @docstring_from(GreeksFDM.lamb)
    def lamb(self, call: bool = True):
        return self._greeks.lamb(call=call)

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        return self._greeks.gamma()

    @docstring_from(GreeksFDM.greeks)
    def greeks(self, call: bool = True):
        # need to override so that the overridden vega is used
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
        }

        return gk

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        # same for both call and put options
        # over-rode parent class vega as it is unstable for larger step sizes of sigma.
        fd = self._greeks._make_partial_der(
            "sigma", True, self, n=1, step=self._sigma / 10
        )
        return float(fd(self._sigma))


class BSAmericanApproxOption(_Option):
    """
    BSAmericanApproxOption evaluates American calls or puts on stocks, futures, and currencies
    due to the approximation method of Bjerksund and Stensland (1993)

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

    Returns
    -------
    Option object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.basic_american_options.BSAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    [2] Bjerksund P., Stensland G. (1993);Closed Form Approximation of American Options, Scandinavian Journal of Management 9, 87–99
    """

    __name__ = "BSAmericanApproxOption"
    __title__ = "The Bjerksund and Stensland (1993) American Approximation Option"

    def __init__(
        self, S: float, K: float, t: float, r: float, b: float, sigma: float = None
    ):
        # only being used for check_array. Remove __init__ once arrays work.
        if self._check_array(S, K, t, r, b, sigma) == True:
            raise TypeError("Arrays not supported as arguments for this option class")

        super().__init__(S, K, t, r, b, sigma)
        self._greeks = GreeksFDM(self)

        # override make_partial_der because call() and put() return dicts
        self._greeks._make_partial_der = self._make_partial_der

    def _make_partial_der(self, wrt, call, opt, **kwargs):
        """
        Create monad from Option methods call and put for use
        in calculating the partial derivatives or greeks with
        respect to wrt.
        """
        # need to override since call/put method return dicts.
        def _func(x):
            tmp = opt.copy()
            tmp.set_param(wrt, x)
            if call == True:
                return tmp.call()["OptionPrice"]
            else:
                return tmp.put()["OptionPrice"]

        fd = _nd.Derivative(_func, **kwargs)

        return fd

    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        return self._greeks.delta(call=call)

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.rho)
    def rho(self, call: bool = True):
        return self._greeks.rho(call=call)

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        return self._greeks.gamma()

    @docstring_from(GreeksFDM.greeks)
    def greeks(self, call: bool = True):
        # need to override so that the overridden lamb is used
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
        }

        return gk

    @docstring_from(GreeksFDM.lamb)
    def lamb(self, call: bool = True):
        if call == True:
            price = self.call()["OptionPrice"]
        else:
            price = self.put()["OptionPrice"]
        return self.delta(call=call) * self._S / price

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        # same for both call and put options, overriden as it needs smaller step sizes
        fd = self._make_partial_der("sigma", True, self, n=1, step=self._sigma / 10)
        return fd(self._sigma) * 1

    def call(self):
        """
        Returns the calculated price of a call option according to the
        The Bjerksund and Stensland (1993) American Approximation option price model.

        Returns
        -------
        dict(str:float) with OptionPrice and TriggerPrice

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.BSAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        return self._BSAmericanCallApprox(
            self._S, self._K, self._t, self._r, self._b, self._sigma
        )

    def put(self):
        """
        Returns the calculated price of a put option according to the
        The Bjerksund and Stensland (1993) American Approximation option price model.

        Returns
        -------
        dict(str:float) with OptionPrice and TriggerPrice

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.BSAmericanApproxOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        # Use the Bjerksund and Stensland put-call transformation
        return self._BSAmericanCallApprox(
            self._K, self._S, self._t, self._r - self._b, -self._b, self._sigma
        )

    def _BSAmericanCallApprox(self, S, X, Time, r, b, sigma):
        # Call Approximation:

        if b >= r:
            # Never optimal to exersice before maturity
            result = dict(
                OptionPrice=_GBSOption(S, X, Time, r, b, sigma).call(),
                TriggerPrice=_np.nan,
            )
        else:
            Beta = (1 / 2 - b / sigma ** 2) + _np.sqrt(
                (b / sigma ** 2 - 1 / 2) ** 2 + 2 * r / sigma ** 2
            )
            BInfinity = Beta / (Beta - 1) * X
            B0 = max(X, r / (r - b) * X)
            ht = -(b * Time + 2 * sigma * _np.sqrt(Time)) * B0 / (BInfinity - B0)
            # Trigger Price I:
            I = B0 + (BInfinity - B0) * (1 - _np.exp(ht))
            alpha = (I - X) * I ** (-Beta)
            if S >= I:
                result = dict(OptionPrice=S - X, TriggerPrice=I)
            else:
                result = dict(
                    OptionPrice=alpha * S ** Beta
                    - alpha * self._bsPhi(S, Time, Beta, I, I, r, b, sigma)
                    + self._bsPhi(S, Time, 1, I, I, r, b, sigma)
                    - self._bsPhi(S, Time, 1, X, I, r, b, sigma)
                    - X * self._bsPhi(S, Time, 0, I, I, r, b, sigma)
                    + X * self._bsPhi(S, Time, 0, X, I, r, b, sigma),
                    TriggerPrice=I,
                )

        return result

    def _bsPhi(self, S, Time, gamma, H, I, r, b, sigma):

        # Utility function phi:

        lamb = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2) * Time
        d = -(_np.log(S / H) + (b + (gamma - 0.5) * sigma ** 2) * Time) / (
            sigma * _np.sqrt(Time)
        )
        kappa = 2 * b / (sigma ** 2) + (2 * gamma - 1)
        result = (
            _np.exp(lamb)
            * S ** gamma
            * (
                self._CND(d)
                - (I / S) ** kappa
                * self._CND(d - 2 * _np.log(I / S) / (sigma * _np.sqrt(Time)))
            )
        )

        return result

    def summary(self, printer=True):
        """
        Print summary report of option

        Parameters
        ----------
        printer : bool
            True to print summary. False to return a string.
        """
        out = f"Title: {self.__title__} Valuation\n\nParameters:\n\n"

        params = self.get_params()

        for p in params:
            out += f"  {p} = {params[p]}\n"

        try:
            # if self._sigma or its variations are not None add call and put prices
            price = f"\nOption Price:\n\n  call: {round(self.call()['OptionPrice'],6)}, trigger: {round(self.call()['TriggerPrice'],6)}\n  put: {round(self.put()['OptionPrice'],6)}, trigger: {round(self.put()['TriggerPrice'],6)}"
            out += price
        except:
            pass

        if printer == True:
            print(out)
        else:
            return out
