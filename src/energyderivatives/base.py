from abc import ABC, abstractmethod
import numpy as _np
from math import atan as _atan
import copy as _copy
import numdifftools as _nd


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

    def _CBND(self, x1, x2, rho):
        """
        Calculate the cumulative bivariate normal distribution function.

        Haug E.G., The Complete Guide to Option Pricing Formulas

        Compute:
        Take care for the limit rho = +/- 1
        """

        a = x1
        b = x2
        if abs(rho) == 1:
            rho = rho - (1e-12) * _np.sign(rho)
        # cat("\n a - b - rho :"); print(c(a,b,rho))
        X = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
        y = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
        a1 = a / _np.sqrt(2 * (1 - rho ** 2))
        b1 = b / _np.sqrt(2 * (1 - rho ** 2))
        if (a <= 0) & (b <= 0) & (rho <= 0):
            Sum1 = 0
            for I in range(0, 5):
                for j in range(0, 5):
                    Sum1 = Sum1 + X[I] * X[j] * _np.exp(
                        a1 * (2 * y[I] - a1)
                        + b1 * (2 * y[j] - b1)
                        + 2 * rho * (y[I] - a1) * (y[j] - b1)
                    )
            result = _np.sqrt(1 - rho ** 2) / _np.pi * Sum1
            return result

        if (a <= 0) & (b >= 0) & (rho >= 0):
            result = self._CND(a) - self._CBND(a, -b, -rho)
            return result

        if (a >= 0) & (b <= 0) & (rho >= 0):
            result = self._CND(b) - self._CBND(-a, b, -rho)
            return result

        if (a >= 0) & (b >= 0) & (rho <= 0):
            result = self._CND(a) + self._CND(b) - 1 + self._CBND(-a, -b, rho)
            return result

        if (a * b * rho) >= 0:
            rho1 = (
                (rho * a - b)
                * _np.sign(a)
                / _np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
            )
            rho2 = (
                (rho * b - a)
                * _np.sign(b)
                / _np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
            )
            delta = (1 - _np.sign(a) * _np.sign(b)) / 4
            result = self._CBND(a, 0, rho1) + self._CBND(b, 0, rho2) - delta
            return result


class Derivative(_Base):
    pass

    @abstractmethod
    def get_params(self):
        pass

    def copy(self):
        return _copy.deepcopy(self)


class Option(Derivative):
    """
    Base class for options

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
    """

    __name__ = "Option"
    __title__ = "Base Option Class"

    def __init__(self, S: float, K: float, t: float, r: float, b: float, sigma: float):
        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma

    def _make_partial_der(self, wrt, call, opt, **kwargs):
        """
        Create monad from Option methods call and put for use
        in calculating the partial derivatives or greeks with 
        respect to wrt.
        """

        def _func(x):
            tmp = opt.copy()
            tmp.set_param(wrt, x)
            if call == True:
                return tmp.call()
            else:
                return tmp.put()

        fd = _nd.Derivative(_func, **kwargs)

        return fd

    def get_params(self):
        return {
            "S": self._S,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b": self._b,
            "sigma": self._sigma,
        }

    def set_param(self, x: str, value: float):
        """
        Method to change a parameter once the class hass been initiatized.

        Parameters
        ----------
        x : str
            Name of parameter to change
        value : float
            New value to give to parameter

        Returns
        -------
        None
        """

        tmp = self.get_params()
        tmp[x] = value

        self.__init__(**tmp)

    def put():
        pass

    def call():
        pass

    def delta(self, call: bool = True):
        """
        Method to return delta greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        call : bool
            Returns delta greek for call option if True, else returns delta greek for put options. By default True.

        Returns
        -------
        float
        """
        fd = self._make_partial_der("S", call, self, n=1)

        return float(fd(self._S))

    def theta(self, call: bool = True):
        """
        Method to return theta greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        call : bool
            Returns theta greek for call option if True, else returns theta greek for put options. By default True.

        Returns
        -------
        float
        """
        fd = self._make_partial_der("t", call, self, n=1, step=1 / 252)

        return float(fd(self._t)) * -1

    def vega(self):
        """
        Method to return vega greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        # same for both call and put options
        fd = self._make_partial_der("sigma", True, self, n=1)
        return float(fd(self._sigma))

    def rho(self, call: bool = True):
        """
        Method to return rho greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        call : bool
            Returns rho greek for call option if True, else returns rho greek for put options. By default True.

        Returns
        -------
        float
        """
        # This only works if the cost to carry b is zero....
        fd = self._make_partial_der("r", call, self, n=1)
        return float(fd(self._r))

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
        """
        if call == True:
            price = self.call()
        else:
            price = self.put()
        return self.delta(call=call) * self._S / price

    def gamma(self):
        """
        Method to return gamma greek for either call or put options.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        # same for both call and put options
        fd = self._make_partial_der("S", True, self, n=2)
        return float(fd(self._S))

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
        """
        return None

    def greeks(self, call: bool = True):
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
            price = f"\nOption Price:\n\n  call: {round(self.call(),6)}\n  put: {round(self.put(),6)}"
            out += price
        except:
            pass

        if printer == True:
            print(out)
        else:
            return out

    def __str__(self):
        return self.summary(printer=False)

    def __repr__(self):
        out = f"{self.__name__}("
        params = self.get_params()

        for p in params:
            out = out + str(params[p]) + ", "

        # get rid of trailing comman and close pararenthesis
        out = out[:-2]
        out += ")"

        return out

