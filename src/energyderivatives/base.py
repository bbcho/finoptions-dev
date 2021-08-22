from abc import ABC, abstractmethod
import numpy as _np
from math import atan as _atan
import copy as _copy
import numdifftools as _nd
import sys as _sys
import warnings as _warnings
from scipy.optimize import root_scalar as _root_scalar
from scipy.optimize import root as _root
import scipy.optimize as _opt


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


class GreeksFDM:
    """
    Greek calculation class for composition of Option classes

    Parameters
    ----------
    opt : Option class
        Option class with Option values to calculate greeks from

    """

    def __init__(self, opt):

        if True:  # issubclass(opt, Option):
            self._opt = opt
        else:
            raise ValueError(
                f"Parameter opt is not of the Option class, it is of the {type(opt)} class"
            )

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
        fd = self._make_partial_der("S", call, self._opt, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        return fd(self._opt._S) * 1

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
        fd = self._make_partial_der("t", call, self._opt, n=1, step=1 / 252)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        return fd(self._opt._t) * -1

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
        fd = self._make_partial_der("sigma", True, self._opt, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        return fd(self._opt._sigma) * 1

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
        fd = self._make_partial_der("r", call, self._opt, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        return fd(self._opt._r) * 1

    def lamb(self, call: bool = True):
        """
        Method to return lambda greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        call : bool
            Returns lambda greek for call option if True, else returns lambda greek for put options. By default True.

        Returns
        -------
        float
        """
        if call == True:
            price = self._opt.call()
        else:
            price = self._opt.put()
        return self.delta(call=call) * self._opt._S / price

    def gamma(self):
        """
        Method to return gamma greek for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        # same for both call and put options
        fd = self._make_partial_der("S", True, self._opt, n=2)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        return fd(self._opt._S) * 1

    def greeks(self, call: bool = True):
        """
        Method to return greeks as a dictiontary for either call or put options using Finite Difference Methods.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
        }

        return gk


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

    @property
    @abstractmethod
    def __name__(self):
        pass

    @property
    @abstractmethod
    def __title__(self):
        pass

    def __init__(self, S: float, K: float, t: float, r: float, b: float, sigma: float):
        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma

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

    @abstractmethod
    def delta(self):
        pass

    @abstractmethod
    def theta(self):
        pass

    @abstractmethod
    def vega(self):
        pass

    @abstractmethod
    def lamb(self):
        pass

    @abstractmethod
    def gamma(self):
        pass

    @abstractmethod
    def greeks(self):
        pass

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
            out += f"  {p} = {self._check_string(params[p])}\n"

        try:
            # if self._sigma or its variations are not None add call and put prices
            if isinstance(self.call(), _np.ndarray):
                c = self._check_string(self.call().round(2))
                p = self._check_string(self.put().round(2))
                price = f"\nOption Price:\n\n  call: {c}\n  put: {p}"
            else:
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
            out = out + str(self._check_string(params[p])) + ", "

        # get rid of trailing comman and close pararenthesis
        out = out[:-2]
        out += ")"

        return out

    def _check_string(self, x):
        """
        helper function for summary method. Checks to see if the variable x is a numpy
        ndarray type and if it's length is greater than 6. If so, shortens the repsentation
        so that it fits.
        """
        if isinstance(x, _np.ndarray):
            if x.shape[0] > 6:
                return _np.array2string(x.round(2), threshold=6)

        return x

    def _check_array(self, *args):
        """
        helper function to return True if any args past are numpy ndarrays
        """
        for a in args:
            # allows size 1 arrays. somethings greeks return size 1 arrays
            if isinstance(a, _np.ndarray) and a.size > 1:
                return True

        return False

    def _max_array(self, *args):
        """
        helper function to get largest ndarray. Assumes at
        least one array.
        """
        maxArray = _np.array([0])
        for a in args:
            if isinstance(a, _np.ndarray):
                if maxArray.size < a.size:
                    maxArray = a

        return maxArray

    def _func(sigma, obj, call, price):
        """
        helper function to be used for volatility root finding.
        """
        temp = obj.copy()
        temp.set_param("sigma", sigma)
        if call == True:
            return price - temp.call()
        else:
            return price - temp.put()

    def _volatility(
        self,
        price: float,
        call: bool = True,
        tol=_sys.float_info.epsilon,
        maxiter=10000,
        verbose=False,
        _func=_func,
    ):
        """
        Compute the implied volatility of the GBSOption.

        Parameters
        ----------
        price : float
            Current price of the option
        call : bool
            Returns implied volatility for call option if True, else returns implied volatility for put options. By default True.
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
        if self._sigma is not None:
            _warnings.warn("sigma is not None but calculating implied volatility.")

        # check to see if arrays were past vs scalars.
        if self._check_array(price, *self.get_params().values()):
            # if arrays, use root function
            a = self._max_array(price, *self.get_params().values())
            if verbose == True:
                sol = _root(_func, args=(self, call, price), x0=_np.ones_like(a))
            else:
                sol = _root(_func, args=(self, call, price), x0=_np.ones_like(a)).x
        else:
            # if scalars use root_scalar function
            if verbose == True:
                sol = _root_scalar(
                    _func,
                    (self, call, price),
                    bracket=[-10, 10],
                    xtol=tol,
                    maxiter=maxiter,
                )
                # sol = _opt.brentq(_func, 1, 10000, xtol=tol, maxiter=maxiter)
            else:
                sol = _root_scalar(
                    _func,
                    (self, call, price),
                    bracket=[-10, 10],
                    xtol=tol,
                    maxiter=maxiter,
                ).root

        return sol
