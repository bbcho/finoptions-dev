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


class NormalDistFunctions:
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


class Display:
    """
    Class for printing summaries of Options classes

    Parameters
    ----------
    opt : Option class
        Option class with Option values to calculate greeks from
    call : bool
        True if to include call option price, False to exclude. By default True.
    put : bool
        True if to include put option price, False to exclude. By default True.
    """

    def __init__(self, opt, call=True, put=True):
        if isinstance(opt, Option):
            self._opt = opt
            self._call = call
            self._put = put
        else:
            raise ValueError("Parameter opt is not of the Option class")

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

    def summary(self, printer=True):
        """
        Print summary report of option

        Parameters
        ----------
        printer : bool
            True to print summary. False to return a string.
        """
        out = f"Title: {self._opt.__title__} Valuation\n\nParameters:\n\n"

        params = self._opt.get_params()

        for p in params:
            out += f"  {p} = {self._opt._check_string(params[p])}\n"

        try:
            # for printing arrays of calls and puts (converts to string first)
            if isinstance(self._opt.call(), _np.ndarray):
                price = f"\nOption Price:\n\n  "

                if self._call == True:
                    c = self._check_string(self._opt.call().round(2))
                    price += f"call: {c}\n  "
                if self._put == True:
                    p = self._check_string(self._opt.put().round(2))
                    price += f"put: {p}"
            else:
                price = f"\nOption Price:\n\n  "
                if self._call == True:
                    price += f"call: {round(self._opt.call(),6)}\n  "
                if self._put == True:
                    price += f"put: {round(self._opt.put(),6)}"
            out += price
        except:
            raise ValueError("call or put option methods failed")

        if printer == True:
            print(out)
        else:
            return out


class Option(ABC):
    """
    Base interface class for all options

    """

    @property
    @abstractmethod
    def __name__(self):
        # abstract attribute
        pass

    @property
    @abstractmethod
    def __title__(self):
        # abstract attribute
        pass

    # @abstractmethod
    # def __str__(self):
    #     pass

    # @abstractmethod
    # def __repr__(self):
    #     pass

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

    def get_params(self):
        return {
            "S": self._S,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b": self._b,
            "sigma": self._sigma,
        }

    def _check_sigma(self, func):
        if self._sigma is None:
            raise ValueError(f"sigma not defined. Required for {func}() method")

    def copy(self):
        return _copy.deepcopy(self)

    # @abstractmethod
    # def get_params(self):
    #     pass

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
    def summary(self):
        pass

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
    def rho(self):
        pass

    @abstractmethod
    def lamb(self):
        pass

    @abstractmethod
    def gamma(self):
        pass

    @abstractmethod
    def volatility(self):
        pass


class ImpliedVolatility:
    """
    Class used to calculate implied volatility for Option classes

    Parameters
    ----------
    opt : Option class
        Option class with Option values to calculate greeks from

    """

    def __init__(self, opt):
        if isinstance(opt, Option):
            self._opt = opt
        else:
            raise ValueError("Parameter opt is not of the Option class")

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

    def _check_array(self, *args):
        """
        helper function to return True if any args past are numpy ndarrays
        """
        for a in args:
            # allows size 1 arrays. somethings greeks return size 1 arrays
            if isinstance(a, _np.ndarray) and a.size > 1:
                return True

        return False

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
        if self._opt._sigma is not None:
            _warnings.warn("sigma is not None but calculating implied volatility.")

        # check to see if arrays were past vs scalars.
        if self._check_array(price, *self._opt.get_params().values()):
            # if arrays, use root function
            a = self._opt._max_array(price, *self._opt.get_params().values())
            if verbose == True:
                sol = _root(_func, args=(self._opt, call, price), x0=_np.ones_like(a))
            else:
                sol = _root(_func, args=(self._opt, call, price), x0=_np.ones_like(a)).x
        else:
            # if scalars use root_scalar function
            if verbose == True:
                sol = _root_scalar(
                    _func,
                    (self._opt, call, price),
                    bracket=[-10, 10],
                    xtol=tol,
                    maxiter=maxiter,
                )
                # sol = _opt.brentq(_func, 1, 10000, xtol=tol, maxiter=maxiter)
            else:
                sol = _root_scalar(
                    _func,
                    (self._opt, call, price),
                    bracket=[-10, 10],
                    xtol=tol,
                    maxiter=maxiter,
                ).root

        return sol


if __name__ == "__main__":

    class TestOpt(Option):
        __name__ = "test"
        __repr__ = "test"

    opt = TestOpt()
