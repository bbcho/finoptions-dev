from abc import ABC, abstractmethod
import numpy as _np
from math import atan as _atan
import copy as _copy
import numdifftools as _nd


def _make_partial_der(wrt, call, opt, **kwargs):
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
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma

    def simulate(self):
        print("sim run")

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

    def copy(self):

        return _copy.deepcopy(self)

    def _check_sigma(self, func):
        if self._sigma is None:
            raise ValueError(f"sigma not defined. Required for {func}() method")

    def put():
        pass

    def call():
        pass

    def greeks():
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
        self._check_sigma("delta")

        fd = _make_partial_der("S", call, self, n=1)

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
        self._check_sigma("theta")

        fd = _make_partial_der("t", call, self, n=1, step=1 / 252)

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
        self._check_sigma("vega")
        # same for both call and put options
        fd = _make_partial_der("sigma", True, self, n=1)
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
        self._check_sigma("rho")

        # This only works if the cost to carry b is zero....
        if self._b == 0:
            fd = _make_partial_der("r", call, self, n=1)
            return float(fd(self._r))
        else:
            raise ValueError(
                "rho calculation versus finite difference method currently only works if b = 0"
            )

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
        self._check_sigma("lamb")
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
        self._check_sigma("gamma")
        # same for both call and put options
        fd = _make_partial_der("S", True, self, n=2)
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
        self._check_sigma("c_of_c")
