import numpy as np

from ..base import GreeksFDM, Option as _Option
from ..utils import docstring_from

class EnergySwaption(_Option):
    """
    European options on energy swaps, also called energy swaptions, are
    options that at maturity give a delivery of an energy swap at the strike
    price (but not necessarilrt physical delivery of any energy). 

    Parameters
    ----------
    F : float
        forward price of the underlying
    K : float
        strike price
    sigma : float
        Annualized volatility
    T : float
        time-to-maturity (from t=0)
    j : float
        Number of compoundings per year (number of settlements in a 
        one year forward contract)
    n : int
        number of time steps
    rj : float
        jump size mean
    Tb : float
        time-to-maturity (for t=0)
    rb : float
        risk-free rate for the borrowing
    re : float
        risk-free rate for the energy
    rp : float
        risk-free rate for the power
    call : bool
        True for call option, False for put option
    """
    
    __name__ = "EnergySwaption"
    __title__ = "Energy Swaption"

    def __init__(self, F, K, sigma, T, j, n, rj, Tb, rb):
        # self._F = S0
        self._K = K
        self._sigma = sigma
        self._T = T
        self._F = F
        self._j = j
        self._n = n
        self._rj = rj
        self._Tb = Tb
        self._rb = rb
        # self._re = re
        # self._rp = rp

        self._d1 = ( np.log(F/K) + sigma**2/2 * T ) / ( sigma * np.sqrt(T) )
        self._d2 = self._d1 - sigma * np.sqrt(T)

        self._greeks = GreeksFDM(self)

    def call(self):
        """
        Returns the calculated price of a call option.
        
        Returns
        -------
        float
            price of the call option

        Examples
        --------

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas, 2nd ed.
        """

        result = (1 - 1/(1 + self._rj/self._j)**self._n)/self._rj * (self._j/self._n)
        result *= np.exp(-self._rj * self._Tb)
        result *= (self._F * self._CND(self._d1) - self._K * self._CND(self._d2))

        return result

    def put(self):
        """
        Returns the calculated price of a put option.

        Returns
        -------
        float
            price of the put options

        Examples
        --------

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas, 2nd ed.
        """

        result = (1 - 1/(1 + self._rj/self._j)**self._n)/self._rj * (self._j/self._n)
        result *= np.exp(-self._rj * self._Tb)
        result *= (self._K * self._CND(-self._d2) - self._F * self._CND(-self._d1))

        return result
    
    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        
        fd = self._greeks._make_partial_der("F", call, self, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        out = fd(self._F) * 1

        return out

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        # same for both call and put options
        fd = self._greeks._make_partial_der("sigma", True, self, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        out = fd(self._sigma) * 1,

        return out

    @docstring_from(GreeksFDM.rho)
    def rho(self, call: bool = True):
        return self._greeks.rho(call=call)

    @docstring_from(GreeksFDM.lamb)
    def lamb(self, call: bool = True):
        if call == True:
            price = self.call()
        else:
            price = self.put()

        out = self.delta(call=call) * self._F / price,
        
        return out

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        # same for both call and put options
        fd = self._greeks._make_partial_der("F", True, self, n=2)
        
        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        out = fd(self._F) * 1,
        
        return out

    @docstring_from(GreeksFDM.greeks)
    def greeks(self, call: bool = True):
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
        }

        return gk

    def get_params(self):
        return {
            "F": self._F,
            "K": self._K,
            "sigma": self._sigma,
            "T": self._T,
            "j": self._j,
            "n": self._n,
            "rj": self._rj,
            "Tb": self._Tb,
            "rb": self._rb,
        }

        
