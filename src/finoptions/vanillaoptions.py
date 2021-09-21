from .base import GreeksFDM, Option
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
from scipy.optimize import root as _root
import sys as _sys
import warnings as _warnings
from .utils import docstring_from


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
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.

    Notes
    -----
    Including a cost of carry term b, the model can
    used to price European Options on:

    b = r       stocks (Black and Scholes’ stock option model)
    b = r - q   stocks and stock indexes paying a continuous dividend yield q (Merton’s stock option model)
    b = 0       futures (Black’s futures option model)
    b = r - rf  currency options with foreign interst rate rf (Garman and Kohlhagen’s currency option model)

    Returns
    -------
    GBSOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "GBSOption"
    __title__ = "Black Scholes Option"

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

        self._greeks = GreeksFDM(self)

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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.call()

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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
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

    def delta(self, call: bool = True, method: str = "analytic"):
        """
        Method to return delta greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns delta greek for call option if True, else returns delta greek for put options. By default True.
        method : str
            'analytic' for analytic solution to delta. 'fdm' for Finite Difference Method. By default 'analytic'

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.delta(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("delta")

        if method == "analytic":
            if call == True:
                return _np.exp((self._b - self._r) * self._t) * self._CND(self._d1)
            else:
                return _np.exp((self._b - self._r) * self._t) * (
                    self._CND(self._d1) - 1
                )
        elif method == "fdm":
            return self._greeks.delta(call=call)

    def theta(self, call: bool = True, method: str = "analytic"):
        """
        Method to return theta greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns theta greek for call option if True, else returns theta greek for put options. By default True.
        method : str
            'analytic' for analytic solution to delta. 'fdm' for Finite Difference Method. By default 'analytic'

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.theta(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("theta")
        if method == "analytic":
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
                    - self._r
                    * self._K
                    * _np.exp(-self._r * self._t)
                    * self._CND(+self._d2)
                )
            else:
                return (
                    Theta1
                    + (self._b - self._r)
                    * self._S
                    * _np.exp((self._b - self._r) * self._t)
                    * self._CND(-self._d1)
                    + self._r
                    * self._K
                    * _np.exp(-self._r * self._t)
                    * self._CND(-self._d2)
                )
        elif method == "fdm":
            return self._greeks.theta(call=call)

    def vega(self, method: str = "analytic"):
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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.vega()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("vega")
        # for both call and put options
        if method == "analytic":
            return (
                self._S
                * _np.exp((self._b - self._r) * self._t)
                * self._NDF(self._d1)
                * _np.sqrt(self._t)
            )
        elif method == "fdm":
            return self._greeks.vega()

    def rho(self, call: bool = True, method: str = "analytic"):
        """
        Method to return rho greek for either call or put options.

        Parameters
        ----------
        call : bool
            Returns rho greek for call option if True, else returns rho greek for put options. By default True.
        method : str
            'analytic' for analytic solution to delta. 'fdm' for Finite Difference Method. By default 'analytic'

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.rho(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("rho")
        if method == "analytic":
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
        elif method == "fdm":
            # This only works if the cost to carry b is zero....
            if self._b == 0:
                return self._greeks.rho(call=call)
            else:
                raise ValueError(
                    "rho calculation versus finite difference method currently only works if b = 0"
                )

    def lamb(self, call: bool = True, method: str = "analytic"):
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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.lambda(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("lamb")
        if method == "analytic":
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
        elif method == "fdm":
            return self._greeks.lamb(call=call)

    def gamma(self, method: str = "analytic"):
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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.gamma(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("gamma")
        # for both call and put options
        if method == "analytic":
            return (
                _np.exp((self._b - self._r) * self._t)
                * self._NDF(self._d1)
                / (self._S * self._sigma * _np.sqrt(self._t))
            )
        elif method == "fdm":
            return self._greeks.gamma()

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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
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
        >>> import finoptions as fo
        >>> opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)
        >>> opt.greeks(call=True)

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        self._check_sigma("greeks")

        # need to define entire greek dict as the base GreekFDM class
        # generates the dict using the FDM method
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "rho": self.rho(call),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
            "CofC": self.c_of_c(call=call),
        }

        return gk

    def volatility(
        self,
        price: float,
        call: bool = True,
        tol=_sys.float_info.epsilon,
        maxiter=10000,
        verbose=False,
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

        sol = self._volatility(price, call, tol, maxiter, verbose)

        return sol


class BlackScholesOption(GBSOption):
    __name__ = "BlackScholesOption"
    __title__ = "Black Scholes Option"


class Black76Option(GBSOption):
    """
    The Black76Option pricing formula is applicable for valuing European call
    and European put options on commodity futures. The exact nature of the
    underlying commodity varies and may be anything from a precious metal such
    as gold or silver to agricultural products.

    Parameters
    ----------
    FT : float
        Futures price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.
    """

    __name__ = "Black76Option"
    __title__ = "Black 1977 Option"

    def __init__(self, FT, K, t, r, sigma=None):
        super().__init__(FT, K, t, r, b=0, sigma=sigma)
        # self._FT = self._S

    def get_params(self):
        return {
            "FT": self._S,
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

        del tmp["b"]  # delete b prior to re-init

        self.__init__(**tmp)


class MiltersenSchwartzOption(Option):
    """
    The MiltersenSchwartzOption class allows for pricing options on commodity futures. The model is a three
    factor model with stochastic futures prices, term structures of convenience yields and interest rates.
    The model is based on lognormal distributed commodity prices and normal distributed continuously compounded
    forward interest rates and futures convenience yields.

    The Miltersen Schwartz Option model is a three factor model with stochastic futures prices,term structures
    and convenience yields, and interest rates. The model is based on log-normal distributed commodity prices
    and normal distributed continuously compounded forward interest rates and future convenience yields.

    Parameters
    ----------
    Pt : float
        The zero coupon bond that expires on the option maturity.
    FT : float
        The futures price.
    K : float
        The strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    T : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    sigmaS : float
        The annualized volatility of the spot commodity price (S), e.g.  0.25 means 25%
    sigmaE : float
        The annualized volatility of the the future convenience yield (E), e.g. 0.25 means 25%
    sigmaF : float
        The annualized volatility of the the forward interest rate (F), e.g.  0.25 means 25%
    rhoSE : float
        The correlations between the spot commodity price and the future convenience yield (SE)
    rhoSF : float
        The correlations between the spot commodity price and the forward interest rate (SF)
    rhoEF : float
        The correlations between the forward interest rate and the future convenience yield (EF)
    KappaE : float
        The speed of mean reversion of the forward interest rate (E)
    KappaF : float
        The speed of mean reversion of the convenience yield (F)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    [2] Miltersen  K.,  Schwartz  E.S.  (1998);Pricing  of  Options  on  Commodity  Futures  with  Stochastic Term Structuures of Convenience Yields and Interest Rates, Journal of Financial and Quantitative Analysis 33, 33–59
    """

    __name__ = "MiltersenSchwartzOption"
    __title__ = "Miltersen Schwartz Option"

    def __init__(
        self,
        Pt: float,
        FT: float,
        K: float,
        t: float,
        T: float,
        sigmaS: float,
        sigmaE: float,
        sigmaF: float,
        rhoSE: float,
        rhoSF: float,
        rhoEF: float,
        KappaE: float,
        KappaF: float,
    ):
        self._Pt = Pt
        self._FT = FT
        self._K = K
        self._t = t
        self._T = T
        self._sigmaS = sigmaS
        self._sigmaE = sigmaE
        self._sigmaF = sigmaF
        self._rhoSE = rhoSE
        self._rhoSF = rhoSF
        self._rhoEF = rhoEF
        self._KappaE = KappaE
        self._KappaF = KappaF

        # fmt: off
        self._vz = (
            self._sigmaS**2*self._t+2*self._sigmaS*(self._sigmaF*self._rhoSF*1/self._KappaF*(self._t-1/self._KappaF*
            _np.exp(-self._KappaF*self._T)*(_np.exp(self._KappaF*self._t)-1))-self._sigmaE*self._rhoSE*1/self._KappaE*
            (self._t-1/self._KappaE*_np.exp(-self._KappaE*self._T)*(_np.exp(self._KappaE*self._t)-1)))+self._sigmaE**2*
            1/self._KappaE**2*(self._t+1/(2*self._KappaE)*_np.exp(-2*self._KappaE*self._T)*(_np.exp(2*self._KappaE*self._t)-
            1)-2*1/self._KappaE*_np.exp(-self._KappaE*self._T)*(_np.exp(self._KappaE*self._t)-1))+self._sigmaF**2*
            1/self._KappaF**2*(self._t+1/(2*self._KappaF)*_np.exp(-2*self._KappaF*self._T)*(_np.exp(2*self._KappaF*self._t)-
            1)-2*1/self._KappaF*_np.exp(-self._KappaF*self._T)*(_np.exp(self._KappaF*self._t)-1))-2*self._sigmaE*
            self._sigmaF*self._rhoEF*1/self._KappaE*1/self._KappaF*(self._t-1/self._KappaE*_np.exp(-self._KappaE*self._T)*
            (_np.exp(self._KappaE*self._t)-1)-1/self._KappaF*_np.exp(-self._KappaF*self._T)*(_np.exp(self._KappaF*self._t)-
            1)+1/(self._KappaE+self._KappaF)*_np.exp(-(self._KappaE+self._KappaF)*self._T)*(_np.exp((self._KappaE+self._KappaF)*
            self._t)-1))
        )
        self._vxz = (
            self._sigmaF*1/self._KappaF*(self._sigmaS*self._rhoSF*(self._t-1/self._KappaF*(1-_np.exp(-self._KappaF*
            self._t)))+self._sigmaF*1/self._KappaF*(self._t-1/self._KappaF*_np.exp(-self._KappaF*self._T)*(_np.exp(self._KappaF*
            self._t)-1)-1/self._KappaF*(1-_np.exp(-self._KappaF*self._t))+1/(2*self._KappaF)*_np.exp(-self._KappaF*
            self._T)*(_np.exp(self._KappaF*self._t)-_np.exp(-self._KappaF*self._t)))-self._sigmaE*self._rhoEF*1/self._KappaE*
            (self._t-1/self._KappaE*_np.exp(-self._KappaE*self._T)*(_np.exp(self._KappaE*self._t)-1)-1/self._KappaF*(1-
            _np.exp(-self._KappaF*self._t))+1/(self._KappaE+self._KappaF)*_np.exp(-self._KappaE*self._T)*
            (_np.exp(self._KappaE*self._t)-_np.exp(-self._KappaF*self._t))))
        )
        
        self._vz = _np.sqrt(self._vz)

        self._d1 = (_np.log(self._FT/self._K)-self._vxz+self._vz**2/2)/self._vz
        self._d2 = (_np.log(self._FT/self._K)-self._vxz-self._vz**2/2)/self._vz

        # fmt: on

        self._greeks = GreeksFDM(self)

    def call(self):
        """
        Returns the calculated price of a call option according to the
        Miltersen Schwartz Option option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> import numpy as np
        >>> opt = fo.MiltersenSchwartzOption(Pt=np.exp(-0.05/4), FT=95, K=80, t=1/4, T=1/2, sigmaS=0.2660,
                    sigmaE=0.2490, sigmaF=0.0096, rhoSE=0.805, rhoSF=0.0805, rhoEF=0.1243, KappaE=1.045, KappaF=0.200)
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        # fmt: off
        result = self._Pt*(self._FT*_np.exp(-self._vxz)*self._CND(self._d1)-self._K*self._CND(self._d2))
        # fmt: on

        return result

    def put(self):
        """
        Returns the calculated price of a call option according to the
        Miltersen Schwartz Option option price model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> import numpy as np
        >>> opt = fo.MiltersenSchwartzOption(Pt=np.exp(-0.05/4), FT=95, K=80, t=1/4, T=1/2, sigmaS=0.2660,
                    sigmaE=0.2490, sigmaF=0.0096, rhoSE=0.805, rhoSF=0.0805, rhoEF=0.1243, KappaE=1.045, KappaF=0.200)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        # fmt: off
        result = self._Pt*(self._K*self._CND(-self._d2)-self._FT*_np.exp(-self._vxz)*self._CND(-self._d1))
        # fmt: on

        return result

    def get_params(self):
        return {
            "Pt": self._Pt,
            "FT": self._FT,
            "K": self._K,
            "t": self._t,
            "T": self._T,
            "sigmaS": self._sigmaS,
            "sigmaE": self._sigmaE,
            "sigmaF": self._sigmaF,
            "rhoSE": self._rhoSE,
            "rhoSF": self._rhoSF,
            "rhoEF": self._rhoEF,
            "KappaE": self._KappaE,
            "KappaF": self._KappaF,
        }

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
        fd = self._greeks._make_partial_der("FT", call, self, n=1)

        return fd(self._FT) * 1

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

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
        fdS = self._greeks._make_partial_der("sigmaS", True, self, n=1)
        fdE = self._greeks._make_partial_der("sigmaE", True, self, n=1)
        fdF = self._greeks._make_partial_der("sigmaF", True, self, n=1)
        result = _np.array(
            (
                fdS(self._sigmaS) * 1,
                fdE(self._sigmaE) * 1,
                fdF(self._sigmaF) * 1,
            )
        )

        # if returning a multi-dim array because arrays were passed as inputs, transpose prior
        # to returing results so that each element of the array input corresponds to a row
        # in the result
        if self._check_array(fdS(self._sigmaS) * 1) == True:
            result = result.T

        return result

    def greeks(self, call: bool = True):
        gk = {
            "delta": self.delta(call),
            "theta": self.theta(call),
            "vega": self.vega(),
            "lambda": self.lamb(call),
            "gamma": self.gamma(),
            # "CofC": self.c_of_c(call),
        }

        return gk

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
        return self.delta(call=call) * self._FT / price

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
        fd = self._greeks._make_partial_der("FT", True, self, n=2)
        return fd(self._FT) * 1
