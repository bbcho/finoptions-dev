import numpy as _np
from ..base import GreeksFDM, Option as _Option
from ..utils import docstring_from


class BionomialSpreadAllTypes(_Option):
    """
    Rubinstein (1994) published a method to construct a 3-dimensional binomial 
    model that can be used to price most types of options that depend on two 
    assets - both American and European.

    Notes
    -----
    This model includes a cost of carry term b, the model can
    used to price European and American Options on:

    b = r       stocks
    b = r - q   stocks and stock indexes paying a continuous dividend yield q
    b = 0       futures
    b = r - rf  currency options with foreign interst rate rf

    Parameters
    ----------
    S1 : float
        Level or index price of asset 1.
    S2 : float
        Level or index price of asset 2.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b1 : float
        Annualized cost-of-carry rate for asset 1, e.g. 0.1 means 10%
    b2 : float
        Annualized cost-of-carry rate for asset 2, e.g. 0.1 means 10%
    sigma1 : float
        Annualized volatility of the underlying asset 1. Optional if calculating implied volatility.
        Required otherwise. By default None.
    sigma2 : float
        Annualized volatility of the underlying asset 2. Optional if calculating implied volatility.
        Required otherwise. By default None.
    rho : float
        Correlation between asset 1 and asset 2.
    K : float
        Strike price. By default None.
    K2 : float
        Strike price. By default None.
    Q1 : float
        Weighting factor for asset 1 for use in payoff formula. By default 1.
    Q2 : float
        Weighting factor for asset 2 for use in payoff formula. By default 1.
    option : int
        Used to select payoff function for different spread option types.
        By default 1 for a spread option.

        option == 1:
            Spread Option
            C: max( 0, (Q1*S1 - Q2*S2) - K )
            P: max( 0, K - (Q1*S1 - Q2*S2) )
        option == 2:
            Options on the maximum
            C: max( 0, max(Q1*S1, Q2*S2) - K )
            P: max( 0, K - max(Q1*S1, Q2*S2) )
        option == 3:
            Options on the minimum
            C: max( 0, min(Q1*S1, Q2*S2) - K )
            P: max( 0, K - min(Q1*S1, Q2*S2) )
        option == 4:
            Dual-strike Option
            C: max( 0, (Q1*S1 - K), (Q2*S2 - K2) )
            P: max( 0, (K - Q1*S1), (K2 - Q2*S2) )
        option == 5:
            Reverse dual-strike option
            C: max( 0, (Q1*S1 - K), (K2 - Q2*S2) )
            P: max( 0, (K - Q1*S1), (Q2*S2 - K2) )
        option == 6:
            Portfolio options
            C: max( 0, (Q1*S1 + Q2*S2) - K )
            P: max( 0, K - (Q2*S2 + Q1*S1) )
        option == 7:
            Options to exchange one asset for another
            max( 0, Q2*S2 - Q1*S1 )
        option == 8:
            Relative performance options
            C: max( 0, (Q1*S1)/(Q2*S2) - K )
            P: max( 0, K - (Q1*S1)/(Q2*S2) )
        option == 9:
            Product options
            C: max( 0, (Q1*S1*Q2*S2) - K )
            P: max( 0, K - (Q1*S1*Q2*S2) )
    otype : str
        "european" to price European options, "american" to price American options. By default "european"
    n : int
        Number of time steps to use. By default 5.

    Returns
    -------
    BinomialSpreadOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """
    __name__ = "BionomialSpreadOption"
    __title__ = "Binomial Tree Spread Option Model"

    def __init__(
        self,
        S1: float,
        S2: float,
        t: float,
        r: float,
        b1: float,
        b2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        K: float = None,
        K2: float = None,
        Q1: float = 1,
        Q2: float = 1,
        option: int = 1,
        otype: str = "european",
        n: int = 5,
    ):
        self._S1 = S1
        self._S2 = S2
        self._Q1 = Q1
        self._Q2 = Q2
        self._K = K
        self._K2 = K2
        self._t = t
        self._r = r
        self._b1 = b1
        self._b2 = b2
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho
        self._option = option
        self._n = n
        self._otype = otype

        self._dt = self._t / self._n
        self._mu1 = self._b1 - 0.5 * self._sigma1 * self._sigma1
        self._mu2 = self._b2 - 0.5 * self._sigma2 * self._sigma2

        self._u = _np.exp(self._mu1 * self._dt + self._sigma1 * _np.sqrt(self._dt))
        self._d = _np.exp(self._mu1 * self._dt - self._sigma1 * _np.sqrt(self._dt))

        self._greeks = GreeksFDM(self)

    def _calc_price(self, z, n, otype, tree):
        # fmt: off
        
        OptionValue = _np.zeros((n+2,n+2))

        for j in range(0,n+1):
            Y1 = (2*j-n)*_np.sqrt(self._dt)
            node_value_S1 = self._S1*self._u**j*self._d**(n-j)

            for i in range(0,n+1):
                node_value_S2 = self._S2*_np.exp(self._mu2*n*self._dt) \
                    *_np.exp(self._sigma2*(self._rho*Y1 + _np.sqrt(1-self._rho**2)* (2*i-n)*_np.sqrt(self._dt)))
                OptionValue[j,i] = self._payoff(self._option, z, node_value_S1, node_value_S2, self._Q1, self._Q2, self._K, self._K2)

        for m in range(0,n)[::-1]:
            
            for j in range(0,m+1):
                Y1 = (2*j-m)*_np.sqrt(self._dt)
                node_value_S1 = self._S1*self._u**j*self._d**(m-j)
                for i in range(0,m+1):
                    y2 = self._rho * Y1 + _np.sqrt(1-self._rho**2) * (2*i-m) * _np.sqrt(self._dt)
                    node_value_S2 = self._S2 * _np.exp(self._mu2*m*self._dt)*_np.exp(self._sigma2*y2)
                    OptionValue[j,i] = 0.25*(OptionValue[j,i]
                        + OptionValue[j+1,i] + OptionValue[j,i+1]
                        + OptionValue[j+1,i+1]
                    )*_np.exp(-self._r*self._dt)

                    if otype == "american":
                        OptionValue[j,i] = _np.maximum(OptionValue[j,i], 
                                self._payoff(self._option, z, node_value_S1, node_value_S2, self._Q1, self._Q2, self._K, self._K2)
                        )


        if tree == True:
            out = OptionValue
        else:
            out = OptionValue[0,0]

        return out
    
    def _payoff(self, option, z, S1, S2, Q1, Q2, K, K2):
        
        if option == 1: # spread option
            return _np.maximum(0, z*((Q1*S1 - Q2*S2) - K))
        elif option == 2: # Max of two assets option
            return _np.maximum(0, z*(_np.maximum(Q1*S1, Q2*S2) - K))
        elif option == 3: # Min of two assets option
            return _np.maximum(0, z*(_np.minimum(Q1*S1, Q2*S2) - K))
        elif option == 4: # Dual strike option
            p1 = _np.maximum(0, z*(Q1*S1 - K))
            p2 = _np.maximum(0, z*(Q2*S2 - K2))
            return  _np.maximum(p1,p2) 
        elif option == 5: # Reverse-dual strike option
            p1 = _np.maximum(0, z*(Q1*S1 - K))
            p2 = _np.maximum(0, z*(K2 - Q2*S2))
            return  _np.maximum(p1,p2) 
        elif option == 6: # Portfolio option
            return _np.maximum(0, z*((Q1*S1 + Q2*S2) - K))
        elif option == 7: # Exchange option
            return _np.maximum(0, z*(Q2*S2 - Q1*S1))
        elif option == 8: # Outperformance option
            return _np.maximum(0, z*((Q1*S1 / Q2*S2) - K))
        elif option == 9: # Product option
            return _np.maximum(0, z*((Q1*S1 * Q2*S2) - K))
        
    def call(self, tree: bool = False):
        """
        Returns the calculated price of a call option according to the
        Rubinstein 3D Binomial Tree option price model.

        Parameters
        ----------
        tree : bool
            If True, returns the bionomial option as a tree-matrix to show the evolution of the option
            value.

        Returns
        -------
        float or tree-matrix

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        z = 1
        return self._calc_price(z, self._n, self._otype, tree)

    def put(self, tree: bool = False):
        """
        Returns the calculated price of a put option according to the
        Rubinstein 3D Binomial Tree option price model.

        Parameters
        ----------
        tree : bool
            If True, returns the bionomial option as a tree-matrix to show the evolution of the option
            value.

        Returns
        -------
        float or tree-matrix

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        z = -1
        return self._calc_price(z, self._n, self._otype, tree)

    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        
        fd1 = self._greeks._make_partial_der("S1", call, self, n=1)
        fd2 = self._greeks._make_partial_der("S2", call, self, n=1)

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
        out = dict(
            S1 = fd1(self._S1) * 1,
            S2 = fd2(self._S2) * 1,
        )

        return out

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        # same for both call and put options
        fd1 = self._greeks._make_partial_der("sigma1", True, self, n=1)
        fd2 = self._greeks._make_partial_der("sigma2", True, self, n=1)

        out = dict(
            sigma1 = fd1(self._sigma1) * 1,
            sigma2 = fd2(self._sigma2) * 1
        )

        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
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

        out = dict(
            S1 = self.delta(call=call)['S1'] * self._S1 / price,
            S2 = self.delta(call=call)['S2'] * self._S2 / price
        )
        return out

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        # same for both call and put options
        fd1 = self._greeks._make_partial_der("S1", True, self, n=2)
        fd2 = self._greeks._make_partial_der("S2", True, self, n=2)
        
        out = dict(
            S1 = fd1(self._S1) * 1,
            S2 = fd2(self._S2) * 1
        )
        # multiple by 1 to return float vs array for single element arrays. Multi-element arrays returned as normal
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
            "S1": self._S1,
            "S2": self._S2,
            "K": self._K,
            "K2": self._K2,
            "t": self._t,
            "r": self._r,
            "b1": self._b1,
            "b2": self._b2,
            "sigma1": self._sigma1,
            "sigma2": self._sigma2,
            "rho": self._rho,
            "Q1": self._Q1,
            "Q2": self._Q2,
            "option": self._option,
            "otype": self._otype,
            "n": self._n,
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
            out += f"  {p} = {self._check_string(params[p])}\n"

        try:
            # if self._sigma or its variations are not None add call and put prices
            if isinstance(self.call(), _np.ndarray):
                c = self._check_string(self.call().round(2))
                p = self._check_string(self.put().round(2))
                price = f"\nOption Price:\n\n  call-{self._otype}: {c}\n  put-{self._otype}: {p}"
            else:
                price = f"\nOption Price:\n\n  call-{self._otype}: {round(self.call(),6)}\n  put-{self._otype}: {round(self.put(),6)}"
            out += price
        except:
            pass

        if printer == True:
            print(out)
        else:
            return out


class BionomialMaxOption(BionomialSpreadAllTypes):
    """
    Rubinstein (1994) published a method to construct a 3-dimensional binomial 
    model that can be used to price most types of options that depend on two 
    assets - both American and European.

    This implementation is for am option on the maximum and is defined as
    the maximum of S1 or S2 vs the strike price K.

    C: max( 0, max(Q1*S1, Q2*S2) - K )
    P: max( 0, K - max(Q1*S1, Q2*S2) )

    Notes
    -----
    This model includes a cost of carry term b, the model can
    used to price European and American Options on:

    b = r       stocks
    b = r - q   stocks and stock indexes paying a continuous dividend yield q
    b = 0       futures
    b = r - rf  currency options with foreign interst rate rf

    Parameters
    ----------
    S1 : float
        Level or index price of asset 1.
    S2 : float
        Level or index price of asset 2.
    K : float
        Strike price. By default None.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b1 : float
        Annualized cost-of-carry rate for asset 1, e.g. 0.1 means 10%
    b2 : float
        Annualized cost-of-carry rate for asset 2, e.g. 0.1 means 10%
    sigma1 : float
        Annualized volatility of the underlying asset 1. Optional if calculating implied volatility.
        Required otherwise. By default None.
    sigma2 : float
        Annualized volatility of the underlying asset 2. Optional if calculating implied volatility.
        Required otherwise. By default None.
    rho : float
        Correlation between asset 1 and asset 2.
    Q1 : float
        Weighting factor for asset 1 for use in payoff formula. By default 1.
    Q2 : float
        Weighting factor for asset 2 for use in payoff formula. By default 1.
    otype : str
        "european" to price European options, "american" to price American options. By default "european"
    n : int
        Number of time steps to use. By default 5.

    Returns
    -------
    BinomialSpreadOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)
    """
    __name__ = "BionomialMaxOption"
    __title__ = "Binomial Tree Maximum Spread Option Model"

    def __init__(
        self,
        S1: float,
        S2: float,
        K: float,
        t: float,
        r: float,
        b1: float,
        b2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        Q1: float = 1,
        Q2: float = 1,
        otype: str = "european",
        n: int = 5,
    ):
        self._S1 = S1
        self._S2 = S2
        self._Q1 = Q1
        self._Q2 = Q2
        self._K = K
        self._K2 = None
        self._t = t
        self._r = r
        self._b1 = b1
        self._b2 = b2
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho
        self._option = 2
        self._n = n
        self._otype = otype

        self._dt = self._t / self._n
        self._mu1 = self._b1 - 0.5 * self._sigma1 * self._sigma1
        self._mu2 = self._b2 - 0.5 * self._sigma2 * self._sigma2

        self._u = _np.exp(self._mu1 * self._dt + self._sigma1 * _np.sqrt(self._dt))
        self._d = _np.exp(self._mu1 * self._dt - self._sigma1 * _np.sqrt(self._dt))

        self._greeks = GreeksFDM(self)

    def get_params(self):
        # need to override so that K2 and option isn't returned
        return {
            "S1": self._S1,
            "S2": self._S2,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b1": self._b1,
            "b2": self._b2,
            "sigma1": self._sigma1,
            "sigma2": self._sigma2,
            "rho": self._rho,
            "Q1": self._Q1,
            "Q2": self._Q2,
            "otype": self._otype,
            "n": self._n,
        }


class BionomialMinOption(BionomialSpreadAllTypes):
    """
    Rubinstein (1994) published a method to construct a 3-dimensional binomial 
    model that can be used to price most types of options that depend on two 
    assets - both American and European.

    This implementation is for am option on the minimum and is defined as
    the maximum of S1 or S2 vs the strike price K.

    C: max( 0, min(Q1*S1, Q2*S2) - K )
    P: max( 0, K - min(Q1*S1, Q2*S2) )

    Notes
    -----
    This model includes a cost of carry term b, the model can
    used to price European and American Options on:

    b = r       stocks
    b = r - q   stocks and stock indexes paying a continuous dividend yield q
    b = 0       futures
    b = r - rf  currency options with foreign interst rate rf

    Parameters
    ----------
    S1 : float
        Level or index price of asset 1.
    S2 : float
        Level or index price of asset 2.
    K : float
        Strike price. By default None.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b1 : float
        Annualized cost-of-carry rate for asset 1, e.g. 0.1 means 10%
    b2 : float
        Annualized cost-of-carry rate for asset 2, e.g. 0.1 means 10%
    sigma1 : float
        Annualized volatility of the underlying asset 1. Optional if calculating implied volatility.
        Required otherwise. By default None.
    sigma2 : float
        Annualized volatility of the underlying asset 2. Optional if calculating implied volatility.
        Required otherwise. By default None.
    rho : float
        Correlation between asset 1 and asset 2.
    Q1 : float
        Weighting factor for asset 1 for use in payoff formula. By default 1.
    Q2 : float
        Weighting factor for asset 2 for use in payoff formula. By default 1.
    otype : str
        "european" to price European options, "american" to price American options. By default "european"
    n : int
        Number of time steps to use. By default 5.

    Returns
    -------
    BinomialSpreadOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)
    """
    __name__ = "BionomialMinOption"
    __title__ = "Binomial Tree Minimum Spread Option Model"

    def __init__(
        self,
        S1: float,
        S2: float,
        K: float,
        t: float,
        r: float,
        b1: float,
        b2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        Q1: float = 1,
        Q2: float = 1,
        otype: str = "european",
        n: int = 5,
    ):
        self._S1 = S1
        self._S2 = S2
        self._Q1 = Q1
        self._Q2 = Q2
        self._K = K
        self._K2 = None
        self._t = t
        self._r = r
        self._b1 = b1
        self._b2 = b2
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho
        self._option = 3
        self._n = n
        self._otype = otype

        self._dt = self._t / self._n
        self._mu1 = self._b1 - 0.5 * self._sigma1 * self._sigma1
        self._mu2 = self._b2 - 0.5 * self._sigma2 * self._sigma2

        self._u = _np.exp(self._mu1 * self._dt + self._sigma1 * _np.sqrt(self._dt))
        self._d = _np.exp(self._mu1 * self._dt - self._sigma1 * _np.sqrt(self._dt))

        self._greeks = GreeksFDM(self)
        
    def get_params(self):
        # need to override so that K2 and option isn't returned
        return {
            "S1": self._S1,
            "S2": self._S2,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b1": self._b1,
            "b2": self._b2,
            "sigma1": self._sigma1,
            "sigma2": self._sigma2,
            "rho": self._rho,
            "Q1": self._Q1,
            "Q2": self._Q2,
            "otype": self._otype,
            "n": self._n,
        }


class BionomialSpreadOption(BionomialSpreadAllTypes):
    """
    Rubinstein (1994) published a method to construct a 3-dimensional binomial 
    model that can be used to price most types of options that depend on two 
    assets - both American and European.

    This implementation is for am option on the minimum and is defined as
    the maximum of S1 or S2 vs the strike price K.

    C: max( 0, (Q1*S1 - Q2*S2) - K )
    P: max( 0, K - (Q1*S1 - Q2*S2) )

    Notes
    -----
    This model includes a cost of carry term b, the model can
    used to price European and American Options on:

    b = r       stocks
    b = r - q   stocks and stock indexes paying a continuous dividend yield q
    b = 0       futures
    b = r - rf  currency options with foreign interst rate rf

    Parameters
    ----------
    S1 : float
        Level or index price of asset 1.
    S2 : float
        Level or index price of asset 2.
    K : float
        Strike price. By default None.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b1 : float
        Annualized cost-of-carry rate for asset 1, e.g. 0.1 means 10%
    b2 : float
        Annualized cost-of-carry rate for asset 2, e.g. 0.1 means 10%
    sigma1 : float
        Annualized volatility of the underlying asset 1. Optional if calculating implied volatility.
        Required otherwise. By default None.
    sigma2 : float
        Annualized volatility of the underlying asset 2. Optional if calculating implied volatility.
        Required otherwise. By default None.
    rho : float
        Correlation between asset 1 and asset 2.
    Q1 : float
        Weighting factor for asset 1 for use in payoff formula. By default 1.
    Q2 : float
        Weighting factor for asset 2 for use in payoff formula. By default 1.
    otype : str
        "european" to price European options, "american" to price American options. By default "european"
    n : int
        Number of time steps to use. By default 5.

    Returns
    -------
    BinomialSpreadOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.spread_options.BinomialSpreadOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, n=100, 
            sigma1=0.2, sigma2=0.2, t=0.1, rho=-0.5, otype="european")
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)
    """
    __name__ = "BionomialMinOption"
    __title__ = "Binomial Tree Minimum Spread Option Model"

    def __init__(
        self,
        S1: float,
        S2: float,
        K: float,
        t: float,
        r: float,
        b1: float,
        b2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        Q1: float = 1,
        Q2: float = 1,
        otype: str = "european",
        n: int = 5,
    ):
        self._S1 = S1
        self._S2 = S2
        self._Q1 = Q1
        self._Q2 = Q2
        self._K = K
        self._K2 = None
        self._t = t
        self._r = r
        self._b1 = b1
        self._b2 = b2
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho
        self._option = 1
        self._n = n
        self._otype = otype

        self._dt = self._t / self._n
        self._mu1 = self._b1 - 0.5 * self._sigma1 * self._sigma1
        self._mu2 = self._b2 - 0.5 * self._sigma2 * self._sigma2

        self._u = _np.exp(self._mu1 * self._dt + self._sigma1 * _np.sqrt(self._dt))
        self._d = _np.exp(self._mu1 * self._dt - self._sigma1 * _np.sqrt(self._dt))

        self._greeks = GreeksFDM(self)

    def get_params(self):
        # need to override so that K2 and option isn't returned
        return {
            "S1": self._S1,
            "S2": self._S2,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b1": self._b1,
            "b2": self._b2,
            "sigma1": self._sigma1,
            "sigma2": self._sigma2,
            "rho": self._rho,
            "Q1": self._Q1,
            "Q2": self._Q2,
            "otype": self._otype,
            "n": self._n,
        }