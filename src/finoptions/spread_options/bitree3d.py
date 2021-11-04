import numpy as _np
from ..base import GreeksFDM, Option as _Option


class BionomialSpreadOption:
    """
    Rubinstein (1994) has published a method to construct a 3-dimensional binomial 
    model that can be used to price most types of options that depend on two assets -
    both American and European.

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
        By default 1.

        option == 1:
            Spread Option
            C: max( 0, Q1*S1 - Q2*S2 - K )
            P: max( 0, K + Q2*S2 - Q1*S1 )
        option == 2:
            Option on the maximum
            C: max( 0, max(Q1*S1, Q2*S2) - K )
            P: max( 0, K - max(Q1*S1, Q2*S2) )


    
    
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
        return OptionValue[0,0]
    
    def _payoff(self, option, z, S1, S2, Q1, Q2, K, K2):
        
        if option == 1: # spread option
            return z*_np.maximum(0, (Q1*S1 - Q2*S2 - K)) 
        elif option == 2: # Max of two assets option
            return z*_np.maximum(0, _np.maximum(Q1*S1, Q2*S2) - K) 
        elif option == 3: # Min of two assets option
            return z*_np.maximum(0, _np.minimum(Q1*S1, Q2*S2) - K) 
        elif option == 4: # Dual strike option
            p1 = _np.maximum(0, Q1*S1 - K)
            p2 = _np.maximum(0, Q2*S2 - K2)
            return  z*_np.maximum(p1,p1) 
        elif option == 5: # Reverse-dual strike option
            p1 = _np.maximum(0, Q1*S1 - K)
            p2 = _np.maximum(0, K2 - Q2*S2)
            return  z*_np.maximum(p1,p1) 
        elif option == 6: # Portfolio option
            return z*_np.maximum(0, (Q1*S1 + Q2*S2) - K)
        elif option == 7: # Exchange option
            return z*_np.maximum(0, (Q2*S2 - Q1*S1))
        elif option == 8: # Outperformance option
            return z*_np.maximum(0, (Q1*S1 / Q2*S2) - K)
        elif option == 9: # Product option
            return z*_np.maximum(0, (Q1*S1 * Q2*S2) - K)
        

    def call(self, tree: bool = False):
        """
        Returns the calculated price of a call option according to the
        Cox-Ross-Rubinstein Binomial Tree option price model.

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
        >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5, type='european')
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        z = 1
        return self._calc_price(z, self._n, self._otype, tree)