import numpy as _np
from ..base import GreeksFDM, Option as _Option
from ..utils import docstring_from

class SpreadApproxOption(_Option):
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
    >>> opt = fo.spread_options.SpreadApproxOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, sigma1=0.2, sigma2=0.25, rho=0.5)
    >>> opt.call()
    >>> opt.put()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "SpreadApproxOption"
    __title__ = "Spread-Option Approximation Model"

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
    ):
        self._S1 = S1
        self._S2 = S2
        self._Q1 = Q1
        self._Q2 = Q2
        self._K = K
        self._t = t
        self._r = r
        self._b1 = b1
        self._b2 = b2
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho

        self._e1 = _np.exp((b1 - r) * t)
        self._e2 = _np.exp((b2 - r) * t)

        self._S = (Q1 * S1 * self._e1) / (Q2 * S2 * self._e2 + K * _np.exp(-r * t))

        F = (Q2 * S2 * self._e2) / (Q2 * S2 * self._e2 + K * _np.exp(-r * t))

        self._sigma = _np.sqrt(
            sigma1 ** 2 + (sigma2 * F) ** 2 - 2 * rho * sigma1 * sigma2 * F
        )

        self._d1 = (_np.log(self._S) + 0.5 * self._sigma ** 2 * t) / (
            self._sigma * _np.sqrt(t)
        )

        self._d2 = self._d1 - self._sigma * _np.sqrt(t)

        self._greeks = GreeksFDM(self)

    def call(self):
        """
        Returns the calculated price of a call option using the Spread
        Option Approximation model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.spread_options.SpreadApproxOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, sigma1=0.2, sigma2=0.25, rho=0.5)
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        # fmt: off
        result = (self._Q2*self._S2*self._e2 + self._K*_np.exp(-self._r*self._t)) \
                    * (self._S*self._CND(self._d1) - self._CND(self._d2))

        return result

    def put(self):
        """
        Returns the calculated price of a put option using the Spread
        Option Approximation model.

        Returns
        -------
        float

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.spread_options.SpreadApproxOption(S1=122, S2=120, K=3, r=0.1, b1=0, b2=0, sigma1=0.2, sigma2=0.25, rho=0.5)
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """
        # fmt: off
        result = (self._Q2*self._S2*self._e2 + self._K*_np.exp(-self._r*self._t)) \
                    * (-self._S*self._CND(-self._d1) + self._CND(-self._d2))

        return result
