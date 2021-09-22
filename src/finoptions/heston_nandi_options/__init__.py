from ..base import Option as _Option
from ..vanillaoptions import GreeksFDM as _GreeksFDM
import numpy as _np
from scipy.integrate import quad as _quad

# def _HNGCharacteristics(lamb, omega, alpha, beta, gamma, S, K, t_in_days, r_daily, call=True):
#     """
#     Characteristics function for Heston Nandi Option
#     """

#     premium = HNGOption(TypeFlag, model, S, X, Time.inDays, r.daily)
#     delta = HNGGreeks("Delta", TypeFlag, model, S, X, Time.inDays, r.daily)
#     gamma = HNGGreeks("Gamma", TypeFlag, model, S, X, Time.inDays, r.daily)

#     # Return Value:
#     list(premium = premium, delta = delta, gamma = gamma)


class HNGGreeks(_GreeksFDM):
    pass


class HestonNandiOption:  # _Option
    """
    Option class for the Heston-Nandi Garch Option Model.

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in days. i.e. 1/252 for 1 business day, 5/252 for 1 week etc...
    r : float
        Daily rate of interest. i.e. 0.25/252 means about 0.001% per day.
    lamb : float

    omega : float
        The GARCH model parameter which specifies the constant coefficient of the variance
        equation.
    alpha : float
        The GARCH model parameter which specifies the autoregressive coefficient
    beta : float
        The GARCH model parameter which specifies the variance coefficient.
    gamma : float
        The GARCH model parameter which specifies the asymmetry coefficient.

    Notes
    -----


    Returns
    -------
    HestonNandiOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.heston_nandi_options.HestonNandiOption(S=80, K=82, t=1/3, td=1/4, r=0.06, D=4, sigma=0.30)
    >>> opt.call()
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "HestonNandiOption"
    __title__ = "The Heston-Nandi Garch Option Model"

    def __init__(
        self,
        S: float,
        K: float,
        t: float,
        r: float,
        lamb: float,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
    ):

        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._lamb = lamb
        self._omega = omega
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def call(self):
        """
        # Integrate:
        call1 = integrate(.fstarHN, 0, Inf, const = 1, model = model,
            S = S, X = X, Time.inDays = Time.inDays, r.daily = r.daily)

        # Compute Call Price:
        call.price = S/2 + exp(-r.daily*Time.inDays) * call1$value -
            X * exp(-r.daily*Time.inDays) * ( 1/2 + call2$value )

        """
        # fmt: off
        call1 = _quad(
            _fstarHN, 0, _np.inf, 
            args=(1, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r)
            )
        call2 = _quad(
            _fstarHN, 0, _np.inf, 
            args=(0, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r)
            )
        # fmt: on

        # Compute Call Price:
        price = (
            self._S / 2
            + _np.exp(-self._r * self._t) * call1[0]
            - self._K * _np.exp(-self._r * self._t) * (1 / 2 + call2[0])
        )

        return price

    def put(self):
        return self.call() + self._K * _np.exp(-self._r * self._t) - self._S


def _fstarHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r):

    # Internal Function:

    # Model Parameters:
    gamma = gamma + lamb + 1 / 2
    lamb = -1 / 2
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma ** 2)
    # Function to be integrated:
    cphi0 = phi * _np.array([1j])
    cphi = cphi0 + const
    a = cphi * r
    b = lamb * cphi + cphi * cphi / 2
    # fmt: off
    for i in range(1, t):
        a = a + cphi*r + b*omega - _np.log(1-2*alpha*b)/2
        b = cphi*(lamb+gamma) - gamma**2/2 + beta*b + 0.5*(cphi-gamma)**2/(1-2*alpha*b)

    f = _np.real(_np.exp(-cphi0*_np.log(K)+cphi*_np.log(S)+a+b*sigma2 )/cphi0)/_np.pi

    # Return Value:
    return f
    # fmt: on
