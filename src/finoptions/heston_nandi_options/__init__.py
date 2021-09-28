from ..base import Option as _Option
from ..vanillaoptions import GreeksFDM as _GreeksFDM
import numpy as _np
from scipy.optimize import minimize as _minimize
from scipy.stats import norm as _norm
from dataclasses import dataclass
# from .hnGARCH import *

# def _HNGCharacteristics(lamb, omega, alpha, beta, gamma, S, K, t_in_days, r_daily, call=True):
#     """
#     Characteristics function for Heston Nandi Option
#     """

#     premium = HNGOption(TypeFlag, model, S, X, Time.inDays, r.daily)
#     delta = HNGGreeks("Delta", TypeFlag, model, S, X, Time.inDays, r.daily)
#     gamma = HNGGreeks("Gamma", TypeFlag, model, S, X, Time.inDays, r.daily)

#     # Return Value:
#     list(premium = premium, delta = delta, gamma = gamma)

_eps = 1.49e-8
low_lim = 0
high_lim = _np.inf

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
        lamb: float = None,
        omega: float = None,
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        model=None,
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
            _fHN, low_lim, high_lim, 
            args=(1, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
            )
        call2 = _quad(
            _fHN, low_lim, high_lim, 
            args=(0, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
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


    def delta(self, call=True):
        # Integrate:
        # fmt: off
        delta1 = _quad(
            _fdeltaHN, low_lim, high_lim, 
            args=(1, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
            )
        
        delta2 = _quad(
            _fdeltaHN, low_lim, high_lim, 
            args=(0, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
            )
       
        # Compute Call and Put Delta :
        greek = 1/2 + _np.exp(-self._r*self._t) * delta1[0] - self._K * _np.exp(-self._r*self._t) * delta2[0]
        if call==False:
            greek = greek - 1

        return greek

    def gamma(self):
        # Integrate:
        # fmt: off
        gamma1 = _quad(
            _fgammaHN, low_lim, high_lim, 
            args=(1, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
            )
        
        gamma2 = _quad(
            _fgammaHN, low_lim, high_lim, 
            args=(0, self._lamb, self._omega, self._alpha, self._beta, self._gamma, self._S, self._K, self._t, self._r),
            epsabs=_eps, epsrel=_eps,
            )
       
        # Compute Call and Put Delta :
        greek = _np.exp(-self._r*self._t) * gamma1[0] - self._K * _np.exp(-self._r*self._t) * gamma2[0]

        return greek


def _fHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r, real=True):
    """
    real=True to return fstarHN, real=False to return fHN
    """

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

    if real == True:
        f = _np.real(_np.exp(-cphi0*_np.log(K)+cphi*_np.log(S)+a+b*sigma2 )/cphi0)/_np.pi
    else:
        f = _np.exp(-cphi0*_np.log(K)+cphi*_np.log(S)+a+b*sigma2 )/cphi0/_np.pi

    # Return Value:
    return f
    # fmt: on


def _fdeltaHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r):
    # Function to be integrated:
    cphi0 = phi * _np.array([1j])
    cphi = cphi0 + const
    fdelta = cphi * _fHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r, real=False) / S

    # Return Value:
    return _np.real(fdelta)


def _fgammaHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r):
    # Function to be integrated:
    cphi0 = phi * _np.array([1j])
    cphi = cphi0 + const
    fgamma = (
        cphi * (cphi - 1) * _fHN(phi, const, lamb, omega, alpha, beta, gamma, S, K, t, r, real=False) / S ** 2
    )

    # Return Value:
    return _np.real(fgamma)

@dataclass
class ParamFit:
    llhHNGarch: float
    z: _np.array
    h: _np.array

    def __sub__(self,b):
        return self.llhHNGarch - b.llhHNGarch


def _llhHNGarch(x0, trace, symmetric, rfr, x, ret_obj=False):

    h = x.copy()
    z = x.copy()

    lamb = x0[0]
    omega = x0[1]
    alpha = x0[2]
    beta = x0[3]
    gamma = x0[4]

    # Transform - to keep them between 0 and 1:
    omega = 1 / (1 + _np.exp(-omega))
    alpha = 1 / (1 + _np.exp(-alpha))
    beta = 1 / (1 + _np.exp(-beta))

    # Add gamma if selected:
    if ~symmetric:
        gam = gamma
    else:
        gam = 0

    # HN Garch Filter:
    h[0] = (omega + alpha) / (1 - alpha * gam * gam - beta)
    z[0] = (x[0] - rfr - lamb * h[0]) / _np.sqrt(h[0])

    # fmt: off
    for i in range(1,len(z)):
        h[i] = omega + alpha * ( z[i-1] - gam * _np.sqrt(h[i-1]) )**2 + beta * h[i-1]
        z[i] = ( x[i] - rfr - lamb*h[i] ) / _np.sqrt(h[i])
    # fmt: on

    # Calculate Log - Likelihood for Normal Distribution:
    llhHNGarch = -_np.sum(_np.log(_norm.pdf(z) / _np.sqrt(h)))
    if trace:
        print("Parameter Estimate\n")
        print(lamb, omega, alpha, beta, gam)

    if ret_obj:
        params = ParamFit(llhHNGarch, z, h)
    else:
        params = llhHNGarch

    # Return Value:
    return params


def hngarch_fit(x, lamb = -0.5, omega = None, alpha = None, beta = 0.1, gamma = 0, rf = 0, symmetric = True, trace = True, **kw_nlm):
    # A function implemented by Diethelm Wuertz

    # Description:
    #   Fits Heston-Nandi Garch(1,1) time series model

    # Parameters:
    if omega is None:
        omega = _np.var(x)
    if alpha is None:
        alpha = 0.1*omega

    rfr = rf
    gam = gamma

    # Continue:
    params = dict(lamb = lamb, omega = omega, alpha = alpha,
        beta = beta, gamma = gam, rf = rfr)

    # Transform Parameters and Calculate Start Parameters:
    par_omega = -_np.log((1-omega)/omega)  # for 2
    par_alpha = -_np.log((1-alpha)/alpha)  # for 3
    par_beta = -_np.log((1-beta)/beta)     # for 4
    par_start = [lamb, par_omega, par_alpha, par_beta]
    if ~symmetric:
        par_start.append(gam)

    # Initial Log Likelihood:
    opt = dict()
    opt['value'] = _llhHNGarch(x0 = par_start, trace = trace, symmetric = symmetric, rfr = rfr, x = x)
    opt['estimate'] = par_start
    
    if trace:
        print(lamb, omega, alpha, beta, gam)
        print(opt['value'])
    
    # Estimate Parameters:
    res = _minimize(_llhHNGarch, par_start, args=(trace, symmetric, rfr, x), method='L-BFGS-B', **kw_nlm)

    # Log-Likelihood:
    opt['minimum'] = -res.fun + len(x)*_np.sqrt(2*_np.pi)
    # opt['params'] = params
    # opt['symmetric'] = symmetric
    opt['estimate'] = res.x

    # LLH, h, and z for Final Estimates:
    final = _llhHNGarch(opt['estimate'], trace = False, symmetric=symmetric, rfr=rfr, x=x, ret_obj=True)
    opt['h'] = final.h
    opt['z'] = final.z

    # Backtransform Estimated parameters:
    lamb = opt['estimate'][0]
    omega = (1 / (1+_np.exp(-opt['estimate'][1])))
    opt['estimate'][1] = omega

    alpha = (1 / (1+_np.exp(-opt['estimate'][2])))
    opt['estimate'][2] = alpha

    beta = (1 / (1+_np.exp(-opt['estimate'][3])))
    opt['estimate'][3] = beta
    
    if symmetric: 
        opt['estimate'][4] = 0
    gam = opt['estimate'][4]

    # names(opt$estimate) = c("lambda", "omega", "alpha", "beta", "gamma")

    # Add to Output:
    opt['model'] = dict(lamb = lamb, omega = omega, alpha = alpha,
        beta = beta, gamma = gam, rf = rfr)
    # opt['x'] = x

    # Statistics - Printing:
    opt['persistence'] = beta + alpha*gam*gam
    opt['sigma2'] = ( omega + alpha ) / ( 1 - opt['persistence'] )

    # Print Estimated Parameters:
    if (trace):
        print(opt['estimate'])

    opt['title'] = "Heston-Nandi Garch Parameter Estimation"

    return opt


def hngarch_sim(lamb = -0.5, omega = None, alpha = None, beta = 0.1, gamma = 0, rf = 0, n = 500, n_start = 0, inno=None, inno_start=None):
    """
    Description:
    Simulates a HN-GARCH time series with user supplied innovations.

    Details:
      The function simulates a Heston Nandi Garch(1,1) process with
      structure parameters specified through the parameters 
      lambda, omega, alpha, beta, gamma, rf.
      The function returns the simulated time series points
      neglecting those from the first "start_innov" innovations.

    Parameters
    ----------
    lamb : float
    omega : float
    alpha : float
    beta : float
    gamma : float
    rf : float
    n : int
    n_start : int
    inno : 
    inno_start :

    Example:
      x = hngarch()
      plot(100*x, type="l", xlab="Day numbers",
        ylab="Daily Returns %", main="Heston Nandi GARCH")
      S0 = 1
      plot(S0*exp(cumsum(x)), type="l", xlab="Day Numbers",
        ylab="Daily Prices", main="Heston Nandi GARCH") }
    """
    # FUNCTION:

    # Innovations:
    if inno is None:
        inno = _np.random.normal(0, 1, n)
    if (inno_start is None) & (n_start > 0):
        inno_start = _np.random.normal(0, 1, n_start)

    if inno_start is not None:
        x = _np.concatenate((inno_start, inno))
    else:
        x = inno.copy()
    h = x.copy()
    Z = x.copy()
    
    nt = n_start + n

    # Recursion:
    h[0] = ( omega + alpha )/( 1 - alpha*gamma*gamma - beta )
    x[0] = rf + lamb*h[0] + _np.sqrt(h[0]) * Z[0]
    for i in range(1,nt):
        h[i] = omega + alpha*(Z[i-1] - gamma*_np.sqrt(h[i-1]))**2 + beta*h[i-1]
        x[i] = rf + lamb*h[i] + _np.sqrt(h[i]) * Z[i]

    # Series:
    x = x[(n_start):]

    # Return Value:
    return x