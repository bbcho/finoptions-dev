from ..base import Option as _Option
from ..vanillaoptions import GreeksFDM as _GreeksFDM
import numpy as _np
from scipy.integrate import quad as _quad
from scipy.stats import norm as _norm

from dataclasses import dataclass


@dataclass
class ParamFit:
    llhHNGarch: float
    Z: _np.array
    h: _np.array


def _llhHNGarch(x, lamb, omega, alpha, beta, gamma, trace, symmetric, rfr):

    h = x
    Z = x

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
    h[1] = (omega + alpha) / (1 - alpha * gam * gam - beta)
    Z[1] = (x[1] - rfr - lamb * h[1]) / _np.sqrt(h[1])
    # fmt: off
    for i in range(1,len(Z)):
        h[i] = omega + alpha * ( Z[i-1] - gam * _np.sqrt(h[i-1]) )**2 + beta * h[i-1]
        Z[i] = ( x[i] - rfr - lamb*h[i] ) / _np.sqrt(h[i])
    # fmt: on

    # Calculate Log - Likelihood for Normal Distribution:
    llhHNGarch = -_np.sum(_np.log(_norm.pdf(Z) / _np.sqrt(h)))
    if trace:
        print("Parameter Estimate\n")
        print(lamb, omega, alpha, beta, gam)

    params = ParamFit()
    params.llhHNGarch = llhHNGarch
    params.Z = Z
    params.h = h

    # Return Value:
    return params


class HNGarch:
    def __init__(self):
        pass
