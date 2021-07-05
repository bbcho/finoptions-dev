from ..base import Option as _Option
from ..vanillaoptions import GBSOption as _GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings
import numdifftools as _nd


class BiTreeOption:
    def plot(self):
        pass


class CRRBinomialTreeOption(_Option, BiTreeOption):
    pass


class JRBinomialTreeOption(_Option, BiTreeOption):
    pass


class TIANBinomialTreeOption(_Option, BiTreeOption):
    pass


class BinomialTreeOption(_Option, BiTreeOption):
    pass
