from abc import ABC, abstractmethod
import numpy as _np


class Payoff(ABC):
    def __init__(
        self, path, S: float, K: float, t: float, r: float, b: float, sigma: float
    ):
        self.path = path
        self.S = S
        self.K = K
        self.t = t
        self.r = r
        self.b = b
        self.sigma = sigma

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def put(self):
        pass


class PlainVanillaPayoff(Payoff):
    def call(self):
        St = self.S * _np.exp(_np.sum(self.path.generate_path(), axis=1))
        return _np.exp(-self.r * self.t) * _np.maximum(St - self.K, 0)

    def put(self):
        St = self.S * _np.exp(_np.sum(self.path.generate_path(), axis=1))
        return _np.exp(-self.r * self.t) * _np.maximum(self.K - St, 0)


class ArithmeticAsianPayoff(Payoff):
    def call(self):
        Sm = self.S * _np.exp(_np.cumsum(self.path.generate_path(), axis=1))
        Sm = _np.mean(Sm, axis=1)
        return _np.exp(-self.r * self.t) * _np.maximum(Sm - self.K, 0)

    def put(self):
        Sm = self.S * _np.exp(_np.cumsum(self.path.generate_path(), axis=1))
        Sm = _np.mean(Sm, axis=1)
        return _np.exp(-self.r * self.t) * _np.maximum(self.K - Sm, 0)
