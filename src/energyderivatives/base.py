from abc import ABC, abstractmethod


class _Base(ABC):
    def __init__(self):
        pass


class Derivative(_Base):
    pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


class Option(Derivative):
    """
    Base class for options
    
    put : bool
        If put is True, initialize a put option object. If put is False, initialize a call option object.
    """

    def __init__(self, S: float, K: float, r: float, t: float, b: float, sigma: float):
        self._S = S
        self._K = K
        self._r = r
        self._t = t
        self._b = b
        self._sigma = sigma

    def simulate(self):
        print("sim run")

    def get_params(self):
        return {
            "level": self._S,
            "strike": self._K,
            "risk-free-rate": self._r,
            "time-to-maturity": self._t,
            "b": self._b,
            "annualized-volatility": self._sigma,
        }

    def put():
        pass

    def call():
        pass


if __name__ == "__main__":

    opt = Option(0, 0, 0, 0, 0, 0)
    opt.simulate()
    print(opt._S)
    print(opt.get_params())

