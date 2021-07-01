from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self):
        pass


class Derivative(Base):
    pass


class Option(Derivative):
    """
    Base class for options
    
    put : bool
        If put is True, initialize a put option object. If put is False, initialize a call option object.
    """

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        t: float,
        b: float,
        sigma: float,
        put: bool = True,
    ):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.b = b
        self.sigma = sigma
        self.put = put

    def simulate(self):
        print("sim run")


class Put(Option):
    def __init__(self, S: float, K: float, r: float, t: float, b: float, sigma: float):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.b = b
        self.sigma = sigma
        self.put = True


class Call(Option):
    def __init__(self, S: float, K: float, r: float, t: float, b: float, sigma: float):
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.b = b
        self.sigma = sigma
        self.put = True


if __name__ == "__main__":

    put = Put(0, 0, 0, 0, 0, 0)
    call = Call(0, 0, 0, 0, 0, 0)
    call.simulate()

