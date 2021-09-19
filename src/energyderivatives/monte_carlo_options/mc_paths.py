from abc import ABC, abstractmethod
import numpy as _np


class Path(ABC):
    def __init__(self, epsilon, sigma, dt, b):
        self.epsilon = epsilon
        self.sigma = sigma
        self.dt = dt
        self.b = b

    @abstractmethod
    def generate_path(self):
        pass


class WienerPath(Path):
    """
    Generate Wiener Paths for use with Monte Carlo Options

    Parameters
    ----------
    epsilon : numpy array
        epilson for paths as numpy array. Output from Innovation class sample_innovation function
    sigma : float
        annualized volatility sigma used to generate paths
    dt : float
        time step dt to generate paths
    b : float
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    """

    def generate_path(self, **kwargs):
        # fmt: off
        return (self.b - (self.sigma ** 2) / 2) * self.dt + self.sigma * _np.sqrt(self.dt)  * self.epsilon
        # fmt: on
