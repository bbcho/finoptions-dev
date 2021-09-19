from abc import ABC, abstractmethod
from scipy.stats import qmc as _qmc, norm as _norm
import numpy as _np


class Innovations(ABC):
    def __init__(self, mc_samples, path_length, eps=None):
        self.mc_samples = mc_samples
        self.path_length = path_length
        self._eps = eps  # for testing only

    @property
    def mc_samples(self):
        return self.__mc_samples

    @mc_samples.setter
    def mc_samples(self, val):
        self.__mc_samples = val

    @property
    def path_length(self):
        return self.__path_length

    @path_length.setter
    def path_length(self, val):
        self.__path_length = val

    @abstractmethod
    def sample_innovation(self, init=False):
        pass


class NormalSobolInnovations(Innovations):
    """
    Normal Sobol Innovation Class for use with Monte Carlo Options Class

    Parameters
    ----------
    mc_samples : int
        Number of monte carlo samples per loop - total number of samples is mc_samples * mc_loops
    path_length : int
        Path length should be a power of 2 (i.e. 2**m) to generate stable paths. See
        statsmodels.stats.qmc.Sobol for more details.

    Returns
    -------
    Innovation Class
    """

    def sample_innovation(self, scramble=True):
        sobol = self._get_sobol(scramble)

        # avoid inf
        while _np.abs(_np.max(sobol)) == _np.inf:
            sobol = self._get_sobol(scramble)

        if self._eps is None:
            return sobol
        else:
            # for testing only since fOptions sobol innovations
            # return different data from statsmodels
            return self._eps

    def _get_sobol(self, scramble):
        sobol = _qmc.Sobol(self.path_length, scramble=scramble).random(self.mc_samples)
        if scramble == False:
            # add new sample since if not scrambled first row is zero which leads to -inf when normalized
            sobol = sobol[1:]
            sobol = _np.append(
                sobol,
                _qmc.Sobol(self.path_length, scramble=scramble)
                .fast_forward(self.mc_samples)
                .random(1),
                axis=0,
            )
        sobol = _norm.ppf(sobol)

        return sobol
