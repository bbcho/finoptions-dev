from ..base import GreeksFDM, Option as _Option
from ..vanillaoptions import GBSOption as _GBSOption
import numpy as _np
from scipy.optimize import root_scalar as _root_scalar
import sys as _sys
import warnings as _warnings
import numdifftools as _nd
from ..utils import docstring_from
from matplotlib import pyplot as _plt


class BiTreePlotter:
    def __init__(self, tree):
        self._tree = tree

    def plot(self, dx=-0.025, dy=0.4, size=12, digits=2, **kwargs):
        depth = self._tree.shape[1] - 1

        fig = _plt.figure(**kwargs)
        for i in range(depth):
            x = [1, 0, 1]
            for j in range(i):
                x.append(0)
                x.append(1)
            x = _np.array(x) + i
            y = _np.arange(-(i + 1), i + 2)[::-1]
            _plt.plot(x, y, "bo-")

            pts = list(zip(x, y))

            for p in pts:
                value = round(self._tree[self._convert_to_index(p)][p[0]], digits)
                _plt.annotate(value, xy=p, xytext=(p[0] + dx, p[1] + dy), fontsize=size)

        _plt.xlabel("n", fontsize=size)
        _plt.ylabel("Option Value", fontsize=size)
        _plt.title("Option Tree", fontsize=int(size * 1.3))
        return fig

    def _convert_to_index(self, p):
        # converts tree y values into indices for the tree matrix
        step = -2
        max = p[0]
        if max > 0:
            a = _np.arange(max, -max + step, step)
            return int(_np.where(a == p[1])[0])
        else:
            return 0


class TriTreePlotter:
    def __init__(self, tree):
        self._tree = tree

    def plot(self, dx=-0.025, dy=0.4, size=12, digits=2, **kwargs):
        n = self._tree.shape[1] - 1
        fig = _plt.figure(**kwargs)
        for i in range(n + 1):
            for j in range(-i + 1, i):
                self._plot_node(i - 1, j, dx, dy, size, digits)

        _plt.annotate(
            round(self._tree[n, 0], digits),
            xy=(0, 0),
            xytext=(+dx, +dy),
            fontsize=size,
        )
        _plt.xlabel("n", fontsize=size)
        _plt.ylabel("Option Value", fontsize=size)
        _plt.title("Option Tree", fontsize=int(size * 1.3))
        return fig

    def _plot_node(self, i, j, dx, dy, size, digits):
        for c in [-1, 0, 1]:
            depth = self._tree.shape[1] - 1
            x = _np.array([0, 1]) + i
            y = _np.array([0, c]) + j
            _plt.plot(x, y, "bo-")
            p = (x[1], y[1])
            # print(p)
            # print(p[0], p[1] + 2)
            _plt.annotate(
                round(self._tree[-y[1] + depth][x[1]], digits),
                xy=p,
                xytext=(p[0] + dx, p[1] + dy),
                fontsize=size,
            )


class CRRBinomialTreeOption(_Option):
    """
    Binomial models were first suggested by Cox, Ross and Rubinstein (1979), CRR,
    and then became widely used because of its intuition and easy implementation. Binomial trees are
    constructed on a discrete-time lattice. With the time between two trading events shrinking to zero,
    the evolution of the price converges weakly to a lognormal diffusion. Within this mode the European
    options value converges to the value given by the Black-Scholes formula.

    Notes
    -----
    The model described here is a version of the CRR Binomial
    Tree model. Including a cost of carry term b, the model can
    used to price European and American Options on:

    b = r       stocks
    b = r - q   stocks and stock indexes paying a continuous dividend yield q
    b = 0       futures
    b = r - rf  currency options with foreign interst rate rf

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b : float
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.

    Returns
    -------
    CRRBinomialTreeOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0., sigma=0.4)
    >>> opt.call()
    >>> opt.put()
    >>> opt.call(type='american')
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "CRRBinomialTreeOption"
    __title__ = "CRR Binomial Tree Model"

    def __init__(
        self,
        S: float,
        K: float,
        t: float,
        r: float,
        b: float,
        sigma: float,
        type: str = "european",
        n: int = 5,
    ):
        self._S = S
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma = sigma
        self._n = n
        self._type = type

        self._greeks = GreeksFDM(self)

    def get_params(self):
        return {
            "S": self._S,
            "K": self._K,
            "t": self._t,
            "r": self._r,
            "b": self._b,
            "sigma": self._sigma,
            "type": self._type,
            "n": self._n,
        }

    def call(self, tree: bool = False):
        """
        Returns the calculated price of a call option according to the
        Cox-Ross-Rubinstein Binomial Tree option price model.

        Parameters
        ----------
        tree : bool
            If True, returns the bionomial option as a tree-matrix to show the evolution of the option
            value.

        Returns
        -------
        float or tree-matrix

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5, type='european')
        >>> opt.call()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        z = 1
        return self._calc_price(z, self._n, self._type, tree)

    def put(self, tree: bool = False):
        """
        Returns the calculated price of a put option according to the
        Cox-Ross-Rubinstein Binomial Tree option price model.

        Parameters
        ----------
        tree : bool
            If True, returns the bionomial option as a tree-matrix to show the evolution of the option
            value.

        Returns
        -------
        float or tree-matrix

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5, type='european')
        >>> opt.put()

        References
        ----------
        [1] Haug E.G., The Complete Guide to Option Pricing Formulas
        """

        z = -1
        return self._calc_price(z, self._n, self._type, tree)

    def summary(self, printer=True):
        """
        Print summary report of option

        Parameters
        ----------
        printer : bool
            True to print summary. False to return a string.
        """
        out = f"Title: {self.__title__} Valuation\n\nParameters:\n\n"

        params = self.get_params()

        for p in params:
            out += f"  {p} = {self._check_string(params[p])}\n"

        try:
            # if self._sigma or its variations are not None add call and put prices
            if isinstance(self.call(), _np.ndarray):
                c = self._check_string(self.call().round(2))
                p = self._check_string(self.put().round(2))
                price = f"\nOption Price:\n\n  call-{self._type}: {c}\n  put-{self._type}: {p}"
            else:
                price = f"\nOption Price:\n\n  call-{self._type}: {round(self.call(),6)}\n  put-{self._type}: {round(self.put(),6)}"
            out += price
        except:
            pass

        if printer == True:
            print(out)
        else:
            return out

    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        return self._greeks.delta(call=call)

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        return self._greeks.vega()

    @docstring_from(GreeksFDM.rho)
    def rho(self, call: bool = True):
        return self._greeks.rho(call=call)

    @docstring_from(GreeksFDM.lamb)
    def lamb(self, call: bool = True):
        return self._greeks.lamb(call=call)

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        return self._greeks.gamma()

    @docstring_from(GreeksFDM.greeks)
    def greeks(self, call: bool = True):
        return self._greeks.greeks(call=call)

    # fmt: off
    def _calc_price(self, z, n, type, tree):
        dt = self._t / n
        u = _np.exp(self._sigma * _np.sqrt(dt))
        d = 1 / u
        p = (_np.exp(self._b * dt) - d) / (u - d)
        Df = _np.exp(-self._r * dt)

        OptionValue = z * (self._S * u ** _np.arange(0, n + 1) * d ** _np.arange(n, -1, -1) - self._K)
        OptionValue = (_np.abs(OptionValue) + OptionValue) / 2

        if type == "european":
            out = self._euro(OptionValue, n, Df, p, tree)
        elif type == "american":
            out = self._amer(OptionValue, n, Df, p, self._K, d, self._S, u, z, tree)

        if tree == False:
            return out[0]
        else:
            return out

    def _euro(self, OptionValue, n, Df, p, tree=False):
        tr = OptionValue.copy()
        for j in _np.arange(0, n)[::-1]:
            tr = _np.append(tr, _np.zeros(n-j))
            for i in _np.arange(0, j + 1):
                OptionValue[i] = (p * OptionValue[i + 1] + (1 - p) * OptionValue[i]) * Df
                tr = _np.append(tr, OptionValue[i])

        if tree == True:
            tr = _np.reshape(tr[::-1],(n+1,n+1)).T
            return tr
        else:
            return OptionValue

    def _amer(self, OptionValue, n, Df, p, K, d, S, u, z, tree=False):
        tr = OptionValue.copy()
        for j in _np.arange(0, n)[::-1]:
            tr = _np.append(tr, _np.zeros(n-j))
            for i in _np.arange(0, j + 1):
                OptionValue[i] = max(
                    (z * (S * u ** i * d ** (abs(i - j)) - K)),
                    (p * OptionValue[i + 1] + (1 - p) * OptionValue[i]) * Df,
                )
                tr = _np.append(tr, OptionValue[i])
        if tree == True:
            tr = _np.reshape(tr[::-1],(n+1,n+1)).T
            return tr
        else:
            return OptionValue
    # fmt: on

    def plot(self, call=True, dx=-0.025, dy=0.4, size=12, digits=2, **kwargs):
        """
        Method to plot the binomial tree values

        Parameters
        ----------
        call : bool
            Set to True to plot a call option. Otherwise set to False for a put option. By default True.
        dx : float
            x-offset for the node values in the same units at the call or put option value.
        dy : float
            y-offset for the node values in the same units at the call or put option value.
        size : float
            Font size for node and axis labels. By default 12. Title is 30% larger.
        digits : int
            Number of significant figures to show for option values. By default 2.
        **kwargs
            paramters to pass to matplotlib

        Returns
        -------
        matplotlib figure object

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5, type='american')
        >>> opt.plot(call=False, figsize=(10,10))
        """

        if call == True:
            tree = self.call(tree=True)
        else:
            tree = self.put(tree=True)

        plotter = BiTreePlotter(tree)

        return plotter.plot(dx, dy, size, digits, **kwargs)


class JRBinomialTreeOption(CRRBinomialTreeOption):
    """
    There exist many extensions of the CRR model. Jarrow and Rudd (1983), JR, adjusted the CRR
    model to account for the local drift term. They constructed a binomial model where the first two
    moments of the discrete and continuous time return processes match. As a consequence a probabil-
    ity measure equal to one half results. Therefore the CRR and JR models are sometimes atrributed
    as equal jumps binomial trees and equal probabilities binomial trees.

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b : float
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.

    Returns
    -------
    JRBinomialTreeOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.binomial_tree_options.JRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0., sigma=0.4)
    >>> opt.call()
    >>> opt.put()
    >>> opt.call(type='american')
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    __name__ = "JRBinomialTreeOption"
    __title__ = "JR Binomial Tree Model"

    def _calc_price(self, z, n, type, tree=False):
        dt = self._t / n
        u = _np.exp((self._b - self._sigma ** 2 / 2) * dt + self._sigma * _np.sqrt(dt))
        d = _np.exp((self._b - self._sigma ** 2 / 2) * dt - self._sigma * _np.sqrt(dt))
        p = 1 / 2
        Df = _np.exp(-self._r * dt)

        OptionValue = z * (
            self._S * u ** _np.arange(0, n + 1) * d ** _np.arange(n, -1, -1) - self._K
        )
        OptionValue = (_np.abs(OptionValue) + OptionValue) / 2

        if type == "european":
            out = self._euro(OptionValue, n, Df, p, tree)
        elif type == "american":
            out = self._amer(OptionValue, n, Df, p, self._K, d, self._S, u, z, tree)

        if tree == False:
            return out[0]
        else:
            return out


class TIANBinomialTreeOption(CRRBinomialTreeOption):
    """
    Tian (1993) suggested to match discrete and continuous local moments up to third order.
    Leisen and Reimer (1996) proved that the order of convergence in pricing European options for all
    three methods is equal to one, and thus the three models are equivalent.

    Parameters
    ----------
    S : float
        Level or index price.
    K : float
        Strike price.
    t : float
        Time-to-maturity in fractional years. i.e. 1/12 for 1 month, 1/252 for 1 business day, 1.0 for 1 year.
    r : float
        Risk-free-rate in decimal format (i.e. 0.01 for 1%).
    b : float
        Annualized cost-of-carry rate, e.g. 0.1 means 10%
    sigma : float
        Annualized volatility of the underlying asset. Optional if calculating implied volatility.
        Required otherwise. By default None.

    Returns
    -------
    JRBinomialTreeOption object.

    Example
    -------
    >>> import finoptions as fo
    >>> opt = fo.binomial_tree_options.TIANBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0., sigma=0.4)
    >>> opt.call()
    >>> opt.put()
    >>> opt.call(type='american')
    >>> opt.greeks(call=True)

    References
    ----------
    [1] Haug E.G., The Complete Guide to Option Pricing Formulas
    """

    def _calc_price(self, z, n, type, tree=False):
        dt = self._t / n
        M = _np.exp(self._b * dt)
        V = _np.exp(self._sigma ** 2 * dt)
        u = (M * V / 2) * (V + 1 + _np.sqrt(V * V + 2 * V - 3))
        d = (M * V / 2) * (V + 1 - _np.sqrt(V * V + 2 * V - 3))
        p = (M - d) / (u - d)
        Df = _np.exp(-self._r * dt)

        OptionValue = z * (
            self._S * u ** _np.arange(0, n + 1) * d ** _np.arange(n, -1, -1) - self._K
        )
        OptionValue = (_np.abs(OptionValue) + OptionValue) / 2

        if type == "european":
            out = self._euro(OptionValue, n, Df, p, tree)
        elif type == "american":
            out = self._amer(OptionValue, n, Df, p, self._K, d, self._S, u, z, tree)

        if tree == False:
            return out[0]
        else:
            return out


class TrinomialTreeOption(CRRBinomialTreeOption):
    """
    Description:
          Calculates option prices from the Trinomial tree model.

        Arguments:
          AmeEurFlag - a character value, either "a" or "e" for
              a European or American style option
          CallPutFlag - a character value, either "c" or "p" for
              a call or put option
          S, X, Time, r, b, sigma - the usual option parameters
          n - an integer value, the depth of the tree

        Value:
          Returns the price of the options.

        Details:
          Trinomial trees in option pricing are similar to
          binomial trees. Trinomial trees can be used to
          price both European and American options on a single
          underlying asset.
          Because the asset price can move in three directions
          from a given node, compared with only two in a binomial
          tree, the number of time steps can be reduced to attain
          the same accuracy as in the binomial tree.

        Reference:
          E.G Haug, The Complete Guide to Option Pricing Formulas
          Chapter 3.2
    """

    __name__ = "TrinomialTreeOption"
    __title__ = "Trinomial Tree Model"

    def _calc_price(self, z, n, type, tree):

        dt = self._t / n

        # Up-and-down jump sizes:
        u = _np.exp(self._sigma * _np.sqrt(2 * dt))
        d = _np.exp(-self._sigma * _np.sqrt(2 * dt))
        # Probabilities of going up and down:
        pu = (
            (_np.exp(self._b * dt / 2) - _np.exp(-self._sigma * _np.sqrt(dt / 2)))
            / (
                _np.exp(self._sigma * _np.sqrt(dt / 2))
                - _np.exp(-self._sigma * _np.sqrt(dt / 2))
            )
        ) ** 2
        pd = (
            (_np.exp(self._sigma * _np.sqrt(dt / 2)) - _np.exp(self._b * dt / 2))
            / (
                _np.exp(self._sigma * _np.sqrt(dt / 2))
                - _np.exp(-self._sigma * _np.sqrt(dt / 2))
            )
        ) ** 2

        # Probability of staying at the same asset price level:
        pm = 1 - pu - pd
        Df = _np.exp(-self._r * dt)

        # init tree as 1-D array
        OptionValue = _np.repeat(0.0, 2 * n + 1)
        for i in _np.arange(0, (2 * n + 1)):
            a = 0
            b = z * (
                self._S
                * u ** _np.maximum(i - n, 0)
                * d ** _np.maximum(n * 2 - n - i, 0)
                - self._K
            )
            OptionValue[i] = _np.maximum(a, b)

        list = []
        tr = _np.array(list)
        tr = OptionValue.copy()
        for j in _np.arange(0, n)[::-1]:
            for i in _np.arange(0, j * 2 + 1):
                # fmt: off
                if type == "european":
                    OptionValue[i] = self._euro(i, OptionValue, Df, pu, pd, pm)
                elif type == "american":
                    OptionValue[i] = self._amer(i, j, OptionValue, Df, pu, pd, pm, self._K, d, self._S, u, z)
                # fmt: on
                tr = _np.append(tr, OptionValue[i])

        if tree == True:
            tr = self._reshape(tr, n + 1)
            return tr
        else:
            return OptionValue[0]

    def _euro(self, i, OptionValue, Df, pu, pd, pm):
        # fmt: off
        out = (pu * OptionValue[i + 2] + pm * OptionValue[i + 1] + pd * OptionValue[i]) * Df
        # fmt: on
        return out

    def _amer(self, i, j, OptionValue, Df, pu, pd, pm, K, d, S, u, z):
        # fmt: off
        a = (z * (S * u ** _np.maximum(i - j, 0) * d ** _np.maximum(j - i, 0) - K))
        b = ( pu * OptionValue[i + 2] + pm * OptionValue[i + 1] + pd * OptionValue[i])* Df
        out = _np.maximum(a, b)
        # fmt: on

        return out

    @docstring_from(GreeksFDM.delta)
    def delta(self, call: bool = True):
        return self._greeks.delta(call=call)

    @docstring_from(GreeksFDM.theta)
    def theta(self, call: bool = True):
        return self._greeks.theta(call=call)

    @docstring_from(GreeksFDM.vega)
    def vega(self):
        return self._greeks.vega()

    @docstring_from(GreeksFDM.rho)
    def rho(self, call: bool = True):
        return self._greeks.rho(call=call)

    @docstring_from(GreeksFDM.lamb)
    def lamb(self, call: bool = True):
        return self._greeks.lamb(call=call)

    @docstring_from(GreeksFDM.gamma)
    def gamma(self):
        return self._greeks.gamma()

    @docstring_from(GreeksFDM.greeks)
    def greeks(self, call: bool = True):
        return self._greeks.greeks(call=call)

    def _reshape(self, tr, n):
        out = _np.zeros((2 * n - 1, n))
        for i in range(0, out.shape[1]):
            if i > 0:
                out[n - i - 1 : n + i, i] = tr[::-1][i ** 2 : i ** 2 + 2 * i + 1]
            else:
                out[n - 1][i] = tr[::-1][0]

        return out

    def plot(self, call=True, dx=-0.2, dy=0.2, size=12, digits=2, **kwargs):
        """
        Method to plot the trinomial tree values

        Parameters
        ----------
        call : bool
            Set to True to plot a call option. Otherwise set to False for a put option. By default True.
        dx : float
            x-offset for the node values in the same units at the call or put option value.
        dy : float
            y-offset for the node values in the same units at the call or put option value.
        size : float
            Font size for node and axis labels. By default 12. Title is 30% larger.
        digits : int
            Number of significant figures to show for option values. By default 2.
        **kwargs
            paramters to pass to matplotlib

        Returns
        -------
        matplotlib figure object

        Example
        -------
        >>> import finoptions as fo
        >>> opt = fo.binomial_tree_options.CRRBinomialTreeOption(S=50, K=50, t=5/12, r=0.1, b=0.1, sigma=0.4, n=5, type='american')
        >>> opt.plot(call=False, figsize=(10,10))
        """

        if call == True:
            tree = self.call(tree=True)
        else:
            tree = self.put(tree=True)

        plotter = TriTreePlotter(tree)

        return plotter.plot(dx, dy, size, digits, **kwargs)
