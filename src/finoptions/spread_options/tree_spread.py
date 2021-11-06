import numpy as _np


class TrinomialSpreadOption:
    __name__ = "TrinomialSpreadOption"
    __title__ = "Trinomial Tree Spread Option Model"

    def __init__(
        self,
        S1: float,
        S2: float,
        K: float,
        t: float,
        r: float,
        b: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        type: str = "european",
        n: int = 5,
    ):
        self._S1 = S1
        self._S2 = S2
        self._K = K
        self._t = t
        self._r = r
        self._b = b
        self._sigma1 = sigma1
        self._sigma2 = sigma2
        self._rho = rho
        self._n = n
        self._type = type

    def call(self):
        return self._calc_price(z=1, type=self._type, tree=False)

    def _calc_price(self, z, type, tree):
        n = self._n
        S1 = self._S1
        sd1 = self._sigma1
        S2 = self._S2
        sd2 = self._sigma2
        rho = self._rho
        dt = self._t / n
        r = self._r
        b = self._b

        pm = _np.array([
            [1,4,1],
            [4,16,4],
            [1,4,1]
        ])/36

        sdp = sd1 * sd1 + sd2 * sd2
        # fmt: off
        lam1 = 0.5 * (sdp + _np.sqrt(sdp * sdp - 4 * (1 - rho * rho) * sd1 * sd1 * sd2 * sd2))
        lam2 = 0.5 * (sdp - _np.sqrt(sdp * sdp - 4 * (1 - rho * rho) * sd1 * sd1 * sd2 * sd2))
        the = _np.arctan((lam1-sd1*sd1)/(rho*sd1*sd2))
        Df = _np.exp(b-r*dt)
        h = _np.exp(-1*_np.sin(the)*_np.sqrt(lam2)* _np.sqrt(3 * dt) - 0.5*sd1*sd1*dt)
        v = _np.exp(_np.cos(the)*_np.sqrt(lam1)* _np.sqrt(3 * dt) - 0.5*sd1*sd1*dt)

        S1_T = S1*self._get_gauss(h,v,n)

        h = _np.exp(_np.cos(the)*_np.sqrt(lam2)* _np.sqrt(3 * dt) - 0.5*sd2*sd2*dt)
        v = _np.exp(_np.sin(the)*_np.sqrt(lam1)* _np.sqrt(3 * dt) - 0.5*sd2*sd2*dt)

        S2_T = S2*self._get_gauss(h,v,n)

        OptionValue = _np.maximum(0,((S1_T-S2_T)-self._K)*z)

        OptionValue = self._euro(OptionValue, n, Df, pm)

        OptionValue = _np.sum(_np.concatenate(_np.multiply(OptionValue[-1][n-1:n+2, n-1:n+2],pm)))*Df

        return OptionValue
        
    def _euro(self, OptionValue, n, Df, pm):
        tr = OptionValue.copy()
        old = tr.copy()

        tr = list()

        # step back in time from t=T to t=0 
        for j in _np.arange(1,n):

            # create new smaller matrix for expected value cale from passed, larger matrix
            latest = _np.zeros((2*(n-j)+1,2*(n-j)+1))
            latest = self._calc_nodes(old, latest, Df, pm)

            old = latest.copy()
            # create new matrix of same size as original matrix with new matrix
            # centered inside and stack
            latest = _np.pad(latest, (j, j))

            tr.append(latest)

        return tr

    def _calc_nodes(self, tr, ntr, Df, pm):
        # loop through elements 1 element away from the edge and calc
        # weighted sum
        for i in range(1,tr.shape[0]-1):
            for j in range(1,tr.shape[0]-1):
                ntr[i-1,j-1] = _np.sum(_np.concatenate(_np.multiply(tr[i-1:i+2, j-1:j+2],pm)))*Df
        return ntr





    def _get_gauss(self, h, v, n):
        """
        Calculate the joint innovation for one of the assets
        """
        # fmt: off
        OptionValue1 = _np.power(h, _np.arange(0, 2 * n + 1)) \
                            * (1 / h) ** _np.arange(2 * n, -1, -1)
        OptionValue1 = _np.broadcast_to(OptionValue1, (2 * n + 1, 2 * n + 1))

        OptionValue2 = _np.power(v, _np.arange(0, 2 * n + 1)) \
                            * (1 / v) ** _np.arange(2 * n, -1, -1)
        OptionValue2 = _np.broadcast_to(_np.flip(OptionValue2), (2 * n + 1, 2 * n + 1)).T
        # fmt: on

        OptionValue = OptionValue2 * OptionValue1

        return OptionValue


if __name__ == "__main__":

    opt = TrinomialSpreadOption(70, 60, 0, 1 / 12, 0.03, 0.03, 0.2, 0.1, 0.5, n=100)
    ret = opt._calc_price(1, type="euro", tree=False)

    print(ret)
