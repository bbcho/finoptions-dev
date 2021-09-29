import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo

S = 100
K = 110
t = 0.5
r = 0.1
b = 0.1
sigma = 0.27
n = 30


def test_trinomial_options():

    opt = fo.binomial_tree_options.TrinomialTreeOption(
        S=S, K=K, t=t, r=r, b=b, sigma=sigma, n=n, type="american"
    )

    test = opt.put()
    assert np.allclose(test, 11.64931), "Trinomial Tree American Put Failed"

    test = opt.call()
    assert np.allclose(test, 5.657877), "Trinomial Tree American Call Failed"

    opt = fo.binomial_tree_options.TrinomialTreeOption(
        S=S, K=K, t=t, r=r, b=b, sigma=sigma, n=n, type="european"
    )

    test = opt.put()
    assert np.allclose(test, 10.29311), "Trinomial Tree European Put Failed"

    test = opt.call()
    assert np.allclose(test, 5.657877), "Trinomial Tree European Call Failed"
