import sys
import os
import numpy as np
from matplotlib import figure


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import energyderivatives as ed


def test_CRR_Tree():

    opt = ed.binomial_tree_options.CRRBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="european"
    )

    assert np.allclose(
        opt.call(), 12.62964
    ), "CRRBinomialTreeOption call-euro price does not match fOptions. CRRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call() should equal 12.62964"
    assert np.allclose(
        opt.put(), 0.9972167
    ), "CRRBinomialTreeOption put-euro price does not match fOptions. CRRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put() should equal 0.9972167"

    opt = ed.binomial_tree_options.CRRBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="american"
    )

    assert np.allclose(
        opt.call(), 12.62964
    ), "CRRBinomialTreeOption call-amer price does not match fOptions. CRRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call(type='american') should equal 12.62964"
    assert np.allclose(
        opt.put(), 1.016134
    ), "CRRBinomialTreeOption put-amer price does not match fOptions. CRRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put(type='american') should equal 1.016134"

    assert isinstance(opt.plot(), figure.Figure), "Binomial tree plotter not working"


def test_JR_Tree():

    opt = ed.binomial_tree_options.JRBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="european"
    )

    assert np.allclose(
        opt.call(), 12.63021
    ), "JRBinomialTreeOption call-euro price does not match fOptions. JRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call() should equal 12.63021"
    assert np.allclose(
        opt.put(), 1.001478
    ), "JRBinomialTreeOption put-euro price does not match fOptions. JRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put() should equal 1.001478"

    opt = ed.binomial_tree_options.JRBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="american"
    )

    assert np.allclose(
        opt.call(), 12.63021
    ), "JRBinomialTreeOption call-amer price does not match fOptions. JRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call(type='american') should equal 12.63021"
    assert np.allclose(
        opt.put(), 1.021516
    ), "JRBinomialTreeOption put-amer price does not match fOptions. JRBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put(type='american') should equal 1.021516"


def test_TIAN_Tree():

    opt = ed.binomial_tree_options.TIANBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="european"
    )

    assert np.allclose(
        opt.call(), 12.36126
    ), "TIANBinomialTreeOption call-euro price does not match fOptions. TIANBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call() should equal 12.36126"
    assert np.allclose(
        opt.put(), 0.7288429
    ), "TIANBinomialTreeOption put-euro price does not match fOptions. TIANBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put() should equal 0.7288429"

    opt = ed.binomial_tree_options.TIANBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="american"
    )

    assert np.allclose(
        opt.call(), 12.36126
    ), "TIANBinomialTreeOption call-amer price does not match fOptions. TIANBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).call(type='american') should equal 12.36126"
    assert np.allclose(
        opt.put(), 0.7666983
    ), "TIANBinomialTreeOption put-amer price does not match fOptions. TIANBinomialTreeOption(S=50, K=40, t=5/12, r=0.1, b=0.1, sigma=0.4).put(type='american') should equal 0.7666983"


def test_CRR_tree():

    rm = np.array(
        [
            [12.62964, 17.695763, 23.976793, 31.360265, 39.68471, 49.065609],
            [0.00000, 7.627502, 11.528673, 16.781187, 23.32114, 30.699123],
            [0.00000, 0.000000, 3.739973, 6.315909, 10.33195, 16.120045],
            [0.00000, 0.000000, 0.000000, 1.151023, 2.28782, 4.547363],
            [0.00000, 0.000000, 0.000000, 0.000000, 0.00000, 0.000000],
            [0.00000, 0.000000, 0.000000, 0.000000, 0.00000, 0.000000],
        ]
    )
    print(rm)
    opt = ed.binomial_tree_options.CRRBinomialTreeOption(
        S=50, K=40, t=5 / 12, r=0.1, b=0.1, sigma=0.4, n=5, type="european"
    )

    assert np.allclose(
        rm, opt.call(tree=True)
    ), "CRRBionomialTreeOption matrix tree for a call option does not match fOptions"
