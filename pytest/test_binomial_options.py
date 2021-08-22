import sys
import os
import numpy as np


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
