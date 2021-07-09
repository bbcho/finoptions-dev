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
