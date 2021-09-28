import sys
import os
import numpy as np
import pandas as pd


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo

ts = pd.read_csv("./pytest/garch_ts.csv").x
ts = np.array(ts)

lamb = 4
omega = 8e-5
alpha = 6e-5
beta = 0.7
gamma = 0
rf = 0

rfr = rf

par_omega = -np.log((1 - omega) / omega)
par_alpha = -np.log((1 - alpha) / alpha)
par_beta = -np.log((1 - beta) / beta)

trace = False
symmetric = True


def test_llhHNGarch():
    opt = fo.heston_nandi_options._llhHNGarch(
        [lamb, par_omega, par_alpha, par_beta, 0], trace, symmetric, rfr, ts, True
    )
    assert np.allclose(opt.llhHNGarch, -1226.474), "HNGarch test 1 failed"


def test_hngarchSim():
    ts = np.genfromtxt("./pytest/ts.csv")[1:]
    inno = np.genfromtxt("./pytest/inno.csv")[1:]
    start_inno = np.genfromtxt("./pytest/start_inno.csv")[1:]

    x = fo.heston_nandi_options.hngarch_sim(
        lamb=4,
        omega=8e-5,
        alpha=6e-5,
        beta=0.7,
        gamma=0,
        rf=0,
        n=500,
        n_start=100,
        inno=inno,
        inno_start=start_inno,
    )

    assert np.allclose(ts, x), "HNGarchSim test 1 failed"

    x = fo.heston_nandi_options.hngarch_sim(
        lamb=4, omega=8e-5, alpha=6e-5, beta=0.7, gamma=0, rf=0, n=500
    )

    assert x.shape[0] == 500, "HNGarchSim test 2 failed"
