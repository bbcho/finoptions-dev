if __name__ == "__main__":

    import finoptions as fo
    import numdifftools as nd

    opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    # print(opt.delta())

    # opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    # # # print(opt.greeks())

    # # vol = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)

    # # # print(vol.volatility(3).root)

    # # # opt.summary()

    # # # print(opt)

    # print(opt.__repr__())

    # opt = fo.BlackScholesOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    # # print(opt.greeks())

    # vol = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)

    # # print(vol.volatility(3).root)

    # # opt.summary()

    # # print(opt)

    # print(opt.__repr__())

    # opt = fo.Black76Option(10.0, 8.0, 1.0, 0.02, 0.1)

    # # print(opt.greeks())

    # vol = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)

    # # print(vol.volatility(3).root)

    # # opt.summary()

    # # print(opt)

    # print(opt.__repr__())

    # import numpy as np

    # opt = fo.MiltersenSchwartzOption(
    #     Pt=np.exp(-0.05 / 4),
    #     FT=95,
    #     K=80,
    #     t=1 / 4,
    #     T=1 / 2,
    #     sigmaS=0.2660,
    #     sigmaE=0.2490,
    #     sigmaF=0.0096,
    #     rhoSE=0.805,
    #     rhoSF=0.0805,
    #     rhoEF=0.1243,
    #     KappaE=1.045,
    #     KappaF=0.200,
    # )
    # # print(opt.call())

    # # print(opt.get_params())

    # opt.summary()
