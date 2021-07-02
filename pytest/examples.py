if __name__ == "__main__":

    import energyderivatives as ed

    opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    # print(opt.greeks())

    vol = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)

    # print(vol.volatility(3).root)

    # opt.summary()

    # print(opt)

    print(opt.__repr__())
