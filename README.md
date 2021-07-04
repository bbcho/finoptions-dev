# energyderivatives

Python implementation of the R package fOptions for use in energy trading. Improvements include coverting the package to OOP as well as Finite Difference Methods for Option greeks for all Options.

To install package run:

```
pip install energyderivatives
```

## Working with energyderivatives

Vanilla Options are found at the root of the package. For example, to run a Generalized Black Scholes Option:

```python
import energyderivatives as ed

opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

opt.call() # to get call price
opt.put() # to get put price
opt.summary() # for a printed summary of the option
opt.greeks() # to get the greeks for the option
```

All options follow the same format for calls, puts, greeks and summaries. GBSOption uses the analytic solution to calculate to the greeks, but for all other options the finite difference method is used.