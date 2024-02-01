# finoptions

Python implementation of the R package fOptions for use in energy trading. Changes include coverting the package to OOP as well as Finite Difference Methods for Option greeks for all Options.

To install package run:

```
pip install finoptions
```

## Working with finoptions

Vanilla Options are found at the root of the package. For example, to run a Generalized Black Scholes Option:

```python
import finoptions as fo

opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

opt.call() # to get call price
opt.put() # to get put price
opt.summary() # for a printed summary of the option
opt.greeks() # to get the greeks for the option

# to calculate implied volatility, omit the sigma argument and then 
# call the volatility method
opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)

opt.volatility(2)
```

All options follow the same format for calls, puts, greeks and summaries. GBSOption uses the analytic solution to calculate to the greeks, but for all other options the finite difference method is used.

## Calculating Options for Multiple Inputs

The vanilla options are capable of calculating calls, puts, vols and greeks for multiple inputs at the same time by passing numpy arrays of values as parameters. Currently this only works for the vanilla options.

```python
import finoptions as fo
import numpy as np

opt = fo.GBSOption(10.0, np.arange(5,15), 1.0, 0.02, 0.01, 0.1)

opt.call() # to get call price
opt.put() # to get put price
opt.summary() # for a printed summary of the option
opt.greeks() # to get the greeks for the option
```

## Implemented Methods

### Vanilla Options
- Black-Scholes-Merton Option
- Black 1976 Option
- Miltersen Schwartz Option

### American Options
- Roll-Geske-Whaley Calls on Dividend Paying Stocks
- Barone-Adesi and Whaley Approximation
- The Bjerksund and Stensland (1993) American Approximation Option

### Garch Options
- The Heston-Nandi Garch Option Model

### Tree Options
- Cox, Ross and Rubinstein (1979) Binomial Tree Model
- Jarrow and Rudd (1983) Binomial Tree Model
- Tian (1993) Binomial Tree Model
- Trinomial Tree Model

### Spread Options
- Rubinstein Binomial Tree Generic Spread Option Model

#### Bionomial Tree Spread Options
- Maximum Spread Option Model
- Minimum Spread Option Model
- Spread Option Model
- Dual-strike Option
- Reverse dual-strike option
- Portfolio options
- Options to exchange one asset for another
- Relative performance options
- Product options

### Monte Carlo Options
- Monte Carlo simulation framework (see example)

## Notebooks
To see example notebooks, please see github repo found here:

https://github.com/bbcho/finoptions-dev/tree/main/notebooks
