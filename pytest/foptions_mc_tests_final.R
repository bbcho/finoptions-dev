library(fOptions)

## How to perform a Monte Carlo Simulation?
## First Step:
# Write a function to generate the option's innovations.
# Use scrambled normal Sobol numbers:
sobolInnovations <- function(mcSteps, pathLength, init, ...) {
  # Create and return Normal Sobol Innovations:
  rnorm.sobol(mcSteps, pathLength, init, ...)
}
## Second Step:
# Write a function to generate the option's price paths.
# Use a Wiener path:
wienerPath <- function(eps) {
  # Note, the option parameters must be globally defined!
  
  # Generate and return the Paths:
  (b-sigma*sigma/2)*delta.t + sigma*sqrt(delta.t)*eps
}
## Third Step:
# Write a function for the option's payoff
# Example 1: use the payoff for a plain Vanilla Call or Put:
plainVanillaPayoff <- function(path) {
  # Note, the option parameters must be globally defined!
  # Compute the Call/Put Payoff Value:
  ST <- S*exp(sum(path))
  if (TypeFlag == "c") payoff <- exp(-r*Time)*max(ST-X, 0)
  if (TypeFlag == "p") payoff <- exp(-r*Time)*max(0, X-ST)
  # Return Value:
  payoff
}
# Example 2: use the payoff for an arithmetic Asian Call or Put:
arithmeticAsianPayoff <- function(path) {
  # Note, the option parameters must be globally defined!
  # Compute the Call/Put Payoff Value:
  SM <- mean(S*exp(cumsum(path)))
  if (TypeFlag == "c") payoff <- exp(-r*Time)*max(SM-X, 0)
  if (TypeFlag == "p") payoff <- exp(-r*Time)*max(0, X-SM)
  # Return Value:
  payoff
}
## Final Step:
# Set Global Parameters for the plain Vanilla / arithmetic Asian Options:
TypeFlag <- "c"; S <- 100; X <- 100
Time <- 1/12; sigma <- 0.4; r <- 0.10; b <- 0.1
# Do the Asian Simulation with scrambled random numbers:
mc <- MonteCarloOption(delta.t = 1/360, pathLength = 30, mcSteps = 5000,
                       mcLoops = 50, init = TRUE, innovations.gen = sobolInnovations,
                       path.gen = wienerPath, payoff.calc = arithmeticAsianPayoff,
                       antithetic = TRUE, standardization = FALSE, trace = TRUE,
                       scrambling = 2, seed = 4711)
