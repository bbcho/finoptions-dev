library(fOptions)

# Monte Carlo Sims

pathLength <- 30
mcSteps <- 5000
mcLoops <- 50
delta.t = 1/360
TypeFlag <- "c"; S <- 100; X <- 100
Time <- 1/12; sigma <- 0.4; r <- 0.10; b <- 0.1


## First Step:
# Write a function to generate the option's innovations.
# Use scrambled normal Sobol numbers:
sobolInnovations <- function(mcSteps, pathLength, init, ...) {
  # Create and return Normal Sobol Innovations:
  runif.sobol(mcSteps, pathLength, init, ...)
}
inno <- sobolInnovations(mcSteps, pathLength, init=TRUE)
inno

# save to csv to test vs python. Python seems to generate different 
# sobol sequences, even when unscrambled...
write.table(inno, "sobol.csv", row.names = FALSE, sep=",",  col.names=FALSE)
inno <-  read.csv("sobol.csv", header=FALSE)

# scrambled innos
inno <- sobolInnovations(mcSteps, pathLength, init=TRUE, scrambling=2)

write.table(inno, "sobol_scrambled.csv", row.names = FALSE, sep=",",  col.names=FALSE)
inno <-  read.csv("sobol.csv", header=FALSE)




# Second Step:
# Write a function to generate the option's price paths.
# Use a Wiener path:
wienerPath <- function(eps) {
  # Note, the option parameters must be globally defined!
  # Generate and return the Paths:
  (b-sigma*sigma/2)*delta.t + sigma*sqrt(delta.t)*eps
}

wienerPath(inno) %>% write.table("wiener.csv", row.names = FALSE, sep=",",  col.names=FALSE)

wp <- wienerPath(inno)

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


plainVanillaPayoff(t(wp[1,]))
plainVanillaPayoff(t(wp[10,]))

TypeFlag <- "p"
plainVanillaPayoff(t(wp[1,]))
plainVanillaPayoff(t(wp[10,]))

X <- 140
plainVanillaPayoff(t(wp[1,]))
plainVanillaPayoff(t(wp[10,]))

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
TypeFlag <- "c"
X <- 100

arithmeticAsianPayoff(t(wp[1,]))
arithmeticAsianPayoff(t(wp[10,]))

TypeFlag <- "p"
arithmeticAsianPayoff(t(wp[1,]))
arithmeticAsianPayoff(t(wp[10,]))

X <- 140
arithmeticAsianPayoff(t(wp[1,]))
arithmeticAsianPayoff(t(wp[10,]))

## Final Step:
# Set Global Parameters for the plain Vanilla / arithmetic Asian Options:

pathLength <- 30
mcSteps <- 5000
mcLoops <- 50
delta.t = 1/360
TypeFlag <- "c"; S <- 100; X <- 100
Time <- 1/12; sigma <- 0.4; r <- 0.10; b <- 0.1

MonteCarloOption = function(delta.t, pathLength, mcSteps, mcLoops, 
                            init = TRUE, innovations.gen, path.gen, payoff.calc, antithetic = TRUE, 
                            standardization = FALSE, trace = TRUE, ...)
{   
  
  # Monte Carlo Simulation:
  delta.t <<- delta.t
  if (trace) cat("\nMonte Carlo Simulation Path:\n\n")
  iteration = rep(0, length = mcLoops)
  # MC Iteration Loop:
  cat("\nLoop:\t", "No\t")
  for ( i in 1:mcLoops ) {
    if ( i > 1) init = FALSE
    # Generate Innovations:
    eps =  innovations.gen(mcSteps, pathLength, init = init, ...)
    # Use Antithetic Variates if requested:
    if (antithetic) 
      eps = rbind(eps, -eps)
    # Standardize Variates if requested:
    if (standardization) eps = 
      (eps-mean(eps))/sqrt(var(as.vector(eps)))
    # Calculate for each path the option price:
    path = t(path.gen(eps))
    payoff = NULL
    k = 0
    
    for (j in 1:dim(path)[2])
      payoff = c(payoff, payoff.calc(path[, j]))
    iteration[i] = mean(payoff)
    # Trace the Simualtion if desired:
    if (trace) 
      cat("\nLoop:\t", i, "\t:", iteration[i], sum(iteration)/i ) 
  }
  if (trace) cat("\n")
  
  # Return Value:
  iteration
}



# Do the Asian Simulation with scrambled random numbers:
mc <- MonteCarloOption(delta.t = delta.t, pathLength = pathLength, mcSteps = mcSteps,
                       mcLoops = mcLoops, init = TRUE, innovations.gen = sobolInnovations,
                       path.gen = wienerPath, payoff.calc = plainVanillaPayoff,
                       antithetic = FALSE, standardization = FALSE, trace = TRUE,
                       scrambling = 2, seed = 4711)
mc
dim(mc)
mc[,1]
