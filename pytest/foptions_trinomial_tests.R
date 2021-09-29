# Exercise: Trinomial Tree

#   Write a function to compute the call and put price of an 
#   European or American style option using a trinomial tree
#   approach.

# For original code see here:
# https://r-forge.r-project.org/scm/viewvc.php/pkg/fOptions/demo/xmpDWChapter081.R?view=markup&root=rmetrics&pathrev=130

TrinomialTreeOption  = 
  function(AmeEurFlag, CallPutFlag, S, X, Time, r, b, sigma, n)
  {   # A function implemented by Diethelm Wuertz           
    
    # Description:
    #   Calculates option prices from the Trinomial tree model.
    
    # Arguments:
    #   AmeEurFlag - a character value, either "a" or "e" for 
    #       a European or American style option
    #   CallPutFlag - a character value, either "c" or "p" for 
    #       a call or put option
    #   S, X, Time, r, b, sigma - the usual option parameters
    #   n - an integer value, the depth of the tree
    
    # Value:
    #   Returns the price of the options.
    
    # Details:
    #   Trinomial trees in option pricing are similar to
    #   binomial trees. Trinomial trees can be used to 
    #   price both European and American options on a single 
    #   underlying asset.
    #   Because the asset price can move in three directions 
    #   from a given node, compared with only two in a binomial
    #   tree, the number of time steps can be reduced to attain
    #   the same accuracy as in the binomial tree. 
    
    # Reference:
    #   E.G Haug, The Complete Guide to Option Pricing Formulas
    #   Chapter 3.2
    
    # FUNCTION:
    
    # Settings:            
    OptionValue  =  rep(0, times=2*n+1)  
    
    # Call-Put Flag:
    if (CallPutFlag == "c") z  =  +1 
    if (CallPutFlag == "p") z  =  -1  
    
    # Time Interval: 
    dt  =  Time/n
    
    # Up-and-down jump sizes:
    u  =  exp(+sigma * sqrt(2*dt))
    d  =  exp(-sigma * sqrt(2*dt)) 
    
    # Probabilities of going up and down:  
    pu  =  ((exp(b * dt/2) - exp( -sigma * sqrt(dt/2))) / 
              (exp(sigma * sqrt(dt/2)) - exp(-sigma * sqrt(dt/2)))) ^ 2
    pd  =  (( exp(sigma * sqrt(dt/2)) - exp( b * dt/2)) / 
              (exp(sigma * sqrt(dt/2)) - exp(-sigma * sqrt(dt/2)))) ^ 2
    
    # Probability of staying at the same asset price level:
    pm  =  1 - pu - pd
    Df  =  exp(-r*dt)   
    for (i in 0:(2*n)) {
      OptionValue[i+1]  =  max(0, z*(S*u^max(i-n, 0) * 
                                       d^max(n*2-n-i, 0) - X))}
    
    for (j in (n-1):0) {
      print(j)
      for (i in 0:(j*2)) {
        # European Type:
        if (AmeEurFlag == "e") {
          OptionValue[i+1]  =  (
            pu * OptionValue[i+3] + 
              pm * OptionValue[i+2] + 
              pd * OptionValue[i+1]) * Df }
        # American Type:
        if (AmeEurFlag == "a") {
          a <- (z*(S*u^max(i-j, 0) * 
                     d ^ max(j*2-j-i, 0) - X))
          b <- (
            pu * OptionValue[i+3] + 
              pm * OptionValue[i+2] + 
              pd * OptionValue[i+1]) * Df
          
          OptionValue[i+1]  =  max(a, b)
          } } }
    TrinomialTree  =  OptionValue[1]
    
    # Return Value:
    TrinomialTree
  }

# Example:  
TrinomialTreeOption(AmeEurFlag = "a", CallPutFlag = "p", S = 100, 
                    X = 110, Time = 0.5, r = 0.1, b = 0.1, sigma = 0.27, n = 30)

TrinomialTreeOption(AmeEurFlag = "a", CallPutFlag = "c", S = 100, 
                    X = 110, Time = 0.5, r = 0.1, b = 0.1, sigma = 0.27, n = 30)

TrinomialTreeOption(AmeEurFlag = "e", CallPutFlag = "p", S = 100, 
                    X = 110, Time = 0.5, r = 0.1, b = 0.1, sigma = 0.27, n = 30)

TrinomialTreeOption(AmeEurFlag = "e", CallPutFlag = "c", S = 100, 
                    X = 110, Time = 0.5, r = 0.1, b = 0.1, sigma = 0.27, n = 30)

TrinomialTreeOption(AmeEurFlag = "a", CallPutFlag = "c", S = 100, 
                    X = 100, Time = 3, r = 0.03, b = -0.04, sigma = 0.2, n = 9)

