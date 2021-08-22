library(fOptions)

GBSOption(c('c'), 10, 8, 1, 0.02, 0.01, 0.1)

GBSOption(c('p'), 10, 8, 1, 0.02, 0.01, 0.1)

# S = 10
# X = 8
# Time = 1
# r = 0 
# b = 0
# sigma = 0.1
# 
# d1 = ( log(S/X) + (b+sigma*sigma/2)*Time ) / (sigma*sqrt(Time))

p <- GBSOption(c('p'), 10, 8, 1, 0.02, 0.01, 0.1)

gk <- c('Delta','Theta','Vega','Rho','Lambda','Gamma','CofC')

for (g in gk) {
  put <- GBSGreeks(Selection = g, TypeFlag = c('c'),  10, 8, 1, 0.02, 0.01, 0.1)
  call <- GBSGreeks(Selection = g, TypeFlag = c('p'),  10, 8, 1, 0.02, 0.01, 0.1)
  
  print(paste(g, round(put,6)))
  print(paste(g, round(call,6)))
}

GBSGreeks(Selection = "Delta", TypeFlag = c('c'),  10, 8, 1, 0.02, 0.01, 0.1)
GBSGreeks(Selection = "Delta", TypeFlag = c('p'),  10, 8, 1, 0.02, 0.01, 0.1)

GBSGreeks(Selection = "Theta", TypeFlag = c('c'),  10, 8, 1, 0.02, 0.01, 0.1)
GBSGreeks(Selection = "Delta", TypeFlag = c('p'),  10, 8, 1, 0.02, 0.01, 0.1)

GBSVolatility(2.5, c('c'), 10, 8, 1, 0.02, 0.01)
GBSVolatility(2.5, c('p'), 10, 8, 1, 0.02, 0.01)


Black76Option(c('c'), 10, 8, 1, 0.02, 0.1)


MiltersenSchwartzOption(TypeFlag = "c", Pt = exp(-0.05/4), FT = 95,X = 80, time = 1/4, Time = 1/2, sigmaS = 0.2660, sigmaE = 0.2490,sigmaF = 0.0096, rhoSE = 0.805, rhoSF = 0.0805, rhoEF = 0.1243,KappaE = 1.045, KappaF = 0.200)
MiltersenSchwartzOption(TypeFlag = "p", Pt = exp(-0.05/4), FT = 95,X = 80, time = 1/4, Time = 1/2, sigmaS = 0.2660, sigmaE = 0.2490,sigmaF = 0.0096, rhoSE = 0.805, rhoSF = 0.0805, rhoEF = 0.1243,KappaE = 1.045, KappaF = 0.200)


RollGeskeWhaleyOption(S = 80, X = 82, Time2 = 1/3, time1 = 1/4, r = 0.06, D = 4, sigma = 0.30)

BAWAmericanApproxOption(TypeFlag = "c", S = 100,X = 100, Time = 0.5, r = 0.10, b = 0, sigma = 0.25)
BAWAmericanApproxOption(TypeFlag = "p", S = 100,X = 100, Time = 0.5, r = 0.10, b = 0, sigma = 0.25)

.bawKp(X=100, Time=0.5, r=0.1, b=0, sigma=0.25) 

BAWAmPutApproxOptiontest <- function(S, X, Time, r, b, sigma) 
  {
    # Internal Function - The Put:
    
    # Compute:
    Sk = .bawKp(X, Time, r, b, sigma)
    n = 2*b/sigma^2
    K = 2*r/(sigma^2*(1-exp(-r*Time)))
    d1 = (log(Sk/X)+(b+sigma^2/2)*Time)/(sigma*sqrt(Time))
    Q1 = (-(n-1)-sqrt((n-1)^2+4*K))/2
    a1 = -(Sk/Q1)*(1-exp((b-r)*Time)*CND(-d1))
    if(S > Sk) {
      result = GBSOption("p", S, X, Time, r, b, sigma)@price + a1*(S/Sk)^Q1 
    } else {
      result = X-S 
    }  
    
    # Return Value:
    GBSOption("p", S, X, Time, r, b, sigma)
  }
BAWAmPutApproxOptiontest( S = 100,X = 100, Time = 0.5, r = 0.10, b = 0, sigma = 0.25)

CRRBinomialTreeOption(TypeFlag = "ce", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
CRRBinomialTreeOption(TypeFlag = "pe", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
CRRBinomialTreeOption(TypeFlag = "ca", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
CRRBinomialTreeOption(TypeFlag = "pa", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)


JRBinomialTreeOption(TypeFlag = "ce", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
JRBinomialTreeOption(TypeFlag = "pe", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
JRBinomialTreeOption(TypeFlag = "ca", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
JRBinomialTreeOption(TypeFlag = "pa", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)

TIANBinomialTreeOption(TypeFlag = "ce", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
TIANBinomialTreeOption(TypeFlag = "pe", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
TIANBinomialTreeOption(TypeFlag = "ca", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)
TIANBinomialTreeOption(TypeFlag = "pa", S = 50, X = 40,Time = 5/12, r = 0.1, b = 0.1, sigma = 0.4, n = 5)

