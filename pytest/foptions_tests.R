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
