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
