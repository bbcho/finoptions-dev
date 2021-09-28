library(fOptions)

model = list(lambda = -0.5, omega = 2.3e-6, alpha = 2.9e-6,
             beta = 0.85, gamma = 184.25)
S = X = 100
Time.inDays = 252
r.daily = 0.05/Time.inDays
sigma.daily = sqrt((model$omega + model$alpha) /
                     (1 - model$beta - model$alpha * model$gamma^2))
data.frame(S, X, r.daily, sigma.daily)

.fstarHN(phi=20, const=0, model=model, S=S, X=X, Time.inDays = Time.inDays, r.daily = r.daily)


integrate(.fstarHN, 0, Inf, const = 1, model = model,
                  S = S, X = X, Time.inDays = Time.inDays, r.daily = r.daily)


HNGOption("c", model = model, S = S, X = X,
         Time.inDays = Time.inDays, r.daily = r.daily)

HNGOption("p", model = model, S = S, X = X,
          Time.inDays = Time.inDays, r.daily = r.daily)

HNGOption("c", model = model, S = S, X = 90,
          Time.inDays = Time.inDays, r.daily = r.daily)

HNGOption("p", model = model, S = S, X = 90,
          Time.inDays = Time.inDays, r.daily = r.daily)

HNGGreeks("Delta", "c", model = model, S = S, X = X,
          Time.inDays = Time.inDays, r.daily = r.daily)

HNGGreeks("Delta", "p", model = model, S = S, X = X,
          Time.inDays = Time.inDays, r.daily = r.daily)


HNGGreeks("Gamma", "c", model = model, S = S, X = X,
          Time.inDays = Time.inDays, r.daily = r.daily)

HNGGreeks("Gamma", "p", model = model, S = S, X = X,
          Time.inDays = Time.inDays, r.daily = r.daily)
.Machine$double.eps^0.25


#####

library(fOptions)


## hngarchSim -
# Simulate a Heston Nandi Garch(1,1) Process:
# Symmetric Model - Parameters:
model = list(lambda = 4, omega = 8e-5, alpha = 6e-5,
             beta = 0.7, gamma = 0, rf = 0)

n = 500
n.start = 100
inno = rnorm(n)
start.inno = rnorm(n.start)

write.csv(inno, file='inno.csv', row.names=FALSE)
write.csv(start.inno, file='start_inno.csv', row.names=FALSE)

ts = hngarchSim(model = model, n = 500, n.start = 100, inno=inno, start.innov = start.inno)
write.csv(ts, file='ts.csv', row.names=FALSE)



par(mfrow = c(2, 1), cex = 0.75)
ts.plot(ts, col = "steelblue", main = "HN Garch Symmetric Model")
grid()


mle = hngarchFit(x = ts, symmetric = TRUE)
mle


symmetric=TRUE
rfr = model$rf
lambda = model$lambda
omega = model$omega
alpha = model$alpha
beta = model$beta
gam = model$gamma

# Continue:
params = c(lambda = lambda, omega = omega, alpha = alpha,
           beta = beta, gamma = gam, rf = rfr)

# Transform Parameters and Calculate Start Parameters:
par.omega = -log((1-omega)/omega)  # for 2
par.alpha = -log((1-alpha)/alpha)  # for 3
par.beta = -log((1-beta)/beta)     # for 4
par.start = c(lambda, par.omega, par.alpha, par.beta)
if (!symmetric) par.start = c(par.start, gam)

# Initial Log Likelihood:
opt = list()
opt$value = .llhHNGarch(par = par.start,
                        trace = trace, symmetric = symmetric, rfr = rfr, x = x)
opt$estimate = par.start
if (trace) {
  print(c(lambda, omega, alpha, beta, gam))
  print(opt$value)
}

# Estimate Parameters:
opt = nlm(.llhHNGarch, par.start,
          trace = trace, symmetric = symmetric, rfr = rfr, x = x)


opt
