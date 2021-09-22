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
