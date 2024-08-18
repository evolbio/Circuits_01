module Data
using DifferentialEquations, Plots
export gbm, ema, ema!, rw

ema!(data, alpha) = ema(data, alpha; in_place = true)

function ema(data, alpha; in_place = false)
    y = in_place ? data : deepcopy(data)
    n = length(y)
    result = zeros(n)
    result[1] = y[1]
    for i = 2:n
        y[i] = alpha * y[i] + (1 - alpha) * y[i-1]
    end
    return y
end

# random walk
function rw(T; sigma = 0.2, saveat = 0.1, alpha = 1, norm_rng = true)
    prob = SDEProblem((u, p, t) -> 0, (u, p, t) -> sigma, 0, (0, T))
    sol = solve(prob, EM(), dt = saveat / 10; saveat = saveat)
    u = norm_rng ? normalize!(sol.u) : sol.u
    return alpha == 1 ? u : ema!(u, alpha), sol.t
end

# Discrete approximation
function geom_brownian_ema(T, dt; mu=0.05, sigma=0.2, init=32.0, theta=log2(init),
				kappa=0.3, alpha=1.0)
    n = Int(round(T / dt))
    dW = sqrt(dt) * randn(n)
    data = zeros(n)
    data[1] = init
    for i = 2:n
        data[i] =
            data[i-1] +
            kappa * (theta - log2(data[i-1])) * data[i-1] * dt +
            mu * data[i-1] * dt +
            sigma * data[i-1] * dW[i]
    end
    data .= log2.(data)
    for i = 2:n
        data[i] = alpha * data[i] + (1 - alpha) * data[i-1]
    end
    data
end

# DiffEq version
# mean is mu*2^(theta/mu), or on log2 scale, log2(mu) + theta/mu
# theta = 0 implies no mean reversion, 

function drift(u, p, t)
    theta, kappa, mu, sigma = p
    mu + kappa * (log2(theta) - log2(u))
end

function diffusion(u, p, t)
    sigma = p[4]
    sigma * u
end

# log2(mean) = theta + mu/kappa
# Params: reversion target theta, reversion speed kappa, drift mu, diffusion sigma
function gbm(T; theta=32.0, kappa=0.3, mu=0.05, sigma=0.2, dt=0.01,
				saveat=0.1, alpha=1)
    params = (theta, kappa, mu, sigma)
    u0 = theta
    tspan = (0.0, T)
    prob = SDEProblem(drift, diffusion, u0, tspan, params)
    sol = solve(prob, EM(), dt = dt, saveat = saveat)
    u = log2.(sol.u)
    return alpha == 1 ? u : ema(u, alpha), sol.t
end

function normalize!(x; low = 0, high = 1)
    x .= ((high - low) / (maximum(x) - minimum(x)) .* x)
    x .= x .+ (low - minimum(x))
end

end # module Data
