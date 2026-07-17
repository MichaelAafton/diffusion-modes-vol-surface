import numpy as np
rng = np.random.default_rng(0)

gamma, sigma, dt = 0.5, 1.0, 1.0      # one mode, medium spring
V = sigma**2 / (2*gamma)
a = np.exp(-gamma*dt)

f = rng.normal(0, np.sqrt(V), size=1_000_000)   # start in balance
eta = rng.normal(0, np.sqrt(V*(1 - a**2)), size=1_000_000)
f_new = f*a + eta

print("measured Var(Δf):", np.var(f_new - f))
print("formula 2V(1-a): ", 2*V*(1 - a))