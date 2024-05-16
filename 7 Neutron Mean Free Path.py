import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

# We want to compute the mean free path of a single neutron crossing a slab of material.

# Parameters of the problem:
rho = 18.71 * 10**6  # [g/m^3]. Density
sigma = 1.235e-28  # [m^2]. Neutron cross-sectional area
n = (rho * 6.022e23 / 235.04)  # [1/m^3]. Number density of nuclei in the slab

# Calculating the mean free path:
xmfp_integrand = lambda x: x * sigma * n * np.exp(- sigma * n * x)
xmfp, _ = quad(xmfp_integrand, 0, math.inf) 

print(f"The mean free path is: {xmfp} m")

# Plotting the probabilities:
x_vals = np.linspace(0, 2, 100)
prob_direct_escape = np.exp(- sigma * n * x_vals)
prob_react = 1 - prob_direct_escape

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(x_vals, x_vals * prob_direct_escape * sigma * n)
plt.title('Mean free path distribution')
plt.xlabel('x')
plt.ylabel(r'$\sigma ~ n ~ x ~ e^{-\sigma n x}$') 

plt.subplot(312)
plt.plot(x_vals, prob_direct_escape)
plt.title('Probability of Direct Escape')
plt.xlabel('x')
plt.ylabel('Probability')

plt.subplot(313)
plt.plot(x_vals, prob_react)
plt.title('Probability of Reaction')
plt.xlabel('x')
plt.ylabel('Probability')

plt.tight_layout()
plt.show()
