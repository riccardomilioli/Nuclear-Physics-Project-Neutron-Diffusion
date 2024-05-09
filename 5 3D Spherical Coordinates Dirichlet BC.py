import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# We want to solve the 3D diffusion equation for a neutron in spherical coordinates with Dirichlet boundary conditions.

# Initial condition for the function that enters the integral for a_p:
def f(r, r1):
    return 1 - (r / r1)**2

# Parameters of the problem:
rho = 18.71e6  # [g/m^3]. Density
r1_val = 0.115  # [m]. We want criticality so we work with R > Rcrit (see below)
N = 30 # Number of a_p that are computed
mu_val = 2.3446e5 # [m^2/s]. Diffusion constant
eta_val = 1.8958e8 # [1/s]. Neutron diffusion rate 

# Computig the critical radius, volume and mass:
Rcrit = np.pi * np.sqrt(mu_val / eta_val)
print(f"The critical radius is {Rcrit} m")

Vcrit = 4/3 * np.pi * Rcrit**3
print(f"The critical volume is {Vcrit} m^3")

Mcrit = rho * Vcrit
print(f"The critical mass is {Mcrit} g")

# Calculation of the integral to get a_p.
# First define a function to be integrated:
def integrand(r, p):
    return r * f(r, r1_val) * np.sin(p * np.pi * r / r1_val)

# Computing values for a_p.
# Results are different from the one on paper but Mathematica agrees with Python.

aa_vals = np.zeros(N)
for p in range(1, N + 1):
    integral_result, _ = quad(integrand, 0, r1_val, args=(p,))
    aa_vals[p - 1] = 2 / (r1_val) * integral_result
    print(f"p = {p}, a({p}) = {aa_vals[p-1]}") 

# Solving the diffusion equation:
def n_func(r, t):
    n = np.zeros_like(r) # Initializing the result
    for p in range(1, N + 1):
        n += (aa_vals[p - 1] / r) * np.exp(eta_val * t - mu_val * ((p) * np.pi / r1_val)**2 * t) * np.sin((p) * np.pi * r / r1_val)
    return n

# Plotting the solution:
r_vals = np.linspace(-r1_val, r1_val, 100)
t_vals = np.linspace(0, 2e-6, 100)
R, T = np.meshgrid(r_vals, t_vals)
N_vals = n_func(R, T)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T, N_vals, cmap='viridis')
ax.set_title("Neutron diffusion, r=11.5 cm, N=30")
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_zlabel("n(t,r)", fontsize = 10, labelpad = 0.1)
plt.xticks(rotation = 10)
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.show()

# Plotting the error for t=0:
error_vals = f(r_vals, r1_val) - n_func(r_vals, 0)
plt.plot(r_vals, error_vals)
plt.title("Error Plot: Neutron diffusion at t=0")
plt.xlabel("r")
plt.ylabel("Error")
plt.show()
