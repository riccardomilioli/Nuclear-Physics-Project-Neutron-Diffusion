import numpy as np
import matplotlib.pyplot as plt

# We want to solve the 1D diffusion equation for a neutron in cartesian coordinates with Dirichlet boundary conditions.

# Gaussian initial condition for the function f that enters the integral for a_p:
def f(x, L):
    return np.exp(-lambda_val * ((x - L / 2) / (L / 2)) ** 2)

# Parameters of the problem:
L_val = 0.111 # [m]. We want supercriticality so we work with L > Lcrit (see below)
lambda_val = 100
mu_val = 2.3446e+05 # [m^2/s]. Diffusion constant
eta_val = 1.8958e+08 # [1/s]. Neutron diffusion rate 
N = 30 # Number of a_p that are computed

# Computing the critical lenght:
Lcrit = np.pi * np.sqrt(mu_val / eta_val)
print(f"The critical lenght is {Lcrit} m")

# Calculation of the integral to find a_p. 
# First initialize: 
x_vals = np.linspace(0, L_val, 1000)
p_vals = np.arange(1, N + 1)
ap_vals = np.zeros(N)

# Computing and printing a_p:
for i, p_val in enumerate(p_vals): 
    integrand = f(x_vals, L_val) * np.sin(p_val * np.pi * x_vals / L_val)
    ap_vals[i] = 2 / L_val * np.trapz(integrand, x_vals)
for i, p_val in enumerate(p_vals[:N]): # Print a_p up to p = 30
    if p_val % 2 != 0: # We only print odd values, the even ones are all zero
        print(f"p = {p_val}: a({p_val}) = {ap_vals[i]}")

# Solving the diffusion equation:
def n_func(x, t):
    n = np.zeros_like(x) # Initializing the result 
    for i, p_val in enumerate(p_vals):
        n += ap_vals[i] * np.exp(eta_val * t - mu_val * (p_val * np.pi / L_val) ** 2 * t) * np.sin(p_val * np.pi * x / L_val)
    return n


# Plotting the solution:
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
x_vals_plot = np.linspace(0, L_val, 100)
t_vals_plot = np.linspace(0, 2e-5, 100)
X, T = np.meshgrid(x_vals_plot, t_vals_plot)
N_vals = n_func(X, T)
ax.plot_surface(X, T, N_vals, cmap='viridis')
ax.set_title("Neutron diffusion for L=11.1 cm, N=30")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("n(t, x)", fontsize = 10, labelpad = 0.1)
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.show()

# Computing errors at t=0:
t_val = 0
n_t0_vals = n_func(x_vals_plot, t_val)
error_vals = n_t0_vals - f(x_vals_plot, L_val)

# Plotting errors at t=0:
fig_error = plt.figure()
plt.plot(x_vals_plot, error_vals, label="Error at t=0")
plt.title("Neutron diffusion error for t=0")
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()
