import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# We want to solve the 3D diffusion equation for a neutron in spherical coordinates with Neumann boundary conditions.

# Parameters of the problem: 
lambda_t = 3.6  # [cm]. Transport free path
rho = 18.71  # [g/cm^3]. Density
mu_val = 2.3446e9 # [cm^2/s]. Diffusion constant
eta_val = 1.8958e8 # [1/s]. Neutron diffusion rate 

# Define the cotangent function. Needed to compute Rcrit, which means alpha = 0:
def cot(x):
    return 1 / np.tan(x)

def eqn_rcrit(r): 
    return -1 + r * np.sqrt(eta_val/ mu_val) * cot(r * np.sqrt(eta_val / mu_val)) + 3 / (2 * lambda_t) * r

# Guess to find the solution of equation_cot:
guess = 8 # Followed suggestion from paper

# Solving numerically to find the critical radius and hence critical volume and mass
Rcrit = (fsolve(eqn_rcrit, guess)).item() # The zero closest to "guess" is found
                                             
print(f"The critical radius is {Rcrit} cm")

Vcrit = (4 / 3) * np.pi * Rcrit ** 3
print(f"The critical volume is {Vcrit} cm^3")

Mcrit = rho * Vcrit  
print(f"The critical mass is {Mcrit} g")

# We want supercriticality so we work with R > Rcrit (see below). This implies a non-zero alpha:
R0 = 8.5  # [cm]

# To find such alpha we go back to "equation_cot(r)" but now the variable is alpha:
def eqn_alpha(x):
    return -1 + R0 * np.sqrt((eta_val + x)/mu_val) * cot(R0 * np.sqrt((eta_val + x)/mu_val)) + 3/2 * R0/lambda_t

guess2 = 1 
alpha = fsolve(eqn_alpha, guess2)[0] # Different from the one on paper 

# Constants needed to define the neutron density:
k = np.sqrt((eta_val + alpha) / mu_val)
A = R0 / np.sin(k*R0) # Follows from imposing an initial condition on n(r, t): n(R0, 0)=1 (see below)

# Defining the neutron density n(r, t):
def n(r, t):
    return A * np.exp(-alpha * t) * np.sin(k * r) / r

# Plotting the equation for alpha:
x_vals = np.linspace(-475e4, -450e4, 100000)
eqn_alpha_vals = eqn_alpha(x_vals)
fig = plt.figure()
plt.plot(x_vals, eqn_alpha_vals, label="f(alpha)")
plt.title("Equation for alpha")
plt.xlabel("alpha")
plt.ylabel("f(alpha)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting the solution at t=0
r_vals = np.linspace(-R0, R0, 1000)
n_vals_t0 = n(r_vals, 0)
plt.plot(r_vals, n_vals_t0, linewidth=2)
plt.title("Neutron diffusion, R=8.5 cm, t=0")
plt.xlabel("r")
plt.ylabel("n(r, t=0)")
plt.grid(True)
plt.show()

# Plotting 3D surface:
t_vals = np.linspace(0, 3e-6, 100)
R, T = np.meshgrid(r_vals, t_vals)
N_vals = n(R, T)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T, N_vals, cmap='viridis')
ax.set_title("Neutron diffusion, R=8.5 cm")
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_zlabel("n(r, t)", fontsize = 9, labelpad = 0.1)
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.show()
