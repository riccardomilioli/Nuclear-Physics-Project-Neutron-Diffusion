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

def equation_cot(r): 
    return -1 + r * np.sqrt(eta_val/ mu_val) * cot(r * np.sqrt(eta_val / mu_val)) + 3 / (2 * lambda_t) * r

# Guess to find the solution of equation_cot:
guess = 8 # Followed suggestion from paper

# Solving numerically to find the critical radius and hence critical volume and mass
Rcrit = (fsolve(equation_cot, guess)).item() # The zero closest to "guess" is found
                                             
print(f"The critical radius is {Rcrit} cm")

Vcrit = (4 / 3) * np.pi * Rcrit ** 3
print(f"The critical volume is {Vcrit} cm^3")

Mcrit = rho * Vcrit  
print(f"The critical mass is {Mcrit} g")

# We want criticality so we work with R > Rcrit (see below). This implies a non-zero alpha:
R0 = 8.5  # [cm]

# To find such alpha we go back to "equation_cot(r)" but now the variable is alpha:
def eqn(x):
    return -1 + R0 * np.sqrt((eta_val + x)/mu_val) * cot(R0 * np.sqrt((eta_val + x)/mu_val)) + 3/2 * R0/lambda_t

guess = - 4.5797e6 # Can check by hand that for this guess the value of "eqn" is close to 0
alpha = fsolve(eqn, guess)[0] # Different from the one in paper, but that one is not coherent with the rest 

# Constants needed to define n (see below):
k = np.sqrt((eta_val + alpha) / mu_val)
A = R0 / np.sin(k*R0) # Follows from imposing Neumann BC on n(r, t): n(R0, 0)=1 (see below)

# Defining the neutron density n(r, t):
def n(r, t):
    return A * np.exp(-alpha * t) * np.sin(k * r) / r

# Plotting at t=0
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
ax.set_title("Neutron diffusion, R=8.5 cm, t=3e-6")
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_zlabel("n(r, t)", fontsize = 9, labelpad = 0.1)
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.show()
