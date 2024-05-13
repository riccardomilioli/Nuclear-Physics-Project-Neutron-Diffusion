import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros
import pandas as pd

# We want to solve the 3D diffusion equation for a neutron in cylindrical coordinates with Dirichlet boundary conditions.
# In this case we're working with fixed p = 1. Indeed one can prove analytically that all a_pq are 0 for p>1 

# Initial condition for the function that enters the integral for a_pq:
def f(r, z, r1, L):
    return (1 - (r / r1)**2) * np.sin(np.pi * z / L)

# Parameters of the problem:
rho = 18.71e6  # [g/m^3]. Density
L_val = 0.192 # [m]. We want supercriticality so we work with L > Lcrit (see below)
r1_val = 0.104 # [m]. We want supercriticality so we work with R > Rcrit (see below)
Np = 1 # Number of a_pq that are computed for p. Coherent with the fact that all a_pq are 0 for p>1 
Nq = 10 # Number of a_pq that are computed for q
mu_val = 2.3446e5 # [m^2/s]. Diffusion constant
eta_val = 1.8958e8 # [1/s]. Neutron diffusion rate 

# Computig the critical lenght, radius, volume and mass:
Lcrit = np.pi * np.sqrt(3 * mu_val / eta_val)
print(f"The critical lenght is {Lcrit} m")

alpha = (jn_zeros(0, 1)).item() 
Rcrit = alpha * np.sqrt(3 * mu_val /(2 * eta_val))
print(f"The critical radius is {Rcrit} m")

Vcrit = np.pi * Rcrit**2 * Lcrit
print(f"The critical volume is {Vcrit} m^3")

Mcrit = rho * Vcrit
print(f"The critical mass is {Mcrit} g")

# Define zeros of Bessel J0 function (needed later)
beta_vals = jn_zeros(0, Nq)

# Calculation of the integral to get a_pq.
# In this case we import results computed by mathematica and saved in 'integral_results.txt'

# Loading the text file into a dataframe:
integ_df = pd.read_csv('integral_results.txt', delim_whitespace=True, names=['k', 'Integral_Result'], index_col = 0)

# Extracting the results from the dataframe:
integral_results = integ_df['Integral_Result'].values

# Reshaping integral_results to match the shape of aa_vals:
aa_vals = integral_results.reshape((Np, -1))  

# Printing the values of a_1p:
print('p is fixed p = 1')
for p in range(1, Np + 1):
    for q in range(1, Nq + 1):
        print(f"q = {q}, a{(1, q)} = {aa_vals[0, q-1]}")
        
        
# Solving the diffusion equation:
def n_func(r, z, t):
    n = np.zeros_like(r) # Initializing the result 
    for p in range(1, Np + 1):
        for q in range(1, Nq + 1):
            n += aa_vals[p-1, q-1] * np.exp((eta_val * r1_val**2 * L_val**2 - mu_val * beta_vals[q-1]**2 * L_val**2 -
                                              mu_val * p**2 * np.pi**2 * r1_val**2) * t / (r1_val**2 * L_val**2)) * \
                                              jn(0, r * beta_vals[q-1] / r1_val) * np.sin(p * np.pi * z / L_val)
    return n

# Plotting the solution:
r_vals = np.linspace(-r1_val, r1_val, 100)
z_vals = np.linspace(0, L_val, 100)
R, Z = np.meshgrid(r_vals, z_vals)
T = 1e-5
N_vals = n_func(R, Z, T)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, Z, N_vals, cmap='viridis')
ax.set_title(f"Neutron diffusion, L={L_val}, r1={r1_val}, Np={Np}, Nq={Nq}, t={T}")
ax.set_xlabel("r")
ax.set_ylabel("z")
ax.set_zlabel("n(r,z)")
plt.show()

# Plotting the initial condition:
f_vals = f(R, Z, r1_val, L_val)
fig2 = plt.figure(constrained_layout=True)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(R, Z, f_vals, cmap='viridis')
ax2.set_title(f"Initial condition f for L={L_val} m, r1={r1_val} m")
ax2.set_xlabel("r")
ax2.set_ylabel("z")
ax2.set_zlabel("f(r,z)")
ax2.set_box_aspect(aspect=None, zoom=0.82)
plt.show()
