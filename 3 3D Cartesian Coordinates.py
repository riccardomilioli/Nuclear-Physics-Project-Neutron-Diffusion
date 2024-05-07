import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import tplquad

# We want to solve the 3D diffusion equation for a neutron in cartesian coordinates with Dirichlet boundary conditions.

# Initial condition for the function that enters the integral for a_pqr:
def f(x, y, z, L):
    return (1 - (x / L)) * (1 - (y / L)) * (1 - (z / L)) * (8 * x * y * z) / (L**3)

# Parameters of the problem:
rho = 18.71e6  # [g/m^3]. Density
L_val = 0.192 # [m]. We want criticality so we work with L > Lcrit (see below)
N_val = 4 # Number of a_pqr that get printed (and computed). In this case we get up to a_444 
mu_val = 2.3446e5 # [m^2/s]. Diffusion constant
eta_val = 1.8958e8 # [1/s]. Neutron diffusion rate 

# Computing the critical lenght, volume and mass:
Lcrit = np.pi * np.sqrt(3 * mu_val / eta_val)
print(f"The critical lenght is {Lcrit} m")

Vcrit = Lcrit ** 3
print(f"The critical volume is {Vcrit} m^3")

Mcrit = rho * Vcrit
print(f"The critical mass is {Mcrit} g")

# Calculation of the integral to get a_pqr.
# First initialize a_pqr:
aa_vals = np.zeros((N_val, N_val, N_val), dtype=float)

# Defining the function to be integrated:
def integrand(x, y, z, p, q, r):
    return f(x, y, z, L_val) * np.sin(p * np.pi * x / L_val) * np.sin(q * np.pi * y / L_val) * np.sin(r * np.pi * z / L_val)

# Computing and printing a_pqr:
for p in range(1, N_val + 1):
    for q in range(1, N_val + 1):
        for r in range(1, N_val + 1):
            integral, _ = tplquad(integrand, 0, L_val, lambda x: 0, lambda x: L_val,
                                  lambda x, y: 0, lambda x, y: L_val, args=(p, q, r))
            aa_vals[p-1, q-1, r-1] = (8 / L_val**3) * integral
for p in range(1, N_val + 1):
    for q in range(1, N_val + 1):
        for r in range(1, N_val + 1):
            print(f"p={p}, q={q}, r={r} : a{(p, q, r)} = {aa_vals[p-1, q-1, r-1]}")


# Solving the diffusion equation:
def n_func(x, y, z, t):
    n = np.zeros_like(x) # Initializing the result 
    for i in range(N_val):
        for j in range(N_val):
            for k in range(N_val):
                n += aa_vals[i, j, k] * np.exp(eta_val * t - mu_val * np.pi * ((i/L_val)**2 + (j/L_val)**2 + (k/L_val)**2) * t) * \
                     np.sin(i * np.pi * x / L_val) * np.sin(j * np.pi * y / L_val) * np.sin(k * np.pi * z / L_val)
    return n

# Plotting the solution:
x_vals = np.linspace(0, L_val, 100)
y_vals = np.linspace(0, L_val, 100)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
T = 2e-7
N_func_vals = n_func(X, Y, L_val/2, T) # Computing n_func values for the density plot:

# Density plot for n_func at z=L/2 and t=2e-7:
plt.figure(constrained_layout=True)
plt.title("Density Plot of n(x, y, L/2) at t=2e-7")
plt.xlabel("x")
plt.ylabel("y")
plt.imshow(N_func_vals, extent=[0, L_val, 0, L_val], cmap='viridis', origin='lower')
plt.colorbar(label="Value")
plt.show()  # The result doesn't make sense physically, even though the initial condition does
            # With Mathematica the plot is ok instead
            # My guess is that this has to do with how Python and Mathematica treat floating point
            # Indeed the values of small a_pqr (<1e-15) are different

# Plotting f at z=L/2:
fslice = f(X, Y, L_val/2, L_val)
fig2 = plt.figure(constrained_layout=True)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, fslice, cmap='viridis')
ax2.set_title("Function f for L=19.2 cm, N=4, z=L/2")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y, L/2)", fontsize = 10, labelpad=0.2)
plt.xticks(rotation = 10)
plt.show()

# Density plot for initial condition f:
plt.figure(constrained_layout=True)
plt.title("Density Plot of f(x, y, L/2)")
plt.xlabel("x")
plt.ylabel("y")
plt.imshow(fslice,  extent=[0, L_val, 0, L_val], cmap='viridis', origin='lower')
plt.colorbar(label="Value")
plt.show()
