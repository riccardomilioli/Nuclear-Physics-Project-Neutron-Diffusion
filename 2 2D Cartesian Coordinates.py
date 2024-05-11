import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# We want to solve the 2D diffusion equation for a neutron in cartesian coordinates with Dirichlet boundary conditions.

# Initial condition for the function that enters the integral for a_pq:
def f(x, y, L):
    return (1 - (x / L)) * (1 - (y / L)) * x * y / (L / 4)**2

# Parameters of the problem:
L_val = 0.157 # [m]. We want criticality so we work with L > Lcrit (see below)
N = 5 # Number of a_pq that get printed (and computed). In this case we get up to a_55
mu_val = 2.3446e5 # [m^2/s]. Diffusion constant
eta_val = 1.8958e8 # [1/s]. Neutron diffusion rate

# Computing the critical lenght:
Lcrit = np.pi * np.sqrt(2 * mu_val / eta_val)
print(f"The critical lenght is {Lcrit} m")

# Calculation of the integral to find a_pq. 
# First define the function to be integrated:
def integrand(x, y, p, q, L):
    return f(x, y, L) * np.sin(p * np.pi * x / L) * np.sin(q * np.pi * y / L)

# Initializing a_pq:
aa = np.zeros((N, N))

# Computing and printing a_pq:
for p in range(N):
    for q in range(N):
        integral_val, _ = dblquad(integrand, 0, L_val, lambda x: 0, lambda x: L_val, args=(p+1, q+1, L_val))
        aa[p, q] = 4 / L_val**2 * integral_val
for p in range(1, N+1): # We only print in the range 1 <= p <= 5 with q <= p.
    for q in range(1, N+1):
        print(f"p = {p}, q = {q}: a{(p, q)} = {aa[p-1, q-1]}") 

# Solving the diffusion equation:
def n_func(x, y, t, L):
    result = np.zeros_like(x) # Initializing the result 
    for i in range(N):
        for j in range(N):
            result += aa[i, j] * np.exp(eta_val * t - mu_val * ((i + 1) * np.pi / L)**2 * t - mu_val * ((j + 1) * np.pi / L_val)**2 * t) * np.sin((i + 1) * np.pi * x / L_val) * np.sin((j + 1) * np.pi * y / L_val)
    return result

# Plotting the solution surface at t=1e-7:
t_val = 1e-7
x_vals = np.linspace(0, L_val, 100)
y_vals = np.linspace(0, L_val, 100)
X, Y = np.meshgrid(x_vals, y_vals)
N_vals = n_func(X, Y, t_val, L_val)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, N_vals, cmap='viridis')
ax.set_title("Neutron diffusion for L=15.7 cm, N=5, t=1e-7")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("n(t=1e-7)", fontsize = 9, labelpad=0.2)
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.show()

# Plotting the initial condition:
f_vals = f(X, Y, L_val)
fig2 = plt.figure(constrained_layout=True)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, f_vals, cmap='viridis')
ax2.set_title("Initial condition f for L=15.7 cm")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x,y)")
ax2.set_box_aspect(aspect=None, zoom=0.82)
plt.show()

