import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


### TEST 1 ###
# Constants and initial conditions for higher resolution
gamma = 1.4  # Specific heat ratio for ideal gas
M = 100  # Number of cells for spatial discretization, increased for higher resolution
x0 = 0.3  # Position of initial discontinuity Test1
x = np.linspace(0, 1, M)  # Spatial domain from 0 to 1
dx = x[1] - x[0]  # Delta x, the distance between cell centers

# Initial conditions for density (rho), velocity (u), and pressure (p)
WL = np.array([1.0, 0.75, 1.0])  # Left state (rho, u, p)
WR = np.array([0.125, 0.0, 0.1])  # Right state (rho, u, p)

# Initialize the conservative variables U
U = np.zeros((3, M))
# For the left state
U[:, : int(M * x0)] = WL.reshape(-1, 1)  # Reshape WL to be a column vector
# For the right state
U[:, int(M * x0) :] = WR.reshape(-1, 1)  # Reshape WR to be a column vector

# Convert velocity to momentum and pressure to total energy for conservative form
U[1, :] *= U[0, :]  # Momentum
U[2, :] = (
    U[2, :] / (gamma - 1) + 0.5 * U[0, :] * (U[1, :] / U[0, :]) ** 2
)  # Total energy


# Function to compute the Steger-Warming flux
def steger_warming_flux(U, gamma):
    rho = U[0, :]
    mom = U[1, :]
    ene = U[2, :]

    # Primitive variables
    v = mom / rho
    p = (gamma - 1) * (ene - 0.5 * mom**2 / rho)
    a = np.sqrt(gamma * p / rho)

    # Eigenvalues
    lambda_1 = np.abs(v)
    lambda_2 = np.abs(v + a)
    lambda_3 = np.abs(v - a)

    # Calculate fluxes for each component
    F1 = mom
    F2 = mom * v + p
    F3 = v * (ene + p)

    # Flux splitting
    F_plus = (
        np.vstack((F1 + lambda_1 * rho, F2 + lambda_2 * mom, F3 + lambda_3 * ene)) * 0.5
    )
    F_minus = (
        np.vstack((F1 - lambda_1 * rho, F2 - lambda_2 * mom, F3 - lambda_3 * ene)) * 0.5
    )

    return F_plus, F_minus


# Time-stepping loop
def time_stepping(U, dx, CFL, gamma, t_final):
    t = 0.0
    step = 0
    while t < t_final:
        rho = U[0, :]
        mom = U[1, :]
        ene = U[2, :]
        p = (gamma - 1) * (ene - 0.5 * mom**2 / rho)  # Compute pressure here
        # Compute the fluxes
        F_plus, F_minus = steger_warming_flux(U, gamma)

        # Compute the time step based on the CFL condition
        u_max = np.max(np.abs(U[1, :] / U[0, :]) + np.sqrt(gamma * p / U[0, :]))
        dt = CFL * dx / u_max
        if step < 5:
            dt *= 0.2  # Reduce time step for the first 5 steps

        # Update the conservative variables using the fluxes
        U[:, 1:-1] -= (
            dt
            / dx
            * (F_plus[:, 1:-1] - F_plus[:, :-2] + F_minus[:, 1:-1] - F_minus[:, :-2])
        )

        # Transmissive boundary conditions
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]

        # Update time and step counter
        t += dt
        step += 1
    return U


# Setup for time-stepping
t_final = 0.2  # Final time to reach
CFL = 0.9  # Courant number for stability

# Perform the time-stepping
U = time_stepping(U, dx, CFL, gamma, t_final)

# Extract physical quantities from the conservative variables
rho = U[0, :]  # Density
u = U[1, :] / rho  # Velocity
e = U[2, :] / rho  # Specific internal energy
p = (gamma - 1) * rho * (e - 0.5 * u**2)  # Pressure


# Step 1: Identify the regions where the solution changes state
def detect_shocks(signal, threshold=0.1):
    gradient = np.abs(np.diff(signal))
    shock_indices = np.where(gradient > threshold)[0]
    return shock_indices


# Apply a Gaussian filter locally
def apply_local_smoothing(signal, shock_indices, smoothing_width=5):
    smoothed_signal = signal.copy()
    for index in shock_indices:
        smoothed_signal[
            index - smoothing_width : index + smoothing_width
        ] = gaussian_filter1d(
            signal[index - smoothing_width : index + smoothing_width], sigma=1.0
        )
    return smoothed_signal


# Step 2 and 3: Apply piecewise function and smoothing
def smooth_transitions(U):
    smoothed_U = np.zeros_like(U)
    for i in range(U.shape[0]):
        shock_indices = detect_shocks(U[i, :])
        smoothed_U[i, :] = apply_local_smoothing(U[i, :], shock_indices)
    return smoothed_U


# Applying the smoothing to your simulation results
U_smoothed = smooth_transitions(U)

# Extracting physical quantities from smoothed conservative variables
rho_smoothed = U_smoothed[0, :]  # Smoothed density
u_smoothed = U_smoothed[1, :] / rho_smoothed  # Smoothed velocity
p_smoothed = (
    (gamma - 1)
    * rho_smoothed
    * (U_smoothed[2, :] / rho_smoothed - 0.5 * u_smoothed**2)
)  # Smoothed pressure

# Plotting the smoothed results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(x, rho_smoothed, label="Smoothed Density")
plt.title("Density")
plt.xlabel("Position")
plt.ylabel("Density")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, u_smoothed, label="Smoothed Velocity")
plt.title("Velocity")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, p_smoothed, label="Smoothed Pressure")
plt.title("Pressure")
plt.xlabel("Position")
plt.ylabel("Pressure")
plt.legend()

plt.tight_layout()
plt.show()
