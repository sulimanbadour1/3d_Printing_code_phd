import numpy as np
import matplotlib.pyplot as plt

# Constants and initial conditions with increased resolution
gamma = 1.4  # Ratio of specific heats (Gamma)
M_high_res = 100  # Increased number of cells for higher resolution
x0 = 0.3  # Initial position of discontinuity
x_high_res = np.linspace(0, 1, M_high_res)  # High resolution spatial domain [0,1]
dx_high_res = x_high_res[1] - x_high_res[0]  # Smaller cell size

# Initial left (WL) and right (WR) states
WL = np.array([1.0, 0.75, 1.0])  # Left state (rho, u, p)
WR = np.array([0.125, 0.0, 0.1])  # Right state (rho, u, p)

# Convert pressure to total energy for the conservative form
UL = np.array([WL[0], WL[0] * WL[1], WL[2] / (gamma - 1) + 0.5 * WL[0] * WL[1] ** 2])
UR = np.array([WR[0], WR[0] * WR[1], WR[2] / (gamma - 1) + 0.5 * WR[0] * WR[1] ** 2])


# Initialize the conservative variables U with high resolution
U_high_res = np.zeros((3, M_high_res))
U_high_res[:, : int(M_high_res * x0)] = UL.reshape(3, 1)
U_high_res[:, int(M_high_res * x0) :] = UR.reshape(3, 1)


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
    F_plus = np.vstack(
        (
            0.5 * (F1 + lambda_1 * rho),
            0.5 * (F2 + lambda_2 * mom),
            0.5 * (F3 + lambda_3 * ene),
        )
    )
    F_minus = np.vstack(
        (
            0.5 * (F1 - lambda_1 * rho),
            0.5 * (F2 - lambda_2 * mom),
            0.5 * (F3 - lambda_3 * ene),
        )
    )

    return F_plus, F_minus


# Courant number and time setup
CFL = 0.9
t_final = 0.2
t = 0.0

# Time-stepping loop
while t < t_final:
    # Compute the fluxes
    F_plus, F_minus = steger_warming_flux(U_high_res, gamma)

    # Compute the time step based on the CFL condition
    dt_high_res = (
        CFL
        * dx_high_res
        / np.max(
            np.abs(U_high_res[1, :] / U_high_res[0, :])
            + np.sqrt(gamma * (gamma - 1) * U_high_res[2, :] / U_high_res[0, :])
        )
    )
    if t < 5 * dt_high_res:
        dt_high_res *= 0.2  # Reduce the time step for the first 5 steps

    # Update the conservative variables using the fluxes
    U_high_res[:, 1:-1] -= (
        dt_high_res
        / dx_high_res
        * (F_plus[:, 1:-1] - F_plus[:, :-2] + F_minus[:, 1:-1] - F_minus[:, :-2])
    )

    # Transmissive boundary conditions
    U_high_res[:, 0] = U_high_res[:, 1]
    U_high_res[:, -1] = U_high_res[:, -2]

    # Update time
    t += dt_high_res

# Extract the physical variables from the conservative form with high resolution
rho_high_res = U_high_res[0, :]
v_high_res = U_high_res[1, :] / rho_high_res
p_high_res = (gamma - 1) * (U_high_res[2, :] - 0.5 * rho_high_res * v_high_res**2)
e_high_res = p_high_res / ((gamma - 1) * rho_high_res)  # Internal energy

# Plotting the numerical results with high resolution
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(x_high_res, rho_high_res, "-", label="Density")
plt.title("Density vs Position")
plt.xlabel("Position")
plt.ylabel("Density")

plt.subplot(2, 2, 2)
plt.plot(x_high_res, v_high_res, "-", label="Velocity")
plt.title("Velocity vs Position")
plt.xlabel("Position")
plt.ylabel("Velocity")

plt.subplot(2, 2, 3)
plt.plot(x_high_res, p_high_res, "-", label="Pressure")
plt.title("Pressure vs Position")
plt.xlabel("Position")
plt.ylabel("Pressure")

plt.subplot(2, 2, 4)
plt.plot(x_high_res, e_high_res, "-", label="Internal Energy")
plt.title("Internal Energy vs Position")
plt.xlabel("Position")
plt.ylabel("Internal Energy")

plt.tight_layout()
plt.legend()
plt.show()
