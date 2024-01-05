
# Numerical Solution of 1D Euler Equations using Steger-Warming Scheme

This project involves the implementation of the Steger-Warming flux vector splitting (FVS) scheme to solve the one-dimensional Euler equations for ideal gases. The scheme is applied to a Riemann problem with given initial conditions.

## Description

The one-dimensional, time-dependent Euler equations for ideal gases are solved using the Steger-Warming FVS scheme. The problem setup is based on the Sod shock tube problem, which is a standard benchmark problem in computational fluid dynamics for testing numerical solvers.

## Initial Conditions

The computational domain is the interval [0, 1], with a discontinuity at \(x_0 = 0.3\). The domain is discretized into 500 cells for higher resolution.

The left (\(W_L\)) and right (\(W_R\)) states are given by:
- \(W_L = (1.0, 0.75, 1.0)^T\)
- \(W_R = (0.125, 0.0, 0.1)^T\)

## Numerical Scheme

The Steger-Warming FVS scheme is used to compute the numerical fluxes at the cell interfaces. The scheme splits the fluxes based on the sign of the eigenvalues of the flux Jacobian, resulting in a stable and accurate method for capturing shock waves and expansion fans.

## Boundary Conditions

Transmissive boundary conditions are used to allow waves to exit the computational domain without reflecting back.

## Time Stepping

The time stepping is performed using a CFL number of 0.9 to ensure numerical stability. The final time is set to \(t = 0.3\) units.

## Outputs

The numerical solution is plotted for the following physical quantities as functions of position:
- Density (\(
ho\))
- Velocity (\(u\))
- Pressure (\(p\))
- Internal Energy (\(e\))

## Usage

To run the simulation, copy the provided Python code into a Google Colab cell or your local Python environment and execute it.

## Dependencies

- Numpy
- Matplotlib

Ensure these packages are installed in your Python environment before running the simulation.

## Author

The numerical solver was implemented based on the some test requirements and computational fluid dynamics principles.

---

**Note:** The exact solutions are not included in this implementation. To compare the numerical results with exact solutions, a Riemann solver or equivalent analytical methods would be required.
