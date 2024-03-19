# Riemann Solvers
Code snippets follow from ___Riemann Solvers and Numerical Methods for Fluid Dynamics___ by Eleuterio F. Toro, where essentials of CFD are discussed in detail.



# 1- Static Code plots using Python only  for three __(1,2,3)__ of the cases


# Numerical Solution of 1D Euler Equations using Steger-Warming Scheme

This project involves the implementation of the Steger-Warming flux vector splitting (FVS) scheme to solve the one-dimensional Euler equations for ideal gases. The scheme is applied to a Riemann problem with given initial conditions.

## Description

The one-dimensional, time-dependent Euler equations for ideal gases are solved using the Steger-Warming FVS scheme. The problem setup is based on the Sod shock tube problem, which is a standard benchmark problem in computational fluid dynamics for testing numerical solvers.

## Initial Conditions

The computational domain is the interval [0, 1], with a discontinuity at \(x_0 = 0.3\). The domain is discretized into 100 cells for higher resolution.

For Test 1, the left (\(W_L\)) and right (\(W_R\)) states are given by:
- \(W_L = (1.0, 0.75, 1.0)^T\)
- \(W_R = (0.125, 0.0, 0.1)^T\)


For other tests, the name of the files are Test __2__ raw , Test __3__ raw, it's the same files but with different input parameters.


## Numerical Scheme

The Steger-Warming FVS scheme is used to compute the numerical fluxes at the cell interfaces. The scheme splits the fluxes based on the sign of the eigenvalues of the flux Jacobian, resulting in a stable and accurate method for capturing shock waves and expansion fans.

## Boundary Conditions

Transmissive boundary conditions are used to allow waves to exit the computational domain without reflecting back.

## Time Stepping

The time stepping is performed using a CFL number of 0.9 to ensure numerical stability. The final time is set to \(t = 0.3\) units.

## Outputs

The numerical solution is plotted for the following physical quantities as functions of position:
- Density (\(ho\))
- Velocity (\(u\))
- Pressure (\(p\))
- Internal Energy (\(e\))

## Usage

To run the simulation, copy the provided Python code into a Google Colab cell or your local Python environment and execute it.
- Link to google colab file for test1, https://colab.research.google.com/drive/1aMAs_xN2xH4RDKr4XrhP8qc3Af7PACu_?usp=sharing

## Dependencies

- Numpy
- Matplotlib
- Scipy

Ensure these packages are installed in your Python environment before running the simulation.

## Author

The numerical solver was implemented based on selected test requirements and computational fluid dynamics principles, for any related questions, contact **S.Badour**.

---

<br/>
<br/>

# 2- Animated Code plots using Cpp, and Python for all the cases


## Euler Equation
1-D Euler equation with ideal gases.


### Exact solution(ch4)
The general solution of the exact solution follows the 3-wave pattern, where the _contact_ must lies in between, _shock_ and _rarefaction_ waves stay at left or right.  
The exact solution can be calculate numerically, where a iterative procedure is necessary for solving the _pressure_. The exact solution requies much computational effort and this is why approximate riemann solvers are studied extensively back in 1980s.
Especially, vacuum condition is considered for completeness.

Usage:  
> * Compile: `g++ main.cc -o Exact.out`  
> * Execute: `./Exact.out < inp.dat`  
> * Plot: `python3 animate.py`

### Godunov's Method(ch6)
The essential ingredient of Godunov's method is to solve _Riemann Problem_ locally, and the keypoint in numerical parctice is to identify all of the 10 possible wave patterns so that the inter-cell flux can be calculated properly.

Usage:
> * Compile: `g++ main.cc -std=c++11 -o Godunov.out`  
> * Execute: `./Godunov.out < inp.dat`  
> * Plot: `python3 animate.py`

### FVS(ch8)
Here, the inter-cell flux is not calculatd directly from the exact solution of Riemann Problem. Instead, the flux at each point is splitted into 2 parts: __upstream traveling__ and __downstream traveling__, then, for each inter-cell, the flux is seen as the sum of the upstream traveling part from the __left__ point and the downstream traveling part from the __right__ point.  
3 typical splitting techniques are introduced.  

#### Steger-Warming

Usage:
> * Compile: `g++ main.cc -std=c++11 -o SW.out`  
> * Execute: `./SW.out < inp.dat`  
> * Plot: `python3 animate.py`

<br/>

## For any change in the parameters, please use the cpp file (main.cc) and compile the code again.



