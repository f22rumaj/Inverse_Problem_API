# Inverse_Problem_API: Bayesian Inference for Nanomechanical Spectrometry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Inverse_Problem_API** is a Python API designed to solve the inverse problem in nanomechanical spectrometry. It infers the adsorption position and the physical properties of an adsorbing analyte—such as its mass and stiffness tensor components from the relative frequency shifts of a mechanical resonator's vibrational modes.

Because experimental measurements are intrinsically stochastic due to noise, this API utilizes a **Bayesian probabilistic framework** to recover both the optimal parameter values and their associated uncertainties (marginal probability density functions).

---

## 📖 Theoretical Background

When an analyte adsorbs onto a mechanical resonator, it shifts the resonance frequencies of the device. For a general 2D resonator in the mass and stiffness sensing regime, the relative frequency shift of the $n$-th mode is given by:

$$\frac{\Delta f_n}{f_{n0}} = -\frac{M_a}{2M}\vert{}\Psi_n(x,y)\vert{}^2 + \frac{K_{mqrs}}{2K}\varepsilon_{mq}^{(n)}(x,y)\varepsilon_{rs}^{(n)}(x,y)$$

Where:
* $M$ is the mass of the resonator.
* $K$ is the stiffness of the resonator.
* $M_a$ is the mass of the analyte.
* $K_{mqrs}$ are the components of the analyte's stiffness tensor.
* $\Psi_n(x,y)$ and $\varepsilon^{(n)}(x,y)$ are the mode shapes and in-plane strains (normalized by the square of the eigenvalue) at the point of contact.

This API reconstructs these physical parameters using a **Bayesian probabilistic framework**. The engine mathematically projects the physical parameters ($\Theta$) as normal distributions over a dynamically refined, iterative spatial grid. This allows for the rapid resolution of up to 7 interacting parameters (Mass + 6 Stiffness components) on a standard CPU.

---

## 🚀 Features

* **Advanced Solvers:** Choose between `profile_likelihood` (fast, highly scalable, CPU-optimized iterative refinement) and `brute_force` algorithms.
* **1D & 2D Devices:** Built-in analytical models for `Cantilevers`, `Doubly Clamped Beams`, `Strings`and `Rectangular Membranes`.
* **Custom Resonators:** Supply your own FEM-derived mode shapes ($\Psi$) and strain fields ($\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{xy}$) to model arbitrary device geometries.
* **Batch Processing:** Seamlessly process single frequency-jump events or massive datasets of continuous measurements.
* **Visualization:** Built-in dynamic plotting class to extract spatial contour maps and 1D probability density functions (PDFs) for all parameters.

---

## 🛠️ Installation

You can install this package directly from GitHub into your local Python or Conda environment.

1. Activate your conda environment

2. Install via pip:
```
pip install git+[https://github.com/f22rumaj/Inverse_Problem_Project.git](https://github.com/f22rumaj/Inverse_Problem_Project.git)
```

## 📂 Project Architecture**

### **devices.py module:**
Contains the mathematical descriptions of the mechanical resonators.
It has 4 pre-defined resonator classes and 1 custom resonator class:

**CantileverBeam:** Models a 1D clamped-free beam tracking Mass (`Theta_Ma`) and Stiffness (`Theta_Ka`). If `parameters` is not specified, the default is mass and stiffness (`['Theta_Ma','Theta_Ka'`). It contains methods `psi(x,n)` and `curv(x,n)` that return the mode shape and curvature of mode `n` at position `x`. The position is normalized from 0 (clamped) to 1 (free end).

**DoublyClampedBeam:** Models a 1D clamped-clamped beam tracking Mass (`Theta_Ma`) and Stiffness (`Theta_Ka`). If `parameters` is not specified, the default is mass and stiffness (`['Theta_Ma','Theta_Ka'`). It contains methods `psi(x,n)` and `curv(x,n)` that return the mode shape and curvature of mode `n` at position `x`. The position is normalized from 0 (clamped) to 0.5 (middle of the beam).

**String:** Models a 1D stressed clamped-clamped beam tracking Mass (`Theta_Ma`). It contains the method `psi(x,n)` that returns the mode shape of mode `n` at position `x`. The position is normalized from 0 (clamped) to 0.5 (middle of the beam).

**Membrane:** Models a 2D stressed rectangular membrane tracking Mass (`Theta_Ma`). It contains the method `psi(x,y,m,n)` that returns the mode shape of mode `(m,n)` at position `(x,y)`. Both `x` and `y` coordinates are normalized from 0 (middle) to 0.5 (clamped).

**CustomResonator:** Models a 1D or 2D custom resonator tracking Mass (`Theta_Ma`) and different components of the stiffness tensor (`Theta_Kxx`, `Theta_Kyy`, `Theta_Kxy`, `Theta_Kxxyy`, `Theta_Kxxxy`, `Theta_Kxyyy`). It contains the methods `psi(x,y,,n)`, `epsilon_xx(x,y,n)`, `epsilon_yy(x,y,n)` and `epsilon_xy(x,y,n)` that return the mode shape and normalized in-plane strains of the mode `n` at position `x,y`. The mode shapes and in-plane strains must be provided by the user when initiate the class

### **measurements.py module:**
Contains the Measurement class: A secure data container that strictly validates frequency shifts and precomputes inverse covariance matrices ($\boldsymbol{\Sigma}^{-1}$) to avoid computational bottlenecks during solver iterations.

### **solvers.py module:**
Contains the InverseSolver class: The mathematica engine. Contains the method `run` to start the computation with the followng inputs: the measurements object, position_grids, parameter_grids and return_position_marginals (keep position marginals if desired, default:True)  It takes the physical resonator and the measurements and computes the optimal values and marginals (Individual Pdfs of the different parameters). Supports the `profile_likelihood` method for all cases (recommended) and the `brute_force` method for up to 3 parameters in total (including position coordinates). The `brute force` method requires high resolution in position and physical parameters in order to be accurate, and therefore, it supports computation on GPU. Returns a Result class object.

### **results.py module:**
Post-processing and visualization. Contains the final results in the variables `optimal_parameters` and `marginals`. Contains the methods `plot_single_measurement(index)` to plot results of an individual measurement and `plot_aggregated_measurements()` to plot all measurements aggregated in one Pdf.

## 💻 Quick Start Guide
Here is a complete example of how to configure a device, input experimental data, run the solver, and plot the results.

**1. Import the API**
```
import numpy as np
from iproblem import measurements, solvers
from iproblem.devices import CantileverBeam
```
**2. Define the Resonator & Parameter Bounds:**
Initialize your device and define the bounds of the parameter space you want to explore.
```
# Initialize a 1D Cantilever using the first 3 modes
resonator = devices.CantileverBeam(modes=3)

# Define the spatial grid and physical parameter limits
xgrid = [np.linspace(0, 1, 500)]
bounds = {
    'Theta_Ma': {'limits': [0, 100], 'points': 1000},  
    'Theta_Ka': {'limits': [0, 10], 'points': 1000}
}

pgrids = [np.linspace(b['limits'][0], b['limits'][1], b['points']) for b in bounds.values()]
```

**3. Load Measurement Data:**
Input the relative frequency shifts ($\Delta f / f$) and the experimental covariance matrix (noise).

```
# Example: 1 measurement across 3 modes
sigma = 0.1
freqs_shifts = np.asarray([[  8.64707471,   5.36683501,   2.38447922],
       [ -0.81818077, -49.70348839, -57.56146848]]) 
covariance_matrix = (np.eye(3) * sigma)**2

data = measurements.Measurement(freq_shifts=freq_shifts, covariance_matrix=covariance_matrix)
```

**4. Run the Inverse Solver:**
Pass the configuration to the engine. The solver dynamically zooms into the optimal spatial coordinates and isolates the physical parameters.

```
# Initialize the engine
solver = solvers.InverseSolver(resonator=resonator, method='profile_likelihood')

# Solve
results_obj = solver.run(measurement=data, position_grids=xgrid, pgrids=pgrids)
```

**5. Visualize the Results:**
Use the built-in Results object to get optimal parameters, marginals and to instantly generate publication-ready marginal distribution plots.

```
# Get optimal parameters and marginals
opt_param = results_obj.optimal_parameters
marg = results_obj.marg

# Plot the spatial and physical marginals for the first measurement
results_obj.plot_single_measurement(index=0)

# If analyzing batch measurements, plot the aggregated statistical sums
results_obj.plot_aggregated_measurements()
```
