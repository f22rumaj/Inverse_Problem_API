# Inverse_Problem_API: Bayesian Inference for Nanomechanical Spectrometry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Inverse_Problem_API** is a Python API designed to solve the inverse problem in nanomechanical spectrometry. It infers the adsorption position and the physical properties of an adsorbing analyte—such as its mass and stiffness tensor components from the relative frequency shifts of a mechanical resonator's vibrational modes.

Because experimental measurements are intrinsically stochastic due to noise, this API utilizes a **Bayesian probabilistic framework** to recover both the optimal parameter values and their associated uncertainties (marginal probability density functions).

---

## 📖 Theoretical Background

When an analyte adsorbs onto a mechanical resonator, it shifts the resonance frequencies of the device. For a general 2D resonator in the mass and stiffness sensing regime, the relative frequency shift of the $n$-th mode is given by:

$$\frac{\Delta f_n}{f_{n0}} = -\frac{M_a}{2M}\vert{}\Psi_n(x,y)\vert{}^2 + \frac{K_{mqrs}}{2K}\frac{\varepsilon_{mq}^{(n)}(x,y)\varepsilon_{rs}^{(n)}(x,y)}{\alpha_n^4}$$

Where:
* $M_a$ is the mass of the analyte.
* $K_{mqrs}$ are the components of the analyte's stiffness tensor.
* $\Psi_n(x,y)$ and $\varepsilon^{(n)}(x,y)$ are the mode shapes and in-plane strains at the point of contact.

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

## 💻 Quick Start Guide
Here is a complete example of how to configure a device, input experimental data, run the solver, and plot the results.

**1. Import the API**
```
import numpy as np
from iproblem import measurements, solvers
from iproblem.devices import CantileverBeam
```
**2. Define the Resonator & Parameter Bounds**
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

**3. Load Measurement Data**
Input the relative frequency shifts ($\Delta f / f$) and the experimental covariance matrix (noise).

```
# Example: 1 measurement across 3 modes
sigma = 0.1
freqs_shifts = np.asarray([[  8.64707471,   5.36683501,   2.38447922],
       [ -0.81818077, -49.70348839, -57.56146848]]) 
covariance_matrix = (np.eye(3) * sigma)**2

data = measurements.Measurement(freq_shifts=freq_shifts, covariance_matrix=covariance_matrix)
```

**4. Run the Inverse Solver**
Pass the configuration to the engine. The solver dynamically zooms into the optimal spatial coordinates and isolates the physical parameters.

```
# Initialize the engine
solver = solvers.InverseSolver(resonator=resonator, method='profile_likelihood')

# Solve
results_obj = solver.run(measurement=data, position_grids=xgrid, pgrids=pgrids)
```

**5. Visualize the Results**
Use the built-in Results object to get optimal parameters, marginals and to instantly generate publication-ready marginal distribution plots.

```
# Get optimal parameaters and marginals
opt_param = results_obj.optimal_parameters
marg = results_obj.marg

# Plot the spatial and physical marginals for the first measurement
results_obj.plot_single_measurement(index=0)

# If analyzing batch measurements, plot the aggregated statistical sums
results_obj.plot_aggregated_measurements()
```

