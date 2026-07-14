# Inverse_Problem_API: Bayesian Inference for Nanomechanical Spectrometry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Inverse_Problem_API** is a Python package for solving the inverse problem in nanomechanical spectrometry. It infers the adsorption position and the physical properties of an adsorbed analyte—such as its mass and stiffness tensor components—by analyzing the relative frequency shifts of a mechanical resonator's vibrational modes.

Because experimental measurements are inherently noisy, the API employs a **Bayesian probabilistic framework** to estimate both the optimal parameter values and their associated uncertainties through marginal probability density functions (PDFs).

---

## 📖 Theoretical Background

When an analyte adsorbs onto a mechanical resonator, it shifts the resonance frequencies of the device. For a general two-dimensional resonator operating in the mass and stiffness sensing regime, the relative frequency shift of the $n$-th vibrational mode is given by:

$$
\frac{\Delta f_n}{f_{n0}} = -\frac{M_a}{2M}\left|\Psi_n(x,y)\right|^2 + \frac{K_{mqrs}}{2K}\varepsilon_{mq}^{(n)}(x,y)\varepsilon_{rs}^{(n)}(x,y)
$$

where

* $M$ is the mass of the resonator.
* $K$ is the effective stiffness of the resonator.
* $M_a$ is the mass of the adsorbed analyte.
* $K_{mqrs}$ are the components of the analyte's stiffness tensor.
* $\Psi_n(x,y)$ and $\varepsilon^{(n)}(x,y)$ are the normalized mode shapes and in-plane strain fields (normalized by the square of the corresponding eigenvalue) evaluated at the adsorption position.

The API reconstructs these physical parameters using a **Bayesian probabilistic framework**. The inference engine models the physical parameters ($\Theta$) as Gaussian probability distributions over a dynamically refined spatial grid, allowing the efficient inference of up to seven coupled parameters (mass plus six stiffness tensor components) on a standard CPU.

---

## 🚀 Features

* **Advanced Solvers:** Choose between the `profile_likelihood` solver (fast, highly scalable, and CPU-optimized through iterative refinement) and the `brute_force` solver.
* **Built-in Resonator Models:** Analytical models for `CantileverBeam`, `DoublyClampedBeam`, `String`, and `Membrane`.
* **Custom Resonators:** Supply your own FEM-derived mode shapes ($\Psi$) and strain fields ($\varepsilon_{xx}$, $\varepsilon_{yy}$, $\varepsilon_{xy}$) to model arbitrary resonator geometries.
* **Batch Processing:** Process either individual frequency-jump events or large datasets of continuous measurements.
* **Visualization:** Built-in plotting utilities for generating spatial contour maps and one-dimensional marginal probability density functions (PDFs).

---

## 🛠️ Installation

You can install the package directly from GitHub into your Python or Conda environment.

1. Activate your Conda environment.

2. Install the package using pip:

```bash
pip install git+https://github.com/f22rumaj/Inverse_Problem_Project.git
```

---

## 📂 Project Architecture

### `devices.py`

Contains the mathematical models of the supported mechanical resonators.

#### `CantileverBeam`

Models a one-dimensional clamped-free beam for simultaneous mass (`Theta_Ma`) and stiffness (`Theta_Ka`) inference. If `parameters` is omitted, the default is:

```python
['Theta_Ma', 'Theta_Ka']
```

It provides the methods:

* `psi(x, n)` – returns the normalized mode shape of mode `n` at position `x`.
* `curv(x, n)` – returns the normalized curvature of mode `n` at position `x`.

The position coordinate is normalized from 0 (clamped end) to 1 (free end).

---

#### `DoublyClampedBeam`

Models a one-dimensional clamped-clamped beam for simultaneous mass (`Theta_Ma`) and stiffness (`Theta_Ka`) inference. If `parameters` is omitted, the default is:

```python
['Theta_Ma', 'Theta_Ka']
```

It provides the methods:

* `psi(x, n)` – returns the normalized mode shape of mode `n` at position `x`.
* `curv(x, n)` – returns the normalized curvature of mode `n` at position `x`.

The position coordinate is normalized from 0 (clamped end) to 0.5 (center of the beam).

---

#### `String`

Models a one-dimensional tensioned clamped-clamped string for mass inference.

It provides the method:

* `psi(x, n)` – returns the normalized mode shape of mode `n` at position `x`.

The position coordinate is normalized from 0 (clamped end) to 0.5 (center of the string).

---

#### `Membrane`

Models a two-dimensional rectangular membrane under tension for mass inference.

It provides the method:

* `psi(x, y, m, n)` – returns the normalized mode shape corresponding to mode `(m, n)` at position `(x, y)`.

Both `x` and `y` coordinates are normalized from 0 (center of the membrane) to 0.5 (clamped edge).

---

#### `CustomResonator`

Models arbitrary one- or two-dimensional resonators using user-provided FEM data. It supports mass (`Theta_Ma`) together with any combination of stiffness tensor components:

* `Theta_Kxx`
* `Theta_Kyy`
* `Theta_Kxy`
* `Theta_Kxxyy`
* `Theta_Kxxxy`
* `Theta_Kxyyy`

It provides the methods:

* `psi(x, y, n)`
* `epsilon_xx(x, y, n)`
* `epsilon_yy(x, y, n)`
* `epsilon_xy(x, y, n)`

These return the normalized mode shape and normalized in-plane strain fields for mode `n` at position `(x, y)`. The mode shapes and strain fields must be supplied by the user when initializing the class.

---

### `measurements.py`

Contains the `Measurement` class, a data container that validates the measured frequency shifts and precomputes the inverse covariance matrix ($\boldsymbol{\Sigma}^{-1}$) to avoid computational bottlenecks during solver iterations.

---

### `solvers.py`

Contains the `InverseSolver` class, which implements the Bayesian inference engine.

The main entry point is the `run()` method, which accepts:

* a `Measurement` object,
* `position_grids`,
* `pgrids`,
* `return_position_marginals` (optional, default `True`).

Given a resonator model and a set of measurements, the solver estimates the optimal physical parameters together with their marginal probability density functions.

The `profile_likelihood` method is recommended and supports all available resonator models. The `brute_force` method is intended for problems involving up to three unknown parameters (including spatial coordinates). Since the brute-force approach requires a high-resolution discretization of both position and parameter space, it also supports GPU acceleration.

The `run()` method returns a `Result` object.

---

### `results.py`

Contains the `Result` class for post-processing and visualization.

The inferred parameters are stored in:

* `optimal_parameters`
* `marginals`

The class also provides:

* `plot_single_measurement(index)` – visualizes the posterior distributions for an individual measurement.
* `plot_aggregated_measurements()` – visualizes the aggregated posterior distributions across all measurements.

---

# 💻 Quick Start Guide

The following example demonstrates how to configure a resonator, load experimental measurements, run the Bayesian solver, and visualize the inferred parameters.

## 1. Import the package

```python
import numpy as np

from iproblem import measurements, solvers
from iproblem.devices import (
    CantileverBeam,
    DoublyClampedBeam,
    String,
    Membrane,
    CustomResonator,
)
```

## 2. Define the resonator

```python
# One-dimensional cantilever using the first four modes
# for simultaneous mass and stiffness inference
resonator = CantileverBeam(modes=4)

# One-dimensional cantilever measuring only mass using the first 3 modes
resonator = CantileverBeam(modes=3, parameters=["Theta_Ma"])

# Two-dimensional rectangular membrane using 6 modes
resonator = Membrane(modes=((1,1), (1,2), (2,1), (2,2), (2,3), (3,2)))

# One-dimensional custom resonator
# phi and curv are NumPy arrays of shape (nmodes, npoints)
# For 2D custom resonators, shape must be (nmodes, npoints_X, npoints_Y)
x = np.linspace(0, 1, 500)

resonator = CustomResonator(
    parameter_map=["Theta_Ma", "Theta_Kxx"],
    position_coordinates=[x],
    psi=phi,
    epsilon_xx=curv
)
```

---

## 3. Load the measurement data

```python
# Create a measurement container with two measurements
# across three vibrational modes

sigma = 0.1  # ppm

freq_shifts = np.asarray([
    [  8.64707471,   5.36683501,   2.38447922],
    [ -0.81818077, -49.70348839, -57.56146848]
])  # ppm

covariance_matrix = (np.eye(len(freq_shifts[0])) * sigma) ** 2

data = measurements.Measurement(freq_shifts=freq_shifts, covariance_matrix=covariance_matrix)
```

---

## 4. Run the inverse solver

```python
# Create a solver using the profile likelihood method
solver = solvers.InverseSolver(resonator=resonator, method="profile_likelihood")

# Create a brute-force solver running on the GPU
solver = solvers.InverseSolver(resonator=resonator, method="brute_force", device="GPU")

# Define the search grids
xgrid = [np.linspace(0, 1, 500)]

mgrid = np.linspace(0, 100, 1000)
kgrid = np.linspace(0, 10, 1000)

pgrids = [mgrid, kgrid]

# Run the inference
results_obj = solver.run(measurement=data, position_grids=xgrid, pgrids=pgrids)

# Run the inference without storing the position marginals
results_obj = solver.run(measurement=data, position_grids=xgrid, pgrids=pgrids, return_position_marginals=False)
```

---

## 5. Visualize the results

```python
# Retrieve the inferred parameters
optimal_parameters = results_obj.optimal_parameters
marginals = results_obj.marginals

# Plot the posterior distributions for the first measurement
results_obj.plot_single_measurement(index=0)

# Plot the aggregated posterior distributions
results_obj.plot_aggregated_measurements()
```
