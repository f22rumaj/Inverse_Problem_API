# Inverse_Problem_API: Bayesian Inference for Nanomechanical Spectrometry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Inverse_Problem_API** is a Python API designed to solve the inverse problem in nanomechanical spectrometry. It infers the physical properties of an adsorbing analyte—such as its mass and anisotropic stiffness tensor components—and its adsorption position from the relative frequency shifts of a mechanical resonator's vibrational modes.

Because experimental measurements are intrinsically stochastic due to noise, this API utilizes a **Bayesian probabilistic framework** to recover both the optimal parameter values and their associated uncertainties (marginal probability density functions).

---

## 📖 Theoretical Background

When an analyte adsorbs onto a mechanical resonator, it shifts the resonance frequencies of the device. For a general 2D resonator in the mass and stiffness sensing regime, the relative frequency shift of the $n$-th mode is given by:

$$\frac{\Delta f_n}{f_{n0}} = -\frac{M_a}{2M}\vert{}\Psi_n(x,y)\vert{}^2 + \frac{K_{mqrs}}{2K}\frac{\varepsilon_{mq}^{(n)}(x,y)\varepsilon_{rs}^{(n)}(x,y)}{\alpha_n^4}$$

Where:
* $M_a$ is the mass of the analyte.
* $K_{mqrs}$ are the components of the analyte's stiffness tensor.
* $\Psi_n(x,y)$ and $\varepsilon^{(n)}(x,y)$ are the mode shapes and in-plane strains at the point of contact.

This API reconstructs these physical parameters using **Profile Likelihood**. Instead of calculating an impossibly massive joint probability space (Brute Force), the engine mathematically projects the physical parameters ($\Theta$) as normal distributions over a dynamically refined, iterative spatial grid. This allows for the rapid resolution of up to 7 interacting parameters (Mass + 6 Stiffness components) on a standard CPU.

---

## 🚀 Features

* **Advanced Solvers:** Choose between `profile_likelihood` (fast, highly scalable, CPU-optimized iterative refinement) and `brute_force` algorithms.
* **1D & 2D Devices:** Built-in analytical models for `Cantilevers`, `Doubly clamped beams`, `Strings`and `Rectangular membranes`.
* **Custom Resonators:** Supply your own FEM-derived mode shapes ($\Psi$) and strain fields ($\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{xy}$) to model arbitrary device geometries.
* **Batch Processing:** Seamlessly process single frequency-jump events or massive datasets of continuous measurements.
* **Visualization:** Built-in dynamic plotting class to extract spatial contour maps and 1D probability density functions (PDFs) for all parameters.

---

## 🛠️ Installation

You can install this package directly from GitHub into your local Python or Conda environment.

1. Activate your environment:
   ```bash
   conda activate my_env
