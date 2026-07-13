# Nanomechanical Spectrometry API

A high-performance Python package for solving the inverse problem in nanomechanical spectrometry. This API uses Bayesian inference to map experimentally measured resonant frequency shifts to the spatial position and physical properties (such as mass and stiffness) of adsorbates on nanomechanical resonators.

---

## 📖 Theory & Physics

When a particle adsorbs onto a nanomechanical resonator, it induces a shift in the resonant frequencies of the device's vibrational modes. The fractional frequency shift for the $m$-th mode, $\Delta_m$, can be linearly approximated using a spatial A-matrix $A(\mathbf{r})$ and a parameter vector $\Theta$:

$$\Delta_m \approx \sum_i A_{m,i}(\mathbf{r}) \Theta_i$$

Here, $\mathbf{r}$ represents the spatial coordinates (1D or 2D) of the adsorbate, and $\Theta$ represents the physical parameters (e.g., mass, stiffness, curvature). 

This API solves the resulting **inverse problem** using Bayesian inference, minimizing the Mahalanobis distance weighted by the experimental covariance matrix of the frequency noise to find the most probable position and properties of the adsorbate.

---

## ✨ Features

* **Versatile Resonator Models:** Includes built-in analytical models (Cantilever, Doubly Clamped Beam, Membrane, String) and a `CustomResonator` class to ingest 1D or 2D Finite Element Method (FEM) mode shapes and strains.
* **Batch Processing:** Seamlessly process single measurements or large batches of high-throughput experimental data.
* **Robust Solvers:** Choose between a highly optimized CPU-based Profile Likelihood solver or a GPU-accelerated Brute Force solver for rigorous global minimum searches.
* **Memory Management:** Dynamically chunks parameter grids during brute-force operations to prevent Out-Of-Memory (OOM) errors.
* **Visualization Suite:** Built-in plotting tools to visualize spatial and parameter probability marginals.

---

## ⚙️ Installation

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/nanomechanical-spectrometry-api.git](https://github.com/YOUR-USERNAME/nanomechanical-spectrometry-api.git)
   cd nanomechanical-spectrometry-api
