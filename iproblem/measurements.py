import numpy as np

class Measurement:
    """
    A data container for experimental nanomechanical spectrometry measurements.
    Supports single or batch measurements with a shared covariance matrix.

    Inputs:
    - freq_shifts: 1D array for a single measurement or 2D array for batch measurements (shape: (num_measurements, num_modes))
    - covariance_matrix: 2D array representing the covariance matrix of the frequency noise (shape: (num_modes, num_modes))
    """
    def __init__(self, freq_shifts, covariance_matrix):
        # Convert inputs to numpy arrays for standard handling
        self.freq_shifts = np.array(freq_shifts, dtype=float)
        self.covariance_matrix = np.array(covariance_matrix, dtype=float)
        
        # Handle 1D (single measurement) or 2D (batch measurements)
        if self.freq_shifts.ndim == 1:
            # Reshape to (1, N_modes) for consistent downstream processing
            self.freq_shifts = self.freq_shifts[np.newaxis, :]
        elif self.freq_shifts.ndim != 2:
            raise ValueError("freq_shifts must be a 1D array (single) or 2D array (batch).")
        
        # Extract dimensions
        self.num_measurements, self.num_modes = self.freq_shifts.shape
        
        # Validate covariance matrix
        if self.covariance_matrix.ndim != 2:
            raise ValueError("covariance_matrix must be a 2D array.")
            
        if self.covariance_matrix.shape != (self.num_modes, self.num_modes):
            raise ValueError(
                f"Shape mismatch: freq_shifts indicates {self.num_modes} modes, "
                f"but covariance_matrix shape is {self.covariance_matrix.shape}. "
                f"Expected ({self.num_modes}, {self.num_modes})."
            )
            
        # Precompute the inverse covariance matrix
        try:
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("The covariance matrix is singular and cannot be inverted.")