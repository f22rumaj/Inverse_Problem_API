import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator, interp1d

class CustomResonator:
    """
    Class for a custom resonator.
    
    Inputs:
    - parameter_map: List of parameter names (e.g., ['Theta_Ma', 'Theta_Kxx', 'Theta_Kyy'])
    - position_coordinates: List of position coordinates (1D or 2D) [x] or [x, y]
    - psi: Mode shape function: Must be an array of shape (num_modes, npoints_X, npoints_Y) for 2D or (num_modes, npoints_X) for 1D
    - epsilon_xx, epsilon_yy, epsilon_xy: In-plane strains functions normalized by the square of the eigenvalue. 
      Must be an array of shape (num_modes, npoints_X, npoints_Y) for 2D or (num_modes, npoints_X) for 1D
    """

    def __init__(self, parameter_map, position_coordinates, modes=None, psi=None, epsilon_xx=None, epsilon_yy=None, epsilon_xy=None):
        self.parameter_map = parameter_map
        self.num_parameters = len(parameter_map)
        
        # Base grids from the FEM data (1D or 2D)
        self.position_coordinates = [np.atleast_1d(np.squeeze(c)) for c in position_coordinates]
        self.dimension = len(self.position_coordinates)
        
        # 1. Identify num_modes from whatever array is provided
        data_arrays = [psi, epsilon_xx] if self.dimension == 1 else [psi, epsilon_xx, epsilon_yy, epsilon_xy]
        valid_arrays = [a for a in data_arrays if a is not None]
        if not valid_arrays:
            raise ValueError("At least one mode shape or strain array must be provided.")
        
        self.num_modes = np.asarray(valid_arrays[0]).shape[0]
        self.modes = modes if modes is not None else list(range(1, self.num_modes + 1))
        
        # Expected array shape: (n_modes, N_base_x, [N_base_y])
        expected_shape = (self.num_modes,) + tuple(len(c) for c in self.position_coordinates)
        
        # 2. Store internal data safely
        self._psi_data = self._validate_array(psi, expected_shape)
        self._exx_data = self._validate_array(epsilon_xx, expected_shape)
        
        if self.dimension == 2:
            self._eyy_data = self._validate_array(epsilon_yy, expected_shape)
            self._exy_data = self._validate_array(epsilon_xy, expected_shape)
        else:
            self._eyy_data = None
            self._exy_data = None
        
        # Initialize the interpolators
        self._setup_interpolators()

    def _validate_array(self, arr, expected_shape):
        """Ensures all arrays match the spatial grid and mode dimensions."""
        if arr is None:
            return np.zeros(expected_shape, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != expected_shape:
            raise ValueError(f"Data shape {arr.shape} does not match expected (n_modes, *spatial_dims): {expected_shape}")
        return arr
    
    def _setup_interpolators(self):
        """
        Prepares SciPy interpolators.
        """
        axes_transpose = tuple(range(1, self.dimension + 1)) + (0,)
        
        psi_T = np.transpose(self._psi_data, axes_transpose)
        exx_T = np.transpose(self._exx_data, axes_transpose)

        if self.dimension == 1:
            x_base = self.position_coordinates[0]
            self._interp_psi = interp1d(x_base, psi_T, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
            self._interp_exx = interp1d(x_base, exx_T, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
            
        elif self.dimension == 2:
            eyy_T = np.transpose(self._eyy_data, axes_transpose)
            exy_T = np.transpose(self._exy_data, axes_transpose)
            
            grid = tuple(self.position_coordinates)
            self._interp_psi = RegularGridInterpolator(grid, psi_T, method='linear', bounds_error=False, fill_value=0.0)
            self._interp_exx = RegularGridInterpolator(grid, exx_T, method='linear', bounds_error=False, fill_value=0.0)
            self._interp_eyy = RegularGridInterpolator(grid, eyy_T, method='linear', bounds_error=False, fill_value=0.0)
            self._interp_exy = RegularGridInterpolator(grid, exy_T, method='linear', bounds_error=False, fill_value=0.0)

    # ---------------------------------------------------------
    # Callable Functions (Mimicking Analytical Resonators)
    # ---------------------------------------------------------
    def _parse_args(self, args, n):
        """
        Helper to cleanly allow calling with or without the mode 'n'.
        Allows for both func(x, n=1) and func(x, 1).
        """
        if n is not None:
            return args, n
            
        # If 'n' is passed positionally at the end (e.g., psi(x, y, 2))
        if len(args) == self.dimension + 1:
            return args[:-1], args[-1]
        # If 'n' is not passed at all (used internally by construct_Amatrix)
        elif len(args) == self.dimension:
            return args, None
        else:
            raise ValueError(f"Expected {self.dimension} spatial coordinates, plus an optional mode 'n'.")

    def psi(self, *args, n=None):
        """Returns interpolated mode shape. Specify 'n' for a specific mode."""
        coords, mode = self._parse_args(args, n)
        return self._evaluate_interpolator(self._interp_psi, coords, mode)
        
    def epsilon_xx(self, *args, n=None):
        """Returns interpolated xx-strain. Specify 'n' for a specific mode."""
        coords, mode = self._parse_args(args, n)
        return self._evaluate_interpolator(self._interp_exx, coords, mode)
        
    def epsilon_yy(self, *args, n=None):
        """Returns interpolated yy-strain. Specify 'n' for a specific mode."""
        if self.dimension == 1:
            raise AttributeError("epsilon_yy is not defined for 1D devices.")
        coords, mode = self._parse_args(args, n)
        return self._evaluate_interpolator(self._interp_eyy, coords, mode)
        
    def epsilon_xy(self, *args, n=None):
        """Returns interpolated xy-strain. Specify 'n' for a specific mode."""
        if self.dimension == 1:
            raise AttributeError("epsilon_xy is not defined for 1D devices.")
        coords, mode = self._parse_args(args, n)
        return self._evaluate_interpolator(self._interp_exy, coords, mode)

    def _evaluate_interpolator(self, interpolator, coords, n):
        """Evaluates the interpolator and extracts the specific mode if requested."""
        coords = [np.atleast_1d(np.squeeze(c)) for c in coords]
        
        # Evaluate for all modes simultaneously 
        if self.dimension == 1:
            res = interpolator(coords[0])         # Outputs: (Nx, n_modes)
            res_all = np.transpose(res)           # Transforms to: (n_modes, Nx)
        else:
            x, y = coords
            X, Y = np.meshgrid(x, y, indexing='ij')
            pts = np.stack([X, Y], axis=-1)
            res = interpolator(pts)               # Outputs: (Nx, Ny, n_modes)
            res_all = np.transpose(res, (2, 0, 1))# Transforms to: (n_modes, Nx, Ny)
            
        # If a specific mode is requested, return only that slice
        if n is not None:
            modes_list = list(self.modes)
            if n not in modes_list:
                raise ValueError(f"Mode {n} not found. Available modes are: {modes_list}")
            
            mode_idx = modes_list.index(n)
            return res_all[mode_idx]
            
        # Otherwise, return the full stack (used by construct_Amatrix)
        return res_all

    # ---------------------------------------------------------
    # A-Matrix Construction Engine
    # ---------------------------------------------------------
    def construct_Amatrix(self, *position_grids):
        """
        Constructs the A matrix for the Bayesian solver evaluated on the targeted spatial grids.
        """
        if len(position_grids) != self.dimension:
            raise ValueError(f"Expected {self.dimension} coordinate grids, got {len(position_grids)}")

        # 1. Evaluate the base 1D/2D interpolations
        psi_val = self.psi(*position_grids)
        exx_val = self.epsilon_xx(*position_grids)

        # 2. Build the parameter map dynamically based on dimension
        param_to_func = {
            'Theta_Ma': -psi_val**2,
            'Theta_Kxx': exx_val**2
        }

        # Only evaluate and add 2D strain components if the device is 2D
        if self.dimension == 2:
            eyy_val = self.epsilon_yy(*position_grids)
            exy_val = self.epsilon_xy(*position_grids)
            
            param_to_func.update({
                'Theta_Kyy': eyy_val**2,
                'Theta_Kxy': exy_val**2,
                'Theta_Kxxyy': exx_val * eyy_val,
                'Theta_Kxxxy': exx_val * exy_val,
                'Theta_Kxyyy': eyy_val * exy_val
            })

        # 3. Stack selected parameters based on user's parameter_map
        try:
            A_matrix_raw = np.stack([param_to_func[param] for param in self.parameter_map], axis=0)
        except KeyError as e:
            raise ValueError(f"Parameter {e} is not valid for a {self.dimension}D CustomResonator.")

        # 4. Transpose to target solver shape: (*spatial_dims, n_modes, n_params)
        if self.dimension == 1:
            # (n_params, n_modes, Nx) -> (Nx, n_modes, n_params)
            A_matrix = np.transpose(A_matrix_raw, (2, 1, 0))
        elif self.dimension == 2:
            # (n_params, n_modes, Nx, Ny) -> (Nx, Ny, n_modes, n_params)
            A_matrix = np.transpose(A_matrix_raw, (2, 3, 1, 0))

        return A_matrix

    
class CantileverBeam:
    """
    1D Euler-Bernoulli Cantilever Beam (Clamped-Free).

    - Position coordinates are assumed to be normalized from 0 (clamped) to 1 (free end).
    
    Inputs:
    - modes: List of mode numbers (e.g., [2, 3, 5]) or an integer specifying the number of modes starting from mode 1
    - parameters: List of parameter names (default: ['Theta_Ma', 'Theta_Ka']), ['Theta_Ma'] for only mass)
    """

    def __init__(self, modes, parameters=None):
        if parameters is None:
            parameters = ['Theta_Ma', 'Theta_Ka']

        self.dimension = 1
        self.parameter_map = parameters
        self.num_parameters = len(parameters)
        self.position_coordinates = [np.linspace(0,1,num=500)]
        
        # Handle integer vs list of modes
        if isinstance(modes, int):
            self.modes = list(range(1, modes + 1))
        else:
            self.modes = modes
            
        self.num_modes = len(self.modes)

    # Construct the A matrix for the Bayesian solver
    def construct_Amatrix(self,x):
        A_matrix = np.zeros((1,len(x), self.num_modes, self.num_parameters))
        for i, n in enumerate(self.modes):

            col_idx = 0
            if 'Theta_Ma' in self.parameter_map:
                A_matrix[0,:, i, col_idx] = -self.psi(x, n)**2
                col_idx += 1
            if 'Theta_Ka' in self.parameter_map:
                A_matrix[0, :, i, col_idx] = self.curv(x, n)**2
        return A_matrix

    def _get_beta(self, n):
        f = lambda b: np.cos(b) + 1.0 / np.cosh(b)
        bracket = [1.0, 2.0] if n == 1 else [(n - 0.5) * np.pi - 0.2, (n - 0.5) * np.pi + 0.2]
        return root_scalar(f, bracket=bracket).root
    
    def psi(self, x, n):
        """Mode shape function for the nth mode at position x."""
        beta = self._get_beta(n)
        C = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
        return (np.cosh(beta * x) - np.cos(beta * x)) - C * (np.sinh(beta * x) - np.sin(beta * x))
    
    def curv(self, x, n):
        """Normalized curvature function for the nth mode at position x."""
        beta = self._get_beta(n)
        C = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
        return (np.cosh(beta * x) + np.cos(beta * x)) - C * (np.sinh(beta * x) + np.sin(beta * x))

class DoublyClampedBeam:
    """
    1D Euler-Bernoulli Doubly Clamped Beam (Clamped-Clamped).

    - Position coordinates are assumed to be normalized from 0 (clamped) to 0.5 (middle of the beam).
    
    Inputs:
    - modes: List of mode numbers (e.g., [2, 3, 5]) or an integer specifying the number of modes starting from mode 1
    - parameters: List of parameter names (default: ['Theta_Ma', 'Theta_Ka']), ['Theta_Ma'] for only mass)

    """
    def __init__(self, modes, parameters=None):
        if parameters is None:
            parameters = ['Theta_Ma', 'Theta_Ka']
        self.dimension = 1
        self.parameter_map = parameters
        self.num_parameters = len(parameters)
        self.position_coordinates = [np.linspace(0,0.5,num=500)]
        
        # Handle integer vs list of modes
        if isinstance(modes, int):
            self.modes = list(range(1, modes + 1))
        else:
            self.modes = modes
            
        self.num_modes = len(self.modes)

    # Construct the A matrix for the Bayesian solver
    def construct_Amatrix(self,x):
        A_matrix = np.zeros((1,len(x), self.num_modes, self.num_parameters))
        for i, n in enumerate(self.modes):

            col_idx = 0
            if 'Theta_Ma' in self.parameter_map:
                A_matrix[0,:, i, col_idx] = -self.psi(x, n)**2
                col_idx += 1
            if 'Theta_Ka' in self.parameter_map:
                A_matrix[0, :, i, col_idx] = self.curv(x, n)**2
        return A_matrix

    def _get_beta(self, n):
        f = lambda b: np.cos(b) - 1.0 / np.cosh(b)
        bracket = [4.0, 5.0] if n == 1 else [(n + 0.5) * np.pi - 0.2, (n + 0.5) * np.pi + 0.2]
        return root_scalar(f, bracket=bracket).root
    
    def psi(self, x, n):
        """Mode shape function for the nth mode at position x."""
        beta = self._get_beta(n)
        C = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))
        return (np.cosh(beta * x) - np.cos(beta * x)) - C * (np.sinh(beta * x) - np.sin(beta * x))
    
    def curv(self, x, n):
        """Normalized curvature function for the nth mode at position x."""
        beta = self._get_beta(n)
        C = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))
        return (np.cosh(beta * x) + np.cos(beta * x)) - C * (np.sinh(beta * x) + np.sin(beta * x))

class Membrane:
    """
    2D Stressed Rectangular Membrane (Mass Sensing Only).

    - Position coordinates are assumed to be normalized from -0.5 to 0.5 in both x and y directions.
    
    Inputs:
    - modes: List of mode tuples (m, n) for the (m,n) mode, e.g., [(1,1), (2,1), (1,2)]
    """

    def __init__(self, modes):
        self.dimension = 2
        self.parameter_map=['Theta_Ma']
        self.modes = modes  
        self.num_modes = len(modes)
        self.num_parameters = len(self.parameter_map)
        x = np.linspace(0,0.5,num=500)
        y = np.linspace(0,0.5,num=500)
        self.position_coordinates = [x,y]

    # Construct the A matrix for the Bayesian solver
    def construct_Amatrix(self,x,y):
        x = np.atleast_1d(np.squeeze(x))
        y = np.atleast_1d(np.squeeze(y))
        A_matrix = np.zeros((len(x), len(y), self.num_modes, 1))
        X, Y = np.meshgrid(x, y, indexing='ij')
        for i, (m, n) in enumerate(self.modes):
            psi = self.psi(X, Y, m, n)
            A_matrix[:, :, i, 0] = -psi**2
        return A_matrix

    def psi(self, x, y, m, n):
        """Mode shape function for the (m,n) mode at position (x,y)."""
        return 2 * np.sin(m * np.pi * (x + 0.5)) * np.sin(n * np.pi * (y + 0.5))

class String:
    """
    1D Stressed Doubly Clamped Beam (Clamped-Clamped) (Mass Sensing Only).

    - Position coordinates are assumed to be normalized from 0 (clamped) to 0.5 (middle of the beam).
    
    Inputs:
    - modes: List of mode numbers (e.g., [2, 3, 5]) or an integer specifying the number of modes starting from mode 1

    """
    def __init__(self, modes):
        self.dimension = 1
        self.parameter_map = ['Theta_Ma']
        self.num_parameters = len(self.parameter_map)
        self.position_coordinates = [np.linspace(0,0.5,num=500)]
        
        # Handle integer vs list of modes
        if isinstance(modes, int):
            self.modes = list(range(1, modes + 1))
        else:
            self.modes = modes
            
        self.num_modes = len(self.modes)

    # Construct the A matrix for the Bayesian solver
    def construct_Amatrix(self,x):
        A_matrix = np.zeros((1,len(x), self.num_modes, self.num_parameters))
        for i, n in enumerate(self.modes):
            A_matrix[0,:, i, 0] = -self.psi(x, n)**2
        return A_matrix
    
    def psi(self, x, n):
        """Mode shape function for the nth mode at position x."""
        return np.sin(n*np.pi*x)