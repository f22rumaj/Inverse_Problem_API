import numpy as np
import psutil
import gc
from tqdm.auto import tqdm
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import label
from .results import Results
from .measurement import Measurement
try:
    import cupy as cp
    cp.zeros(1)
except:
    pass


class InverseSolver:
    """
    Engine for solving the nanomechanical inverse problem.
    Supports 'profile_likelihood' and 'brute_force' methods.
    """

    def __init__(self, resonator, method='profile_likelihood', device='CPU'):

        self.device = device.upper()
        if self.device == 'GPU':
            try:
                cp.zeros(1)
                available_memory = cp.cuda.Device(0).mem_info[0]
                print("GPU is available. Memory: ", available_memory/1e9, " GB")
            except:
                print("WARNING: CuPy not found or GPU unavailable! Falling back to CPU.")
                self.device = 'CPU'
        else:
            self.device = 'CPU'


        self.resonator = resonator
        self.method = method.lower()
        
        
        if self.method not in ['profile_likelihood', 'brute_force']:
            raise ValueError("Method must be 'profile_likelihood' or 'brute_force'.")
        
        if self.resonator.dimension + self.resonator.num_parameters > 3 and self.method == 'brute_force':
            raise ValueError("Brute force method is only supported for up to 3 combined dimensions.")

        if self.method == 'profile_likelihood' and self.device == 'GPU':
            print("GPU is not available for profile likelihood method. Falling back to CPU.")
            self.device = 'CPU'
            

    def run(self, measurement: 'Measurement', position_grids: list[np.ndarray], pgrids: list[np.ndarray], return_position_marginals: bool = True) -> Results:
        """
        Runs the solver for the inverse problem.
        
        Inputs:
        - measurement: An instance of the Measurement class containing frequency shifts and covariance.
        - position_grids: List of spatial grids for the resonator.
        - pgrids: List of parameter grids.
        - return_position_marginals: Boolean indicating whether to return position marginals.

        Outputs:
        - Results object
        """

        # Check if the number of modes in the measurement matches the number of modes in the resonator.
        if measurement.num_modes != self.resonator.num_modes:
            raise ValueError("Number of modes in measurement must match number of modes in resonator.")

        # Validate that the number of parameter grids matches the expected physical parameters
        if len(pgrids) != self.resonator.num_parameters:
            raise ValueError(f"Expected {self.resonator.num_parameters} parameter grids based on the resonator, but received {len(pgrids)}.")

        # Route to the specific solver method.
        if self.method == 'profile_likelihood':
            return self._run_profile_likelihood(measurement, position_grids, pgrids, return_position_marginals)
        elif self.method == 'brute_force':
            return self._run_brute_force(measurement, position_grids, pgrids, return_position_marginals)

    
    # Get optimal chunk size for the brute force method to avoid memory OOM errors
    def get_optimal_chunk_size(self, parameter_shape, total_spatial_points, nmodes):
        """Calculates safe chunk sizes, properly tracking CPU/GPU fallback state."""
        if self.device == 'GPU':
            available_memory = cp.cuda.Device(0).mem_info[0]
            safety_factor = 0.8
        else:
            available_memory = psutil.virtual_memory().available
            safety_factor = 0.3
            
        safe_mem = available_memory * safety_factor 
        base_elements = np.prod(parameter_shape)
        elements_per_slice = base_elements * ((3 * nmodes) + 2)
        einsum_overhead_multiplier = 2

        bytes_per_slice = elements_per_slice * 8 * einsum_overhead_multiplier
        max_slices = max(1, int(safe_mem / bytes_per_slice))
        
        return min(max_slices, total_spatial_points)
    
    # Brute force solver
    def _run_brute_force(self, measurement, position_grids, pgrids, return_position_marginals):
        """
        The central router for brute force. Pre-computes all heavy matrices ONCE
        and shares them between Pass 1 and Pass 2 via a Context Dictionary.
        """
        freq_shifts_np = np.atleast_2d(measurement.freq_shifts)
        n_measurements, nmodes = freq_shifts_np.shape
        
        # 1. Setup Device & Fallback
        if self.device == 'GPU':
            xp = cp
        else:
            xp = np

        # Squeeze out extra dimensions and ensure they are flat 1D arrays
        pgrids = [np.atleast_1d(np.squeeze(p)) for p in pgrids]
        
        # 2. Generalize Spatial Grids & Volumes
        if self.resonator.dimension == 2:
            xgrid = np.atleast_1d(np.squeeze(position_grids[0]))
            ygrid = np.atleast_1d(np.squeeze(position_grids[1]))
            shape_spatial = (xgrid.size, ygrid.size)
            dx = xgrid[1] - xgrid[0] if xgrid.size > 1 else 1.0
            dy = ygrid[1] - ygrid[0] if ygrid.size > 1 else 1.0
            spatial_dV = dx * dy
        else:
            xgrid = np.atleast_1d(np.squeeze(position_grids[0]))
            ygrid = None
            shape_spatial = (xgrid.size,)
            dx = xgrid[1] - xgrid[0] if xgrid.size > 1 else 1.0
            spatial_dV = dx
            
        # 3. Construct and Flatten Amatrix
        Amatrix = self.resonator.construct_Amatrix(*position_grids)
        n_params = Amatrix.shape[-1]
        n_spatial = Amatrix.size // (nmodes * n_params)
        Amatrix_flat = Amatrix.reshape(1, n_spatial, nmodes, n_params)

        # 4. Construct Parameter Grids
        f_mesh = np.meshgrid(*pgrids, indexing='ij')
        f_Theta = np.stack(f_mesh, axis=0)
        parameter_shape = f_Theta.shape[1:] 
        
        dp_list = [p[1] - p[0] if len(p) > 1 else 1.0 for p in pgrids]
        dV_params = np.prod(dp_list)

        # Number of chunks for memory management
        num_chunks = -(-n_spatial // self.get_optimal_chunk_size(parameter_shape, n_spatial, nmodes))

        # 5. Move Static Arrays to Device
        ctx = {
            'xp': xp,
            'n_measurements': n_measurements,
            'nmodes': nmodes,
            'S_inv_d': xp.asarray(measurement.inv_covariance_matrix),
            'freq_shifts_d': xp.asarray(freq_shifts_np),
            'f_Theta_d': xp.asarray(f_Theta),
            'Amatrix_flat': Amatrix_flat, # Remains on CPU, chunked dynamically later
            'xgrid': xgrid,
            'ygrid': ygrid,
            'pgrids': pgrids,
            'shape_spatial': shape_spatial,
            'n_spatial': n_spatial,
            'spatial_dV': spatial_dV,
            'n_params': n_params,
            'dp_list': dp_list,
            'dV_params': dV_params,
            'chunk_size': self.get_optimal_chunk_size(parameter_shape, n_spatial, nmodes),
            'num_chunks': num_chunks
        }

        # --- Initialize Granular Progress Bar ---
        # Total steps = (chunks * measurements) for Pass 1 + (chunks * measurements) for Pass 2
        total_steps = 2 * num_chunks * n_measurements
        pbar = tqdm(total=total_steps, desc=f"Brute Force ({self.device})", unit="step", leave=False)
        ctx['pbar'] = pbar

        # --- Execute Passes using Shared Context ---
        optimal_parameters = self._pass_1_global_minimum(ctx)
        marginals = self._pass_2_marginals(ctx, optimal_parameters, return_position_marginals)

        pbar.close()
        
        return Results(optimal_parameters, marginals)

    # Pass 1: Scans for the global minimum across all spatial points for brute force method.
    def _pass_1_global_minimum(self, ctx):
        
        xp, n_meas, nmodes = ctx['xp'], ctx['n_measurements'], ctx['nmodes']
        chunk_size, num_chunks, n_spatial = ctx['chunk_size'], ctx['num_chunks'], ctx['n_spatial']
        
        global_min_D = [np.inf] * n_meas
        best_indices = [None] * n_meas

        ctx['pbar'].set_postfix_str(f"Pass 1: Global Minimum")

        for chunk_idx, start_idx in enumerate(range(0, n_spatial, chunk_size)):
                
            end_idx = min(start_idx + chunk_size, n_spatial)
            A_chunk_d = xp.asarray(ctx['Amatrix_flat'][:, start_idx:end_idx, :, :])
            B_chunk = xp.tensordot(A_chunk_d, ctx['f_Theta_d'], axes=([3], [0]))
            del A_chunk_d
            
            for m_idx in range(n_meas):
                Delta_exp_chunk = ctx['freq_shifts_d'][m_idx].reshape(1, 1, nmodes, *([1] * (ctx['f_Theta_d'].ndim - 1)))
                
                R_chunk = Delta_exp_chunk - B_chunk
                tmp_chunk = xp.einsum('yxm...,mn->yxn...', R_chunk, ctx['S_inv_d'])
                D_chunk = xp.einsum('yxk...,yxk...->yx...', tmp_chunk, R_chunk)

                del R_chunk, tmp_chunk, Delta_exp_chunk

                D_chunk_sq = xp.squeeze(D_chunk, axis=0)
                
                chunk_min = float(xp.min(D_chunk_sq))
                
                if chunk_min < global_min_D[m_idx]:
                    global_min_D[m_idx] = chunk_min
                    
                    local_flat_idx = xp.argmin(D_chunk_sq)
                    local_multi_idx = xp.unravel_index(local_flat_idx, D_chunk_sq.shape)
                    if self.device == 'GPU': local_multi_idx = tuple(idx.item() for idx in local_multi_idx)
                        
                    global_spatial_idx = start_idx + local_multi_idx[0]
                    param_indices = local_multi_idx[1:]
                    
                    if self.resonator.dimension == 2:
                        x_idx, y_idx = np.unravel_index(global_spatial_idx, ctx['shape_spatial'])
                        best_indices[m_idx] = (x_idx, y_idx, *param_indices)
                    else:
                        best_indices[m_idx] = (global_spatial_idx, *param_indices)

                del D_chunk_sq

                # Update the progress bar
                ctx['pbar'].update(1)

            del B_chunk

            if self.device == 'GPU':
                nclim = 5 
                xp.get_default_memory_pool().free_all_blocks()
            else:
                nclim = 1
                gc.collect()
            
        # --- Compile Dynamic Results ---
        batch_results = []
        for m_idx in range(n_meas):
            idx_tuple = best_indices[m_idx]
            res = {'min_D': global_min_D[m_idx]}
            
            if self.resonator.dimension == 2:
                res['x'], res['y'] = ctx['xgrid'][idx_tuple[0]], ctx['ygrid'][idx_tuple[1]]
                param_offset = 2 
            else:
                res['x'] = ctx['xgrid'][idx_tuple[0]]
                param_offset = 1
                
            for i, param_name in enumerate(self.resonator.parameter_map):
                res[param_name] = ctx['pgrids'][i][idx_tuple[param_offset + i]]
                
            batch_results.append(res)
        
        return batch_results[0] if n_meas == 1 else batch_results


    # Pass 2: Computes the marginal distributions for each parameter and optionally for spatial positions for the brute force method.
    def _pass_2_marginals(self, ctx, optimal_parameters, return_position_marginals):
        
        xp, n_meas, nmodes = ctx['xp'], ctx['n_measurements'], ctx['nmodes']
        chunk_size, n_spatial = ctx['chunk_size'], ctx['n_spatial']

        # Update progress bar status text
        ctx['pbar'].set_postfix_str(f"Pass 2: Marginals")
        
        # Smart min_D extraction
        if isinstance(optimal_parameters, dict) and 'min_D' in optimal_parameters:
            global_min_D = [optimal_parameters['min_D']]
        elif isinstance(optimal_parameters, list) and len(optimal_parameters) > 0 and isinstance(optimal_parameters[0], dict):
            global_min_D = [res['min_D'] for res in optimal_parameters]
        else:
            global_min_D = [optimal_parameters]
            
        # Pre-allocate accumulators
        batch_results = []
        for _ in range(n_meas):
            m_dict = {
                'total_prob_d': xp.array(0.0, dtype=xp.float64), 
                'param_accums_d': [xp.zeros(p.shape) for p in ctx['pgrids']]
            }
            if return_position_marginals:
                m_dict['marg_spatial_full_d'] = xp.zeros(n_spatial)
            batch_results.append(m_dict)

        for chunk_idx, start_idx in enumerate(range(0, n_spatial, chunk_size)):
            end_idx = min(start_idx + chunk_size, n_spatial)
            A_chunk_d = xp.asarray(ctx['Amatrix_flat'][:, start_idx:end_idx, :, :])
            B_chunk = xp.tensordot(A_chunk_d, ctx['f_Theta_d'], axes=([3], [0]))
            del A_chunk_d
            
            for m_idx in range(n_meas):
                Delta_exp_chunk = ctx['freq_shifts_d'][m_idx].reshape(1, 1, nmodes, *([1] * (ctx['f_Theta_d'].ndim - 1)))
                
                R_chunk = Delta_exp_chunk - B_chunk
                tmp_chunk = xp.einsum('yxm...,mn->yxn...', R_chunk, ctx['S_inv_d'])
                D_chunk = xp.einsum('yxk...,yxk...->yx...', tmp_chunk, R_chunk)

                del R_chunk, tmp_chunk, Delta_exp_chunk

                D_chunk_sq = xp.squeeze(D_chunk, axis=0)
                del D_chunk
                
                prob_chunk = xp.exp(-0.5 * (D_chunk_sq - global_min_D[m_idx]))
                del D_chunk_sq
                
                chunk_total_prob = xp.sum(prob_chunk) * (ctx['spatial_dV'] * ctx['dV_params'])
                batch_results[m_idx]['total_prob_d'] += chunk_total_prob
                
                for i in range(ctx['n_params']):
                    axes_to_sum = tuple([0] + [j+1 for j in range(ctx['n_params']) if j != i])
                    dV_other = ctx['spatial_dV'] * np.prod([ctx['dp_list'][j] for j in range(ctx['n_params']) if j != i])
                    batch_results[m_idx]['param_accums_d'][i] += xp.sum(prob_chunk, axis=axes_to_sum) * dV_other
                
                if return_position_marginals:
                    axes_to_sum_spatial = tuple(range(1, ctx['n_params'] + 1))
                    marg_spatial_chunk = xp.sum(prob_chunk, axis=axes_to_sum_spatial) * ctx['dV_params']
                    batch_results[m_idx]['marg_spatial_full_d'][start_idx:end_idx] = marg_spatial_chunk

                del prob_chunk

                # Update the progress bar
                ctx['pbar'].update(1)
                
            del B_chunk
            
            if self.device == 'GPU':
                xp.get_default_memory_pool().free_all_blocks()
            else:
                gc.collect()

        # --- Finalization & Normalization ---
        final_returns = []
        for m_idx in range(n_meas):
            total_prob_mass = float(batch_results[m_idx]['total_prob_d'].get() if self.device == 'GPU' else batch_results[m_idx]['total_prob_d'])
            if total_prob_mass == 0: total_prob_mass = 1.0 

            res = {'xgrid': ctx['xgrid'], 'pgrids': ctx['pgrids']}

            for i, param_name in enumerate(self.resonator.parameter_map):
                param_marg = batch_results[m_idx]['param_accums_d'][i].get() if self.device == 'GPU' else batch_results[m_idx]['param_accums_d'][i]
                res[f'marg_{param_name}'] = param_marg / total_prob_mass

            if return_position_marginals:
                marg_spatial_full = batch_results[m_idx]['marg_spatial_full_d'].get() if self.device == 'GPU' else batch_results[m_idx]['marg_spatial_full_d']

                if self.resonator.dimension == 2:
                    marg_xy = marg_spatial_full.reshape(ctx['shape_spatial'])
                    res['marg_xy'] = marg_xy / total_prob_mass
                    res['ygrid'] = ctx['ygrid']
                else:
                    res['marg_x'] = marg_spatial_full / total_prob_mass

            final_returns.append(res)
            
        return final_returns[0] if n_meas == 1 else final_returns


    # Profile likelihood solver for the nanomechanical inverse problem.
    def _run_profile_likelihood(self, measurement, position_grids, pgrids, return_position_marginals):
        
        freq_shifts_np = np.atleast_2d(measurement.freq_shifts)
        n_measurements = freq_shifts_np.shape[0]
        S_inv_d = measurement.inv_covariance_matrix

        # Squeeze out extra dimensions and ensure they are flat 1D arrays
        pgrids = [np.atleast_1d(np.squeeze(p)) for p in pgrids]
        
        batch_optimal = []
        batch_marginals = []
        
        # Iterating through all measurements
        for m_idx in tqdm(range(n_measurements), desc="Profile Likelihood (CPU)"):
            Delta_d = freq_shifts_np[m_idx]
            Amatrix = self.resonator.construct_Amatrix(*position_grids)
            
            # Step 1: Refine the spatial grid and compute Profile Likelihood
            if self.resonator.dimension == 2:
                new_pos, Lprof, Theta_hat_clipped, Sigma_Theta, min_D = self._Lprof_refine_2D(m_idx,
                    self.resonator, Delta_d, S_inv_d, Amatrix, pgrids, position_grids
                )
            else:
                new_pos, Lprof, Theta_hat_clipped, Sigma_Theta, min_D = self._Lprof_refine_1D(m_idx,
                    self.resonator, Delta_d, S_inv_d, Amatrix, pgrids, position_grids
                )
                
            # Step 2: Extract Global Minimum index
            max_flat_idx = np.argmax(Lprof)
            
            # Step 3: Initialize the separate result dictionaries
            opt_res = {'min_D': float(min_D)}
            marg_res = {'xgrid': new_pos[0], 'pgrids': pgrids}
            
            if self.resonator.dimension == 2:
                idx_x, idx_y = np.unravel_index(max_flat_idx, Lprof.shape)
                opt_res['x'] = new_pos[0][idx_x]
                opt_res['y'] = new_pos[1][idx_y]
                marg_res['ygrid'] = new_pos[1]
            else:
                opt_res['x'] = new_pos[0][max_flat_idx]

            # Save optimal parameters
            for i, param_name in enumerate(self.resonator.parameter_map):
                opt_res[param_name] = float(Theta_hat_clipped[max_flat_idx, i])
                
            # Step 4: Construct Marginals
            if return_position_marginals:
                if self.resonator.dimension == 2:
                    dx = new_pos[0][1] - new_pos[0][0] if len(new_pos[0]) > 1 else 1.0
                    dy = new_pos[1][1] - new_pos[1][0] if len(new_pos[1]) > 1 else 1.0
                    spatial_dV = dx * dy
                    marg_xy = Lprof / (np.sum(Lprof) * spatial_dV)
                    marg_res['marg_xy'] = marg_xy
                else:
                    dx = new_pos[0][1] - new_pos[0][0] if len(new_pos[0]) > 1 else 1.0
                    marg_x = Lprof / (np.sum(Lprof) * dx)
                    marg_res['marg_x'] = marg_x

            # Parameter Marginals: Constructed as a Gaussian around \hat{\Theta}
            for i, param_name in enumerate(self.resonator.parameter_map):
                pgrid = pgrids[i]
                dp = pgrid[1] - pgrid[0] if len(pgrid) > 1 else 1.0
                mean = Theta_hat_clipped[max_flat_idx, i]
                variance = Sigma_Theta[max_flat_idx, i, i]
                
                if variance <= 0:
                    marg_param = np.zeros_like(pgrid)
                    marg_param[np.argmin(np.abs(pgrid - mean))] = 1.0 / dp
                else:
                    marg_param = np.exp(-0.5 * ((pgrid - mean)**2) / variance)
                    marg_param /= (np.sum(marg_param) * dp) 
                    
                marg_res[f'marg_{param_name}'] = marg_param

            batch_optimal.append(opt_res)
            batch_marginals.append(marg_res)
            
        if n_measurements == 1:
            return Results(batch_optimal[0], batch_marginals[0])
        else:
            return Results(batch_optimal, batch_marginals)

    # Core mathematical engine for finding the profile likelihood.
    def _Lprof_calculation(self, Delta_d, S_inv_d, Amatrix, pgrids):
        
        n_params = Amatrix.shape[-1]
        nmodes = len(Delta_d)

        S_inv_Delta = np.dot(S_inv_d, Delta_d)
        n_spatial = int(np.prod(Amatrix.shape[:-2]))
        A_d = np.asarray(Amatrix.reshape(n_spatial, nmodes, n_params))

        Sigma_inv = np.einsum('smp,mn,snq->spq', A_d, S_inv_d, A_d)
        lam = 1e-20 * np.eye(n_params, dtype=np.float64)
        Sigma_inv_reg = Sigma_inv + lam

        Sigma_Theta = np.linalg.inv(Sigma_inv_reg)
        A_T_S_inv_Delta = np.einsum('smp,m->sp', A_d, S_inv_Delta) 
        Theta_hat = np.einsum('spq,sq->sp', Sigma_Theta, A_T_S_inv_Delta) 

        Theta_hat_clipped = np.zeros_like(Theta_hat)
        for i in range(n_params):
            p_min, p_max = pgrids[i][0], pgrids[i][-1]
            Theta_hat_clipped[:, i] = np.clip(Theta_hat[:, i], min(p_min, p_max), max(p_min, p_max))

        residual = Delta_d[None, :] - np.einsum('smp,sp->sm', A_d, Theta_hat_clipped)
        tmp_res = np.einsum('mn,sn->sm', S_inv_d, residual)
        Gamma = np.einsum('sm,sm->s', residual, tmp_res)

        min_Gamma = np.min(Gamma)
        Lprof = np.exp(-0.5 * (Gamma - min_Gamma))

        return Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma

    # Iterative spatial grid refinement for 1D resonators.
    def _Lprof_refine_1D(self, m_idx, resonator, Delta_d, S_inv_d, Amatrix, pgrids, position_grids):
        Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma = self._Lprof_calculation(Delta_d, S_inv_d, Amatrix, pgrids)
        Lprof = np.squeeze(Lprof)

        global_max = np.max(Lprof)
        min_height = 0.01 * global_max
        peak_indices, _ = find_peaks(Lprof, height=min_height, prominence=0.05 * global_max)

        if len(peak_indices) == 0:
            return position_grids, Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma
        elif len(peak_indices) > 1:
            print(f'Warning: More than one peak found in measurement index {m_idx}. Tracking the highest peak...')

        maxpeak = np.argmax(Lprof[peak_indices])
        results_half = peak_widths(Lprof, peak_indices, rel_height=0.99)
        _, _, left_ips, right_ips = results_half

        npeak = maxpeak
        lidx = max(int(left_ips[npeak]) - 5, 0)
        ridx = min(int(right_ips[npeak]) + 5, len(position_grids[0])-1)
        new_position_grids = [np.linspace(position_grids[0][lidx], position_grids[0][ridx], num=500)]
        Amatrix = resonator.construct_Amatrix(*new_position_grids)

        Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma = self._Lprof_calculation(Delta_d, S_inv_d, Amatrix, pgrids)
        Lprof = np.squeeze(Lprof)

        cond = np.sum(Lprof > min_height) < 50

        while cond:
            global_max = np.max(Lprof)
            min_height = 0.01 * global_max
            peak_indices, _ = find_peaks(Lprof, height=min_height, prominence=0.05 * global_max)
            
            if len(peak_indices) == 0:
                break
                
            maxpeak = np.argmax(Lprof[peak_indices])
            results_half = peak_widths(Lprof, peak_indices, rel_height=0.99)
            _, _, left_ips, right_ips = results_half
            npeak = maxpeak
            lidx = max(int(left_ips[npeak]) - 5, 0)
            ridx = min(int(right_ips[npeak]) + 5, len(new_position_grids[0])-1)

            new_position_grids = [np.linspace(new_position_grids[0][lidx], new_position_grids[0][ridx], num=500)]
            Amatrix = resonator.construct_Amatrix(*new_position_grids)

            Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma = self._Lprof_calculation(Delta_d, S_inv_d, Amatrix, pgrids)
            Lprof = np.squeeze(Lprof)

            min_height = 0.01 * np.max(Lprof)
            if np.sum(Lprof > min_height) >= 50:
                cond = False

        return new_position_grids, Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma

    # Iterative spatial grid refinement for 2D resonators (Multi-Peak Tracking).
    def _Lprof_refine_2D(self, m_idx, resonator, Delta_d, S_inv_d, Amatrix, pgrids, position_grids):
        
        Lprof_np, Theta_hat_clipped, Sigma_Theta, min_Gamma = self._Lprof_calculation(Delta_d, S_inv_d, Amatrix, pgrids)
        
        x_grid = np.atleast_1d(np.squeeze(position_grids[0]))
        y_grid = np.atleast_1d(np.squeeze(position_grids[1]))
        
        Nx, Ny = len(x_grid), len(y_grid)
        Lprof = Lprof_np.reshape((Nx, Ny))

        global_max = np.max(Lprof)
        min_height = 0.01 * global_max 
        
        # Identify all candidate regions
        mask = Lprof > min_height
        labeled_array, num_features = label(mask)

        if num_features == 0:
            print(f"Warning: No region found above threshold in measurement index {m_idx}.")
            return [x_grid, y_grid], Lprof, Theta_hat_clipped, Sigma_Theta, min_Gamma

        if num_features > 1:
            print(f"Warning: {num_features} candidate peaks found in measurement index {m_idx}. Tracking the highest peak...")

        best_peak_data = None
        best_min_Gamma = np.inf  # We track Gamma (error) because Lprof is normalized to 1
        buffer = 5 

        # Refine each peak independently
        for feature_id in range(1, num_features + 1):
            
            # Extract initial bounding box for this specific peak
            idx_x, idx_y = np.where(labeled_array == feature_id)
            
            lidx_x = max(np.min(idx_x) - buffer, 0)
            ridx_x = min(np.max(idx_x) + buffer, len(x_grid) - 1)
            lidx_y = max(np.min(idx_y) - buffer, 0)
            ridx_y = min(np.max(idx_y) + buffer, len(y_grid) - 1)
            
            new_x_grid = np.linspace(x_grid[lidx_x], x_grid[ridx_x], num=500)
            new_y_grid = np.linspace(y_grid[lidx_y], y_grid[ridx_y], num=500)
            
            new_pos = [new_x_grid, new_y_grid]
            
            # Local refinement loop for this peak
            while True:
                Amatrix_local = resonator.construct_Amatrix(*new_pos)
                Lprof_local_np, Theta_local, Sigma_local, Gamma_local = self._Lprof_calculation(Delta_d, S_inv_d, Amatrix_local, pgrids)
                Lprof_local = Lprof_local_np.reshape((500, 500))
                
                local_min_height = 0.01 * np.max(Lprof_local)
                
                if np.sum(Lprof_local > local_min_height) >= 2500:
                    # Break loop, this peak is fully resolved
                    break
                else:
                    # Not resolved enough, zoom in again on this specific sub-peak
                    mask_local = Lprof_local > local_min_height
                    lbl_local, num_feat_local = label(mask_local)
                    
                    if num_feat_local > 1:
                        max_flat = np.argmax(Lprof_local)
                        max_x, max_y = np.unravel_index(max_flat, Lprof_local.shape)
                        target_lbl = lbl_local[max_x, max_y]
                        loc_x, loc_y = np.where(lbl_local == target_lbl)
                    else:
                        loc_x, loc_y = np.where(mask_local)
                        
                    if len(loc_x) == 0:
                        break 
                        
                    lidx_x = max(np.min(loc_x) - buffer, 0)
                    ridx_x = min(np.max(loc_x) + buffer, len(new_x_grid) - 1)
                    lidx_y = max(np.min(loc_y) - buffer, 0)
                    ridx_y = min(np.max(loc_y) + buffer, len(new_y_grid) - 1)
                    
                    new_x_grid = np.linspace(new_x_grid[lidx_x], new_x_grid[ridx_x], num=500)
                    new_y_grid = np.linspace(new_y_grid[lidx_y], new_y_grid[ridx_y], num=500)
                    new_pos = [new_x_grid, new_y_grid]
            
            # Compare fully resolved peaks
            # We compare Gamma (the actual residual error), not Lprof, 
            # because Lprof max is always normalized to 1 in each sub-grid.
            if Gamma_local < best_min_Gamma:
                best_min_Gamma = Gamma_local
                best_peak_data = (new_pos, Lprof_local, Theta_local, Sigma_local, Gamma_local)

        return best_peak_data
        
