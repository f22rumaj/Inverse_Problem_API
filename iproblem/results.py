import numpy as np
import matplotlib.pyplot as plt
import math

class Results:
    """
    A container with the solution of the inverse problem. It contains the optimal parameters and the marginals and provides defferent visualization tools.  
    """

    def __init__(self, optimal_parameters, marginals):
        # Normalize inputs to lists to seamlessly handle both single and batch measurements
        self.optimal_parameters = [optimal_parameters] if isinstance(optimal_parameters, dict) else optimal_parameters
        self.marginals = [marginals] if isinstance(marginals, dict) else marginals
        self.n_meas = len(self.optimal_parameters)

        # Dynamically identify physical parameters (excluding spatial variables)
        self.param_names = [
            k.replace('marg_', '') for k in self.marginals[0].keys() 
            if k.startswith('marg_') and k not in ['marg_x', 'marg_xy']
        ]
        
        # Map parameter grids to their names for easy access
        self.pgrids_map = {p_name: self.marginals[0]['pgrids'][i] for i, p_name in enumerate(self.param_names)}
        
        # Detect dimensionality
        self.is_2d = 'marg_xy' in self.marginals[0]

    def plot_single_measurement(self, index, true_values=None):
        """
        Plots the spatial and parameter marginals for a specific measurement index.

        Inputs:
        index: int, index of the measurement to plot
        true_values (optional): dictionary mapping parameter names (and 'x', 'y') to their true values.
        """
        if index >= self.n_meas or index < 0:
            raise ValueError(f"Measurement index {index} is out of bounds (0 to {self.n_meas - 1}).")

        opt_m = self.optimal_parameters[index]
        marg_m = self.marginals[index]

        # Calculate layout grid (1 spatial plot + N parameter plots)
        total_plots = 1 + len(self.param_names)
        cols = 4 if total_plots > 4 else total_plots
        rows = math.ceil(total_plots / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.atleast_1d(axs).flatten()  # Flatten to iterate easily

        # 1. Plot Spatial Marginal
        if self.is_2d:
            X, Y = np.meshgrid(marg_m['xgrid'], marg_m['ygrid'], indexing='ij')
            cf = axs[0].contourf(X, Y, marg_m['marg_xy'], levels=50, cmap='viridis')
            fig.colorbar(cf, ax=axs[0], label='Prob Density')
            
            axs[0].plot(opt_m['x'], opt_m['y'], 'w+', markersize=12, markeredgewidth=2, label='Opt Pos')
            if true_values and 'x' in true_values and 'y' in true_values:
                axs[0].plot(true_values['x'], true_values['y'], 'rx', markersize=10, markeredgewidth=2, label='True Pos')
            
            axs[0].set_title(f"Measurement {index}: Spatial Marginal (x,y)")
            axs[0].set_xlabel('Position x')
            axs[0].set_ylabel('Position y')
        else:
            axs[0].plot(marg_m['xgrid'], marg_m['marg_x'], 'b-', label='Marginal PDF')
            axs[0].axvline(opt_m['x'], color='g', linestyle=':', linewidth=2, label=f"Opt x: {opt_m['x']:.4f}")
            if true_values and 'x' in true_values:
                axs[0].axvline(true_values['x'], color='r', linestyle='--', label=f"True x: {true_values['x']:.4f}")
            
            axs[0].set_title(f"Measurement {index}: Spatial Marginal (x)")
            axs[0].set_xlabel('Position x')
            axs[0].set_ylabel('Probability Density')
            
        axs[0].legend(loc='best')

        # 2. Plot Parameter Marginals
        for i, p_name in enumerate(self.param_names):
            ax = axs[i + 1]
            p_grid = self.pgrids_map[p_name]
            prob_p = marg_m[f'marg_{p_name}']
            
            ax.plot(p_grid, prob_p, 'b-', label='Marginal PDF')
            ax.axvline(opt_m[p_name], color='g', linestyle=':', linewidth=2, label=f'Opt: {opt_m[p_name]:.2f}')
            
            if true_values and p_name in true_values:
                ax.axvline(true_values[p_name], color='r', linestyle='--', label=f'True: {true_values[p_name]:.2f}')
                
            ax.set_title(f"{p_name.replace('Theta_', '')} Marginal")
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Probability Density')
            ax.legend(fontsize=9, loc='best')

        # Hide any unused subplots
        for j in range(total_plots, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def plot_aggregated_measurements(self):
        """
        Aggregates (sums and renormalizes) the parameter marginals across ALL measurements.
        """
        if self.n_meas <= 1:
            print("Warning: Only 1 measurement found. Aggregation is equivalent to single measurement plot.")

        total_plots = len(self.param_names)
        cols = 4 if total_plots > 4 else total_plots
        rows = math.ceil(total_plots / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.atleast_1d(axs).flatten()

        print(f"Aggregating parameter marginals over {self.n_meas} measurements...")

        for i, p_name in enumerate(self.param_names):
            ax = axs[i]
            p_grid = self.pgrids_map[p_name]
            
            # Sum probabilities across all measurements
            agg_prob = np.zeros_like(p_grid)
            for marg_m in self.marginals:
                agg_prob += marg_m[f'marg_{p_name}']
                
            # Normalize the aggregated probability
            dp = p_grid[1] - p_grid[0] if len(p_grid) > 1 else 1.0
            area = np.sum(agg_prob) * dp
            if area > 0:
                agg_prob /= area
                
            # Plot ONLY the aggregated marginal PDF
            ax.plot(p_grid, agg_prob, 'b-', linewidth=2, label='Aggregated PDF')
            
            ax.set_title(f"Aggregated {p_name.replace('Theta_', '')}")
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Probability Density')
            ax.legend(fontsize=9, loc='best')

        # Hide any unused subplots
        for j in range(total_plots, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()