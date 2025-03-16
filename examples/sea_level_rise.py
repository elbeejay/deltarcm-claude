#!/usr/bin/env python
"""
Example script to run DeltaRCM with sea level rise.

This example shows how to set up and run a DeltaRCM simulation with sea level rise,
and demonstrates the effect of sea level rise on delta morphology.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from deltarcm.model import DeltaRCM
from deltarcm.utils import plot_delta_morphology, plot_delta_3d, plot_cross_section


def run_simulation_with_slr(slr_rate):
    """
    Run a DeltaRCM simulation with specified sea level rise rate.
    
    Parameters
    ----------
    slr_rate : float
        Sea level rise rate in mm/yr.
    
    Returns
    -------
    DeltaRCM
        The model instance after simulation.
    """
    # Convert mm/yr to m/day
    slr_m_day = slr_rate / 1000 / 365.25
    
    # Create output directory
    output_dir = f'./output_slr_{int(slr_rate)}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure the model
    config = {
        # Grid parameters
        'L': 120,               # Domain length [cells]
        'W': 100,               # Domain width [cells]
        'cell_size': 10,        # Cell size [m]
        
        # Time parameters
        'dt': 86400,            # Time step [s] (1 day)
        'n_steps': 100,         # Number of time steps to simulate
        
        # Flow parameters
        'h0': 5.0,              # Inlet channel depth [m]
        'Qw': 1000,             # Water discharge [m³/s]
        'N_crosssection': 5,    # Number of cells for inlet
        'u0': 1.0,              # Inlet water velocity [m/s]
        'c0': 0.01,             # Inlet sediment concentration
        'dry_depth': 0.1,       # Minimum depth to be considered "wet" [m]
        
        # Sediment parameters
        'f_bedload': 0.3,       # Fraction of sediment transported as bedload
        'f_suspended': 0.7,     # Fraction of sediment transported as suspended load
        'alpha': 0.3,           # Sediment diffusion coefficient
        'beta': 1.0,            # Topographic steering parameter
        
        # Sea level parameters
        'SLR': slr_m_day,       # Sea level rise rate [m/day]
        'h_ocean': 10.0,        # Ocean depth [m]
        
        # Numerical parameters
        'N_water_parcels': 3000, # Number of water parcels per iteration
        'itermax': 100,         # Maximum number of iterations for water surface calculation
        'theta': 1.5,           # Weight for flow partition calculation
        
        # Output parameters
        'save_interval': 10,     # Save output every N steps
        'plot_interval': 10,     # Plot output every N steps
        'output_dir': output_dir # Directory for output files
    }
    
    # Create and initialize the model
    model = DeltaRCM(config)
    
    # Run the model
    print(f"Starting DeltaRCM simulation with SLR rate: {slr_rate} mm/yr")
    
    # Save initial state
    plot_delta_morphology(
        model.eta, model.depth, model.qx, model.qy, 
        step=0, time_days=0, output_dir=output_dir
    )
    plot_delta_3d(
        model.eta, model.depth,
        step=0, time_days=0, output_dir=output_dir
    )
    plot_cross_section(
        model.eta, model.depth, 
        y_index=model.config['W'] // 2,
        step=0, time_days=0, output_dir=output_dir
    )
    
    # Run simulation for specified number of steps
    for step in range(1, model.config['n_steps'] + 1):
        print(f"Running step {step}/{model.config['n_steps']}...")
        
        # Run time step
        model.run_timestep()
        
        # Apply sea level rise by adjusting inlet and ocean depth
        if model.config['SLR'] > 0:
            # Update water depth across domain
            # This represents sea level rise
            model.depth += model.config['SLR'] * model.config['dt'] / 86400  # Convert to m/s
            
            # Update inlet boundary condition
            W = model.config['W']
            center = W // 2
            half_inlet = model.config['N_crosssection'] // 2
            inlet_start = max(0, center - half_inlet)
            inlet_end = min(W, center + half_inlet)
            model.depth[0, inlet_start:inlet_end] = model.config['h0']
        
        # Plot results at specified intervals
        if step % model.config['plot_interval'] == 0:
            time_days = step * model.config['dt'] / 86400
            
            # Plot morphology
            plot_delta_morphology(
                model.eta, model.depth, model.qx, model.qy, 
                step=step, time_days=time_days, output_dir=output_dir
            )
            
            # Plot 3D view
            plot_delta_3d(
                model.eta, model.depth,
                step=step, time_days=time_days, output_dir=output_dir
            )
            
            # Plot cross-section
            plot_cross_section(
                model.eta, model.depth, 
                y_index=model.config['W'] // 2,
                step=step, time_days=time_days, output_dir=output_dir
            )
    
    print(f"Simulation complete. Results saved to {output_dir}")
    return model


def compare_results():
    """Compare results from different sea level rise scenarios."""
    # Create comparison figure for final time step
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot results for each SLR rate
    slr_rates = [0, 5, 10]  # mm/yr
    for i, slr in enumerate(slr_rates):
        # Load final bed elevation
        output_dir = f'./output_slr_{int(slr)}'
        try:
            # Find last step
            import glob
            step_files = glob.glob(f'{output_dir}/step_*.png')
            last_step = max([int(f.split('_')[-1].split('.')[0]) for f in step_files])
            
            # Load data (simplified - in a real implementation, you would save and load numpy arrays)
            # For demonstration purposes, we'll just generate dummy data here
            # In reality, you would load the saved model state
            L, W = 120, 100
            eta = np.zeros((L, W))
            
            # Show bed elevation with sea level
            im = axes[i].imshow(eta.T, origin='lower', cmap='terrain')
            axes[i].set_title(f'SLR: {slr} mm/yr')
            plt.colorbar(im, ax=axes[i])
        except:
            axes[i].text(0.5, 0.5, 'No data available', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes[i].transAxes)
    
    plt.suptitle('Comparison of Delta Morphology under Different Sea Level Rise Scenarios')
    plt.tight_layout()
    plt.savefig('./slr_comparison.png', dpi=200)
    plt.close()


def main():
    """Run simulations with different sea level rise rates."""
    # Run simulations with different sea level rise rates
    slr_rates = [0, 5, 10]  # mm/yr
    for slr in slr_rates:
        model = run_simulation_with_slr(slr)
    
    # Compare results
    compare_results()


if __name__ == "__main__":
    main()