#!/usr/bin/env python
"""
Run script for the DeltaRCM model.

This script sets up and runs a simulation of delta formation using the DeltaRCM model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from deltarcm.model import DeltaRCM
from deltarcm.utils import plot_delta_morphology, plot_delta_3d, plot_cross_section


def main():
    """Set up and run a DeltaRCM simulation."""
    # Create output directory
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure the model
    config = {
        # Grid parameters
        'L': 120,               # Domain length [cells]
        'W': 100,               # Domain width [cells]
        'cell_size': 50,        # Cell size [m]

        # Time parameters
        'dt': 86400,            # Time step [s] (1 day)
        'n_steps': 100,         # Number of time steps to simulate

        # Flow parameters
        'h0': 5.0,              # Inlet channel depth [m]
        'Qw': 1000,             # Water discharge [m³/s]
        'N_crosssection': 5,    # Number of cells for inlet
        'u0': 1.0,              # Inlet water velocity [m/s]
        'c0': 0.001,            # Inlet sediment concentration
        'dry_depth': 0.1,       # Minimum depth to be considered "wet" [m]

        # Sediment parameters
        'f_bedload': 0.3,       # Fraction of sediment transported as bedload
        'f_suspended': 0.7,     # Fraction of sediment transported as suspended load
        'alpha': 0.3,           # Sediment diffusion coefficient
        'beta': 1.0,            # Topographic steering parameter

        # Sea level parameters
        'SLR': 0.0,             # Sea level rise rate [m/yr]
        'h_ocean': 10.0,        # Ocean depth [m]

        # Numerical parameters
        'N_water_parcels': 2000, # Number of water parcels per iteration (reduced for performance)
        'itermax': 10,          # Maximum number of iterations for water surface calculation (reduced)
        'theta': 1.5,           # Weight for flow partition calculation

        # Output parameters
        'save_interval': 5,     # Save output every N steps
        'plot_interval': 5,     # Plot output every N steps
        'output_dir': output_dir # Directory for output files
    }

    # Create and initialize the model
    model = DeltaRCM(config)

    # Run the model
    print("Starting DeltaRCM simulation...")

    # Save initial state
    model._plot_output(0)
    # plot_delta_morphology(
    #     model.eta, model.depth, model.qx, model.qy,
    #     step=0, time_days=0, output_dir=output_dir
    # )
    # plot_delta_3d(
    #     model.eta, model.depth,
    #     step=0, time_days=0, output_dir=output_dir
    # )
    # plot_cross_section(
    #     model.eta, model.depth,
    #     y_index=model.config['W'] // 2,
    #     step=0, time_days=0, output_dir=output_dir
    # )

    # Run simulation for specified number of steps
    for step in range(1, model.config['n_steps'] + 1):
        print(f"Running step {step}/{model.config['n_steps']}...")

        # Run time step
        model.run_timestep()

        # Plot results at specified intervals
        if step % model.config['plot_interval'] == 0:
            time_days = step * model.config['dt'] / 86400

            model._plot_output(step)
            # # Plot morphology
            # plot_delta_morphology(
            #     model.eta, model.depth, model.qx, model.qy,
            #     step=step, time_days=time_days, output_dir=output_dir
            # )

            # # Plot 3D view
            # plot_delta_3d(
            #     model.eta, model.depth,
            #     step=step, time_days=time_days, output_dir=output_dir
            # )

            # # Plot cross-section
            # plot_cross_section(
            #     model.eta, model.depth,
            #     y_index=model.config['W'] // 2,
            #     step=step, time_days=time_days, output_dir=output_dir
            # )

    print(f"Simulation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()