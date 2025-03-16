"""
Core DeltaRCM model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any


class DeltaRCM:
    """
    A reduced-complexity model (RCM) for river delta formation.

    This model uses a rule-based cellular morphodynamic approach with
    stochastic parcel-based routing schemes for water and sediment.

    The model consists of four main phases:
    1. Water routing
    2. Water surface calculation
    3. Sediment transport and bed topography update
    4. Update routing direction

    Parameters
    ----------
    config : dict, optional
        Configuration parameters for the model. If not provided, default values are used.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the DeltaRCM model with default or user-provided parameters."""
        # Default configuration parameters
        self.default_config = {
            # Grid parameters
            'L': 100,                # Domain length [cells]
            'W': 80,                 # Domain width [cells]
            'cell_size': 10,         # Cell size [m]

            # Time parameters
            'dt': 1000,              # Time step [s]
            'n_steps': 1000,         # Number of time steps to simulate

            # Flow parameters
            'h0': 1.0,               # Inlet channel depth [m]
            'Qw': 1000,              # Water discharge [m³/s]
            'N_crosssection': 10,    # Number of cells for inlet
            'u0': 1.0,               # Inlet water velocity [m/s]
            'c0': 0.1,               # Inlet sediment concentration
            'dry_depth': 0.1,        # Minimum depth to be considered "wet" [m]
            'g': 9.81,               # Gravity acceleration [m/s²]

            # Sediment parameters
            'f_bedload': 0.5,        # Fraction of sediment transported as bedload
            'f_suspended': 0.5,      # Fraction of sediment transported as suspended load
            'alpha': 0.1,            # Sediment diffusion coefficient
            'beta': 0.5,             # Topographic steering parameter

            # Sea level parameters
            'SLR': 0.0,              # Sea level rise rate [m/yr]
            'h_ocean': 5.0,          # Ocean depth [m]

            # Numerical parameters
            'N_water_parcels': 1000, # Number of water parcels per iteration
            'itermax': 1000,         # Maximum number of iterations for water surface calculation
            'theta': 1.0,            # Weight for flow partition calculation

            # Output parameters
            'save_interval': 10,     # Save output every N steps
            'plot_interval': 10,     # Plot output every N steps
            'output_dir': './output' # Directory for output files
        }

        # Update with user configuration if provided
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)

        # Initialize model domain and variables
        self._initialize_domain()

        # Initialize output tracking
        self.output_data = []
        self.time_steps = []

    def _initialize_domain(self):
        """Initialize model domain and state variables."""
        L, W = self.config['L'], self.config['W']

        # Initialize arrays for model state
        self.eta = np.zeros((L, W))       # Bed elevation [m]
        self.stage = np.zeros((L, W))     # Water stage (h + eta) [m]
        self.depth = np.zeros((L, W))     # Water depth [m]
        self.qx = np.zeros((L, W))        # Unit water discharge in x-direction [m²/s]
        self.qy = np.zeros((L, W))        # Unit water discharge in y-direction [m²/s]
        self.ux = np.zeros((L, W))        # Depth-averaged velocity in x-direction [m/s]
        self.uy = np.zeros((L, W))        # Depth-averaged velocity in y-direction [m/s]
        self.weight = np.zeros((L, W, 8)) # Routing weights for 8 neighboring cells
        self.cell_type = np.zeros((L, W), dtype=int)  # Cell type (0=land, 1=channel, 2=ocean)

        # Initialize stratigraphy (optional)
        self.strata_age = np.zeros((L, W, 1))      # Age of deposits
        self.strata_sand_frac = np.zeros((L, W, 1)) # Sand fraction of deposits

        # Initialize trajectory arrays
        self.water_trajectories = []
        self.sediment_trajectories = []

        # Initialize the domain topography with a smooth transition from inlet to ocean
        self._initialize_topography()

        # Initialize inlet (upstream) boundary
        self._setup_boundary_conditions()

        # Initialize ocean (downstream) boundary
        self._setup_ocean()

    def _initialize_topography(self):
        """Initialize domain topography with a realistic gradient."""
        L, W = self.config['L'], self.config['W']
        h0 = self.config['h0']  # Inlet channel depth
        h_ocean = self.config['h_ocean']  # Ocean depth

        # Define domain regions
        ocean_start = int(0.05 * L)  # Ocean starts at 5% of domain length
        inlet_length = 10  # Number of cells for initial inlet channel

        # Create a slight depression for the initial channel pathway
        # This helps guide the flow initially to prevent immediate sediment deposition at inlet

        # 1. Start with flat terrain at sea level
        self.eta.fill(0.0)

        # 2. Create a smooth channel depression from inlet to ocean
        # Calculate linear transition from inlet to ocean depth
        center_width = W // 2
        channel_width = min(W // 4, 10)  # Width of the initial channel

        # Define the channel region
        channel_left = max(0, center_width - channel_width // 2)
        channel_right = min(W, center_width + channel_width // 2)

        # Create a slight slope from the inlet to the ocean
        for x in range(L):
            # Determine the depression depth based on position
            if x < inlet_length:
                # Inlet region - full channel depth
                channel_depth = h0
            elif x >= ocean_start:
                # Ocean region - ocean depth
                channel_depth = h_ocean
            else:
                # Transition region - linear interpolation
                frac = (x - inlet_length) / (ocean_start - inlet_length)
                channel_depth = h0 + frac * (h_ocean - h0)

            # Apply the depression to the channel area with smoothed edges
            for y in range(W):
                # Distance from channel centerline (normalized)
                dist_from_center = abs(y - center_width) / (channel_width / 2)

                if dist_from_center < 1.0:
                    # Inside the channel - full depth
                    depth_factor = 1.0
                elif dist_from_center < 2.0:
                    # Transition zone - smooth edges using cosine
                    edge_frac = dist_from_center - 1.0
                    depth_factor = 0.5 * (1.0 + np.cos(np.pi * edge_frac))
                else:
                    # Outside the channel
                    depth_factor = 0.0

                # Apply the depression
                self.eta[x, y] = -channel_depth * depth_factor

    def _setup_boundary_conditions(self):
        """Set up the inlet boundary conditions."""
        # Set inlet properties
        W = self.config['W']
        h0 = self.config['h0']
        N_crosssection = self.config['N_crosssection']

        # Calculate center of domain width
        center = W // 2
        half_inlet = N_crosssection // 2

        # Create inlet channel at upstream boundary
        inlet_start = max(0, center - half_inlet)
        inlet_end = min(W, center + half_inlet)

        # Set inlet bed elevation and depth
        # Make sure the inlet has correct depth
        self.eta[0, inlet_start:inlet_end] = -h0
        self.depth[0, inlet_start:inlet_end] = h0

        # Extend the inlet channel a few cells downstream to ensure good flow establishment
        inlet_extension = 5  # Number of cells to extend inlet
        for i in range(1, inlet_extension):
            transition_factor = 1.0 - (i / inlet_extension)

            # Apply a gradual transition for the inlet channel
            for y in range(inlet_start, inlet_end):
                # More natural transition for channel edges
                edge_dist = min(y - inlet_start, inlet_end - 1 - y)
                if edge_dist < 2:
                    edge_factor = 0.7 + 0.3 * edge_dist / 2
                else:
                    edge_factor = 1.0

                # Set bed and depth
                self.eta[i, y] = -h0 * transition_factor * edge_factor
                self.depth[i, y] = h0 * transition_factor * edge_factor

        # Set cell types for inlet and channel
        self.cell_type[0:inlet_extension, inlet_start:inlet_end] = 1  # Channel cells

    def _setup_ocean(self):
        """Set up the ocean (downstream) boundary."""
        # Set ocean depth and bed elevation
        L, W = self.config['L'], self.config['W']
        h_ocean = self.config['h_ocean']

        # Define ocean region (last 95% of domain)
        ocean_start = int(0.05 * L)

        # Set ocean bed elevation and depth
        self.eta[ocean_start:, :] = -h_ocean
        self.depth[ocean_start:, :] = h_ocean

        # Set cell types for ocean
        self.cell_type[ocean_start:, :] = 2  # Ocean cells

        # Initialize the full water depth field - ensures a proper initial condition
        # This fills the domain with water up to the stage of the ocean
        for x in range(L):
            for y in range(W):
                # Skip inlet and ocean cells that are already set
                if self.cell_type[x, y] != 0:
                    continue

                # Calculate water depth to reach the ocean water stage
                # This ensures a gradual hydraulic gradient from inlet to ocean
                ocean_stage = 0.0  # Ocean water surface at sea level
                target_depth = ocean_stage - self.eta[x, y]

                # Make sure depth is non-negative
                if target_depth > 0:
                    self.depth[x, y] = target_depth
                else:
                    self.depth[x, y] = 0.0

        # Update stage
        self.stage = self.eta + self.depth

    def run_model(self):
        """Run the model for the specified number of time steps."""
        n_steps = self.config['n_steps']
        save_interval = self.config['save_interval']
        plot_interval = self.config['plot_interval']

        for step in range(n_steps):
            # Perform a single time step
            self.run_timestep()

            # Save output at specified intervals
            if step % save_interval == 0:
                self._save_output(step)

            # Plot output at specified intervals
            if step % plot_interval == 0:
                self._plot_output(step)

    def run_timestep(self):
        """Run a single time step of the model."""
        # Track model time
        if not hasattr(self, 'time'):
            self.time = 0.0
        self.time += self.config['dt']

        # Phase 1: Water routing
        self._water_routing()

        # Phase 2: Water surface calculation
        self._water_surface_calculation()

        # Phase 3: Sediment transport and bed topography update
        self._sediment_routing()

        # Phase 4: Update routing direction
        self._update_routing_weights()

    def _water_routing(self):
        """
        Phase 1: Route water parcels through the domain using a weighted random walk.

        This implements the water routing component of the model using the stochastic
        parcel routing scheme described in the papers.
        """
        # Import water routing module if not already imported
        from deltarcm.water_routing import WaterRouter

        # Create water router if it doesn't exist
        if not hasattr(self, 'water_router'):
            self.water_router = WaterRouter(self)

        # Route water parcels
        self.water_router.route_water()

    def _water_surface_calculation(self):
        """
        Phase 2: Calculate water surface elevations.

        Solves for the water surface profile taking into account backwater effects.
        """
        # Use the water router to calculate water surface
        if not hasattr(self, 'water_router'):
            from deltarcm.water_routing import WaterRouter
            self.water_router = WaterRouter(self)

        # Make sure stage is updated before calculation
        self.stage = self.eta + self.depth

        # Calculate water surface
        self.water_router.calculate_water_surface()

        # After calculation, update velocities based on the new depth field
        # This ensures velocities are consistent with the updated water depth
        dry_depth = self.config['dry_depth']
        wet_cells = self.depth > dry_depth

        # Calculate velocities in wet cells
        if np.any(wet_cells):
            self.ux[wet_cells] = self.qx[wet_cells] / self.depth[wet_cells]
            self.uy[wet_cells] = self.qy[wet_cells] / self.depth[wet_cells]

        # Set velocities to zero in dry cells
        self.ux[~wet_cells] = 0.0
        self.uy[~wet_cells] = 0.0

    def _sediment_routing(self):
        """
        Phase 3: Route sediment parcels and update bed topography.

        Routes bedload and suspended sediment parcels and updates the bed elevation
        based on erosion and deposition patterns.
        """
        # Import sediment routing module if not already imported
        from deltarcm.sediment_routing import SedimentRouter

        # Create sediment router if it doesn't exist
        if not hasattr(self, 'sediment_router'):
            self.sediment_router = SedimentRouter(self)

        # Route sediment parcels and update bed topography
        self.sediment_router.route_sediment()

    def _update_routing_weights(self):
        """
        Phase 4: Update routing directions/weights.

        Updates the weights used for routing water and sediment parcels based on
        the updated flow field and bed topography.
        """
        # Recalculate routing weights using water router
        if not hasattr(self, 'water_router'):
            from deltarcm.water_routing import WaterRouter
            self.water_router = WaterRouter(self)

        self.water_router._calculate_routing_weights()

    def _save_output(self, step):
        """Save model state at the current time step."""
        # Save key model state variables
        output = {
            'step': step,
            'time': step * self.config['dt'],
            'eta': self.eta.copy(),
            'depth': self.depth.copy(),
            'qx': self.qx.copy(),
            'qy': self.qy.copy()
        }
        self.output_data.append(output)
        self.time_steps.append(step)

    def _plot_output(self, step):
        """Plot the current model state with enhanced visualizations."""
        from deltarcm.utils import (plot_delta_morphology, plot_delta_3d,
                                    plot_cross_section, plot_parcel_trajectories,
                                    plot_deposition_patterns)

        # Calculate time in days for titles
        time_days = step * self.config["dt"] / 86400
        output_dir = self.config["output_dir"]

        # Basic plots - bed elevation and water depth with flow vectors
        plot_delta_morphology(
            self.eta, self.depth, self.qx, self.qy,
            step=step, time_days=time_days, output_dir=output_dir
        )

        # Plot deposition and erosion patterns if they exist
        if hasattr(self, 'sediment_router'):
            plot_deposition_patterns(
                self.eta, self.sediment_router.deposition, self.sediment_router.erosion,
                step=step, time_days=time_days, output_dir=output_dir
            )

        # Plot parcel trajectories if they exist
        water_trajectories = getattr(self, 'water_trajectories', [])
        sediment_trajectories = getattr(self, 'sediment_trajectories', [])

        if water_trajectories and sediment_trajectories:
            plot_parcel_trajectories(
                water_trajectories, sediment_trajectories,
                self.eta, self.depth,
                step=step, time_days=time_days, output_dir=output_dir
            )