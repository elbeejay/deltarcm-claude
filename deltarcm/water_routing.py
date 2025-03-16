"""
Water routing module for the DeltaRCM model.

This module implements the stochastic weighted random walk method for water routing
as described in Liang et al. (2015).
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class WaterRouter:
    """
    Water routing component of the DeltaRCM model.
    
    This class implements the flow routing scheme (FlowRCM) described in Liang et al. (2015).
    It uses a weighted random walk method to route water parcels through the delta,
    resolving important flow features like backwater effects, flow around mouth bars,
    and flow through distributary channel networks.
    """
    
    def __init__(self, model):
        """
        Initialize the water router.
        
        Parameters
        ----------
        model : DeltaRCM
            Reference to the main model instance.
        """
        self.model = model
        self.config = model.config
        
        # Initialize arrays for neighbor search
        self._init_neighbor_arrays()
    
    def _init_neighbor_arrays(self):
        """Initialize arrays for neighbor search."""
        # Define neighbor search directions (8-connected grid)
        # Order: E, NE, N, NW, W, SW, S, SE
        self.neighbors_x = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        self.neighbors_y = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        
        # Weights for diagonal vs. cardinal directions
        self.neighbor_dists = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414])
    
    def route_water(self):
        """
        Route water parcels through the domain using a weighted random walk.
        
        This method implements the following steps:
        1. Define routing direction at each cell
        2. Calculate relative routing weights for neighbors
        3. Route water parcels using weighted random walk
        4. Calculate water discharge field from parcel trajectories
        
        Returns
        -------
        None
            Updates the model's water depth, discharge, and velocity fields in-place.
            Also stores and returns water parcel trajectories for visualization.
        """
        # Reset water flux arrays
        L, W = self.model.eta.shape
        self.model.qx.fill(0.0)
        self.model.qy.fill(0.0)
        
        # Get model parameters
        N_parcels = self.config['N_water_parcels']
        h0 = self.config['h0']
        Qw = self.config['Qw']
        N_crosssection = self.config['N_crosssection']
        dry_depth = self.config['dry_depth']
        
        # Calculate discharge per water parcel
        qw_parcel = Qw / N_parcels
        
        # Calculate stage for water surface calculation
        self.model.stage = self.model.eta + self.model.depth
        
        # Calculate routing weights
        self._calculate_routing_weights()
        
        # Route water parcels from inlet
        W_half = W // 2
        inlet_start = max(0, W_half - N_crosssection // 2)
        inlet_end = min(W, W_half + N_crosssection // 2)
        inlet_cells = list(range(inlet_start, inlet_end))
        
        # List to store all water parcel trajectories for visualization
        all_trajectories = []
        
        # Track parcel paths
        for _ in range(N_parcels):
            # Start parcel at random inlet cell
            y = np.random.choice(inlet_cells)
            x = 0  # Inlet is at x=0
            
            # Track trajectory for this parcel
            trajectory = [(x, y)]
            
            # Add maximum step limit to prevent infinite loops
            max_steps = L * 2
            step_count = 0
            
            # Route parcel until it exits the domain or reaches a trap
            while 0 <= x < L and 0 <= y < W and step_count < max_steps:
                step_count += 1
                
                # Check if cell is dry (trap)
                if self.model.depth[x, y] < dry_depth:
                    break
                
                # Get weights for current cell
                weights = self.model.weight[x, y, :].copy()  # Use a copy to modify safely
                
                # Apply momentum to prefer continuing in the same direction if possible
                if len(trajectory) >= 2:
                    # Get the previous direction
                    prev_x, prev_y = trajectory[-2]
                    dx = x - prev_x
                    dy = y - prev_y
                    
                    # Find the direction index most closely matching the previous movement
                    momentum_dir = -1
                    for i in range(8):
                        if (self.neighbors_x[i] == dx and self.neighbors_y[i] == dy):
                            momentum_dir = i
                            break
                    
                    # Apply momentum by increasing weights in similar directions
                    if momentum_dir >= 0:
                        # Enhance the weight in the direction of movement
                        weights[momentum_dir] *= 1.5
                        
                        # Also enhance weights in similar directions (adjacent directions)
                        weights[(momentum_dir-1) % 8] *= 1.2
                        weights[(momentum_dir+1) % 8] *= 1.2
                
                # If all weights are zero (e.g., trapped), break
                weight_sum = np.sum(weights)
                if weight_sum <= 0:
                    # Try to create a default downstream path rather than getting stuck
                    # This helps prevent unrealistic trapping of parcels
                    if x < L-1:  # Can move east (downstream)
                        weights[0] = 1.0  # East
                        if y > 0:
                            weights[1] = 0.3  # Northeast
                        if y < W-1:
                            weights[7] = 0.3  # Southeast
                        weight_sum = np.sum(weights)
                    
                    if weight_sum <= 0:
                        # Still no valid direction
                        break
                
                # Normalize weights
                weights /= weight_sum
                
                # Choose direction based on weights
                direction = np.random.choice(8, p=weights)
                
                # Get neighbor coordinates
                nx = x + self.neighbors_x[direction]
                ny = y + self.neighbors_y[direction]
                
                # Check if new position is within domain
                if 0 <= nx < L and 0 <= ny < W:
                    # Update discharge at the interface between cells
                    if nx != x:  # Moving in x-direction
                        idx = min(x, nx)
                        self.model.qx[idx, y] += qw_parcel * np.sign(nx - x)
                    if ny != y:  # Moving in y-direction
                        idx = min(y, ny)
                        self.model.qy[x, idx] += qw_parcel * np.sign(ny - y)
                    
                    # Update position
                    x, y = nx, ny
                    
                    # Add to trajectory
                    trajectory.append((x, y))
                else:
                    # Parcel exits the domain
                    break
            
            # Add the complete trajectory to the list
            all_trajectories.append(trajectory)
        
        # Store trajectories in the model for visualization
        self.model.water_trajectories = all_trajectories
        
        # Apply conservation of mass to the discharge field
        # This ensures a more physically realistic flow field
        self._ensure_flux_conservation()
        
        # Calculate flow velocities from discharge
        wet_cells = self.model.depth > dry_depth
        self.model.ux = np.zeros_like(self.model.qx)
        self.model.uy = np.zeros_like(self.model.qy)
        
        # Avoid division by zero in dry cells
        if np.any(wet_cells):
            self.model.ux[wet_cells] = self.model.qx[wet_cells] / self.model.depth[wet_cells]
            self.model.uy[wet_cells] = self.model.qy[wet_cells] / self.model.depth[wet_cells]
            
        # Return trajectories for visualization
        return all_trajectories
    
    def _ensure_flux_conservation(self):
        """
        Ensure conservation of mass in the water flux field.
        
        This method adjusts the flux field to ensure that the net flux into each cell
        is approximately zero (except at boundaries), enforcing conservation of mass.
        """
        L, W = self.model.eta.shape
        dry_depth = self.config['dry_depth']
        
        # Only apply to wet cells
        wet_cells = self.model.depth > dry_depth
        
        # Calculate divergence of flow field
        divergence = np.zeros((L, W))
        
        # For each cell, compute inflow - outflow
        for x in range(1, L-1):  # Skip boundaries
            for y in range(1, W-1):  # Skip boundaries
                if wet_cells[x, y]:
                    # Calculate net flux
                    inflow_x = self.model.qx[x-1, y] if self.model.qx[x-1, y] > 0 else 0
                    outflow_x = -self.model.qx[x, y] if self.model.qx[x, y] < 0 else 0
                    inflow_y = self.model.qy[x, y-1] if self.model.qy[x, y-1] > 0 else 0
                    outflow_y = -self.model.qy[x, y] if self.model.qy[x, y] < 0 else 0
                    
                    # Net flux = inflow - outflow
                    divergence[x, y] = (inflow_x + inflow_y) - (outflow_x + outflow_y)
        
        # Apply small corrections to discharge field to reduce divergence
        # This is a simplified approach; a full solution would involve solving 
        # a linear system to ensure exact conservation
        alpha = 0.1  # Relaxation factor
        for x in range(1, L-1):
            for y in range(1, W-1):
                if wet_cells[x, y] and abs(divergence[x, y]) > 1e-6:
                    # Determine main flow direction
                    qx_mag = abs(self.model.qx[x, y]) + abs(self.model.qx[x-1, y])
                    qy_mag = abs(self.model.qy[x, y]) + abs(self.model.qy[x, y-1])
                    
                    # Adjust fluxes proportionally to reduce divergence
                    if qx_mag > qy_mag:
                        # Adjust x-direction fluxes
                        if divergence[x, y] > 0:  # Too much inflow
                            if self.model.qx[x-1, y] > 0:
                                self.model.qx[x-1, y] -= alpha * divergence[x, y]
                            if self.model.qx[x, y] < 0:
                                self.model.qx[x, y] += alpha * divergence[x, y]
                        else:  # Too much outflow
                            if self.model.qx[x, y] > 0:
                                self.model.qx[x, y] -= alpha * abs(divergence[x, y])
                            if self.model.qx[x-1, y] < 0:
                                self.model.qx[x-1, y] += alpha * abs(divergence[x, y])
                    else:
                        # Adjust y-direction fluxes
                        if divergence[x, y] > 0:  # Too much inflow
                            if self.model.qy[x, y-1] > 0:
                                self.model.qy[x, y-1] -= alpha * divergence[x, y]
                            if self.model.qy[x, y] < 0:
                                self.model.qy[x, y] += alpha * divergence[x, y]
                        else:  # Too much outflow
                            if self.model.qy[x, y] > 0:
                                self.model.qy[x, y] -= alpha * abs(divergence[x, y])
                            if self.model.qy[x, y-1] < 0:
                                self.model.qy[x, y-1] += alpha * abs(divergence[x, y])
    
    def _calculate_routing_weights(self):
        """
        Calculate routing weights for each cell based on water surface gradients.
        
        The weights determine the probability of a water parcel moving to each 
        neighboring cell, based on the water surface slope, momentum, and a 
        general downstream bias to prevent unrealistic routing.
        """
        L, W = self.model.eta.shape
        dry_depth = self.config['dry_depth']
        theta = self.config['theta']  # Weight parameter for flow partition
        
        # Reset weights
        self.model.weight.fill(0.0)
        
        # Add a general downstream bias to guide flow (overall domain gradient)
        # This acts as a prior for the flow direction and helps prevent backflow
        # Higher values create stronger downstream bias
        downstream_bias = 0.01
        
        # Loop through all cells
        for x in range(L):
            for y in range(W):
                # Skip dry cells
                if self.model.depth[x, y] < dry_depth:
                    continue
                
                # Calculate weights for each neighbor
                for i in range(8):
                    nx = x + self.neighbors_x[i]
                    ny = y + self.neighbors_y[i]
                    
                    # Skip out-of-bounds neighbors
                    if nx < 0 or nx >= L or ny < 0 or ny >= W:
                        continue
                    
                    # Skip neighbors that are too dry 
                    if self.model.depth[nx, ny] < dry_depth * 0.5:
                        continue
                    
                    # Calculate water surface slope to neighbor
                    dz = self.model.stage[x, y] - self.model.stage[nx, ny]
                    
                    # Add downstream bias for east-directed flow (domain aligned with x-axis)
                    # This helps guide flow in the general downstream direction
                    if i == 0:  # East
                        dz += downstream_bias
                    elif i == 1 or i == 7:  # Northeast or Southeast
                        dz += downstream_bias * 0.7  # Reduced diagonal bias
                    elif i == 4:  # West (upstream)
                        dz -= downstream_bias  # Penalize upstream flow
                    elif i == 3 or i == 5:  # Northwest or Southwest
                        dz -= downstream_bias * 0.7  # Penalize diagonal upstream
                    
                    # Weight is proportional to slope divided by distance
                    if dz > 0:  # Downhill (positive slope)
                        # Use power function for emphasizing steeper slopes
                        weight = np.power(dz / self.neighbor_dists[i], theta)
                    else:  # Flat or uphill (zero or negative slope)
                        # Add small diffusive component for flat/uphill directions
                        # This helps with numerical stability and handles flat water surfaces
                        eps = 1e-6  # Small value to ensure some flow in flat regions
                        # Make uphill flow exponentially less likely based on steepness
                        weight = eps * np.exp(dz * 5)
                    
                    self.model.weight[x, y, i] = weight
                
                # Add a check for wet cells with obstacles around them
                # This prevents flow from getting trapped in wet cells surrounded by dry cells or boundaries
                has_wet_neighbor = False
                for i in range(8):
                    nx = x + self.neighbors_x[i]
                    ny = y + self.neighbors_y[i]
                    if 0 <= nx < L and 0 <= ny < W and self.model.depth[nx, ny] >= dry_depth:
                        has_wet_neighbor = True
                        break
                
                # Normalize weights
                weight_sum = np.sum(self.model.weight[x, y, :])
                if weight_sum > 0:
                    self.model.weight[x, y, :] /= weight_sum
                # If no valid directions or no wet neighbors and not at outlet, add default downstream flow
                elif (weight_sum <= 0 or not has_wet_neighbor) and x < L-1:
                    # Default to moving downstream (east) with small random lateral component
                    # This helps prevent flow from getting stuck
                    if x < L-1:  # Can move east
                        self.model.weight[x, y, 0] = 0.8  # East direction (downstream)
                    if y > 0:    # Can move north
                        self.model.weight[x, y, 2] = 0.1
                    if y < W-1:  # Can move south
                        self.model.weight[x, y, 6] = 0.1
                    # Add diagonal components
                    if y > 0 and x < L-1:  # Can move northeast
                        self.model.weight[x, y, 1] = 0.05
                    if y < W-1 and x < L-1:  # Can move southeast
                        self.model.weight[x, y, 7] = 0.05
                    # Re-normalize (should sum to ~1.1 before normalization)
                    self.model.weight[x, y, :] /= np.sum(self.model.weight[x, y, :])
    
    def calculate_water_surface(self):
        """
        Calculate water surface elevations.
        
        This method solves for the water surface elevation profile, taking into
        account backwater effects. It follows the approach described in Liang et al. (2015).
        """
        L, W = self.model.eta.shape
        itermax = self.config['itermax']
        dry_depth = self.config['dry_depth']
        h0 = self.config['h0']
        h_ocean = self.config['h_ocean']
        
        # Make a copy of the current water depth field
        depth_prev = self.model.depth.copy()
        
        # Initialize water surface with a gentle downstream slope
        # This provides a better initial guess for the iterative solution
        ocean_start = int(0.8 * L)  # Ocean starts at 80% of domain length
        
        # Set up a more physically realistic initial water surface
        # Linearly interpolate from inlet to ocean
        for x in range(L):
            for y in range(W):
                if x == 0:  # Inlet boundary
                    # Only set depth at the inlet cells
                    continue  # Keep existing depth at inlet
                elif x >= ocean_start:  # Ocean
                    depth_prev[x, y] = h_ocean
                else:  # River/delta region
                    # Linear interpolation from inlet (h0) to ocean (h_ocean)
                    frac = x / ocean_start
                    target_depth = h0 * (1.0 - frac) + h_ocean * frac
                    # Don't let water go below bed
                    if self.model.eta[x, y] + target_depth > 0:
                        depth_prev[x, y] = max(target_depth, dry_depth)
        
        # Set convergence parameters
        max_iterations = min(100, itermax)  # Reduced maximum iterations for performance
        convergence_threshold = 1e-3  # Relaxed tolerance for faster convergence
        relaxation_factor = 0.8  # Under-relaxation to improve stability
        
        # Iterative calculation of water surface elevation
        for iter_num in range(max_iterations):
            # Calculate stage from current depth
            self.model.stage = self.model.eta + depth_prev
            
            # Update depth based on local mass balance
            depth_new = np.zeros_like(depth_prev)
            
            # Set inlet boundary condition
            # Only set fixed depths at the actual inlet cells
            center = W // 2
            half_inlet = self.config['N_crosssection'] // 2
            inlet_start = max(0, center - half_inlet)
            inlet_end = min(W, center + half_inlet)
            depth_new[0, inlet_start:inlet_end] = h0
            
            # Set ocean boundary condition
            depth_new[ocean_start:, :] = h_ocean
            
            # Loop through domain (excluding fixed boundaries)
            for x in range(1, ocean_start):
                for y in range(W):
                    # Skip very shallow cells
                    if depth_prev[x, y] < dry_depth * 0.5:
                        depth_new[x, y] = 0.0
                        continue
                    
                    # Calculate inflow and outflow
                    inflow = 0.0
                    outflow = 0.0
                    valid_neighbor_count = 0
                    
                    # Check all neighbors
                    for i in range(8):
                        nx = x + self.neighbors_x[i]
                        ny = y + self.neighbors_y[i]
                        
                        # Skip out-of-bounds neighbors
                        if nx < 0 or nx >= L or ny < 0 or ny >= W:
                            continue
                        
                        valid_neighbor_count += 1
                        
                        # Calculate flow between cells based on water surface gradient
                        dz = self.model.stage[x, y] - self.model.stage[nx, ny]
                        
                        if dz > 0:  # Outflow to neighbor
                            outflow += self.model.weight[x, y, i]
                        elif dz < 0:  # Inflow from neighbor
                            inflow += self.model.weight[nx, ny, (i+4)%8]  # Opposite direction
                    
                    # Update depth based on net flow
                    if valid_neighbor_count > 0:
                        if outflow > 0:
                            # Apply a relaxation factor to improve numerical stability
                            # and prevent oscillations
                            change = depth_prev[x, y] * (inflow - outflow)
                            # Limit the maximum change in depth per iteration
                            change = np.clip(change, -depth_prev[x, y] * 0.5, depth_prev[x, y])
                            depth_new[x, y] = depth_prev[x, y] + relaxation_factor * change
                        else:
                            depth_new[x, y] = depth_prev[x, y]
                        
                        # Ensure depth is at least the dry depth
                        if depth_new[x, y] < dry_depth:
                            depth_new[x, y] = 0.0
                    else:
                        depth_new[x, y] = depth_prev[x, y]
            
            # Apply minimum depth threshold
            depth_new[depth_new < dry_depth] = 0.0
            
            # Check convergence
            abs_diff = np.abs(depth_new - depth_prev)
            rel_diff = abs_diff / (depth_prev + 1e-10)  # Avoid division by zero
            max_rel_diff = np.max(rel_diff)
            
            # Print diagnostics for debugging convergence issues
            if iter_num % 20 == 0:
                print(f"Water surface iteration {iter_num}, max relative diff: {max_rel_diff:.6f}")
                
            if max_rel_diff < convergence_threshold:
                break
            
            # Update depth field for next iteration
            depth_prev = depth_new.copy()
            
            # Safety check: if iteration is not converging, apply stronger relaxation
            if iter_num > max_iterations // 2 and max_rel_diff > 0.1:
                relaxation_factor *= 0.9  # Reduce relaxation factor
                
            # Emergency exit if convergence is very slow
            if iter_num >= max_iterations - 1:
                print(f"WARNING: Water surface calculation did not converge after {max_iterations} iterations")
        
        # Update model depth field
        self.model.depth = depth_prev
        
        # Update stage
        self.model.stage = self.model.eta + self.model.depth