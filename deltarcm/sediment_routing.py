"""
Sediment routing module for the DeltaRCM model.

This module implements the sediment transport and bed topography update 
components of the DeltaRCM model as described in Liang et al. (2015).
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class SedimentRouter:
    """
    Sediment routing component of the DeltaRCM model.
    
    This class implements the sediment transport rules described in Liang et al. (2015).
    It handles both bedload and suspended sediment transport, erosion, deposition,
    and updating the bed topography based on these processes.
    """
    
    def __init__(self, model):
        """
        Initialize the sediment router.
        
        Parameters
        ----------
        model : DeltaRCM
            Reference to the main model instance.
        """
        self.model = model
        self.config = model.config
        
        # Initialize arrays for neighbor search
        self._init_neighbor_arrays()
        
        # Initialize sediment transport arrays
        self._init_sediment_arrays()
    
    def _init_neighbor_arrays(self):
        """Initialize arrays for neighbor search."""
        # Define neighbor search directions (8-connected grid)
        # Order: E, NE, N, NW, W, SW, S, SE
        self.neighbors_x = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        self.neighbors_y = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        
        # Weights for diagonal vs. cardinal directions
        self.neighbor_dists = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414])
    
    def _init_sediment_arrays(self):
        """Initialize arrays for sediment transport."""
        L, W = self.model.eta.shape
        
        # Erosion-deposition arrays
        self.erosion = np.zeros((L, W))     # Erosion at each cell
        self.deposition = np.zeros((L, W))  # Deposition at each cell
        
        # Sediment transport arrays
        self.qs_x = np.zeros((L, W))        # Sediment discharge in x-direction
        self.qs_y = np.zeros((L, W))        # Sediment discharge in y-direction
        
        # Sediment concentration field
        self.concentration = np.zeros((L, W))
    
    def route_sediment(self):
        """
        Route sediment parcels and update bed topography.
        
        This method routes both bedload and suspended sediment parcels and 
        updates the bed elevation based on erosion and deposition patterns.
        
        Returns
        -------
        None
            Updates the model's bed elevation field in-place.
            Also stores sediment parcel trajectories for visualization.
        """
        # Reset sediment transport arrays
        self.erosion.fill(0.0)
        self.deposition.fill(0.0)
        self.qs_x.fill(0.0)
        self.qs_y.fill(0.0)
        
        # Get model parameters
        L, W = self.model.eta.shape
        N_water_parcels = self.config['N_water_parcels']
        Qw = self.config['Qw']
        c0 = self.config['c0']
        N_crosssection = self.config['N_crosssection']
        dry_depth = self.config['dry_depth']
        f_bedload = self.config['f_bedload']
        f_suspended = self.config['f_suspended']
        alpha = self.config['alpha']  # Sediment diffusion coefficient
        beta = self.config['beta']    # Topographic steering parameter
        dt = self.config['dt']
        
        # Calculate total sediment input
        Qs = Qw * c0
        
        # Calculate sediment per parcel
        qs_parcel = Qs / N_water_parcels
        
        # Route sediment parcels from inlet
        W_half = W // 2
        inlet_start = max(0, W_half - N_crosssection // 2)
        inlet_end = min(W, W_half + N_crosssection // 2)
        inlet_cells = list(range(inlet_start, inlet_end))
        
        # Calculate number of bedload and suspended load parcels
        n_bedload = int(N_water_parcels * f_bedload)
        n_suspended = N_water_parcels - n_bedload
        
        # Lists to store all sediment parcel trajectories for visualization
        all_trajectories = []
        
        # Track parcel paths
        # 1. Route bedload parcels
        bedload_trajectories = self._route_bedload_parcels(n_bedload, qs_parcel, inlet_cells)
        all_trajectories.extend(bedload_trajectories)
        
        # 2. Route suspended load parcels
        suspended_trajectories = self._route_suspended_parcels(n_suspended, qs_parcel, inlet_cells)
        all_trajectories.extend(suspended_trajectories)
        
        # Store trajectories in the model for visualization
        self.model.sediment_trajectories = all_trajectories
        
        # Add erosion based on flow velocities
        self._calculate_erosion()
        
        # Update bed topography based on erosion and deposition
        net_change = self.deposition - self.erosion
        self.model.eta += net_change * dt
        
        # Record stratigraphy (optional)
        self._record_stratigraphy(net_change, dt)
        
        # Return trajectories for visualization
        return all_trajectories
        
    def _calculate_erosion(self):
        """
        Calculate erosion based on flow velocities.
        
        This adds a physically-based erosion component to complement the deposition
        process, creating a more realistic sediment transport model.
        """
        L, W = self.model.eta.shape
        dry_depth = self.config['dry_depth']
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(self.model.ux**2 + self.model.uy**2)
        
        # Critical erosion thresholds
        v_critical = 0.3  # Critical velocity for erosion onset [m/s]
        v_max = 1.5      # Velocity at which erosion reaches maximum [m/s]
        
        # Erodibility parameter (controls erosion rate)
        # Higher values = more erosion
        erodibility = 0.002  # Adjusted to balance with deposition
        
        # Calculate erosion rate based on excess velocity
        # No erosion below critical velocity, maximum erosion above v_max
        excess_vel = np.maximum(0, vel_mag - v_critical)
        erosion_factor = np.minimum(1.0, excess_vel / (v_max - v_critical))
        
        # Apply erosion only to wet cells
        wet_cells = self.model.depth > dry_depth
        self.erosion[wet_cells] += erodibility * erosion_factor[wet_cells]
        
        # Restrict erosion in inlet zone to prevent inlet degradation
        inlet_zone = np.zeros((L, W), dtype=bool)
        inlet_zone[0:3, :] = True  # First few rows
        self.erosion[inlet_zone] = 0.0
    
    def _route_bedload_parcels(self, n_parcels, qs_parcel, inlet_cells):
        """
        Route bedload sediment parcels.
        
        Parameters
        ----------
        n_parcels : int
            Number of bedload parcels to route.
        qs_parcel : float
            Sediment discharge per parcel.
        inlet_cells : list
            List of inlet cell indices.
            
        Returns
        -------
        list
            List of sediment parcel trajectories for visualization.
        """
        L, W = self.model.eta.shape
        dry_depth = self.config['dry_depth']
        beta = self.config['beta']
        cell_size = self.config['cell_size']
        
        # List to store all trajectories
        trajectories = []
        
        # Route bedload parcels
        for _ in range(n_parcels):
            # Start parcel at random inlet cell
            y = np.random.choice(inlet_cells)
            x = 0  # Inlet is at x=0
            
            # Track trajectory
            trajectory = [(x, y)]
            
            # Initialize parcel sediment mass
            mass = qs_parcel
            
            # Maximum number of steps to prevent infinite loops
            max_steps = L * 2
            step_count = 0
            
            # Route parcel until it deposits all its sediment or exits the domain
            while mass > 0 and 0 <= x < L and 0 <= y < W and step_count < max_steps:
                step_count += 1
                
                # Check if cell is dry (trap)
                if self.model.depth[x, y] < dry_depth:
                    # Deposit all remaining sediment and break
                    self.deposition[x, y] += mass
                    break
                
                # Calculate routing weights for bedload:
                # - Water surface gradient (from model.weight)
                # - Topographic steering (slope of bed)
                # - Flow velocity
                weights = np.zeros(8)
                
                for i in range(8):
                    nx = x + self.neighbors_x[i]
                    ny = y + self.neighbors_y[i]
                    
                    # Skip out-of-bounds neighbors
                    if nx < 0 or nx >= L or ny < 0 or ny >= W:
                        continue
                    
                    # Skip neighbors that are too dry
                    if self.model.depth[nx, ny] < dry_depth:
                        continue
                    
                    # Base weight from water routing (flow field)
                    w_flow = self.model.weight[x, y, i]
                    
                    # Add topographic steering effect - slope of bed
                    deta = self.model.eta[x, y] - self.model.eta[nx, ny]
                    # Normalize by distance and cell size
                    w_topo = max(0, deta / (self.neighbor_dists[i] * cell_size))
                    
                    # Add velocity component - higher velocity increases transport
                    # Use average velocity at the interface
                    if i == 0:  # East
                        v_mag = abs(self.model.ux[x, y]) if x < L-1 else 0
                    elif i == 4:  # West
                        v_mag = abs(self.model.ux[x-1, y]) if x > 0 else 0
                    elif i == 2:  # North
                        v_mag = abs(self.model.uy[x, y]) if y < W-1 else 0
                    elif i == 6:  # South
                        v_mag = abs(self.model.uy[x, y-1]) if y > 0 else 0
                    else:  # Diagonal - use average of components
                        vx = abs(self.model.ux[min(x, nx), y]) if 0 <= min(x, nx) < L-1 else 0
                        vy = abs(self.model.uy[x, min(y, ny)]) if 0 <= min(y, ny) < W-1 else 0
                        v_mag = np.sqrt(vx**2 + vy**2) / np.sqrt(2)  # Scale diagonal
                    
                    # Normalize velocity weight
                    v_max = 1.0  # Max expected velocity
                    w_vel = min(v_mag / v_max, 1.0)
                    
                    # Combine weights
                    # Higher beta values increase influence of bed slope
                    weights[i] = (1.0 - beta) * (0.7 * w_flow + 0.3 * w_vel) + beta * w_topo
                    
                    # Add small positive weight to ensure some valid paths exist
                    weights[i] = max(1e-6, weights[i])
                
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
                    
                    # Apply momentum by increasing weights in the same and similar directions
                    if momentum_dir >= 0:
                        # Enhance the weight in the direction of movement
                        weights[momentum_dir] *= 1.3
                        
                        # Also enhance weights in similar directions
                        weights[(momentum_dir-1) % 8] *= 1.1
                        weights[(momentum_dir+1) % 8] *= 1.1
                
                # Normalize weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights /= weight_sum
                else:
                    # If no valid routing direction, deposit all sediment and break
                    self.deposition[x, y] += mass
                    break
                
                # Choose direction based on weights
                direction = np.random.choice(8, p=weights)
                
                # Calculate deposition at current cell
                # Bedload deposits more in shallow water
                # Use a more physically realistic deposition function:
                # - More deposition in shallow water
                # - Less deposition in deep water
                # - Less deposition in high-velocity areas
                # - Scaled by cell size to account for grid resolution
                h_scale = 0.5  # Characteristic depth scale [m]
                v_mag = np.sqrt(self.model.ux[x, y]**2 + self.model.uy[x, y]**2)
                v_scale = 0.25  # Velocity scale for deposition reduction
                
                # Base deposition fraction from depth
                depth_factor = h_scale / max(h_scale, self.model.depth[x, y])
                
                # Velocity factor (less deposition in high-velocity areas)
                vel_factor = 1.0 / (1.0 + v_mag / v_scale)
                
                # Combined deposition fraction
                deposit_frac = 0.03 * depth_factor * vel_factor
                deposit_frac = np.clip(deposit_frac, 0.005, 0.1)  # Limit range
                
                deposit = mass * deposit_frac
                self.deposition[x, y] += deposit
                mass -= deposit
                
                # Update sediment transport
                nx = x + self.neighbors_x[direction]
                ny = y + self.neighbors_y[direction]
                
                # Check if new position is within domain
                if 0 <= nx < L and 0 <= ny < W:
                    # Update sediment discharge at the interface between cells
                    if nx != x:  # Moving in x-direction
                        idx = min(x, nx)
                        self.qs_x[idx, y] += mass * np.sign(nx - x)
                    if ny != y:  # Moving in y-direction
                        idx = min(y, ny)
                        self.qs_y[x, idx] += mass * np.sign(ny - y)
                    
                    # Update position
                    x, y = nx, ny
                    
                    # Add to trajectory
                    trajectory.append((x, y))
                else:
                    # Parcel exits the domain
                    break
            
            # If parcel has deposited all sediment or reached max steps, add final deposition
            if mass > 0 and 0 <= x < L and 0 <= y < W:
                self.deposition[x, y] += mass
            
            # Add the complete trajectory to the list
            trajectories.append(trajectory)
        
        return trajectories
    
    def _route_suspended_parcels(self, n_parcels, qs_parcel, inlet_cells):
        """
        Route suspended sediment parcels.
        
        Parameters
        ----------
        n_parcels : int
            Number of suspended parcels to route.
        qs_parcel : float
            Sediment discharge per parcel.
        inlet_cells : list
            List of inlet cell indices.
            
        Returns
        -------
        list
            List of sediment parcel trajectories for visualization.
        """
        L, W = self.model.eta.shape
        dry_depth = self.config['dry_depth']
        alpha = self.config['alpha']  # Diffusion coefficient
        cell_size = self.config['cell_size']
        
        # List to store all trajectories
        trajectories = []
        
        # Route suspended load parcels
        for _ in range(n_parcels):
            # Start parcel at random inlet cell
            y = np.random.choice(inlet_cells)
            x = 0  # Inlet is at x=0
            
            # Track trajectory
            trajectory = [(x, y)]
            
            # Initialize parcel sediment mass
            mass = qs_parcel
            
            # Maximum number of steps to prevent infinite loops
            max_steps = L * 2
            step_count = 0
            
            # Route parcel until it deposits all its sediment or exits the domain
            while mass > 0 and 0 <= x < L and 0 <= y < W and step_count < max_steps:
                step_count += 1
                
                # Check if cell is dry (trap)
                if self.model.depth[x, y] < dry_depth:
                    # Deposit all remaining sediment and break
                    self.deposition[x, y] += mass
                    break
                
                # Calculate routing weights for suspended load:
                # - Water surface gradient (from model.weight)
                # - Flow velocity (higher velocities transport suspended sediment better)
                # - Random diffusion component
                weights = np.zeros(8)
                valid_directions = 0
                
                for i in range(8):
                    nx = x + self.neighbors_x[i]
                    ny = y + self.neighbors_y[i]
                    
                    # Skip out-of-bounds neighbors
                    if nx < 0 or nx >= L or ny < 0 or ny >= W:
                        continue
                    
                    # Skip neighbors that are too dry (suspended load needs water)
                    if self.model.depth[nx, ny] < dry_depth:
                        continue
                    
                    valid_directions += 1
                    
                    # Base weight from water routing
                    w_flow = self.model.weight[x, y, i]
                    
                    # Add velocity component - higher velocity transports sediment better
                    # Use average velocity at the interface
                    if i == 0:  # East
                        v_mag = abs(self.model.ux[x, y]) if x < L-1 else 0
                    elif i == 4:  # West
                        v_mag = abs(self.model.ux[x-1, y]) if x > 0 else 0
                    elif i == 2:  # North
                        v_mag = abs(self.model.uy[x, y]) if y < W-1 else 0
                    elif i == 6:  # South
                        v_mag = abs(self.model.uy[x, y-1]) if y > 0 else 0
                    else:  # Diagonal - use average of components
                        vx = abs(self.model.ux[min(x, nx), y]) if 0 <= min(x, nx) < L-1 else 0
                        vy = abs(self.model.uy[x, min(y, ny)]) if 0 <= min(y, ny) < W-1 else 0
                        v_mag = np.sqrt(vx**2 + vy**2) / np.sqrt(2)  # Scale diagonal
                    
                    # Normalize velocity weight
                    v_max = 1.0  # Max expected velocity
                    w_vel = min(v_mag / v_max, 1.0) ** 2  # Square to emphasize velocity effect
                    
                    # Add diffusion component
                    # Make diffusion inversely proportional to distance for more realistic patterns
                    w_diff = 1.0 / self.neighbor_dists[i]
                    
                    # Combine weights - for suspended load, diffusion and velocity are important
                    # Higher alpha means more random/diffusive behavior
                    weights[i] = (1.0 - alpha) * (0.6 * w_flow + 0.4 * w_vel) + alpha * w_diff
                    
                    # Add small positive weight to ensure some valid paths exist
                    weights[i] = max(1e-6, weights[i])
                
                # Apply momentum to prefer continuing in the same direction if possible
                # For suspended load, momentum is less important than for bedload but still relevant
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
                    
                    # Apply momentum by increasing weights in the same and similar directions
                    if momentum_dir >= 0:
                        # Enhance the weight in the direction of movement
                        weights[momentum_dir] *= 1.2  # Less than bedload (1.3)
                        
                        # Also enhance weights in similar directions
                        weights[(momentum_dir-1) % 8] *= 1.05  # Less than bedload (1.1)
                        weights[(momentum_dir+1) % 8] *= 1.05  # Less than bedload (1.1)
                
                # Normalize weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights /= weight_sum
                elif valid_directions > 0:
                    # If no flow-based weights but valid directions exist,
                    # use uniform diffusion for all valid directions
                    for i in range(8):
                        nx = x + self.neighbors_x[i]
                        ny = y + self.neighbors_y[i]
                        if 0 <= nx < L and 0 <= ny < W and self.model.depth[nx, ny] >= dry_depth:
                            weights[i] = 1.0 / valid_directions
                else:
                    # If no valid routing direction, deposit all sediment and break
                    self.deposition[x, y] += mass
                    break
                
                # Choose direction based on weights
                direction = np.random.choice(8, p=weights)
                
                # Calculate deposition at current cell
                # Suspended load deposits based on water velocity and depth
                # Use a more physically realistic deposition function:
                # - More deposition in low velocity areas
                # - Less deposition in high velocity areas
                # - More deposition in shallow water (but less sensitive than bedload)
                # - Add a settling velocity component
                vel_mag = np.sqrt(self.model.ux[x, y]**2 + self.model.uy[x, y]**2)
                v_critical = 0.3  # Critical velocity for deposition [m/s]
                h_scale = 1.0  # Characteristic depth scale for suspended load [m] - larger than for bedload
                
                # Settling component - always some deposition
                settling_rate = 0.01  # Base settling rate - lower than bedload
                
                # Velocity-dependent component
                if vel_mag < v_critical:
                    vel_factor = 1.0 - (vel_mag / v_critical)
                else:
                    vel_factor = 0.0
                
                # Depth-dependent component (less important for suspended load)
                depth_factor = np.sqrt(h_scale / max(h_scale, self.model.depth[x, y]))
                
                # Combined deposition fraction
                deposit_frac = settling_rate + (0.05 * vel_factor * depth_factor)
                deposit_frac = np.clip(deposit_frac, 0.005, 0.05)  # Lower range than bedload
                
                deposit = mass * deposit_frac
                self.deposition[x, y] += deposit
                mass -= deposit
                
                # Update sediment transport
                nx = x + self.neighbors_x[direction]
                ny = y + self.neighbors_y[direction]
                
                # Check if new position is within domain
                if 0 <= nx < L and 0 <= ny < W:
                    # Update sediment discharge at the interface between cells
                    if nx != x:  # Moving in x-direction
                        idx = min(x, nx)
                        self.qs_x[idx, y] += mass * np.sign(nx - x)
                    if ny != y:  # Moving in y-direction
                        idx = min(y, ny)
                        self.qs_y[x, idx] += mass * np.sign(ny - y)
                    
                    # Update position
                    x, y = nx, ny
                    
                    # Add to trajectory
                    trajectory.append((x, y))
                else:
                    # Parcel exits the domain
                    break
            
            # If parcel has deposited all sediment or reached max steps, add final deposition
            if mass > 0 and 0 <= x < L and 0 <= y < W:
                self.deposition[x, y] += mass
            
            # Add the complete trajectory to the list
            trajectories.append(trajectory)
        
        return trajectories
    
    def _record_stratigraphy(self, net_change, dt):
        """
        Record stratigraphy information.
        
        Parameters
        ----------
        net_change : ndarray
            Net change in bed elevation.
        dt : float
            Time step duration.
        """
        L, W = self.model.eta.shape
        
        # Limit the number of stratigraphic layers to prevent memory issues
        max_strata_layers = 100  # Limit maximum number of layers stored
        
        # Check if any deposition occurred this time step
        if np.any(net_change > 0):
            # Only record if we're under the maximum layers limit
            current_layers = self.model.strata_age.shape[2]
            
            if current_layers < max_strata_layers:
                # Record age of deposition
                self.model.strata_age = np.append(
                    self.model.strata_age, 
                    np.zeros((L, W, 1)), 
                    axis=2
                )
                self.model.strata_age[:, :, -1] = self.model.time if hasattr(self.model, 'time') else 0
                
                # Record sand fraction (simplified - could be more complex)
                sand_frac = 0.7  # Assuming 70% sand in this example
                self.model.strata_sand_frac = np.append(
                    self.model.strata_sand_frac, 
                    np.zeros((L, W, 1)), 
                    axis=2
                )
                self.model.strata_sand_frac[:, :, -1] = sand_frac
            else:
                # Log message about reaching maximum layers
                print(f"WARNING: Maximum number of stratigraphy layers ({max_strata_layers}) reached. Not recording additional layers.")