"""
Utility functions for the DeltaRCM model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Dict, List, Any, Optional, Tuple


def create_output_directory(output_dir: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Parameters
    ----------
    output_dir : str
        Path to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def plot_delta_morphology(
    eta: np.ndarray, 
    depth: np.ndarray, 
    qx: Optional[np.ndarray] = None, 
    qy: Optional[np.ndarray] = None,
    step: int = 0, 
    time_days: float = 0.0, 
    output_dir: str = './output',
    cmap_topo: str = 'terrain',
    cmap_depth: str = 'Blues'
) -> None:
    """
    Plot delta morphology and flow field.
    
    Parameters
    ----------
    eta : ndarray
        Bed elevation.
    depth : ndarray
        Water depth.
    qx : ndarray, optional
        Water discharge in x-direction.
    qy : ndarray, optional
        Water discharge in y-direction.
    step : int, optional
        Time step number.
    time_days : float, optional
        Time in days.
    output_dir : str, optional
        Output directory.
    cmap_topo : str, optional
        Colormap for topography.
    cmap_depth : str, optional
        Colormap for water depth.
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot bed elevation
    im1 = axes[0].imshow(eta.T, origin='lower', cmap=cmap_topo)
    axes[0].set_title('Bed Elevation [m]')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot water depth
    im2 = axes[1].imshow(depth.T, origin='lower', cmap=cmap_depth)
    axes[1].set_title('Water Depth [m]')
    plt.colorbar(im2, ax=axes[1])
    
    # Add quiver plot of flow field if provided
    if qx is not None and qy is not None:
        # Downsample velocity field for clarity
        stride = 5
        L, W = eta.shape
        x, y = np.meshgrid(np.arange(0, L, stride), np.arange(0, W, stride))
        u = qx[::stride, ::stride].T
        v = qy[::stride, ::stride].T
        
        # Normalize vectors
        magnitude = np.sqrt(u**2 + v**2)
        u_norm = np.zeros_like(u)
        v_norm = np.zeros_like(v)
        
        # Avoid division by zero
        mask = magnitude > 0
        u_norm[mask] = u[mask] / magnitude[mask]
        v_norm[mask] = v[mask] / magnitude[mask]
        
        # Plot flow vectors
        axes[1].quiver(x, y, u_norm, v_norm, color='white', scale=30)
    
    plt.suptitle(f'Time step: {step}, Time: {time_days:.1f} days')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{output_dir}/step_{step:04d}.png', dpi=200)
    plt.close()


def plot_delta_3d(
    eta: np.ndarray,
    depth: np.ndarray,
    step: int = 0,
    time_days: float = 0.0,
    output_dir: str = './output',
    cmap_topo: str = 'terrain',
    elev: float = 25,
    azim: float = 210
) -> None:
    """
    Create a 3D plot of the delta topography.
    
    Parameters
    ----------
    eta : ndarray
        Bed elevation.
    depth : ndarray
        Water depth.
    step : int, optional
        Time step number.
    time_days : float, optional
        Time in days.
    output_dir : str, optional
        Output directory.
    cmap_topo : str, optional
        Colormap for topography.
    elev : float, optional
        Elevation angle for 3D view.
    azim : float, optional
        Azimuth angle for 3D view.
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plot
    L, W = eta.shape
    x, y = np.meshgrid(np.arange(L), np.arange(W))
    
    # Plot water surface
    water_surface = eta + depth
    water_surface_above = water_surface > eta
    
    # Plot bed
    surf1 = ax.plot_surface(
        x, y, eta.T, 
        cmap=cmap_topo, 
        linewidth=0, 
        antialiased=True, 
        alpha=1.0,
        label='Bed'
    )
    
    # Plot water surface
    if np.any(water_surface_above):
        surf2 = ax.plot_surface(
            x, y, water_surface.T,
            cmap='Blues',
            linewidth=0,
            antialiased=True,
            alpha=0.5,
            label='Water Surface'
        )
    
    # Set labels
    ax.set_xlabel('X [cells]')
    ax.set_ylabel('Y [cells]')
    ax.set_zlabel('Elevation [m]')
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    plt.title(f'Delta Topography - Time step: {step}, Time: {time_days:.1f} days')
    
    # Save figure
    plt.savefig(f'{output_dir}/3d_step_{step:04d}.png', dpi=200)
    plt.close()


def plot_cross_section(
    eta: np.ndarray,
    depth: np.ndarray,
    y_index: int,
    step: int = 0,
    time_days: float = 0.0,
    output_dir: str = './output'
) -> None:
    """
    Plot a cross-section of the delta.
    
    Parameters
    ----------
    eta : ndarray
        Bed elevation.
    depth : ndarray
        Water depth.
    y_index : int
        Index for y-coordinate of cross-section.
    step : int, optional
        Time step number.
    time_days : float, optional
        Time in days.
    output_dir : str, optional
        Output directory.
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Get dimensions
    L = eta.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract cross-section data
    x = np.arange(L)
    bed = eta[:, y_index]
    water_surface = bed + depth[:, y_index]
    
    # Plot cross-section
    ax.plot(x, bed, 'k-', linewidth=2, label='Bed Elevation')
    ax.plot(x, water_surface, 'b-', linewidth=1.5, label='Water Surface')
    ax.fill_between(x, bed, water_surface, color='lightblue', alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Distance [cells]')
    ax.set_ylabel('Elevation [m]')
    ax.legend()
    
    plt.title(f'Cross-section at y={y_index} - Time step: {step}, Time: {time_days:.1f} days')
    plt.grid(True)
    
    # Save figure
    plt.savefig(f'{output_dir}/cross_section_y{y_index}_step_{step:04d}.png', dpi=200)
    plt.close()


def plot_parcel_trajectories(
    water_trajectories: List[List[Tuple[int, int]]],
    sed_trajectories: List[List[Tuple[int, int]]],
    eta: np.ndarray,
    depth: np.ndarray,
    step: int = 0,
    time_days: float = 0.0,
    output_dir: str = './output',
    max_trajectories: int = 50
) -> None:
    """
    Plot water and sediment parcel trajectories.
    
    Parameters
    ----------
    water_trajectories : list of lists of tuples
        List of water parcel trajectories. Each trajectory is a list of (x, y) points.
    sed_trajectories : list of lists of tuples
        List of sediment parcel trajectories. Each trajectory is a list of (x, y) points.
    eta : ndarray
        Bed elevation for background.
    depth : ndarray
        Water depth for background.
    step : int, optional
        Time step number.
    time_days : float, optional
        Time in days.
    output_dir : str, optional
        Output directory.
    max_trajectories : int, optional
        Maximum number of trajectories to plot to avoid overcrowding.
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Create figure for water trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot background as bed elevation
    im = ax.imshow(eta.T, origin='lower', cmap='terrain', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Bed Elevation [m]')
    
    # Draw wet area boundaries
    wet_mask = depth > 0.1  # Using the usual dry depth threshold
    ax.contour(wet_mask.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.5)
    
    # Limit the number of trajectories to avoid overcrowding
    n_water = min(len(water_trajectories), max_trajectories)
    
    # Sample trajectories randomly if there are too many
    if len(water_trajectories) > max_trajectories:
        indices = np.random.choice(len(water_trajectories), n_water, replace=False)
        water_sample = [water_trajectories[i] for i in indices]
    else:
        water_sample = water_trajectories[:n_water]
    
    # Plot water trajectories
    for traj in water_sample:
        if len(traj) > 1:  # Only plot if trajectory has at least 2 points
            x, y = zip(*traj)
            ax.plot(x, y, '-', color='blue', linewidth=0.8, alpha=0.6)
            # Mark the end point
            ax.plot(x[-1], y[-1], 'o', color='blue', markersize=3, alpha=0.7)
    
    ax.set_title('Water Parcel Trajectories')
    ax.set_xlabel('X [cells]')
    ax.set_ylabel('Y [cells]')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/water_trajectories_step_{step:04d}.png', dpi=200)
    plt.close()
    
    # Create figure for sediment trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot background as bed elevation
    im = ax.imshow(eta.T, origin='lower', cmap='terrain', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Bed Elevation [m]')
    
    # Draw wet area boundaries
    ax.contour(wet_mask.T, levels=[0.5], colors='blue', linewidths=1, alpha=0.5)
    
    # Limit the number of trajectories
    n_sed = min(len(sed_trajectories), max_trajectories)
    
    # Sample trajectories randomly if there are too many
    if len(sed_trajectories) > max_trajectories:
        indices = np.random.choice(len(sed_trajectories), n_sed, replace=False)
        sed_sample = [sed_trajectories[i] for i in indices]
    else:
        sed_sample = sed_trajectories[:n_sed]
    
    # Plot sediment trajectories
    for traj in sed_sample:
        if len(traj) > 1:  # Only plot if trajectory has at least 2 points
            x, y = zip(*traj)
            ax.plot(x, y, '-', color='orange', linewidth=0.8, alpha=0.6)
            # Mark the deposition point
            ax.plot(x[-1], y[-1], 'o', color='red', markersize=3, alpha=0.7)
    
    ax.set_title('Sediment Parcel Trajectories')
    ax.set_xlabel('X [cells]')
    ax.set_ylabel('Y [cells]')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sediment_trajectories_step_{step:04d}.png', dpi=200)
    plt.close()
    
    # Create a combined plot with both water and sediment
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot background as bed elevation with water depth overlay
    im1 = ax.imshow(eta.T, origin='lower', cmap='terrain', alpha=0.7)
    
    # Create a blue overlay for water depth
    cmap_water = plt.cm.Blues
    norm_water = colors.Normalize(vmin=0, vmax=np.max(depth) if np.max(depth) > 0 else 1)
    im2 = ax.imshow(depth.T, origin='lower', cmap=cmap_water, alpha=0.3, norm=norm_water)
    
    # Create colorbars
    plt.colorbar(im1, ax=ax, label='Bed Elevation [m]')
    
    # Plot water trajectories in blue
    for traj in water_sample:
        if len(traj) > 1:
            x, y = zip(*traj)
            ax.plot(x, y, '-', color='blue', linewidth=0.8, alpha=0.4)
            ax.plot(x[-1], y[-1], 'o', color='blue', markersize=3, alpha=0.5)
    
    # Plot sediment trajectories in orange/red
    for traj in sed_sample:
        if len(traj) > 1:
            x, y = zip(*traj)
            ax.plot(x, y, '-', color='orange', linewidth=0.8, alpha=0.6)
            ax.plot(x[-1], y[-1], 'o', color='red', markersize=3, alpha=0.7)
    
    ax.set_title(f'Water and Sediment Parcel Trajectories - Step: {step}, Time: {time_days:.1f} days')
    ax.set_xlabel('X [cells]')
    ax.set_ylabel('Y [cells]')
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, alpha=0.7, label='Water Parcels'),
        Line2D([0], [0], color='orange', lw=2, alpha=0.7, label='Sediment Parcels'),
        Line2D([0], [0], marker='o', color='red', label='Deposition Points',
               markersize=5, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_trajectories_step_{step:04d}.png', dpi=200)
    plt.close()


def plot_deposition_patterns(
    eta: np.ndarray,
    deposition: np.ndarray,
    erosion: np.ndarray,
    step: int = 0,
    time_days: float = 0.0,
    output_dir: str = './output'
) -> None:
    """
    Plot sediment deposition and erosion patterns.
    
    Parameters
    ----------
    eta : ndarray
        Bed elevation for background.
    deposition : ndarray
        Sediment deposition field.
    erosion : ndarray
        Sediment erosion field.
    step : int, optional
        Time step number.
    time_days : float, optional
        Time in days.
    output_dir : str, optional
        Output directory.
    """
    # Create output directory if it doesn't exist
    create_output_directory(output_dir)
    
    # Create figure for deposition patterns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot background as bed elevation
    im1 = axes[0].imshow(eta.T, origin='lower', cmap='terrain')
    axes[0].set_title('Bed Elevation [m]')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot deposition field
    if np.max(deposition) > 0:
        im2 = axes[1].imshow(deposition.T, origin='lower', cmap='Greens')
        axes[1].set_title('Deposition [m]')
        plt.colorbar(im2, ax=axes[1])
    else:
        axes[1].imshow(np.zeros_like(deposition).T, origin='lower', cmap='Greens')
        axes[1].set_title('No Deposition')
    
    # Plot erosion field or net change
    net_change = deposition - erosion
    if np.any(erosion > 0) or np.any(net_change < 0):
        # Use a diverging colormap for net change
        cmap_div = plt.cm.RdBu_r
        abs_max = max(abs(np.min(net_change)), abs(np.max(net_change)))
        if abs_max > 0:
            norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)
            im3 = axes[2].imshow(net_change.T, origin='lower', cmap=cmap_div, norm=norm)
            axes[2].set_title('Net Change [m] (+ deposition, - erosion)')
            plt.colorbar(im3, ax=axes[2])
        else:
            axes[2].imshow(np.zeros_like(net_change).T, origin='lower', cmap=cmap_div)
            axes[2].set_title('No Net Change')
    else:
        axes[2].imshow(np.zeros_like(erosion).T, origin='lower', cmap='Reds_r')
        axes[2].set_title('No Erosion')
    
    plt.suptitle(f'Sediment Patterns - Time step: {step}, Time: {time_days:.1f} days')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{output_dir}/sediment_patterns_step_{step:04d}.png', dpi=200)
    plt.close()