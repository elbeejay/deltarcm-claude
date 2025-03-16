# DeltaRCM Model Implementation Details

## Overview

This document provides technical details on the implementation of the DeltaRCM model. The model is a reduced-complexity model for simulating river delta formation, based on the work of Liang et al. (2015).

## Model Architecture

The model is implemented as a Python package with a modular structure:

1. **Core Model Class** (`deltarcm/model.py`): Handles initialization, time stepping, and coordination of the different model components.
2. **Water Routing Module** (`deltarcm/water_routing.py`): Implements the flow routing scheme (FlowRCM) using a weighted random walk method.
3. **Sediment Routing Module** (`deltarcm/sediment_routing.py`): Handles sediment transport and bed topography updates.
4. **Utility Functions** (`deltarcm/utils.py`): Provides visualization and output functions.

## Key Algorithms

### Water Routing

The water routing scheme is based on a stochastic parcel-based cellular routing approach:

1. **Routing Direction Determination**: At each cell, water parcels have a probability of moving to any of the 8 neighboring cells based on water surface gradients.
2. **Weight Calculation**: The probability is proportional to the water surface slope between the current cell and the neighbor, raised to a power (theta).
3. **Water Parcel Tracking**: Water parcels are initialized at the inlet and traced through the domain until they exit or reach a trap (dry cell).
4. **Flow Field Calculation**: Water discharge fields are derived from the trajectories of all water parcels.

### Water Surface Calculation

The water surface is calculated using an iterative approach that accounts for backwater effects:

1. **Initial Condition**: The initial water surface is set based on the bed topography and initial water depth.
2. **Iteration**: The water surface is updated iteratively based on local mass balance, using the routing weights to determine flux between cells.
3. **Boundary Conditions**: Fixed water levels are maintained at the inlet and ocean boundaries.

### Sediment Transport

The sediment transport module handles both bedload and suspended load:

1. **Bedload Transport**: Bedload parcels are routed with a stronger influence of bed topography (topographic steering).
2. **Suspended Load Transport**: Suspended load parcels are routed with more random diffusion and less influence from bed topography.
3. **Deposition Rules**: Bedload deposits more in shallow water, while suspended load deposits more in slow-moving water.
4. **Bed Topography Update**: The bed elevation is updated based on the net deposition and erosion patterns.

### Stratigraphy Recording

The model includes an optional component for recording stratigraphy information:

1. **Age Recording**: The age of deposits is tracked at each location.
2. **Composition Recording**: The sand fraction of deposits is recorded.

## Model Parameters

The model includes numerous parameters that can be adjusted to simulate different delta environments:

### Grid and Time Parameters
- `L`, `W`: Domain length and width [cells]
- `cell_size`: Cell size [m]
- `dt`: Time step [s]
- `n_steps`: Number of time steps to simulate

### Flow Parameters
- `h0`: Inlet channel depth [m]
- `Qw`: Water discharge [m³/s]
- `N_crosssection`: Number of cells for inlet
- `u0`: Inlet water velocity [m/s]
- `c0`: Inlet sediment concentration
- `dry_depth`: Minimum depth to be considered "wet" [m]

### Sediment Parameters
- `f_bedload`: Fraction of sediment transported as bedload
- `f_suspended`: Fraction of sediment transported as suspended load
- `alpha`: Sediment diffusion coefficient
- `beta`: Topographic steering parameter

### Sea Level Parameters
- `SLR`: Sea level rise rate [m/yr]
- `h_ocean`: Ocean depth [m]

### Numerical Parameters
- `N_water_parcels`: Number of water parcels per iteration
- `itermax`: Maximum number of iterations for water surface calculation
- `theta`: Weight for flow partition calculation

## Model Outputs

The model outputs include:

1. **Bed Topography**: The evolving bed elevation field.
2. **Water Depth**: The water depth field.
3. **Flow Field**: The water discharge and velocity fields.
4. **Sediment Transport**: The sediment discharge fields.
5. **Stratigraphy**: The record of deposit age and composition.

These outputs can be visualized using the utility functions provided in `deltarcm/utils.py`.

## Example Applications

1. **Basic Delta Formation**: Simulation of a delta prograding into a basin with constant sea level.
2. **Sea Level Rise**: Simulation of delta formation under different sea level rise rates.
3. **Varying Sediment Input**: Simulation of delta response to changes in sediment supply.

## References

1. Liang, M., Voller, V. R., & Paola, C. (2015). A reduced-complexity model for river delta formation – Part 1: Modeling deltas with channel dynamics. Earth Surface Dynamics, 3, 67-86.
2. Liang, M., Geleynse, N., Edmonds, D. A., & Passalacqua, P. (2015). A reduced-complexity model for river delta formation – Part 2: Assessment of the flow routing scheme. Earth Surface Dynamics, 3, 87-104.