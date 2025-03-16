"""
DeltaRCM: A reduced-complexity model for river delta formation.

This package implements the DeltaRCM model as described in:
Liang et al. (2015). A reduced-complexity model for river delta formation – Part 1: Modeling deltas with channel dynamics
Liang et al. (2015). A reduced-complexity model for river delta formation – Part 2: Assessment of the flow routing scheme

The model uses a stochastic parcel-based cellular routing scheme for water and sediment,
with four main components:
1. Water routing using a weighted random walk
2. Water surface calculation
3. Sediment transport and bed topography update
4. Routing direction/weight update

This implementation includes:
- Core model class (DeltaRCM)
- Water routing module (WaterRouter)
- Sediment routing module (SedimentRouter)
- Utility functions for visualization and output
"""

__version__ = "0.1.0"

# Import main classes for easier access
from deltarcm.model import DeltaRCM
from deltarcm.water_routing import WaterRouter
from deltarcm.sediment_routing import SedimentRouter