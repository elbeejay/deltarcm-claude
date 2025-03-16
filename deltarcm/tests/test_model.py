"""
Tests for the DeltaRCM model.
"""

import unittest
import numpy as np
from deltarcm.model import DeltaRCM


class TestDeltaRCM(unittest.TestCase):
    """Test case for the DeltaRCM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            # Grid parameters
            'L': 50,                # Domain length [cells]
            'W': 40,                # Domain width [cells]
            'cell_size': 10,        # Cell size [m]
            
            # Time parameters
            'dt': 86400,            # Time step [s] (1 day)
            'n_steps': 2,           # Number of time steps to simulate
            
            # Flow parameters
            'h0': 2.0,              # Inlet channel depth [m]
            'Qw': 500,              # Water discharge [m³/s]
            'N_crosssection': 4,    # Number of cells for inlet
            'u0': 1.0,              # Inlet water velocity [m/s]
            'c0': 0.01,             # Inlet sediment concentration
            'dry_depth': 0.1,       # Minimum depth to be considered "wet" [m]
            
            # Sediment parameters
            'f_bedload': 0.5,       # Fraction of sediment transported as bedload
            'f_suspended': 0.5,     # Fraction of sediment transported as suspended load
            'alpha': 0.2,           # Sediment diffusion coefficient
            'beta': 0.5,            # Topographic steering parameter
            
            # Sea level parameters
            'SLR': 0.0,             # Sea level rise rate [m/yr]
            'h_ocean': 5.0,         # Ocean depth [m]
            
            # Numerical parameters
            'N_water_parcels': 100, # Number of water parcels per iteration
            'itermax': 10,          # Maximum number of iterations for water surface calculation
            'theta': 1.0,           # Weight for flow partition calculation
            
            # Output parameters
            'save_interval': 1,     # Save output every N steps
            'plot_interval': 1,     # Plot output every N steps
            'output_dir': './test_output' # Directory for output files
        }
        
        # Create model instance
        self.model = DeltaRCM(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        # Test dimensions
        self.assertEqual(self.model.eta.shape, (50, 40))
        self.assertEqual(self.model.depth.shape, (50, 40))
        
        # Test inlet configuration
        center = self.config['W'] // 2
        half_inlet = self.config['N_crosssection'] // 2
        inlet_start = max(0, center - half_inlet)
        inlet_end = min(self.config['W'], center + half_inlet)
        
        # Check inlet depth
        for y in range(inlet_start, inlet_end):
            self.assertEqual(self.model.depth[0, y], self.config['h0'])
            self.assertEqual(self.model.eta[0, y], -self.config['h0'])
        
        # Check ocean depth
        ocean_start = int(0.8 * self.config['L'])
        for x in range(ocean_start, self.config['L']):
            for y in range(self.config['W']):
                self.assertEqual(self.model.depth[x, y], self.config['h_ocean'])
                self.assertEqual(self.model.eta[x, y], -self.config['h_ocean'])
    
    def test_timestep(self):
        """Test running a single time step."""
        # Store initial state
        eta_init = self.model.eta.copy()
        
        # Run a time step
        self.model.run_timestep()
        
        # Check that something changed
        self.assertFalse(np.array_equal(eta_init, self.model.eta))
        
        # Check conservation of mass (approximately)
        # In a closed system, the total volume change should be close to the sediment input
        delta_vol = np.sum(self.model.eta - eta_init) * self.config['cell_size']**2
        input_vol = self.config['Qw'] * self.config['c0'] * self.config['dt']
        
        # Allow for some numerical error
        self.assertLess(abs(delta_vol - input_vol) / input_vol, 0.5)


if __name__ == '__main__':
    unittest.main()