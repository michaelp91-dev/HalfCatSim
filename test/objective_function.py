from typing import List, Dict, Any
import numpy as np
from halfcatsim import run_simulation  # Import your existing simulation function

class RocketEngineObjective:
    """
    Wrapper class for rocket engine simulation objective function.
    """
    
    def __init__(self):
        # Define parameter names for clarity
        self.param_names = [
            'throat_diameter',
            'exit_diameter',
            'chamber_diameter',
            'chamber_length',
            'exit_half_angle',
            'ox_number_of_holes',
            'ox_injector_diameter',
            'f_number_of_holes',
            'f_injector_diameter'
        ]
        
    def __call__(self, x: np.ndarray) -> float:
        """
        Compute objective value for given parameters.
        Handles constraints and returns penalty value if constraints are violated.
        
        Args:
            x: Array of parameter values
            
        Returns:
            Objective value (total impulse or other metric)
        """
        # Convert parameters to dictionary
        params = dict(zip(self.param_names, x))
        
        # Check constraints
        if not self._check_constraints(params):
            return -np.inf
        
        try:
            # Run simulation with parameters
            results = run_simulation(
                throat_diameter=params['throat_diameter'],
                exit_diameter=params['exit_diameter'],
                chamber_diameter=params['chamber_diameter'],
                chamber_length=params['chamber_length'],
                exit_half_angle=params['exit_half_angle'],
                ox_number_of_holes=int(params['ox_number_of_holes']),
                ox_injector_hole_diameter=params['ox_injector_diameter'],
                f_number_of_holes=int(params['f_number_of_holes']),
                f_injector_hole_diameter=params['f_injector_diameter']
            )
            
            # Return total impulse as objective
            return results['total_impulse']
            
        except Exception as e:
            print(f"Simulation failed with parameters {params}: {str(e)}")
            return -np.inf
            
    def _check_constraints(self, params: Dict[str, Any]) -> bool:
        """
        Check if parameters satisfy physical constraints.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        # Exit diameter must be larger than throat diameter
        if params['exit_diameter'] <= params['throat_diameter']:
            return False
            
        # Chamber diameter must be larger than throat diameter
        if params['chamber_diameter'] <= params['throat_diameter']:
            return False
            
        # Number of holes must be integers
        if not (params['ox_number_of_holes'].is_integer() and 
                params['f_number_of_holes'].is_integer()):
            return False
            
        return True