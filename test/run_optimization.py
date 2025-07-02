from bayesian_optimizer import BayesianOptimizer
from objective_function import RocketEngineObjective
import numpy as np
import json
from datetime import datetime

def main():
    # Define parameter bounds
    bounds = [
        (0.5, 1.5),    # throat_diameter
        (1.25, 3.25),  # exit_diameter
        (2.0, 4.5),    # chamber_diameter
        (4.0, 6.0),    # chamber_length
        (10, 20),      # exit_half_angle
        (4, 12),       # ox_number_of_holes
        (0.05, 0.4),   # ox_injector_diameter
        (4, 12),       # f_number_of_holes
        (0.05, 0.4)    # f_injector_diameter
    ]
    
    # Create optimizer and objective function
    optimizer = BayesianOptimizer(
        bounds=bounds,
        n_initial=10,
        n_iterations=50,
        random_state=42
    )
    
    objective = RocketEngineObjective()
    
    # Run optimization
    best_params, best_value = optimizer.optimize(objective)
    
    # Save results
    results = {
        'best_parameters': dict(zip(objective.param_names, best_params.tolist())),
        'best_value': float(best_value),
        'optimization_time': datetime.now().isoformat()
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nOptimization completed!")
    print(f"Best total impulse: {best_value:.2f}")
    print("\nBest parameters:")
    for name, value in zip(objective.param_names, best_params):
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()