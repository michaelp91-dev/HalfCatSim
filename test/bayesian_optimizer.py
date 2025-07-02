import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import Tuple, List, Callable, Optional
import logging
from datetime import datetime

class BayesianOptimizer:
    """
    Bayesian Optimization implementation for rocket engine design optimization.
    Uses Gaussian Process Regression with Expected Improvement acquisition function.
    """
    
    def __init__(self, 
                 bounds: List[Tuple[float, float]], 
                 n_initial: int = 10, 
                 n_iterations: int = 50,
                 random_state: Optional[int] = None):
        """
        Initialize the Bayesian Optimizer.
        
        Args:
            bounds: List of tuples (min, max) for each parameter
            n_initial: Number of initial random samples
            n_iterations: Number of optimization iterations
            random_state: Random seed for reproducibility
        """
        self.bounds = np.array(bounds)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.dim = len(bounds)
        
        # Initialize storage for samples and results
        self.X_sample = None
        self.Y_sample = None
        self.best_params = None
        self.best_value = None
        
        # Set up GP regression with Matern kernel
        kernel = Matern(nu=2.5)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state
        )
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the optimization process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'optimization_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _expected_improvement(self, 
                            X: np.ndarray, 
                            xi: float = 0.01) -> np.ndarray:
        """
        Compute the Expected Improvement acquisition function.
        
        Args:
            X: Points at which to evaluate EI
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Expected improvement values
        """
        mu, sigma = self.gpr.predict(X.reshape(-1, self.dim), return_std=True)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        
        mu_sample_opt = np.max(self.Y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = np.zeros_like(sigma)
            mask = sigma > 0
            Z[mask] = imp[mask] / sigma[mask]
            ei = np.zeros_like(Z)
            ei[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])
            
        return ei
        
    def _sample_initial_points(self) -> np.ndarray:
        """Generate initial random samples within bounds."""
        return np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.n_initial, self.dim)
        )
    
    def optimize(self, objective_func: Callable) -> Tuple[np.ndarray, float]:
        """
        Run the Bayesian optimization process.
        
        Args:
            objective_func: Function to optimize, should take array of parameters and return scalar
            
        Returns:
            Tuple of (best parameters, best objective value)
        """
        # Initial sampling
        self.X_sample = self._sample_initial_points()
        self.Y_sample = np.array([objective_func(x) for x in self.X_sample])
        
        self.logger.info(f"Initial sampling completed with {self.n_initial} points")
        
        # Main optimization loop
        for i in range(self.n_iterations):
            # Fit GP model
            self.gpr.fit(self.X_sample, self.Y_sample)
            
            # Find next point to evaluate using EI
            next_x = self._find_next_point()
            next_y = objective_func(next_x)
            
            # Update samples
            self.X_sample = np.vstack((self.X_sample, next_x))
            self.Y_sample = np.append(self.Y_sample, next_y)
            
            # Update best found
            if next_y > np.max(self.Y_sample[:-1]):
                self.best_params = next_x
                self.best_value = next_y
                self.logger.info(f"New best found at iteration {i + 1}: {self.best_value}")
            
            self.logger.info(f"Completed iteration {i + 1}/{self.n_iterations}")
            
        return self.best_params, self.best_value
    
    def _find_next_point(self) -> np.ndarray:
        """Find the next point to evaluate using EI maximization."""
        def negative_ei(x):
            return -self._expected_improvement(x.reshape(1, -1))
        
        best_x = None
        best_ei = -1
        
        # Try multiple random starts to avoid local optima
        n_restarts = 5
        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            res = minimize(
                negative_ei,
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if res.fun < -best_ei:
                best_ei = -res.fun
                best_x = res.x
                
        return best_x