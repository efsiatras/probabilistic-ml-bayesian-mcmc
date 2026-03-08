"""
MCMC Methods: Probabilistic Inference and Decryption

A comprehensive implementation of statistical inference methods including:
- Bayesian parameter estimation (ML/MAP)
- Bayesian model selection
- Expectation-Maximization algorithm for mixture models
- MCMC-based cryptanalysis using Metropolis-Hastings

Author: Efstathios Siatras
"""

__version__ = "1.0.0"

from . import bayesian_inference
from . import model_selection
from . import em_algorithm
from . import mcmc_cryptanalysis

__all__ = [
    "bayesian_inference",
    "model_selection",
    "em_algorithm",
    "mcmc_cryptanalysis",
]
