"""
Bayesian Model Selection via Marginal Likelihood

This module implements Bayesian model comparison for different levels of
parameter sharing in multivariate Bernoulli models. Uses the marginal
likelihood (evidence) to compute posterior probabilities over models.

Author: Efstathios Siatras
"""

import numpy as np
from scipy.special import betaln
from typing import Tuple, Dict


def model_A(X: np.ndarray) -> float:
    """
    Model A: All dimensions from Bernoulli(0.5)

    This is the simplest model with no free parameters. All dimensions
    are generated from a Bernoulli distribution with fixed p=0.5.

    The log marginal likelihood is:
        log p(X|Model A) = N×D×log(0.5)

    Args:
        X: Binary data matrix of shape (N, D)

    Returns:
        Log marginal likelihood under Model A
    """
    num_samples = X.shape[0]  # N
    num_dims = X.shape[1]     # D

    log_marginal_likelihood = num_samples * num_dims * np.log(0.5)

    return log_marginal_likelihood


def model_B(X: np.ndarray) -> float:
    """
    Model B: All dimensions share one unknown Bernoulli parameter

    All D dimensions are generated from Bernoulli distributions with the
    same unknown probability p. Uses Beta(1,1) = Uniform(0,1) prior on p.

    The log marginal likelihood with Beta prior is:
        log p(X|Model B) = log Beta(α + k, β + N×D - k)
    where k = total number of 1's in X, α=1, β=1

    Args:
        X: Binary data matrix of shape (N, D)

    Returns:
        Log marginal likelihood under Model B
    """
    # Total count of 1's across all dimensions and samples
    total_ones = np.sum(X)

    # Total count of 0's
    total_zeros = np.sum(1 - X)

    # Parameters for posterior Beta distribution (with uniform prior α=β=1)
    alpha_post = total_ones + 1
    beta_post = total_zeros + 1

    # Log marginal likelihood using Beta function
    # Note: The full formula is log B(α+k, β+N·D-k) - log B(α, β),
    # but with uniform prior α=β=1, B(1,1) = 1 so the second term vanishes.
    log_marginal_likelihood = betaln(alpha_post, beta_post)

    return log_marginal_likelihood


def model_C(X: np.ndarray) -> float:
    """
    Model C: Each dimension has separate unknown Bernoulli parameter

    Each of D dimensions has its own unknown probability p_d.
    Uses independent Beta(1,1) priors on each p_d.

    The log marginal likelihood is:
        log p(X|Model C) = Σ_d log Beta(α + k_d, β + N - k_d)
    where k_d = number of 1's in dimension d, α=1, β=1

    Args:
        X: Binary data matrix of shape (N, D)

    Returns:
        Log marginal likelihood under Model C
    """
    num_samples = X.shape[0]

    # Count 1's for each dimension
    counts_per_dim = np.sum(X, axis=0)

    # Parameters for posterior Beta distribution for each dimension
    alpha_post = counts_per_dim + 1
    beta_post = num_samples - counts_per_dim + 1

    # Log marginal likelihood for each dimension
    # Note: With uniform prior α=β=1, B(1,1) = 1 so the denominator vanishes.
    log_likelihood_per_dim = betaln(alpha_post, beta_post)

    # Sum over all dimensions
    log_marginal_likelihood = np.sum(log_likelihood_per_dim)

    return log_marginal_likelihood


def compare_models(
    X: np.ndarray,
    prior_probs: Dict[str, float] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compare all three models and compute posterior probabilities.

    Assuming equal priors p(Model A) = p(Model B) = p(Model C) = 1/3,
    the posterior probability is:
        p(Model_i | X) ∝ p(X | Model_i) × p(Model_i)

    Args:
        X: Binary data matrix of shape (N, D)
        prior_probs: Optional dict with prior probabilities for each model
                    Default: {'A': 1/3, 'B': 1/3, 'C': 1/3}

    Returns:
        Tuple of (log_marginal_likelihoods, posterior_probabilities)
        Both are dicts with keys 'A', 'B', 'C'
    """
    if prior_probs is None:
        prior_probs = {'A': 1/3, 'B': 1/3, 'C': 1/3}

    # Compute log marginal likelihoods
    log_ml = {
        'A': model_A(X),
        'B': model_B(X),
        'C': model_C(X)
    }

    # Compute log posteriors (unnormalized)
    log_posteriors = {
        model: log_ml[model] + np.log(prior_probs[model])
        for model in ['A', 'B', 'C']
    }

    # Normalize using log-sum-exp trick for numerical stability
    max_log_post = max(log_posteriors.values())
    posteriors = {
        model: np.exp(log_posteriors[model] - max_log_post)
        for model in ['A', 'B', 'C']
    }
    Z = sum(posteriors.values())  # Normalization constant
    posteriors = {model: p / Z for model, p in posteriors.items()}

    return log_ml, posteriors


def print_model_comparison(X: np.ndarray):
    """
    Print detailed model comparison results.

    Args:
        X: Binary data matrix
    """
    print("Bayesian Model Selection")
    print("=" * 70)
    print(f"Data: {X.shape[0]} samples × {X.shape[1]} dimensions")
    print()

    # Model descriptions
    models_desc = {
        'A': "All dimensions Bernoulli(0.5)",
        'B': "All dimensions share one unknown p",
        'C': "Each dimension has separate unknown p_d"
    }

    log_ml, posteriors = compare_models(X)

    print("Model Descriptions:")
    for model in ['A', 'B', 'C']:
        print(f"  Model {model}: {models_desc[model]}")
    print()

    print("Results:")
    print(f"{'Model':<8} {'Log Marginal Likelihood':<25} {'Posterior Probability':<20}")
    print("-" * 70)
    for model in ['A', 'B', 'C']:
        print(f"{model:<8} {log_ml[model]:>24.2f} {posteriors[model]:>19.4f}")
    print("=" * 70)
    print()

    # Determine best model
    best_model = max(posteriors, key=posteriors.get)
    print(f"Best Model: {best_model} ({models_desc[best_model]})")
    print(f"Posterior Probability: {posteriors[best_model]:.4f}")
    print()

    # Bayes factors
    print("Bayes Factors (relative to Model A):")
    bf_B_vs_A = np.exp(log_ml['B'] - log_ml['A'])
    bf_C_vs_A = np.exp(log_ml['C'] - log_ml['A'])
    print(f"  BF(B vs A) = {bf_B_vs_A:.2e}")
    print(f"  BF(C vs A) = {bf_C_vs_A:.2e}")
    print()

    # Interpretation
    print("Interpretation:")
    if posteriors['A'] > 0.95:
        print("  Strong evidence for Model A (all dims from Bernoulli(0.5))")
    elif posteriors['B'] > 0.95:
        print("  Strong evidence for Model B (shared parameter)")
    elif posteriors['C'] > 0.95:
        print("  Strong evidence for Model C (separate parameters)")
    else:
        print("  Model uncertainty - no single model is strongly preferred")

    return log_ml, posteriors
