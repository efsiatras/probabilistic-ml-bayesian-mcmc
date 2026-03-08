"""
Expectation-Maximization Algorithm for Mixture of Bernoullis

Implements the EM algorithm for clustering binary data using a mixture of
multivariate Bernoulli distributions. Commonly used for document clustering,
binary image classification, and categorical data analysis.

Author: Efstathios Siatras
"""

import os

import numpy as np
from typing import Tuple, List
import plotly.graph_objects as go
import plotly.subplots as sp


def _ensure_save_dir(save_path: str):
    """Create parent directories for save_path if they don't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)


def step_e(X: np.ndarray, pi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    E-step: Compute responsibilities (posterior probabilities over clusters).

    For each data point n and cluster k, compute:
        r_nk = p(z_n=k | x_n) ∝ π_k × p(x_n | z_n=k)

    Args:
        X: Binary data matrix of shape (N, D)
        pi: Mixture weights of shape (K,), sum to 1
        p: Bernoulli parameters of shape (K, D)

    Returns:
        Responsibilities matrix of shape (N, K), rows sum to 1
    """
    num_samples = X.shape[0]
    K = len(pi)

    # Initialize responsibilities
    r = np.zeros((num_samples, K))

    # Clip parameters to avoid numerical issues (0^0, log(0))
    p_safe = np.clip(p, 1e-10, 1 - 1e-10)

    # Compute responsibilities for each sample and cluster
    for n in range(num_samples):
        for k in range(K):
            # Compute p(x_n | z_n=k) using Bernoulli likelihood
            likelihood = np.prod(
                (p_safe[k, :] ** X[n, :]) * ((1 - p_safe[k, :]) ** (1 - X[n, :]))
            )
            r[n, k] = pi[k] * likelihood

        # Normalize to get posterior probabilities
        total = np.sum(r[n])
        if total > 0:
            r[n] = r[n] / total
        else:
            r[n] = 1.0 / K  # uniform fallback

    return r


def step_m(X: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-step: Update parameters given responsibilities.

    Update mixture weights:
        π_k = (1/N) Σ_n r_nk

    Update Bernoulli parameters:
        p_kd = (Σ_n r_nk × x_nd) / (Σ_n r_nk)

    Args:
        X: Binary data matrix of shape (N, D)
        r: Responsibilities matrix of shape (N, K)

    Returns:
        Tuple of (pi, p) where:
            pi: Updated mixture weights of shape (K,)
            p: Updated Bernoulli parameters of shape (K, D)
    """
    num_samples = X.shape[0]
    num_dims = X.shape[1]
    K = r.shape[1]

    # Initialize parameters
    pi = np.zeros(K)
    p = np.zeros((K, num_dims))

    for k in range(K):
        # Effective number of points assigned to cluster k
        N_k = np.sum(r[:, k])

        # Update mixture weight
        pi[k] = N_k / num_samples

        # Update Bernoulli parameters for each dimension
        for d in range(num_dims):
            p[k, d] = np.sum(r[:, k] * X[:, d]) / N_k

    # Clip to prevent exact 0/1 which poison subsequent E-steps
    p = np.clip(p, 1e-10, 1 - 1e-10)

    return pi, p


def compute_log_likelihood(
    X: np.ndarray,
    pi: np.ndarray,
    p: np.ndarray
) -> float:
    """
    Compute log-likelihood of data under current parameters.

    log p(X | π, p) = Σ_n log[Σ_k π_k × p(x_n | p_k)]

    Args:
        X: Binary data matrix of shape (N, D)
        pi: Mixture weights of shape (K,)
        p: Bernoulli parameters of shape (K, D)

    Returns:
        Log-likelihood (scalar)
    """
    num_samples = X.shape[0]
    K = len(pi)

    log_likelihood = 0.0

    p_safe = np.clip(p, 1e-10, 1 - 1e-10)

    for n in range(num_samples):
        # Compute mixture likelihood for sample n
        likelihood_n = 0.0
        for k in range(K):
            # p(x_n | z_n=k)
            likelihood_k = np.prod(
                (p_safe[k, :] ** X[n, :]) * ((1 - p_safe[k, :]) ** (1 - X[n, :]))
            )
            likelihood_n += pi[k] * likelihood_k

        log_likelihood += np.log(max(likelihood_n, 1e-300))

    return log_likelihood


def em_algorithm(
    X: np.ndarray,
    K: int,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-3,
    verbose: bool = False
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Run EM algorithm for mixture of K Bernoullis.

    Args:
        X: Binary data matrix of shape (N, D)
        K: Number of mixture components
        max_iterations: Maximum number of EM iterations
        convergence_threshold: Stop if LL change < threshold
        verbose: Print progress

    Returns:
        Tuple of (log_likelihood_history, pi, p) where:
            log_likelihood_history: List of LL values per iteration
            pi: Final mixture weights of shape (K,)
            p: Final Bernoulli parameters of shape (K, D)
    """
    num_samples, num_dims = X.shape

    # Random initialization
    p = np.random.uniform(0.1, 0.9, size=(K, num_dims))
    pi = np.random.uniform(size=K)
    pi = pi / np.sum(pi)  # Normalize

    log_likelihood_history = []

    for iteration in range(max_iterations):
        # E-step
        r = step_e(X, pi, p)

        # M-step
        pi, p = step_m(X, r)

        # Compute log-likelihood
        ll = compute_log_likelihood(X, pi, p)
        log_likelihood_history.append(ll)

        if verbose and iteration % 5 == 0:
            print(f"Iteration {iteration:3d}: LL = {ll:.2f}")

        # Check convergence
        if iteration > 0:
            ll_change = abs(log_likelihood_history[iteration] -
                           log_likelihood_history[iteration - 1])
            if ll_change < convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

    return log_likelihood_history, pi, p


def visualize_clusters(
    p: np.ndarray,
    K: int,
    pi: np.ndarray = None,
    img_shape: Tuple[int, int] = (8, 8),
    save_path: str = None
) -> go.Figure:
    """
    Visualize learned cluster centers as images.

    Args:
        p: Bernoulli parameters of shape (K, D)
        K: Number of clusters
        pi: Optional mixture weights of shape (K,) for titles
        img_shape: Image shape (height, width)
        save_path: Optional path to save figure

    Returns:
        Plotly figure
    """
    # Create subplot titles with cluster weights if provided
    if pi is not None:
        subplot_titles = [f"<b>Cluster {k+1}</b><br>(π={pi[k]:.2f})" for k in range(K)]
    else:
        subplot_titles = [f"<b>Cluster {k+1}</b>" for k in range(K)]

    fig = sp.make_subplots(
        rows=1,
        cols=K,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02
    )

    for k in range(K):
        p_img = np.flipud(np.reshape(p[k, :], img_shape))
        fig.add_trace(
            go.Heatmap(
                z=p_img,
                colorscale='Greys',
                reversescale=True,
                showscale=(k == K-1),  # Show colorbar only on last subplot
                colorbar=dict(
                    title=dict(text="P(pixel=1)", side="right"),
                    len=0.9
                ) if k == K-1 else None,
                hovertemplate='Row: %{y}<br>Col: %{x}<br>P: %{z:.3f}<extra></extra>'
            ),
            row=1,
            col=k+1
        )

    fig.update_layout(
        template="presentation",
        title=dict(
            text=f"<b>Learned Cluster Centers (K={K})</b>",
            font=dict(size=16)
        ),
        width=max(220*K, 500),
        height=280,
        showlegend=False,
        margin=dict(t=80, b=20, l=20, r=80)
    )

    # Fix aspect ratio and hide axes
    for k in range(K):
        fig.update_xaxes(visible=False, constrain='domain', row=1, col=k+1)
        fig.update_yaxes(visible=False, constrain='domain', scaleanchor=f"x{k+1 if k > 0 else ''}", scaleratio=1, row=1, col=k+1)

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def run_multiple_experiments(
    X: np.ndarray,
    K: int,
    num_experiments: int = 10,
    max_iterations: int = 20,
    save_path: str = None
) -> go.Figure:
    """
    Run EM multiple times with different random initializations.

    Useful for analyzing:
    - Sensitivity to initialization
    - Local optima behavior
    - Consistency of solutions

    Args:
        X: Binary data matrix
        K: Number of clusters
        num_experiments: Number of random restarts
        max_iterations: Max EM iterations per run
        save_path: Optional path to save convergence plot

    Returns:
        Plotly figure showing convergence curves
    """
    all_ll_histories = []

    print(f"Running {num_experiments} experiments with K={K}...")

    for exp in range(num_experiments):
        ll_history, pi, p = em_algorithm(
            X, K, max_iterations,
            convergence_threshold=1e-3,
            verbose=False
        )
        all_ll_histories.append(ll_history)

    # Plot convergence for all experiments
    fig = go.Figure()

    for exp, ll_history in enumerate(all_ll_histories):
        fig.add_trace(go.Scatter(
            x=list(range(len(ll_history))),
            y=ll_history,
            mode='lines+markers',
            name=f'Experiment {exp+1}',
            marker=dict(size=4)
        ))

    fig.update_layout(
        template="presentation",
        title=f'EM Convergence: Multiple Runs (K={K})',
        xaxis_title="Iteration",
        yaxis_title="Log-Likelihood",
        width=900,
        height=600
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    print(f"  Final LLs range: [{min(h[-1] for h in all_ll_histories):.2f}, "
          f"{max(h[-1] for h in all_ll_histories):.2f}]")

    return fig


def plot_initialization_sensitivity(
    X: np.ndarray,
    K_values: List[int] = [3, 7, 10],
    num_experiments: int = 10,
    max_iterations: int = 20,
    save_path: str = None
) -> go.Figure:
    """
    Create a consolidated 3-panel figure showing EM initialization sensitivity.

    Runs multiple experiments for each K value and shows convergence curves
    in a professional publication-grade format.

    Args:
        X: Binary data matrix
        K_values: List of K values to compare (default: [3, 7, 10])
        num_experiments: Number of random restarts per K
        max_iterations: Max EM iterations per run
        save_path: Optional path to save figure

    Returns:
        Plotly figure with subplots for each K
    """
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=len(K_values),
        subplot_titles=[f"<b>K = {K}</b>" for K in K_values],
        horizontal_spacing=0.08
    )

    # Color palette for experiments (colorblind-friendly)
    colors = [
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
        '#95C623', '#5C4D7D', '#44AF69', '#FCAB10', '#2EC4B6'
    ]

    all_results = {}

    for col_idx, K in enumerate(K_values):
        print(f"Running {num_experiments} experiments with K={K}...")
        all_ll_histories = []
        final_lls = []

        for exp in range(num_experiments):
            ll_history, pi, p = em_algorithm(
                X, K, max_iterations,
                convergence_threshold=1e-3,
                verbose=False
            )
            all_ll_histories.append(ll_history)
            final_lls.append(ll_history[-1])

        all_results[K] = {'histories': all_ll_histories, 'final_lls': final_lls}

        # Find best run
        best_exp = np.argmax(final_lls)

        # Add traces for each experiment
        for exp, ll_history in enumerate(all_ll_histories):
            is_best = (exp == best_exp)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(ll_history))),
                    y=ll_history,
                    mode='lines',
                    name=f'Run {exp+1}' if col_idx == 0 else None,
                    line=dict(
                        color=colors[exp % len(colors)],
                        width=3 if is_best else 1.5,
                        dash='solid' if is_best else 'solid'
                    ),
                    opacity=1.0 if is_best else 0.5,
                    showlegend=(col_idx == 0),
                    hovertemplate=f'K={K}, Run {exp+1}<br>Iter: %{{x}}<br>LL: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=col_idx+1
            )

        # Highlight best run with annotation
        best_ll = final_lls[best_exp]
        fig.add_annotation(
            x=len(all_ll_histories[best_exp])-1,
            y=best_ll,
            text=f"Best: {best_ll:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=colors[best_exp % len(colors)],
            font=dict(size=9, color=colors[best_exp % len(colors)]),
            ax=20,
            ay=-20,
            row=1, col=col_idx+1
        )

    # Update axes
    for col_idx in range(len(K_values)):
        fig.update_xaxes(
            title_text="Iteration" if col_idx == 1 else "",
            range=[0, max_iterations + 1],
            dtick=5,
            row=1, col=col_idx+1
        )
        fig.update_yaxes(
            title_text="Log-Likelihood" if col_idx == 0 else "",
            row=1, col=col_idx+1
        )

    fig.update_layout(
        template="presentation",
        title=dict(
            text="<b>EM Initialization Sensitivity Analysis</b>",
            font=dict(size=18)
        ),
        width=1200,
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        ),
        margin=dict(b=100)
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    # Print summary
    print("\nInitialization Sensitivity Summary:")
    for K in K_values:
        lls = all_results[K]['final_lls']
        print(f"  K={K}: Best={max(lls):.1f}, Worst={min(lls):.1f}, Range={max(lls)-min(lls):.1f}")

    return fig


def compare_log_likelihood_bits(
    X: np.ndarray,
    K_values: List[int] = [2, 3, 4, 7, 10],
    max_iterations: int = 20,
    save_path: str = None
) -> go.Figure:
    """
    Express log-likelihoods in bits per dimension and compare to naive encoding.

    Bits per dimension = LL / (N × D × log(2))

    Naive encoding: 1 bit per dimension (no compression)

    Args:
        X: Binary data matrix
        K_values: List of K values to compare
        max_iterations: Max EM iterations
        save_path: Optional path to save figure

    Returns:
        Plotly figure showing bits comparison
    """
    num_samples, num_dims = X.shape
    all_ll_bits = []

    print(f"Computing log-likelihoods in bits for K={K_values}...")

    for K in K_values:
        ll_history, pi, p = em_algorithm(X, K, max_iterations, verbose=False)

        # Convert to bits per dimension
        ll_bits = [ll / (num_samples * num_dims * np.log(2)) for ll in ll_history]
        all_ll_bits.append(ll_bits)

        print(f"  K={K}: Final = {ll_bits[-1]:.4f} bits/dim")

    # Plot
    fig = go.Figure()

    for i, K in enumerate(K_values):
        fig.add_trace(go.Scatter(
            x=list(range(len(all_ll_bits[i]))),
            y=all_ll_bits[i],
            mode='lines+markers',
            name=f'K={K}',
            marker=dict(size=4)
        ))

    # Add naive encoding baseline (-1 bit per dimension)
    fig.add_trace(go.Scatter(
        x=[0, max_iterations],
        y=[-1, -1],
        mode='lines',
        name='Naive Encoding',
        line=dict(dash='dash', color='red', width=2)
    ))

    fig.update_layout(
        template="presentation",
        title='Log-Likelihood per Dimension (bits)',
        xaxis_title="Iteration",
        yaxis_title="Bits per Dimension",
        width=900,
        height=600
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


# ==============================================================================
# MODEL SELECTION: BIC & AIC
# ==============================================================================

def compute_bic(log_likelihood: float, K: int, N: int, D: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = -2 × LL + num_params × log(N)

    Lower BIC indicates better model. Penalizes complexity more heavily
    than AIC, especially for large N.

    Args:
        log_likelihood: Final log-likelihood from EM
        K: Number of clusters
        N: Number of samples
        D: Number of dimensions

    Returns:
        BIC score (lower is better)
    """
    # Parameters: K×D probabilities + (K-1) mixing weights (sum-to-1 constraint)
    num_params = K * D + (K - 1)
    bic = -2 * log_likelihood + num_params * np.log(N)
    return bic


def compute_aic(log_likelihood: float, K: int, D: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = -2 × LL + 2 × num_params

    Lower AIC indicates better model. Penalizes complexity less than BIC.

    Args:
        log_likelihood: Final log-likelihood from EM
        K: Number of clusters
        D: Number of dimensions

    Returns:
        AIC score (lower is better)
    """
    num_params = K * D + (K - 1)
    aic = -2 * log_likelihood + 2 * num_params
    return aic


def model_selection_comparison(
    X: np.ndarray,
    K_values: List[int] = [2, 3, 4, 7, 10],
    max_iterations: int = 20,
    num_runs: int = 5,
    save_path: str = None
) -> Tuple[dict, go.Figure]:
    """
    Compare models using BIC and AIC for different K values.

    Runs EM multiple times per K and selects best run to avoid local optima.

    Args:
        X: Binary data matrix (N, D)
        K_values: List of K values to compare
        max_iterations: Max EM iterations per run
        num_runs: Number of random restarts per K
        save_path: Optional path to save comparison plot

    Returns:
        results: Dictionary with LL, BIC, AIC for each K
        fig: Plotly figure showing model selection curves
    """
    N, D = X.shape
    results = {
        'K': [],
        'log_likelihood': [],
        'BIC': [],
        'AIC': [],
        'num_params': []
    }

    print("Model Selection via Information Criteria")
    print("=" * 70)

    for K in K_values:
        print(f"\nK = {K}...")
        best_ll = float('-inf')
        best_params = None

        # Multiple runs to avoid local optima
        for run in range(num_runs):
            ll_history, pi, p = em_algorithm(
                X, K, max_iterations,
                convergence_threshold=1e-3,
                verbose=False
            )
            final_ll = ll_history[-1]

            if final_ll > best_ll:
                best_ll = final_ll
                best_params = (pi, p)

        # Compute information criteria
        bic = compute_bic(best_ll, K, N, D)
        aic = compute_aic(best_ll, K, D)
        num_params = K * D + (K - 1)

        results['K'].append(K)
        results['log_likelihood'].append(best_ll)
        results['BIC'].append(bic)
        results['AIC'].append(aic)
        results['num_params'].append(num_params)

        print(f"  Best LL: {best_ll:.2f}")
        print(f"  BIC: {bic:.2f} | AIC: {aic:.2f} | Params: {num_params}")

    # Find optimal K for each criterion
    best_k_bic = K_values[np.argmin(results['BIC'])]
    best_k_aic = K_values[np.argmin(results['AIC'])]
    best_k_ll = K_values[np.argmax(results['log_likelihood'])]

    print("\n" + "=" * 70)
    print(f"BEST K (BIC): {best_k_bic}")
    print(f"BEST K (AIC): {best_k_aic}")
    print(f"BEST K (LL):  {best_k_ll} (no penalty)")
    print("=" * 70)

    # Create comparison plot
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Log-Likelihood', 'BIC (lower=better)',
                       'AIC (lower=better)', 'Number of Parameters'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Log-Likelihood
    fig.add_trace(go.Scatter(
        x=results['K'], y=results['log_likelihood'],
        mode='lines+markers', name='Log-Likelihood',
        marker=dict(size=10), line=dict(width=3)
    ), row=1, col=1)

    # BIC
    fig.add_trace(go.Scatter(
        x=results['K'], y=results['BIC'],
        mode='lines+markers', name='BIC',
        marker=dict(size=10, color='red'), line=dict(width=3, color='red')
    ), row=1, col=2)
    fig.add_vline(x=best_k_bic, line_dash="dash", line_color="red",
                  annotation_text=f"Best K={best_k_bic}", row=1, col=2)

    # AIC
    fig.add_trace(go.Scatter(
        x=results['K'], y=results['AIC'],
        mode='lines+markers', name='AIC',
        marker=dict(size=10, color='orange'), line=dict(width=3, color='orange')
    ), row=2, col=1)
    fig.add_vline(x=best_k_aic, line_dash="dash", line_color="orange",
                  annotation_text=f"Best K={best_k_aic}", row=2, col=1)

    # Number of parameters
    fig.add_trace(go.Scatter(
        x=results['K'], y=results['num_params'],
        mode='lines+markers', name='Parameters',
        marker=dict(size=10, color='green'), line=dict(width=3, color='green')
    ), row=2, col=2)

    fig.update_xaxes(title_text="Number of Clusters (K)")
    fig.update_layout(
        template="presentation",
        title_text="Model Selection: Information Criteria Comparison",
        width=1000,
        height=800,
        showlegend=False
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return results, fig


# ==============================================================================
# PERFORMANCE: VECTORIZED E-STEP
# ==============================================================================

def step_e_vectorized(X: np.ndarray, pi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Vectorized E-step: 10-100x faster than loop-based version.

    Uses NumPy broadcasting to compute all responsibilities simultaneously.

    Mathematical equivalence:
        log p(x_n | z_k) = Σ_d [x_{nd} log p_{kd} + (1-x_{nd}) log(1-p_{kd})]

    Args:
        X: Binary data matrix of shape (N, D)
        pi: Mixture weights of shape (K,)
        p: Bernoulli parameters of shape (K, D)

    Returns:
        Responsibilities matrix of shape (N, K)
    """
    # Clip probabilities to avoid log(0)
    p_safe = np.clip(p, 1e-10, 1 - 1e-10)

    # Compute log-likelihoods in log-space for numerical stability
    # Shape: (N, K) = (N, D) @ (D, K)
    log_p = X @ np.log(p_safe).T + (1 - X) @ np.log(1 - p_safe).T

    # Add log mixture weights: log π_k
    log_p += np.log(pi)[np.newaxis, :]  # Broadcasting over N

    # Convert from log-space to probabilities with numerical stability
    # log-sum-exp trick: log Σ exp(x_i) = max(x) + log Σ exp(x_i - max(x))
    log_p_max = np.max(log_p, axis=1, keepdims=True)
    p_unnorm = np.exp(log_p - log_p_max)

    # Normalize to get responsibilities
    r = p_unnorm / np.sum(p_unnorm, axis=1, keepdims=True)

    return r


# ==============================================================================
# CLUSTER INTERPRETATION & QUALITY METRICS
# ==============================================================================

def interpret_clusters(
    p: np.ndarray,
    img_shape: Tuple[int, int] = (8, 8),
    top_k: int = 10
) -> dict:
    """
    Analyze cluster characteristics and interpret what each cluster represents.

    Computes:
    - Most/least active dimensions per cluster
    - Cluster sparsity (how many near-0/1 values)
    - Cluster entropy (uncertainty in parameters)

    Args:
        p: Bernoulli parameters of shape (K, D)
        img_shape: Shape for visualizing dimensions as images
        top_k: Number of top dimensions to report

    Returns:
        Dictionary with interpretation metrics per cluster
    """
    K, D = p.shape
    interpretation = {}

    for k in range(K):
        # Most/least active pixels
        sorted_idx = np.argsort(p[k, :])
        least_active = sorted_idx[:top_k]
        most_active = sorted_idx[-top_k:][::-1]

        # Sparsity: how many dimensions are near 0 or 1
        threshold = 0.1
        sparse_low = np.sum(p[k, :] < threshold)
        sparse_high = np.sum(p[k, :] > (1 - threshold))
        sparsity = (sparse_low + sparse_high) / D

        # Entropy: measure of uncertainty in parameters
        # H(p) = -p log p - (1-p) log(1-p)
        p_safe = np.clip(p[k, :], 1e-10, 1 - 1e-10)
        entropy = -p_safe * np.log(p_safe) - (1 - p_safe) * np.log(1 - p_safe)
        avg_entropy = np.mean(entropy)

        # Average activation
        mean_activation = np.mean(p[k, :])

        interpretation[f'cluster_{k+1}'] = {
            'most_active_dims': most_active.tolist(),
            'least_active_dims': least_active.tolist(),
            'mean_activation': float(mean_activation),
            'sparsity': float(sparsity),
            'avg_entropy': float(avg_entropy),
            'active_pixels': int(np.sum(p[k, :] > 0.5)),
            'inactive_pixels': int(np.sum(p[k, :] < 0.5))
        }

    return interpretation


def compute_cluster_quality(
    X: np.ndarray,
    r: np.ndarray,
    p: np.ndarray
) -> dict:
    """
    Compute cluster quality metrics.

    Metrics:
    - Cluster sizes (effective number of points per cluster)
    - Cluster separation (entropy of responsibility distribution)
    - Within-cluster variance

    Args:
        X: Binary data matrix (N, D)
        r: Responsibilities (N, K)
        p: Cluster centers (K, D)

    Returns:
        Dictionary with quality metrics
    """
    N, D = X.shape
    K = r.shape[1]

    # Effective cluster sizes
    cluster_sizes = np.sum(r, axis=0)

    # Responsibility entropy (how "hard" vs "soft" are assignments?)
    # High entropy = uncertain assignments, low entropy = confident
    r_safe = np.clip(r, 1e-10, 1 - 1e-10)
    assignment_entropy = -np.sum(r_safe * np.log(r_safe), axis=1)
    avg_assignment_entropy = np.mean(assignment_entropy)

    # Cluster separation: average distance between cluster centers
    cluster_distances = []
    for i in range(K):
        for j in range(i+1, K):
            # L2 distance between cluster centers
            dist = np.linalg.norm(p[i, :] - p[j, :])
            cluster_distances.append(dist)
    avg_cluster_separation = np.mean(cluster_distances) if cluster_distances else 0

    return {
        'cluster_sizes': cluster_sizes.tolist(),
        'avg_assignment_entropy': float(avg_assignment_entropy),
        'cluster_separation': float(avg_cluster_separation),
        'min_cluster_size': float(np.min(cluster_sizes)),
        'max_cluster_size': float(np.max(cluster_sizes)),
        'size_imbalance': float(np.std(cluster_sizes) / np.mean(cluster_sizes))
    }
