"""
Bayesian Parameter Estimation for Multivariate Bernoulli Distributions

This module implements Maximum Likelihood (ML) and Maximum A Posteriori (MAP)
estimation for multivariate Bernoulli distributions, commonly used in binary
image classification and clustering tasks.

Author: Efstathios Siatras
"""

import os

import numpy as np
from typing import Tuple
import plotly.graph_objects as go


def _ensure_save_dir(save_path: str):
    """Create parent directories for save_path if they don't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)


def compute_ml(X: np.ndarray) -> np.ndarray:
    """
    Compute Maximum Likelihood (ML) estimate for Bernoulli parameters.

    The ML estimate for a Bernoulli parameter is simply the empirical mean:
        μ_ML = (1/N) Σ x_n

    Args:
        X: Binary data matrix of shape (N, D) where N is number of samples
           and D is number of dimensions (features)

    Returns:
        ML estimate vector of shape (D,) containing probability estimates
        for each dimension

    Example:
        >>> X = np.array([[1, 0], [1, 1], [0, 1]])
        >>> ml_params = compute_ml(X)
        >>> print(ml_params)  # [0.667, 0.667]
    """
    # N samples
    num_samples = X.shape[0]

    # Compute the sum of each column (dimension)
    sum_X_col = np.sum(X, axis=0)

    # Compute the ML estimate
    ml_estimate = sum_X_col / num_samples

    return ml_estimate


def compute_map(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Compute Maximum A Posteriori (MAP) estimate for Bernoulli parameters.

    Uses a Beta(α, β) prior over each Bernoulli parameter. The MAP estimate is:
        μ_MAP = (Σ x_n + α - 1) / (N + α + β - 2)

    Args:
        X: Binary data matrix of shape (N, D)
        alpha: Beta prior shape parameter α (α > 1 shrinks estimates toward prior)
        beta: Beta prior shape parameter β (β > 1 shrinks estimates toward prior)

    Returns:
        MAP estimate vector of shape (D,)

    Notes:
        - α = β = 1 gives uniform prior (equivalent to ML)
        - α = β = 3 gives preference for μ ≈ 0.5 (regularization)
        - Larger α, β values provide stronger regularization

    Example:
        >>> X = np.array([[1, 0], [1, 1]])
        >>> map_params = compute_map(X, alpha=3, beta=3)
    """
    # N samples
    num_samples = X.shape[0]

    # Compute the sum of each column
    sum_X_col = np.sum(X, axis=0)

    # Compute the MAP estimate with Beta prior
    map_estimate = (sum_X_col + alpha - 1) / (alpha + beta + num_samples - 2)

    return map_estimate


def visualize_parameters(
    params: np.ndarray,
    title: str = "Parameter Visualization",
    img_shape: Tuple[int, int] = (8, 8),
    save_path: str = None,
    show_stats: bool = True
) -> go.Figure:
    """
    Visualize learned parameters as a 2D image (heatmap).

    Useful for visualizing Bernoulli parameters learned from binary images
    (e.g., MNIST digits, binary patterns).

    Args:
        params: Parameter vector of shape (D,) where D = height × width
        title: Plot title
        img_shape: Tuple (height, width) for reshaping parameters
        save_path: Optional path to save the figure
        show_stats: Whether to show statistics annotation

    Returns:
        Plotly Figure object

    Example:
        >>> params = compute_ml(X)
        >>> fig = visualize_parameters(params, title="ML Estimate")
        >>> fig.show()
    """
    # Reshape parameters to 2D image and flip for correct orientation
    params_reshaped = np.flipud(np.reshape(params, img_shape))

    # Compute statistics
    p_mean = np.mean(params)
    p_std = np.std(params)
    p_min = np.min(params)
    p_max = np.max(params)

    # Create heatmap with publication-grade styling
    fig = go.Figure(data=go.Heatmap(
        z=params_reshaped,
        colorscale='Greys',
        reversescale=True,
        colorbar=dict(
            title=dict(text="P(pixel=1)", side="right"),
            tickformat=".2f"
        ),
        hovertemplate='Row: %{y}<br>Col: %{x}<br>P: %{z:.3f}<extra></extra>'
    ))

    # Add statistics annotation
    if show_stats:
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Statistics</b><br>μ = {p_mean:.3f}<br>σ = {p_std:.3f}<br>Range: [{p_min:.2f}, {p_max:.2f}]",
            showarrow=False,
            font=dict(size=10, color="#333"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#2E86AB",
            borderwidth=1,
            borderpad=4,
            align="left"
        )

    fig.update_layout(
        template="presentation",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16)
        ),
        xaxis_title="Column",
        yaxis_title="Row",
        width=550,
        height=550,
        autosize=False
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def compare_ml_map_sidebyside(
    X: np.ndarray,
    alpha: float = 3.0,
    beta: float = 3.0,
    img_shape: Tuple[int, int] = (8, 8),
    save_path: str = None
) -> go.Figure:
    """
    Create a publication-grade side-by-side comparison of ML and MAP estimates.

    Args:
        X: Binary data matrix of shape (N, D)
        alpha: Beta prior α parameter
        beta: Beta prior β parameter
        img_shape: Tuple (height, width) for reshaping parameters
        save_path: Optional path to save the figure

    Returns:
        Plotly Figure with side-by-side ML and MAP heatmaps
    """
    import plotly.subplots as sp

    # Compute estimates
    ml_est = compute_ml(X)
    map_est = compute_map(X, alpha, beta)

    # Reshape for visualization
    ml_reshaped = np.flipud(np.reshape(ml_est, img_shape))
    map_reshaped = np.flipud(np.reshape(map_est, img_shape))

    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"<b>ML Estimate</b><br>μ̂ = (1/N)Σxₙ",
            f"<b>MAP Estimate</b><br>α={alpha}, β={beta}"
        ],
        horizontal_spacing=0.12
    )

    # ML heatmap
    fig.add_trace(
        go.Heatmap(
            z=ml_reshaped,
            colorscale='Greys',
            reversescale=True,
            showscale=False,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>P: %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # MAP heatmap
    fig.add_trace(
        go.Heatmap(
            z=map_reshaped,
            colorscale='Greys',
            reversescale=True,
            colorbar=dict(
                title=dict(text="P(pixel=1)", side="right"),
                tickformat=".2f",
                len=0.9
            ),
            hovertemplate='Row: %{y}<br>Col: %{x}<br>P: %{z:.3f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(title_text="Column", row=1, col=col)
        fig.update_yaxes(title_text="Row" if col == 1 else "", row=1, col=col)

    fig.update_layout(
        template="presentation",
        title=dict(
            text="<b>Bayesian Parameter Estimation: ML vs MAP</b>",
            font=dict(size=18),
            x=0.5
        ),
        width=1000,
        height=500,
        margin=dict(b=80)
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def compare_ml_map(
    X: np.ndarray,
    alpha: float = 3.0,
    beta: float = 3.0,
    img_shape: Tuple[int, int] = (8, 8),
    save_dir: str = "results/figures"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare ML and MAP estimates side-by-side.

    Args:
        X: Binary data matrix
        alpha: Beta prior α parameter
        beta: Beta prior β parameter
        img_shape: Shape for visualization
        save_dir: Directory to save figures

    Returns:
        Tuple of (ml_estimate, map_estimate)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Compute estimates
    ml_est = compute_ml(X)
    map_est = compute_map(X, alpha, beta)

    # Visualize and save
    fig_ml = visualize_parameters(
        ml_est,
        title="Maximum Likelihood Estimate",
        img_shape=img_shape,
        save_path=f"{save_dir}/ml_estimate.png"
    )

    fig_map = visualize_parameters(
        map_est,
        title=f"MAP Estimate (α={alpha}, β={beta})",
        img_shape=img_shape,
        save_path=f"{save_dir}/map_estimate.png"
    )

    # Print comparison statistics
    print(f"ML Estimate Statistics:")
    print(f"  Mean: {np.mean(ml_est):.4f}")
    print(f"  Std:  {np.std(ml_est):.4f}")
    print(f"  Min:  {np.min(ml_est):.4f}")
    print(f"  Max:  {np.max(ml_est):.4f}")
    print()
    print(f"MAP Estimate Statistics (α={alpha}, β={beta}):")
    print(f"  Mean: {np.mean(map_est):.4f}")
    print(f"  Std:  {np.std(map_est):.4f}")
    print(f"  Min:  {np.min(map_est):.4f}")
    print(f"  Max:  {np.max(map_est):.4f}")
    print()
    print(f"Difference (|ML - MAP|):")
    print(f"  Mean: {np.mean(np.abs(ml_est - map_est)):.4f}")
    print(f"  Max:  {np.max(np.abs(ml_est - map_est)):.4f}")

    return ml_est, map_est
