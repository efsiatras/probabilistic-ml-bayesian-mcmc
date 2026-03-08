"""
MCMC-Based Substitution Cipher Decryption

Implements Markov Chain Monte Carlo (MCMC) methods using Metropolis-Hastings
algorithm to decrypt substitution ciphers. Uses bigram language models learned
from large text corpora (e.g., War and Peace) to guide the search through the
space of permutations.

Key Idea:
    - Model English text as a Markov chain with bigram transitions
    - Use MCMC to sample permutations (decryption keys) from posterior
    - Proposal: randomly swap two symbols in the key
    - Acceptance: Metropolis-Hastings with likelihood ratio

Author: Efstathios Siatras
"""

import os
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import random
from typing import List, Tuple, Dict
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.figure_factory import create_annotated_heatmap


def _ensure_save_dir(save_path: str):
    """Create parent directories for save_path if they don't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)


# ==============================================================================
# VECTORIZED HELPERS (used internally for fast MCMC/SA inner loops)
# ==============================================================================

def _precompute_cipher_indices(ciphertext: str, symbols: List[str]) -> np.ndarray:
    """Convert ciphertext to integer indices (precompute once per problem)."""
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    return np.array([symbol_to_idx.get(c, -1) for c in ciphertext], dtype=np.int32)


def _compute_log_likelihood_fast(
    cipher_indices: np.ndarray,
    key_perm: np.ndarray,
    log_transition_matrix: np.ndarray
) -> float:
    """Vectorized log-likelihood using precomputed indices.

    Args:
        cipher_indices: Precomputed cipher character indices (from _precompute_cipher_indices)
        key_perm: Key as integer permutation array (key_perm[cipher_idx] = plaintext_idx)
        log_transition_matrix: Precomputed log of transition matrix

    Returns:
        Log-likelihood (scalar)
    """
    valid = cipher_indices >= 0
    plaintext_indices = np.where(valid, key_perm[cipher_indices], -1)

    both_valid = valid[:-1] & valid[1:]
    prev_idx = plaintext_indices[:-1][both_valid]
    curr_idx = plaintext_indices[1:][both_valid]

    return float(np.sum(log_transition_matrix[prev_idx, curr_idx]) - 100 * np.sum(~both_valid))


def _key_str_to_perm(key: str, symbols: List[str]) -> np.ndarray:
    """Convert string key to integer permutation array."""
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    return np.array([symbol_to_idx[c] for c in key], dtype=np.int32)


def _perm_to_key_str(perm: np.ndarray, symbols: List[str]) -> str:
    """Convert integer permutation array back to string key."""
    return ''.join(symbols[i] for i in perm)


def preprocess_symbols(
    symbols_text: str,
    symbols_to_remove: List[str] = None
) -> List[str]:
    """
    Preprocess symbol list by removing unwanted characters.

    Args:
        symbols_text: Raw symbols as newline-separated string
        symbols_to_remove: List of symbols to exclude

    Returns:
        Filtered list of valid symbols
    """
    if symbols_to_remove is None:
        symbols_to_remove = ['=', '*', '']

    symbols_list = symbols_text.split('\n')
    filtered_symbols = [
        s for s in symbols_list
        if s not in symbols_to_remove and s != ''
    ]

    return filtered_symbols


def preprocess_text(text: str, remove_digits: bool = False) -> str:
    """
    Preprocess training corpus text.

    Args:
        text: Raw text corpus
        remove_digits: If True, remove all digit characters (recommended
                      for training to avoid noise from page numbers, etc.)

    Returns:
        Cleaned, lowercase text with normalized whitespace
    """
    # Lowercase
    clean_text = text.lower()

    # Remove newlines
    clean_text = clean_text.replace('\n', ' ')

    # Optionally remove digits (for training corpus only)
    if remove_digits:
        clean_text = ''.join(c for c in clean_text if not c.isdigit())

    # Normalize whitespace
    clean_text = ' '.join(clean_text.split())

    return clean_text


def compute_transition_matrix(
    text: str,
    symbols: List[str],
    pseudocount: float = 1e-10
) -> np.ndarray:
    """
    Compute bigram transition probability matrix from training text.

    Builds matrix where entry [i,j] = P(symbol_j | symbol_i), i.e.,
    the probability of observing symbol_j after symbol_i.

    Args:
        text: Training corpus (preprocessed)
        symbols: List of valid symbols
        pseudocount: Small value to avoid log(0) errors

    Returns:
        Transition matrix of shape (num_symbols, num_symbols) where
        each row sums to 1
    """
    num_symbols = len(symbols)
    transition_count = np.zeros((num_symbols, num_symbols))

    # Count bigram transitions
    for i in range(1, len(text)):
        prev_char = text[i - 1]
        curr_char = text[i]

        if prev_char in symbols and curr_char in symbols:
            prev_idx = symbols.index(prev_char)
            curr_idx = symbols.index(curr_char)
            # Store as [prev, curr] for P(curr | prev)
            transition_count[prev_idx, curr_idx] += 1.0

    # Add pseudocount
    transition_count += pseudocount

    # Normalize rows to get probabilities
    for row_idx in range(num_symbols):
        row_sum = transition_count[row_idx].sum()
        if row_sum > 0:
            transition_count[row_idx] /= row_sum

    return transition_count


def compute_unigram_frequencies(
    text: str,
    symbols: List[str]
) -> np.ndarray:
    """
    Compute stationary distribution (unigram probabilities).

    Args:
        text: Training corpus
        symbols: List of valid symbols

    Returns:
        Probability distribution over symbols
    """
    num_symbols = len(symbols)
    symbol_counts = np.zeros(num_symbols)

    for char in text:
        if char in symbols:
            idx = symbols.index(char)
            symbol_counts[idx] += 1

    total = symbol_counts.sum()
    stationary_dist = symbol_counts / total if total > 0 else np.ones(num_symbols) / num_symbols

    return stationary_dist


def perform_swap(key: str) -> str:
    """
    Propose new key by randomly swapping two positions.

    This is the proposal distribution for Metropolis-Hastings.
    Since we uniformly sample two positions to swap, the proposal
    is symmetric: Q(key' | key) = Q(key | key').

    Args:
        key: Current decryption key (string of symbols)

    Returns:
        Proposed key with two positions swapped
    """
    key_list = list(key)
    i, j = random.sample(range(len(key_list)), 2)
    key_list[i], key_list[j] = key_list[j], key_list[i]
    return ''.join(key_list)


def perform_decryption(
    ciphertext: str,
    symbols: List[str],
    key: str
) -> str:
    """
    Decrypt ciphertext using the given key.

    Args:
        ciphertext: Encrypted message
        symbols: List of symbols (cipher alphabet)
        key: Decryption key where key[i] is the plaintext symbol
             for cipher symbol symbols[i]

    Returns:
        Decrypted plaintext
    """
    decrypted_chars = []

    for char in ciphertext:
        if char in symbols:
            symbol_idx = symbols.index(char)
            plaintext_char = key[symbol_idx]
            decrypted_chars.append(plaintext_char)
        else:
            decrypted_chars.append(char)

    return ''.join(decrypted_chars)


def compute_log_likelihood(
    ciphertext: str,
    symbols: List[str],
    key: str,
    transition_matrix: np.ndarray
) -> float:
    """
    Compute log-likelihood of decrypted text under bigram model.

    Uses the transition matrix to compute:
        log P(decrypted text) = Σ log P(s_i | s_{i-1})

    Args:
        ciphertext: Encrypted message
        symbols: Cipher alphabet
        key: Decryption key
        transition_matrix: Bigram transition probabilities

    Returns:
        Log-likelihood (scalar)
    """
    decrypted_text = perform_decryption(ciphertext, symbols, key)
    log_likelihood = 0.0

    for i in range(1, len(decrypted_text)):
        prev_char = decrypted_text[i - 1]
        curr_char = decrypted_text[i]

        if prev_char in symbols and curr_char in symbols:
            prev_idx = symbols.index(prev_char)
            curr_idx = symbols.index(curr_char)
            prob = transition_matrix[prev_idx, curr_idx]
            log_likelihood += np.log(prob)
        else:
            # Penalty for unknown characters
            log_likelihood -= 100

    return log_likelihood


def mcmc_decrypt(
    ciphertext: str,
    symbols: List[str],
    transition_matrix: np.ndarray,
    n_iterations: int = 100000,
    early_stop_patience: int = 10000,
    verbose: bool = True
) -> Tuple[str, List[Tuple[int, float]]]:
    """
    Decrypt ciphertext using MCMC with Metropolis-Hastings.

    Uses vectorized log-likelihood computation for performance.
    Proposal: randomly swap two symbols in the permutation key.
    Acceptance: Metropolis-Hastings with likelihood ratio (symmetric proposal).

    Args:
        ciphertext: Encrypted message to decrypt
        symbols: Cipher alphabet (list of symbols)
        transition_matrix: Bigram transition probabilities
        n_iterations: Maximum number of MCMC iterations
        early_stop_patience: Stop if no improvement for this many iterations
        verbose: Print progress information

    Returns:
        Tuple of (best_key, convergence_history) where:
            best_key: Best decryption key found (string)
            convergence_history: List of (iteration, log_likelihood) tuples
    """
    n_symbols = len(symbols)

    # Precompute for vectorized inner loop
    cipher_indices = _precompute_cipher_indices(ciphertext, symbols)
    log_trans = np.log(transition_matrix)

    # Initialize key as permutation array
    perm = np.arange(n_symbols, dtype=np.int32)
    np.random.shuffle(perm)

    best_perm = perm.copy()
    best_likelihood = float('-inf')
    iterations_without_improvement = 0
    convergence_history = []

    if verbose:
        print(f"Starting MCMC with max {n_iterations} iterations")
        print(f"Early stopping: patience = {early_stop_patience} iterations")
        print("=" * 80)
        print()

    # Compute initial log-likelihood
    current_ll = _compute_log_likelihood_fast(cipher_indices, perm, log_trans)

    for iteration in range(n_iterations):
        # Propose new key by swapping two symbols (in-place, then revert if rejected)
        i, j = random.sample(range(n_symbols), 2)
        perm[i], perm[j] = perm[j], perm[i]

        # Compute proposed log-likelihood (vectorized)
        proposed_ll = _compute_log_likelihood_fast(cipher_indices, perm, log_trans)

        # Metropolis-Hastings acceptance (symmetric proposal)
        if math.log(random.random()) < proposed_ll - current_ll:
            current_ll = proposed_ll

            if proposed_ll > best_likelihood:
                best_likelihood = proposed_ll
                best_perm = perm.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        else:
            # Reject: revert swap
            perm[i], perm[j] = perm[j], perm[i]
            iterations_without_improvement += 1

        # Record convergence history (every 100 iterations)
        if iteration % 100 == 0:
            convergence_history.append((iteration, best_likelihood))

        # Print progress every 1000 iterations
        if verbose and iteration % 1000 == 0:
            best_key_str = _perm_to_key_str(best_perm, symbols)
            decrypted_preview = perform_decryption(ciphertext, symbols, best_key_str)[:80]
            print(f'Iteration {iteration:6d} | Best LL: {best_likelihood:8.2f} | '
                  f'No improvement: {iterations_without_improvement:5d}')
            print(f'  Preview: {decrypted_preview}')
            print()

        # Early stopping check
        if iterations_without_improvement >= early_stop_patience:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Early stopping at iteration {iteration}")
                print(f"No improvement for {early_stop_patience} iterations")
                print(f"Best log-likelihood: {best_likelihood:.2f}")
                print(f"{'='*80}\n")
            break

    # Add final point
    convergence_history.append((iteration, best_likelihood))

    # Convert best permutation back to string key
    best_key = _perm_to_key_str(best_perm, symbols)
    return best_key, convergence_history


def visualize_transition_matrix(
    transition_matrix: np.ndarray,
    symbols: List[str],
    save_path: str = None
) -> go.Figure:
    """
    Visualize bigram transition matrix as annotated heatmap.

    Args:
        transition_matrix: Transition probability matrix
        symbols: List of symbols
        save_path: Optional path to save figure

    Returns:
        Plotly figure
    """
    # Round for display
    matrix_rounded = np.round(transition_matrix, 3)

    fig = create_annotated_heatmap(
        transition_matrix,
        x=symbols,
        y=symbols,
        annotation_text=matrix_rounded,
        colorscale='Blues'
    )

    fig.update_layout(
        title="Bigram Transition Probabilities P(curr | prev)",
        xaxis_title="Current Symbol",
        yaxis_title="Previous Symbol",
        width=1200,
        height=1200,
        font=dict(size=10)
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def plot_convergence(
    convergence_history: List[Tuple[int, float]],
    save_path: str = None
) -> go.Figure:
    """
    Plot MCMC convergence curve.

    Args:
        convergence_history: List of (iteration, log_likelihood) tuples
        save_path: Optional path to save figure

    Returns:
        Plotly figure
    """
    iterations, log_likelihoods = zip(*convergence_history)
    iterations = list(iterations)
    log_likelihoods = list(log_likelihoods)

    # Trim early burn-in where LL is extremely negative (compresses useful range)
    best_ll_val = max(log_likelihoods)
    trim_threshold = 1.5 * abs(best_ll_val)
    trim_idx = 0
    for k, ll in enumerate(log_likelihoods):
        if abs(ll) <= trim_threshold:
            trim_idx = k
            break
    if trim_idx > 0:
        iterations = iterations[trim_idx:]
        log_likelihoods = log_likelihoods[trim_idx:]

    fig = go.Figure()

    # Main convergence line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=log_likelihoods,
        mode='lines',
        name='Best Log-Likelihood',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='Iteration: %{x}<br>Log-Likelihood: %{y:.2f}<extra></extra>'
    ))

    # Find and annotate best solution
    best_ll = max(log_likelihoods)
    best_idx = log_likelihoods.index(best_ll)

    fig.add_annotation(
        x=iterations[best_idx],
        y=best_ll,
        text=f"Best: {best_ll:.1f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2E86AB",
        font=dict(size=11, color="#2E86AB"),
        ax=-40,
        ay=-30
    )

    fig.update_layout(
        template="presentation",
        title=dict(
            text="MCMC Convergence: Substitution Cipher Decryption",
            font=dict(size=18)
        ),
        xaxis_title="Iteration",
        yaxis_title="Log-Likelihood",
        width=1000,
        height=600,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode="x unified"
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def _run_single_mcmc_restart(args):
    """Worker function for parallel MCMC restarts (must be top-level for pickling)."""
    ciphertext, symbols, transition_matrix, n_iterations, early_stop_patience, restart_id, seed = args
    random.seed(seed)
    np.random.seed(seed)

    best_key, history = mcmc_decrypt(
        ciphertext, symbols, transition_matrix,
        n_iterations=n_iterations,
        early_stop_patience=early_stop_patience,
        verbose=False
    )

    final_ll = history[-1][1]
    decrypted = perform_decryption(ciphertext, symbols, best_key)
    common_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'are', 'was', 'all', 'can']
    words_found = sum(1 for w in common_words if f' {w} ' in decrypted.lower())

    return {
        'restart_id': restart_id, 'key': best_key, 'history': history,
        'final_ll': final_ll, 'words_found': words_found, 'preview': decrypted[:60],
        'full_text': decrypted
    }


def mcmc_decrypt_with_restarts(
    ciphertext: str,
    symbols: List[str],
    transition_matrix: np.ndarray,
    n_restarts: int = 10,
    n_iterations: int = 50000,
    early_stop_patience: int = 5000,
    verbose: bool = True,
    n_workers: int = None
) -> Tuple[str, List[Tuple[int, float]], dict]:
    """
    Run MCMC decryption with multiple random restarts in parallel.

    Uses ProcessPoolExecutor to run independent chains across CPU cores.

    Args:
        ciphertext: Encrypted message to decrypt
        symbols: Cipher alphabet
        transition_matrix: Bigram transition probabilities
        n_restarts: Number of independent runs
        n_iterations: Max iterations per run
        early_stop_patience: Early stopping patience
        verbose: Print progress
        n_workers: Number of parallel workers (default: cpu_count - 1)

    Returns:
        Tuple of (best_key, best_history, summary_dict)
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if verbose:
        print(f"Running MCMC with {n_restarts} random restarts ({n_workers} parallel workers)")
        print(f"Iterations per restart: {n_iterations}, Early stop patience: {early_stop_patience}")
        print("=" * 80)

    # Prepare args for each restart
    args_list = [
        (ciphertext, symbols, transition_matrix, n_iterations,
         early_stop_patience, rid, rid * 12345 + 42)
        for rid in range(n_restarts)
    ]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_results = list(executor.map(_run_single_mcmc_restart, args_list))

    # Print results
    if verbose:
        for r in all_results:
            print(f"\nRestart {r['restart_id'] + 1}/{n_restarts}: "
                  f"LL = {r['final_ll']:.2f}, Words = {r['words_found']}")
            print(f"   Text: {r['full_text']}")

    # Sort by likelihood (higher is better)
    all_results.sort(key=lambda x: x['final_ll'], reverse=True)
    best_result = all_results[0]

    best_decrypted = perform_decryption(ciphertext, symbols, best_result['key'])

    if verbose:
        print("\n" + "=" * 80)
        print(f"BEST RESULT (Restart {best_result['restart_id'] + 1}):")
        print(f"  Log-Likelihood: {best_result['final_ll']:.2f}")
        print(f"  Common words found: {best_result['words_found']}/14")
        print(f"  Decrypted text:")
        print(f"  {best_decrypted}")
        print("=" * 80)

        print("\nTop 3 restarts:")
        for i, r in enumerate(all_results[:3]):
            print(f"  {i+1}. Restart {r['restart_id']+1}: LL = {r['final_ll']:.2f}, Words = {r['words_found']}")

    all_lls = [r['final_ll'] for r in all_results]
    summary = {
        'all_results': [(r['restart_id'], r['final_ll'], r['words_found']) for r in all_results],
        'best_ll': best_result['final_ll'],
        'worst_ll': all_results[-1]['final_ll'],
        'll_range': best_result['final_ll'] - all_results[-1]['final_ll'],
        'all_lls': all_lls,
        'median_ll': float(np.median(all_lls))
    }

    return best_result['key'], best_result['history'], summary


def save_results(
    ciphertext: str,
    symbols: List[str],
    best_key: str,
    convergence_history: List[Tuple[int, float]],
    output_dir: str = "results"
):
    """
    Save decryption results to files.

    Args:
        ciphertext: Original encrypted message
        symbols: Cipher alphabet
        best_key: Best decryption key found
        convergence_history: MCMC convergence history
        output_dir: Directory to save results
    """
    import os

    # Create directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    # Decrypt message
    decrypted_text = perform_decryption(ciphertext, symbols, best_key)

    # Save decrypted message
    with open(f"{output_dir}/logs/decrypted_message.txt", "w") as f:
        f.write("MCMC-Based Substitution Cipher Decryption\n")
        f.write("=" * 80 + "\n\n")
        f.write(decrypted_text)
        f.write("\n\n" + "=" * 80 + "\n")

    # Save cipher mapping
    with open(f"{output_dir}/logs/cipher_mapping.txt", "w") as f:
        f.write("Learned Cipher Mapping (Cipher → Plaintext)\n")
        f.write("=" * 80 + "\n\n")
        for i, cipher_symbol in enumerate(symbols):
            plaintext_symbol = best_key[i]
            f.write(f"  '{cipher_symbol}' → '{plaintext_symbol}'\n")
        f.write("\n" + "=" * 80 + "\n")

    # Plot and save convergence
    fig = plot_convergence(
        convergence_history,
        save_path=f"{output_dir}/figures/mcmc_convergence.png"
    )

    print(f"\n✓ Results saved to {output_dir}/")


# ==============================================================================
# CONVERGENCE DIAGNOSTICS
# ==============================================================================

def compute_gelman_rubin(chains: List[List[float]], warmup_frac: float = 0.5) -> float:
    """
    Compute Gelman-Rubin diagnostic (R-hat) for MCMC convergence.

    R-hat ≈ 1.0 indicates convergence. Values > 1.1 suggest non-convergence.

    The diagnostic compares within-chain variance to between-chain variance.
    If chains have converged to the same distribution, these should be similar.

    Args:
        chains: List of log-likelihood chains from multiple MCMC runs
        warmup_frac: Fraction of chain to discard as burn-in (default: 0.5)

    Returns:
        R-hat statistic (should be close to 1.0 for convergence)
    """
    import numpy as np

    # Convert to numpy and discard warmup
    chains_array = [np.array(chain) for chain in chains]
    n_chains = len(chains_array)

    # Discard warmup period
    chains_trimmed = [chain[int(len(chain) * warmup_frac):] for chain in chains_array]
    n_iterations = len(chains_trimmed[0])

    # Compute chain means and overall mean
    chain_means = np.array([np.mean(chain) for chain in chains_trimmed])
    overall_mean = np.mean(chain_means)

    # Between-chain variance (B)
    B = n_iterations * np.var(chain_means, ddof=1)

    # Within-chain variance (W)
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains_trimmed])
    W = np.mean(chain_vars)

    # Marginal posterior variance estimate
    var_plus = ((n_iterations - 1) / n_iterations) * W + (1 / n_iterations) * B

    # Potential scale reduction factor (R-hat)
    R_hat = np.sqrt(var_plus / W)

    return float(R_hat)


def run_multiple_chains(
    text: str,
    symbols: List[str],
    transition_matrix: np.ndarray,
    n_chains: int = 4,
    n_iterations: int = 50000,
    early_stop_patience: int = 5000,
    verbose: bool = False,
    n_workers: int = None
) -> Tuple[List[str], List[List[Tuple[int, float]]]]:
    """
    Run multiple MCMC chains in parallel for convergence diagnostics.

    Args:
        text: Encrypted message
        symbols: Cipher alphabet
        transition_matrix: Bigram transition probabilities
        n_chains: Number of parallel chains to run
        n_iterations: Max iterations per chain
        early_stop_patience: Early stopping patience
        verbose: Print progress
        n_workers: Number of parallel workers (default: cpu_count - 1)

    Returns:
        best_keys: Best decryption key from each chain
        all_histories: Convergence history for each chain
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    print(f"Running {n_chains} MCMC chains in parallel ({n_workers} workers) for Gelman-Rubin diagnostic")
    print("=" * 70)

    args_list = [
        (text, symbols, transition_matrix, n_iterations,
         early_stop_patience, chain_id, chain_id * 1000)
        for chain_id in range(n_chains)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_run_single_mcmc_restart, args_list))

    # Sort by restart_id to preserve chain ordering
    results.sort(key=lambda x: x['restart_id'])

    best_keys = [r['key'] for r in results]
    all_histories = [r['history'] for r in results]

    for r in results:
        ll_values = [ll for _, ll in r['history']]
        print(f"\n  Chain {r['restart_id'] + 1}/{n_chains}: Final LL = {ll_values[-1]:.2f}")

    return best_keys, all_histories


def plot_gelman_rubin_evolution(
    all_histories: List[List[Tuple[int, float]]],
    window_size: int = 100,
    save_path: str = None
) -> go.Figure:
    """
    Plot how Gelman-Rubin R-hat evolves over iterations.

    Shows convergence of multiple chains over time.

    Args:
        all_histories: List of convergence histories from multiple chains
        window_size: Window for computing R-hat at each point
        save_path: Optional path to save figure

    Returns:
        Plotly figure
    """
    # Extract log-likelihoods
    all_lls = [[ll for _, ll in history] for history in all_histories]

    # Find minimum length across all chains
    min_len = min(len(lls) for lls in all_lls)

    # Compute R-hat at regular intervals
    r_hat_evolution = []
    iterations = []

    for i in range(window_size, min_len, 10):  # Every 10 iterations
        # Get window of values from all chains
        window_chains = [lls[:i] for lls in all_lls]
        r_hat = compute_gelman_rubin(window_chains, warmup_frac=0.2)
        r_hat_evolution.append(r_hat)
        iterations.append(i)

    # Create plot
    fig = go.Figure()

    # Compute y-axis range
    y_max = max(1.5, max(r_hat_evolution) * 1.1) if r_hat_evolution else 2.0

    # Add convergence region shading (R-hat < 1.1 = converged)
    fig.add_shape(
        type="rect",
        x0=iterations[0] if iterations else 0,
        x1=iterations[-1] if iterations else 1000,
        y0=0.95,
        y1=1.1,
        fillcolor="rgba(39, 174, 96, 0.15)",
        line_width=0,
        layer="below"
    )

    # Add non-convergence region shading
    fig.add_shape(
        type="rect",
        x0=iterations[0] if iterations else 0,
        x1=iterations[-1] if iterations else 1000,
        y0=1.1,
        y1=y_max,
        fillcolor="rgba(231, 76, 60, 0.1)",
        line_width=0,
        layer="below"
    )

    # R-hat evolution line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=r_hat_evolution,
        mode='lines',
        name='R̂ Statistic',
        line=dict(width=3, color='#2E86AB'),
        hovertemplate='Iteration: %{x}<br>R̂: %{y:.3f}<extra></extra>'
    ))

    # Add convergence threshold lines
    fig.add_hline(y=1.1, line_dash="dash", line_color="#E74C3C", line_width=2)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#27AE60", line_width=2)

    # Add region labels
    fig.add_annotation(
        x=iterations[-1] if iterations else 1000,
        y=1.03,
        text="<b>Converged (R̂ < 1.1)</b>",
        showarrow=False,
        font=dict(size=10, color="#27AE60"),
        xanchor="right"
    )

    # Final status annotation
    final_r_hat = r_hat_evolution[-1] if r_hat_evolution else float('inf')
    status = "✓ CONVERGED" if final_r_hat < 1.1 else "✗ NOT CONVERGED"
    status_color = "#27AE60" if final_r_hat < 1.1 else "#E74C3C"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Final Status</b><br>R̂ = {final_r_hat:.3f}<br>{status}",
        showarrow=False,
        font=dict(size=11, color=status_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=status_color,
        borderwidth=2,
        borderpad=6,
        align="left"
    )

    fig.update_layout(
        template="presentation",
        title=dict(
            text="Gelman-Rubin Diagnostic: Multi-Chain Convergence",
            font=dict(size=18)
        ),
        xaxis_title="Iteration",
        yaxis_title="R̂ Statistic",
        width=1000,
        height=600,
        yaxis_range=[0.95, y_max],
        showlegend=False
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig


def track_acceptance_rate(
    accepted_proposals: List[bool],
    window_size: int = 100
) -> List[float]:
    """
    Compute rolling acceptance rate over MCMC iterations.

    Target acceptance rate for random walk Metropolis is ~23-44%.
    Too low = poor mixing, too high = random walk too small.

    Args:
        accepted_proposals: Boolean list of accepted/rejected proposals
        window_size: Window for computing rolling average

    Returns:
        List of acceptance rates
    """
    acceptance_rates = []

    for i in range(window_size, len(accepted_proposals)):
        window = accepted_proposals[i-window_size:i]
        rate = np.mean(window)
        acceptance_rates.append(rate)

    return acceptance_rates


# ==============================================================================
# SIMULATED ANNEALING
# ==============================================================================

def simulated_annealing_decrypt(
    text: str,
    symbols: List[str],
    transition_matrix: np.ndarray,
    n_iterations: int = 100000,
    T_initial: float = 10.0,
    T_final: float = 0.01,
    cooling_schedule: str = "exponential",
    verbose: bool = True
) -> Tuple[str, List[Tuple[int, float, float]]]:
    """
    Decrypt ciphertext using Simulated Annealing.

    Uses vectorized log-likelihood computation for performance.
    Temperature decreases over time: high T explores, low T exploits.

    Acceptance probability: min(1, exp((ΔLL)/T))

    Args:
        text: Encrypted message
        symbols: Cipher alphabet
        transition_matrix: Bigram transition probabilities
        n_iterations: Number of iterations
        T_initial: Starting temperature
        T_final: Final temperature
        cooling_schedule: "exponential", "linear", or "logarithmic"
        verbose: Print progress

    Returns:
        best_key: Best decryption key found
        history: List of (iteration, log_likelihood, temperature) tuples
    """
    n_symbols = len(symbols)

    # Precompute for vectorized inner loop
    cipher_indices = _precompute_cipher_indices(text, symbols)
    log_trans = np.log(transition_matrix)

    # Initialize key as permutation array
    perm = np.arange(n_symbols, dtype=np.int32)
    np.random.shuffle(perm)

    best_perm = perm.copy()
    current_likelihood = _compute_log_likelihood_fast(cipher_indices, perm, log_trans)
    best_likelihood = current_likelihood

    history = [(0, best_likelihood, T_initial)]

    if verbose:
        print(f"Simulated Annealing Decryption")
        print(f"Initial Temperature: {T_initial}, Final: {T_final}")
        print(f"Cooling Schedule: {cooling_schedule}")
        print("=" * 80)

    # Precompute cooling rate for exponential schedule
    if cooling_schedule == "exponential":
        exp_alpha = (T_final / T_initial) ** (1 / n_iterations)
    elif cooling_schedule == "linear":
        linear_slope = (T_initial - T_final) / n_iterations
    elif cooling_schedule != "logarithmic":
        raise ValueError(f"Unknown cooling schedule: {cooling_schedule}")

    for i in range(1, n_iterations + 1):
        # Compute temperature
        if cooling_schedule == "exponential":
            temperature = T_initial * (exp_alpha ** i)
        elif cooling_schedule == "linear":
            temperature = T_initial - linear_slope * i
        else:  # logarithmic
            temperature = T_initial / math.log(i + 2)

        # Propose swap (in-place, revert if rejected)
        si, sj = random.sample(range(n_symbols), 2)
        perm[si], perm[sj] = perm[sj], perm[si]

        proposed_likelihood = _compute_log_likelihood_fast(cipher_indices, perm, log_trans)
        delta_ll = proposed_likelihood - current_likelihood

        if delta_ll > 0 or random.random() < math.exp(delta_ll / temperature):
            current_likelihood = proposed_likelihood
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_perm = perm.copy()
        else:
            # Reject: revert swap
            perm[si], perm[sj] = perm[sj], perm[si]

        # Record history
        if i % 100 == 0:
            history.append((i, best_likelihood, temperature))

        # Print progress
        if verbose and i % 1000 == 0:
            best_key_str = _perm_to_key_str(best_perm, symbols)
            decrypted_preview = perform_decryption(text, symbols, best_key_str)[:80]
            print(f'Iteration {i:6d} | Best LL: {best_likelihood:8.2f} | T: {temperature:6.4f}')
            print(f'  Preview: {decrypted_preview}')
            print()

    if verbose:
        print("=" * 80)
        print(f"Final best log-likelihood: {best_likelihood:.2f}")
        print("=" * 80)

    best_key = _perm_to_key_str(best_perm, symbols)
    return best_key, history


def _run_single_sa_restart(args):
    """Worker function for parallel SA restarts (must be top-level for pickling)."""
    ciphertext, symbols, transition_matrix, n_iterations, T_initial, T_final, cooling_schedule, restart_id, seed = args
    random.seed(seed)
    np.random.seed(seed)

    best_key, history = simulated_annealing_decrypt(
        ciphertext, symbols, transition_matrix,
        n_iterations=n_iterations,
        T_initial=T_initial, T_final=T_final,
        cooling_schedule=cooling_schedule,
        verbose=False
    )

    best_ll = max(ll for _, ll, _ in history)
    decrypted = perform_decryption(ciphertext, symbols, best_key)
    common_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'are', 'was', 'all', 'can']
    words_found = sum(1 for w in common_words if f' {w} ' in decrypted.lower())

    return {
        'restart_id': restart_id, 'key': best_key, 'history': history,
        'best_ll': best_ll, 'words_found': words_found, 'preview': decrypted[:60]
    }


def simulated_annealing_with_restarts(
    ciphertext: str,
    symbols: List[str],
    transition_matrix: np.ndarray,
    n_restarts: int = 5,
    n_iterations: int = 100000,
    T_initial: float = 20.0,
    T_final: float = 0.001,
    cooling_schedule: str = "exponential",
    verbose: bool = True,
    n_workers: int = None
) -> Tuple[str, List[Tuple[int, float, float]], dict]:
    """
    Run Simulated Annealing with multiple restarts in parallel.

    Args:
        ciphertext: Encrypted message
        symbols: Cipher alphabet
        transition_matrix: Bigram transition probabilities
        n_restarts: Number of independent SA runs
        n_iterations: Iterations per run
        T_initial: Starting temperature
        T_final: Final temperature
        cooling_schedule: Cooling schedule type
        verbose: Print progress
        n_workers: Number of parallel workers (default: cpu_count - 1)

    Returns:
        Tuple of (best_key, best_history, summary_dict)
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if verbose:
        print(f"Running Simulated Annealing with {n_restarts} restarts ({n_workers} parallel workers)")
        print(f"Iterations: {n_iterations}, T: {T_initial} -> {T_final}")
        print("=" * 80)

    args_list = [
        (ciphertext, symbols, transition_matrix, n_iterations,
         T_initial, T_final, cooling_schedule, rid, rid * 54321 + 99)
        for rid in range(n_restarts)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_results = list(executor.map(_run_single_sa_restart, args_list))

    if verbose:
        for r in all_results:
            print(f"\nSA Restart {r['restart_id'] + 1}/{n_restarts}: "
                  f"LL = {r['best_ll']:.2f}, Words = {r['words_found']}, Preview: {r['preview'][:40]}...")

    all_results.sort(key=lambda x: x['best_ll'], reverse=True)
    best_result = all_results[0]

    if verbose:
        print("\n" + "=" * 80)
        print(f"BEST SA RESULT (Restart {best_result['restart_id'] + 1}):")
        print(f"  Log-Likelihood: {best_result['best_ll']:.2f}")
        print(f"  Common words found: {best_result['words_found']}/14")
        print("=" * 80)

    all_lls = [r['best_ll'] for r in all_results]
    summary = {
        'all_results': [(r['restart_id'], r['best_ll'], r['words_found']) for r in all_results],
        'best_ll': best_result['best_ll'],
        'worst_ll': all_results[-1]['best_ll'],
        'all_lls': all_lls,
        'median_ll': float(np.median(all_lls))
    }

    return best_result['key'], best_result['history'], summary


def plot_simulated_annealing_convergence(
    history: List[Tuple[int, float, float]],
    save_path: str = None
) -> go.Figure:
    """
    Plot Simulated Annealing convergence with temperature schedule.

    Args:
        history: List of (iteration, log_likelihood, temperature) tuples
        save_path: Optional path to save figure

    Returns:
        Plotly figure
    """
    iterations, log_likelihoods, temperatures = zip(*history)
    iterations = list(iterations)
    log_likelihoods = list(log_likelihoods)
    temperatures = list(temperatures)

    # Trim early burn-in where LL is extremely negative (compresses useful range)
    best_ll_val = max(log_likelihoods)
    trim_threshold = 1.5 * abs(best_ll_val)
    trim_idx = 0
    for k, ll in enumerate(log_likelihoods):
        if abs(ll) <= trim_threshold:
            trim_idx = k
            break
    if trim_idx > 0:
        iterations = iterations[trim_idx:]
        log_likelihoods = log_likelihoods[trim_idx:]
        temperatures = temperatures[trim_idx:]

    # Create figure with subplots
    fig = sp.make_subplots(
        rows=2, cols=1,
        subplot_titles=('<b>Log-Likelihood Convergence</b>', '<b>Temperature Schedule</b>'),
        vertical_spacing=0.18
    )

    # Log-likelihood trace
    fig.add_trace(go.Scatter(
        x=iterations,
        y=log_likelihoods,
        mode='lines',
        name='Log-Likelihood',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='Iteration: %{x}<br>LL: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Find and annotate best solution
    best_ll = max(log_likelihoods)
    best_idx = log_likelihoods.index(best_ll)
    fig.add_annotation(
        x=iterations[best_idx],
        y=best_ll,
        text=f"Best: {best_ll:.1f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2E86AB",
        font=dict(size=10, color="#2E86AB"),
        ax=30,
        ay=-25,
        row=1, col=1
    )

    # Temperature trace
    fig.add_trace(go.Scatter(
        x=iterations,
        y=temperatures,
        mode='lines',
        name='Temperature',
        line=dict(color='#E74C3C', width=2),
        hovertemplate='Iteration: %{x}<br>T: %{y:.4f}<extra></extra>'
    ), row=2, col=1)

    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Log-Likelihood", row=1, col=1)
    fig.update_yaxes(title_text="Temperature", type="log", row=2, col=1)

    fig.update_layout(
        template="presentation",
        title=dict(
            text="Simulated Annealing Convergence",
            font=dict(size=18)
        ),
        width=1000,
        height=800,
        showlegend=False
    )

    if save_path:
        _ensure_save_dir(save_path)
        fig.write_image(save_path)

    return fig
