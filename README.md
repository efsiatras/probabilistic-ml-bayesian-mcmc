# Probabilistic Machine Learning: Bayesian Inference and MCMC Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

**Efstathios Siatras**<sup>†</sup>\
<sup>†</sup>Department of Computer Science, University College London (UCL)\
COMP0086 — Probabilistic and Unsupervised Learning

Implementation of foundational probabilistic machine learning methods, including Bayesian parameter estimation, model selection, the Expectation-Maximization algorithm, and MCMC-based cryptanalysis.

## Methods Implemented

### 1. Bayesian Parameter Estimation

Maximum Likelihood (ML) and Maximum A Posteriori (MAP) estimation for multivariate Bernoulli distributions with conjugate Beta priors.

$$\hat{p}\_d^{ML} = \frac{1}{N} \sum\_{n=1}^{N} x\_{nd}, \qquad \hat{p}\_d^{MAP} = \frac{\sum\_n x\_{nd} + \alpha - 1}{N + \alpha + \beta - 2}$$

### 2. Bayesian Model Selection

Model comparison using marginal likelihood (Bayesian evidence) with Beta-Bernoulli conjugacy. Demonstrates the automatic Occam's Razor effect — the most complex model (separate parameters per dimension) is selected with posterior probability ≈ 1.0.

$$p(\mathcal{D}|\mathcal{M}) = \frac{B(\alpha + n\_1,\; \beta + n\_0)}{B(\alpha,\; \beta)}$$

### 3. Expectation-Maximization Algorithm

EM for mixture of Bernoullis with BIC/AIC model selection and vectorized implementation.

$$r\_{nk} = \frac{\pi\_k\, p(x\_n | \theta\_k)}{\sum\_j \pi\_j\, p(x\_n | \theta\_j)}, \qquad \pi\_k^{\text{new}} = \frac{1}{N}\sum\_n r\_{nk}, \qquad p\_{kd}^{\text{new}} = \frac{\sum\_n r\_{nk}\, x\_{nd}}{\sum\_n r\_{nk}}$$

### 4. MCMC Cryptanalysis

Metropolis-Hastings sampling for substitution cipher decryption using bigram language models. Searches over 51! ≈ 10^66 possible keys.

$$\alpha = \min\!\left(1,\; e^{\mathcal{L}(\sigma') - \mathcal{L}(\sigma)}\right), \qquad \mathcal{L}(\sigma) = \sum\_{i=2}^{L} \log P\!\left(\sigma^{-1}(c\_i) \mid \sigma^{-1}(c\_{i-1})\right)$$

---

## Repository Structure

```
mcmc/
├── src/                           # Core implementations
│   ├── __init__.py
│   ├── bayesian_inference.py      # ML/MAP estimation
│   ├── model_selection.py         # Bayesian model comparison
│   ├── em_algorithm.py            # EM for mixture models
│   └── mcmc_cryptanalysis.py      # MCMC cipher decryption
│
├── notebooks/                     # Interactive demonstrations
│   ├── 01_bayesian_parameter_estimation.ipynb
│   ├── 02_bayesian_model_selection.ipynb
│   ├── 03_em_mixture_models.ipynb
│   └── 04_mcmc_cryptanalysis.ipynb
│
├── data/                          # Datasets
│   ├── binarydigits.txt           # 100 binary 8×8 images
│   ├── symbols.txt                # Cipher alphabet (51 symbols)
│   ├── message.txt                # Encrypted message
│   └── corpus/2600-0.txt          # War and Peace (training corpus)
│
└── results/                       # Generated outputs
    ├── figures/                   # Visualizations
    └── logs/                      # Text outputs
```

---

## Usage

### Run Notebooks

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Import as Library

```python
from src.bayesian_inference import compute_ml, compute_map
from src.model_selection import compare_models
from src.em_algorithm import em_algorithm
from src.mcmc_cryptanalysis import mcmc_decrypt
```

---

## Main Results

### Bayesian Model Selection

| Model | Description | Log Marginal Likelihood | Posterior |
|-------|-------------|------------------------|-----------|
| A | Fixed p = 0.5 | -4436 | ≈ 0 |
| B | Shared p | -4284 | ≈ 0 |
| C | Separate p per dimension | -3851 | ≈ 1.0 |

### EM Mixture Models

BIC selects K=4 clusters for the binary digit data, while AIC prefers higher K. Vectorized E-step achieves speedup via NumPy broadcasting.

### MCMC Decryption

The MCMC algorithm successfully decrypts a substitution cipher to reveal the opening of *The Great Gatsby*:

```
in my younger and more vulnerable years my father gave me some advice
that i've been turning over in my mind ever since...
```

With 10 random restarts (50,000 iterations each), MCMC reliably finds the correct decryption. Gelman-Rubin diagnostics (R-hat ≈ 5.7) reveal non-convergence consistent with multimodality: independent chains converge to different local optima.

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
3. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
4. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum likelihood from incomplete data via the EM algorithm." *JRSS-B*, 39(1), 1-38.
5. Chen, J., & Rosenthal, J. S. (2012). "Decrypting classical cipher text using Markov chain Monte Carlo." *Statistics and Computing*, 22(2), 397-413.
6. Gelman, A., & Rubin, D. B. (1992). "Inference from iterative simulation using multiple sequences." *Statistical Science*, 7(4), 457-472.

## License

MIT License — see [LICENSE](LICENSE) file.
