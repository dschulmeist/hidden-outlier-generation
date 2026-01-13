# Hidden Outlier Generation

A Python library for generating synthetic hidden outliers using the bisection algorithm.

## What are Hidden Outliers?

Hidden outliers are data points that exhibit different outlier behavior depending on which feature subspace you examine:

- **H1 (Subspace Hidden)**: Outlier in some feature subspace but NOT in the full feature space
- **H2 (Full-space Hidden)**: Outlier in the full feature space but NOT in any subspace

These are useful for benchmarking outlier detection algorithms, especially subspace-aware methods.

## Installation

```bash
pip install hidden-outlier-generation
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add hidden-outlier-generation
```

## Quick Start

```python
import numpy as np
from pyod.models.lof import LOF
from hog_bisect import BisectHOGen

# Your dataset
data = np.random.randn(200, 5)

# Create generator
generator = BisectHOGen(
    data=data,
    outlier_detection_method=LOF,
    seed=42
)

# Generate hidden outliers
hidden_outliers = generator.fit_generate(gen_points=50)

print(f"Generated {len(hidden_outliers)} hidden outliers")
generator.print_summary()
```

## Features

- **Multiple origin strategies**: centroid, least outlier, random, weighted
- **Flexible detection methods**: Any PyOD detector (LOF, KNN, IForest, etc.)
- **Parallel processing**: Use `n_jobs=-1` for multi-core execution
- **Reproducible**: Seed parameter for deterministic results
- **Type hints**: Full typing support with py.typed marker

## API Reference

### BisectHOGen

```python
BisectHOGen(
    data: np.ndarray,                    # Input dataset (n_samples, n_features)
    outlier_detection_method=LOF,        # PyOD detector class
    seed: int = 5,                       # Random seed
    max_dimensions: int = 11             # Threshold for random subspace sampling
)
```

### fit_generate()

```python
generator.fit_generate(
    gen_points: int = 100,               # Number of candidate points
    check_fast: bool = True,             # Fast subspace checking
    is_fixed_interval_length: bool = True,
    get_origin_type: str = "weighted",   # Origin strategy
    verbose: bool = False,
    n_jobs: int = 1                      # Parallel workers (-1 for all cores)
) -> np.ndarray                          # Array of hidden outliers
```

### Origin Types

| Type | Description |
|------|-------------|
| `"centroid"` | Use data mean as origin (deterministic) |
| `"least outlier"` | Use most normal point (stable) |
| `"random"` | Random inlier each iteration (diverse) |
| `"weighted"` | Weighted random toward normal points (recommended) |

## Examples

The repository includes example scripts in the `examples/` directory:

```bash
# Clone the repo to access examples
git clone https://github.com/dschulmeist/hidden-outlier-generation
cd hidden-outlier-generation

# Run examples
python examples/basic_usage.py
python examples/compare_origins.py
python examples/compare_detectors.py
python examples/visualize_outliers.py  # requires matplotlib
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/dschulmeist/hidden-outlier-generation
cd hidden-outlier-generation
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and release process.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{hidden_outlier_generation,
  title = {Hidden Outlier Generation},
  author = {Schulmeister, Daniel},
  url = {https://github.com/dschulmeist/hidden-outlier-generation},
  year = {2023}
}
```
