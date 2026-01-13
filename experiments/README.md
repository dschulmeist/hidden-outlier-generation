# Experiments

This directory contains research experiments and benchmarks for the hidden outlier generation library.

## Structure

```
experiments/
├── README.md           # This file
├── benchmarks/         # Performance benchmarks
│   └── scalability.py  # Test performance vs data size and dimensions
└── results/            # Output directory (gitignored)
```

## Running Experiments

From the repository root:

```bash
# Run scalability benchmark
uv run python experiments/benchmarks/scalability.py

# Results are printed to console and optionally saved to experiments/results/
```

## Available Experiments

### Benchmarks

- **scalability.py**: Measures generation time across different:
  - Data sizes (100, 500, 1000 samples)
  - Dimensionalities (3, 5, 8 features)
  - Outputs timing results to console

## Adding New Experiments

1. Create a new Python script in the appropriate subdirectory
2. Follow the pattern of existing experiments
3. Document the experiment purpose in this README
4. Ensure results are saved to `experiments/results/` (which is gitignored)
