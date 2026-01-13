"""
Scalability benchmark for hidden outlier generation.

This experiment measures how generation time scales with:
- Number of samples in the dataset
- Number of features (dimensions)

The results help understand the computational complexity and
guide users in choosing appropriate parameters for their use case.
"""

import time
from dataclasses import dataclass

import numpy as np
from pyod.models.lof import LOF

from hog_bisect import BisectHOGen

# Benchmark configurations
SAMPLE_SIZES = [100, 300, 500]
DIMENSIONS = [3, 5, 7]
GEN_POINTS = 20  # Keep low for faster benchmarks


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    n_samples: int
    n_features: int
    gen_points: int
    time_seconds: float
    hidden_found: int


def run_benchmark(n_samples: int, n_features: int, gen_points: int) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)

    generator = BisectHOGen(data=data, outlier_detection_method=LOF, seed=42)

    start = time.time()
    hidden_outliers = generator.fit_generate(gen_points=gen_points, n_jobs=1, verbose=False)
    elapsed = time.time() - start

    return BenchmarkResult(
        n_samples=n_samples,
        n_features=n_features,
        gen_points=gen_points,
        time_seconds=elapsed,
        hidden_found=len(hidden_outliers),
    )


def main():
    print("Hidden Outlier Generation - Scalability Benchmark")
    print("=" * 70)
    print(f"Generating {GEN_POINTS} candidate points per configuration\n")

    results = []

    # Run benchmarks
    total_runs = len(SAMPLE_SIZES) * len(DIMENSIONS)
    current = 0

    for n_samples in SAMPLE_SIZES:
        for n_features in DIMENSIONS:
            current += 1
            print(f"[{current}/{total_runs}] Running: {n_samples} samples, {n_features} features...")

            result = run_benchmark(n_samples, n_features, GEN_POINTS)
            results.append(result)

            print(f"         Time: {result.time_seconds:.2f}s, Hidden found: {result.hidden_found}")

    # Summary table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Samples':<10} {'Features':<10} {'Time (s)':<12} {'Hidden':<10} {'Rate (pts/s)':<12}")
    print("-" * 70)

    for r in results:
        rate = r.gen_points / r.time_seconds if r.time_seconds > 0 else 0
        print(f"{r.n_samples:<10} {r.n_features:<10} {r.time_seconds:<12.2f} {r.hidden_found:<10} {rate:<12.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    # Time vs samples (fixed dimensions)
    print("\nTime scaling with sample size (features=5):")
    for r in [r for r in results if r.n_features == 5]:
        print(f"  {r.n_samples} samples: {r.time_seconds:.2f}s")

    # Time vs dimensions (fixed samples)
    print("\nTime scaling with dimensions (samples=300):")
    for r in [r for r in results if r.n_samples == 300]:
        print(f"  {r.n_features} features: {r.time_seconds:.2f}s")

    print("\nNote: Higher dimensions require fitting more subspace models,")
    print("which increases computational cost exponentially until the")
    print("random sampling threshold (default: 11 dimensions) is reached.")


if __name__ == "__main__":
    main()
