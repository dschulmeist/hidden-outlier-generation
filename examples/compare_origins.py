"""
Compare different origin methods for hidden outlier generation.

The origin is the starting point from which the bisection algorithm
searches outward. Different origin strategies can discover different
hidden outliers depending on the data distribution.

Available origin methods:
- centroid: Use the data mean (simple, deterministic)
- least outlier: Use the most "normal" point (stable)
- random: Random inlier each iteration (diverse results)
- weighted: Weighted random toward normal points (recommended)
"""

import time

import numpy as np
from pyod.models.lof import LOF

from hog_bisect import BisectHOGen

# Origin methods to compare
ORIGIN_METHODS = ["centroid", "least outlier", "random", "weighted"]


def run_with_origin(data: np.ndarray, origin_type: str, gen_points: int = 30) -> dict:
    """Run generation with a specific origin method and collect metrics."""
    generator = BisectHOGen(data=data, outlier_detection_method=LOF, seed=42)

    start = time.time()
    hidden_outliers = generator.fit_generate(
        gen_points=gen_points,
        get_origin_type=origin_type,
        n_jobs=1,
    )
    elapsed = time.time() - start

    # Count H1 and H2 types from the generator's stored results
    h1_count = np.sum(generator.hidden_x_type == "H1") if generator.hidden_x_type is not None else 0
    h2_count = np.sum(generator.hidden_x_type == "H2") if generator.hidden_x_type is not None else 0

    return {
        "origin": origin_type,
        "total_hidden": len(hidden_outliers),
        "h1_count": int(h1_count),
        "h2_count": int(h2_count),
        "time_seconds": elapsed,
    }


def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    data = np.random.randn(n_samples, n_features)

    print("Comparing Origin Methods for Hidden Outlier Generation")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("Generating 30 candidate points with each method\n")

    results = []
    for origin in ORIGIN_METHODS:
        print(f"Running with origin='{origin}'...")
        result = run_with_origin(data, origin)
        results.append(result)
        print(f"  Found {result['total_hidden']} hidden outliers in {result['time_seconds']:.2f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Origin Method':<15} {'Hidden':<8} {'H1':<6} {'H2':<6} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['origin']:<15} {r['total_hidden']:<8} {r['h1_count']:<6} {r['h2_count']:<6} {r['time_seconds']:<10.2f}")

    print("\nNotes:")
    print("- H1: Outlier in subspace but not full space")
    print("- H2: Outlier in full space but not subspaces")
    print("- 'weighted' is generally recommended for diverse results")


if __name__ == "__main__":
    main()
