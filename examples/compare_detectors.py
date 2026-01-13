"""
Compare different outlier detection methods for hidden outlier generation.

The choice of outlier detection method affects which points are classified
as outliers, and therefore which hidden outliers can be discovered.

This example compares several PyOD detectors:
- LOF (Local Outlier Factor): Density-based, good for local outliers
- KNN (K-Nearest Neighbors): Distance-based, simple and effective
- IForest (Isolation Forest): Tree-based, fast and scalable
"""

import time

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from hog_bisect import BisectHOGen

# Detection methods to compare
DETECTORS = [
    ("LOF", LOF),
    ("KNN", KNN),
    ("IForest", IForest),
]


def run_with_detector(data: np.ndarray, name: str, detector_class, gen_points: int = 30) -> dict:
    """Run generation with a specific detector and collect metrics."""
    generator = BisectHOGen(data=data, outlier_detection_method=detector_class, seed=42)

    start = time.time()
    hidden_outliers = generator.fit_generate(
        gen_points=gen_points,
        get_origin_type="weighted",
        n_jobs=1,
    )
    elapsed = time.time() - start

    h1_count = np.sum(generator.hidden_x_type == "H1") if generator.hidden_x_type is not None else 0
    h2_count = np.sum(generator.hidden_x_type == "H2") if generator.hidden_x_type is not None else 0

    return {
        "detector": name,
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

    print("Comparing Detection Methods for Hidden Outlier Generation")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("Generating 30 candidate points with each detector\n")

    results = []
    for name, detector_class in DETECTORS:
        print(f"Running with detector={name}...")
        result = run_with_detector(data, name, detector_class)
        results.append(result)
        print(f"  Found {result['total_hidden']} hidden outliers in {result['time_seconds']:.2f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Detector':<12} {'Hidden':<8} {'H1':<6} {'H2':<6} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['detector']:<12} {r['total_hidden']:<8} {r['h1_count']:<6} {r['h2_count']:<6} {r['time_seconds']:<10.2f}")

    print("\nDetector Characteristics:")
    print("- LOF: Local density-based, sensitive to local structure")
    print("- KNN: Distance-based, simple and interpretable")
    print("- IForest: Isolation-based, fast for large datasets")


if __name__ == "__main__":
    main()
