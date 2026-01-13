"""
Basic usage example for hidden outlier generation.

This script demonstrates the simplest way to generate hidden outliers
using the BisectHOGen class. Hidden outliers are points that exhibit
different outlier behavior in subspaces vs the full feature space.
"""

import numpy as np
from pyod.models.lof import LOF

from hog_bisect import BisectHOGen


def main():
    # Generate synthetic normal data
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    data = np.random.randn(n_samples, n_features)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("-" * 50)

    # Create the hidden outlier generator
    generator = BisectHOGen(
        data=data,
        outlier_detection_method=LOF,  # Local Outlier Factor
        seed=42,
    )

    # Generate hidden outliers
    hidden_outliers = generator.fit_generate(
        gen_points=50,  # Number of candidate points to generate
        n_jobs=1,  # Use single core (increase for parallel processing)
    )

    # Print results
    print(f"\nGenerated {len(hidden_outliers)} hidden outliers")

    if len(hidden_outliers) > 0:
        print(f"Shape: {hidden_outliers.shape}")
        print("\nFirst 3 hidden outliers:")
        for i, point in enumerate(hidden_outliers[:3]):
            print(f"  {i + 1}: {point}")

    # Print detailed summary
    generator.print_summary()


if __name__ == "__main__":
    main()
