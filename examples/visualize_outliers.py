"""
Visualize generated hidden outliers.

This example generates hidden outliers and creates a 2D scatter plot
showing the original data alongside the generated hidden outliers.

Requirements:
    pip install matplotlib
    or: uv add matplotlib
"""

import sys

import numpy as np
from pyod.models.lof import LOF

from hog_bisect import BisectHOGen

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This example requires matplotlib.")
    print("Install it with: uv add matplotlib")
    sys.exit(1)


def main():
    # Generate synthetic 2D data for easy visualization
    np.random.seed(42)
    n_samples = 200
    n_features = 2  # 2D for direct visualization

    # Create clustered data
    data = np.vstack([
        np.random.randn(100, 2) * 0.5 + [0, 0],  # Cluster 1
        np.random.randn(100, 2) * 0.5 + [3, 3],  # Cluster 2
    ])

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("Generating hidden outliers...")

    # Generate hidden outliers
    generator = BisectHOGen(data=data, outlier_detection_method=LOF, seed=42)
    hidden_outliers = generator.fit_generate(gen_points=50, n_jobs=1)

    print(f"Generated {len(hidden_outliers)} hidden outliers")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot original data
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c="blue",
        alpha=0.5,
        s=30,
        label=f"Original data (n={len(data)})",
    )

    # Plot hidden outliers
    if len(hidden_outliers) > 0:
        ax.scatter(
            hidden_outliers[:, 0],
            hidden_outliers[:, 1],
            c="red",
            marker="x",
            s=100,
            linewidths=2,
            label=f"Hidden outliers (n={len(hidden_outliers)})",
        )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Hidden Outlier Generation Visualization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path = "hidden_outliers_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    # Show plot (comment out if running headless)
    plt.show()


if __name__ == "__main__":
    main()
