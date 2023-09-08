# Hidden Outlier Generator with Bisection Algorithm

## Description

This repository hosts a Python-based hidden outlier generator leveraging the bisection algorithm. The project offers various functionalities, including fitting outlier detection models in every potential data subspace and performing interval assessments.

## Installation

To install, clone this repository:

\```bash
git clone <repository_url>
\```

## Usage

Here's how to import and use the main function to generate synthetic hidden outliers:

\```python
from src.bisect import BisectHOGen

# Initialize the generator
generator = BisectHOGen(data, outlier_detection_method=pyod.models.lof.LOF, seed=42)

# Generate hidden outliers
outliers = generator.fit_generate(gen_points=100)
\```

## Features

### Methods and Classes

#### `fit_generate()`

Generate synthetic hidden outliers using a bisection algorithm.

- **Args:**
    - `gen_points` (int): Number of synthetic points to generate.
    - `check_fast` (bool): Fast check flag.
    - `fixed_interval_length` (bool): Flag for fixed interval length.
    - `get_origin_type` (str): Method to determine the origin.
    - `verbose` (bool): Verbose flag.
    - `n_jobs` (int): Number of parallel jobs.

- **Returns:**
    - ndarray: An array containing generated hidden outliers.

#### `BisectHOGen`

A class for generating synthetic hidden outliers.

### Utility Functions

#### `inference()`

Determine if a point is an outlier within a given subspace.

#### `outlier_check()`

Check for outliers in any subspace.

#### `interval_check()`

Perform interval assessments to determine outliers and inliers.

#### `bisect()`

Apply the multi-bisection algorithm to discover hidden outliers.

## Code Structure

### `BisectHOGen` Class

The `BisectHOGen` class initializes the generator and contains utility methods.

\```python
class BisectHOGen:
...
\```

### Function Definitions

Functions like `outlier_check`, `inference`, `interval_check`, and `bisect` are defined to perform various tasks in the outlier detection and generation process.

\```python
def outlier_check(...):
...
def inference(...):
...
def interval_check(...):
...
def bisect(...):
...
\```
