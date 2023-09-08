# Hidden Outlier Generator with Bisection Algorithm

## Description

This repository hosts a Python-based hidden outlier generator leveraging the bisect algorithm. 


## Installation

To install, clone this repository:

\```bash
git clone https://github.com/dschulmeist/hidden-outlier-generation
\```

## Usage

Here's how to import and use the main function to generate synthetic hidden outliers:

\```python
from hog_bisect.bisect import BisectHOGen

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

# Hidden Outlier Generator with Bisection Algorithm

## Description

This repository hosts a Python-based hidden outlier generator leveraging the bisect algorithm.

## Utility Functions

### `random_unif_on_sphere(number, dimensions, r, random_state=5)`

Generates uniformly distributed random points on a sphere.

- **Args:**
  - `number` (int): Number of points to generate.
  - `dimensions` (int): The dimensions of the sphere.
  - `r` (float): Radius of the sphere.
  - `random_state` (int, optional): Random seed.

- **Returns:**
  - ndarray: An array containing the generated points on the sphere.

### `gen_powerset(dims)`

Generates the power set of dimensions, which are sets containing all possible combinations of dimensions.

- **Args:**
  - `dims` (int): Number of dimensions.

- **Returns:**
  - set: The power set of dimensions.

### `subspace_grab(indices, data)`

Grabs a subspace of the data based on the specified indices.

- **Args:**
  - `indices` (list or tuple): Indices of the attributes for the subspace.
  - `data` (ndarray): The original data.

- **Returns:**
  - ndarray: The subspace data.

### `gen_rand_subspaces(dims, upper_limit, include_all_attr=True, seed=5)`

Generates random subspaces based on given dimensions.

- **Args:**
  - `dims` (int): Number of dimensions.
  - `upper_limit` (int): Upper limit for the number of subspaces.
  - `include_all_attr` (bool, optional): Whether to include all attributes.
  - `seed` (int, optional): Random seed.

- **Returns:**
  - set: The generated subspaces.

### `fit_model(subspace, data, outlier_detection_method, tempdir)`

Fits an outlier detection model for a given subspace.

- **Args:**
  - `subspace` (tuple): The subspace to fit the model on.
  - `data` (ndarray): The dataset.
  - `outlier_detection_method` (class): The outlier detection model class.
  - `tempdir` (str): Temporary directory for storing data.

- **Returns:**
  - tuple: The subspace and the fitted model.

### `fit_in_all_subspaces(...)`

Fits models for all possible subspaces of the given data.

- **Args:**
  - `outlier_detection_method` (class): The outlier detection model class.
  - `data` (ndarray): The dataset.
  - `tempdir` (str): Temporary directory for storing data.
  - `subspace_limit` (int): 2^subspace_limit will be the maximum amount of subspaces fitted.
  - `seed` (int, optional): Seed for random number generator.
  - `n_jobs` (int): Number of cores to use.

- **Returns:**
  - dict: Dictionary of models fitted on different subspaces.




