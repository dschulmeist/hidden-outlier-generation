# Multi-Bisect Hidden Outlier Generation

## Description

This repository contains a Python implementation of a synthetic hidden outlier generation algorithm using the Multi-Bisection method. It also includes functions to fit outlier detection models in all possible subspaces of the data and to perform interval checks.

---

## Installation

To install the package, you can simply clone this GitHub repository.

---

## Usage

Import the module and call the `main_multi_bisect()` method:

```python
from src.bisect import MultiBisectHOGen

# Initialize
generator = MultiBisectHOGen(data, outlier_detection_method=OdLOF, seed=42)

# Generate hidden outliers
outliers = generator.fit_generate(gen_points=100)
```

---

## Features

### `fit_model()`

**Fits a model for a given subspace.**

**Args:**
- `subspace` (tuple): The subspace to fit the model on.
- `data` (ndarray): The dataset.
- `outlier_detection_method` (class): The outlier detection model class.
- `tempdir` (str): Temporary directory for storing data.

**Returns:**
- tuple: The subspace and the fitted model.

---

### `fit_in_all_subspaces()`

**Fits models for all possible subspaces of the given data.**

---

### `outlier_check()`

**Checks if a point is an outlier in any subspace.**

---

### `inference()`

**Infer whether a given data point is an outlier in the specified subspace.**

---

### `interval_check()`

**Checks intervals for inliers and outliers.**

---

### `multi_bisect()`

**Performs the multi-bisection algorithm to find hidden outliers.**

---

### `MultiBisectHOGen`

**Class that initializes the algorithm and contains utility methods.**
