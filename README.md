# ABCBoost: Gradient Boosting with Optimizations

This repository contains the source code for `ABCBoost`, a high-performance gradient boosting framework designed for both regression and classification tasks. This implementation includes various optimizations to enhance computation speed and efficiency, particularly suited for large datasets and high-dimensional data.

## Overview

`ABCBoost` is a gradient boosting library implemented in C++. The core of the library is built around training decision trees iteratively, where each tree aims to correct the errors of the previous trees. The model supports classification, regression, and ranking tasks, with specific optimizations to accelerate training, reduce memory usage, and improve overall efficiency.

### Key Features

- **Gradient Boosting Framework**: Implements gradient boosting for various tasks, including regression and classification.
- **Multi-Class Support**: Handles binary and multi-class classification tasks with efficient probability and loss calculations.
- **Optimized for Large Datasets**: Supports large data sizes, with an efficient handling of memory and CPU resources.
- **Custom Logging and Output**: Provides detailed logging options for training metrics, with the ability to save predictions and feature importance scores.
- **Parallelization with OpenMP**: Utilizes OpenMP for parallel processing to speed up training on multi-core CPUs.

## File Structure

- **`model.cc`**: The main implementation file for gradient boosting, including functions for training, prediction, and model evaluation.
- **`tree.h`**: Contains the decision tree structures used in the boosting iterations.
- **`data.h`**: Manages dataset loading and preprocessing.
- **`utils.h`**: Utility functions for mathematical operations, logging, and memory management.
- **`config.h`**: Configurations and parameters for setting up the model and data processing options.

## Optimizations

This implementation includes several optimizations aimed at reducing training time and improving efficiency. Here are the primary optimizations:

### 1. **Enhanced OpenMP Parallelization**

   - We leveraged OpenMP parallel loops to distribute the workload across multiple CPU cores effectively. This approach significantly reduces the training time by parallelizing computations, such as updating the gradient and residual values across data points.
   - A custom OpenMP reduction (`vec_double_plus`) was implemented to optimize vector summations within parallel loops.

### 2. **Memory Preallocation and Reuse**

   - Frequently resized vectors (such as `prob` and `hist` buffers) are now preallocated and reused across iterations, which minimizes memory allocation and deallocation overhead.
   - We reduced dynamic memory allocations within loops and instead allocated necessary memory once at the start of training. This optimization reduced memory fragmentation and improved cache efficiency.

### 3. **Efficient Logging and Conditional Output**

   - Instead of logging at every iteration, conditional logging has been implemented to reduce I/O overhead. Logging can be controlled through configuration parameters, allowing flexibility in logging frequency.
   - Logging is now buffered and written in batches to avoid frequent file access, which was a bottleneck in the original implementation.

### 4. **Optimized Softmax and Loss Calculations**

   - The `softmax` function, used frequently during probability calculations, was optimized by applying numerical stability tricks and avoiding redundant calculations. 
   - Loss calculations were also optimized by caching values and reducing unnecessary calls to expensive mathematical functions like `exp` and `log`.

### 5. **Reduced Sorting and Redundant Calculations**

   - Sorting operations, particularly for feature importance, were reduced or optimized. Where possible, we used indexed structures to avoid repeated sorting of large vectors.
   - Redundant calculations within training loops were minimized, such as caching intermediate results for metrics that don’t need to be recomputed every iteration.

### 6. **Improved Data Access Patterns**

   - Data structures, such as the histogram (`hist`) used in tree training, were reorganized for better memory locality. This reorganization ensures that data access patterns are more CPU cache-friendly, leading to faster read and write operations.

### 7. **Numerical Stability Enhancements**

   - Several numerical stability improvements were implemented, such as clamping values within a reasonable range in softmax calculations to avoid overflow. This ensures that calculations remain stable and accurate, even on large datasets.

## Usage

### Compilation

Ensure that you have a C++ compiler with OpenMP support. To compile the code, you can use the following command:

```bash
g++ -fopenmp -o abcboost model.cc -O3
