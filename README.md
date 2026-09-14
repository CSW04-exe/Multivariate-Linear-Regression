# Multivariate Linear Regression — Boston Housing

**Type:** Team project
**Contributors:** Carter Ward, Boyd Emmons
**Course:** CS 430-1 (Machine Learning) — HW2
**Completed:** 10/07/2025

## Purpose

This assignment extends an earlier single-feature regression project into **multivariate** linear regression: fitting a model with several input features at once. We implemented both gradient descent and the closed-form normal equation from scratch, rather than calling `sklearn.LinearRegression`, to understand what a regression library does under the hood.

## Problem and Approach

The task is the classic Boston Housing dataset: predict `MEDV` (median home value, $1000s) from neighborhood features like crime rate, room count, and pupil-teacher ratio. We avoided `numpy`/`pandas`/`sklearn` entirely, writing the parser, statistics, linear algebra, and training algorithms in pure Python. The assignment has three parts: (1) gradient descent on two standardized features (`AGE`, `TAX`), (2) gradient descent on all 13 standardized features, and (3) the normal equation on the same two-feature (`AGE`, `TAX`) problem, unscaled, as a cross-check.

## Structure and Methodologies

- No dependencies beyond the standard library — `X`/`y` and weight vectors are plain nested Python lists (bias term prepended as a column of 1s).
- Hand-rolled statistics (`mean`/`stdev`) and standardization fit on training data only, then applied to both train and validation sets to avoid leakage.
- From-scratch matrix operations (`mat_transpose`, `mat_mul`, `mat_vec_mul`) with nested loops.
- Gaussian elimination with partial pivoting (`solve_linear_system`) to solve `(XᵀX)w = Xᵀy` for the normal equation, with a small ridge fallback for numerical stability.
- Both a manual two-feature gradient descent update and a generalized matrix-based gradient descent that works for any feature count.

## Process

1. Parse `boston.txt` (whitespace-delimited UCI format) into 506 rows of 14 values.
2. Split into the first 456 rows for training and the last 50 for validation (non-random, per spec).
3. Standardize features (training stats only) for the gradient descent runs; leave `AGE`/`TAX` unscaled for the normal equation run.
4. Train all three models: 2-feature GD, 13-feature GD (both 5000 epochs, `alpha=0.01`), and the normal equation via Gaussian elimination.
5. Evaluate each model's MSE on the 50-row validation set.
6. Write weights and validation MSEs to `output.txt` and print a summary.

## Outcome

Validation MSE: 2-feature gradient descent (`AGE`, `TAX`, standardized) scored **22.046**; the 13-feature gradient descent scored **10.948**, roughly halving the error and confirming home value depends on much more than age and tax rate; the normal equation on the same 2-feature problem (unscaled) matched the 2-feature GD result exactly at **22.046**. That exact agreement between gradient descent and the closed-form solution was the strongest sanity check — it shows the from-scratch GD implementation converges to the true OLS solution rather than just approximating it, and we came away with a working, from-scratch understanding of standardization, matrix algebra, gradient descent, and Gaussian elimination.

## How to run

```bash
python3 boston_linreg_hw2.py
```

Expects `boston.txt` in the same directory; writes results to `output.txt`.
