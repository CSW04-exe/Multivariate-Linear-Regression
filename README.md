# Multivariate Linear Regression — Boston Housing (CS430 HW2)

## 1. Purpose

This is my second linear regression assignment, built on top of an earlier
single-feature (AGE-only) regression project. Where that first assignment
fit one predictor to one target, this one asks for **multivariate** linear
regression: fitting a model with several input features at once, both by
gradient descent and by the closed-form normal equation. Boyd Emmons and I
worked through the assignment together (see the header comment in
`boston_linreg_hw2.py`), and the point of the exercise is to actually
implement the regression machinery ourselves — matrix operations, gradient
descent, and Gaussian elimination — rather than call `sklearn.LinearRegression`
and be done with it. That's the "why": understanding what a regression
library does under the hood before trusting it as a black box.

## 2. Problem and Approach

The problem is the classic **Boston Housing** dataset: predict `MEDV`
(median value of owner-occupied homes, in $1000s) from a set of
neighborhood-level features such as crime rate, room count, pupil-teacher
ratio, distance to employment centers, and so on.

I deliberately avoided `numpy`/`pandas`/`sklearn` and wrote everything —
the file parser, the statistics, the linear algebra, and both training
algorithms — in pure Python using nested lists. The assignment has three
parts, all driven from `main()`:

- **Part 1 (2a):** Gradient descent on just two standardized features,
  `AGE` and `TAX`, predicting `MEDV`. This mirrors the update rule from the
  earlier single-feature project, just extended to two inputs.
- **Part 1 (2b):** Gradient descent on **all 13** standardized features at
  once, to see how much a fuller model improves on the two-feature one.
- **Part 2 (2a):** The same `AGE`/`TAX` problem as 2a, but solved with the
  **normal equation** (closed-form OLS) on the raw, unscaled features
  instead of gradient descent — as a cross-check that the two methods agree.

## 3. Structure and Methodologies

**Dependencies:** none beyond the Python standard library — no `numpy`,
`pandas`, or `sklearn`. Every piece of math is hand-rolled.

**Data structures:**
- `X` is a list of lists (rows × features); `y` is a flat list of targets.
- Weight vectors (`theta` / `w`) are plain Python lists, with the bias term
  stored at index 0 (a leading column of `1.0`s is prepended to `X` via
  `add_bias_column`).

**Core building blocks I implemented:**
- `mean` / `stdev` — from-scratch statistics (population standard
  deviation) used to z-score features.
- `fit_standardizer` / `apply_standardizer` — compute mean/std on the
  **training** columns only, then apply those same stats to both train and
  validation data, so the validation set never leaks into preprocessing.
- `mat_shape`, `mat_transpose`, `mat_mul`, `mat_vec_mul` — basic matrix
  operations built with nested loops (no linear-algebra library).
- `solve_linear_system` — Gaussian elimination with partial pivoting to
  solve `(XᵀX) w = Xᵀy` for the normal equation. It falls back to a tiny
  ridge term (`1e-8`+) if it hits a (near-)singular pivot, which is also
  used proactively as `ridge_lambda=1e-10` in `normal_equation` for
  numerical stability.
- `gradient_descent_two_feature` — the original hand-written 2-feature
  update rule (`theta -= alpha * (1/m) * sum(error * x)`), kept close to
  the earlier single-feature project's formulas.
- `gradient_descent_general` — a vectorized-by-hand generalization that
  works for any number of features using the matrix helpers above
  (`gradient = (2/N) * Xᵀ(Xw - y)`).
- `normal_equation` — closed-form OLS: `w = (XᵀX)⁻¹ Xᵀy`, solved via the
  Gaussian elimination routine rather than an explicit matrix inverse.
- `mse` — mean squared error, used as the loss/evaluation metric throughout.

## 4. Process

1. **Load the data.** `load_boston_txt` reads `boston.txt` (the classic
   UCI-format Boston housing file, whitespace-delimited and wrapped across
   multiple lines per record) line by line, skips the descriptive header
   text, floods all numeric tokens into a running buffer, and slices that
   buffer into rows of 14 values (13 features + `MEDV`) once enough numbers
   have accumulated. This yields 506 complete rows.
2. **Split train/validation.** Per the assignment spec, the split isn't
   random: the **first 456 rows** are training data and the **last 50
   rows** are held out for validation (`train_rows = data[:N-50]`,
   `val_rows = data[N-50:]`).
3. **Preprocess.** For the gradient-descent runs (2a and 2b), I compute the
   mean and standard deviation of each feature column **on the training
   rows only**, then z-score both the training and validation rows using
   those same statistics. For the normal-equation run (Part 2, 2a), I
   intentionally leave `AGE`/`TAX` **unscaled** to compare a scaled vs.
   unscaled fit on the same two features.
4. **Train.**
   - 2a GD: `gradient_descent_general` on `[bias, AGE_z, TAX_z]` for 5000
     epochs at learning rate `alpha=0.01`.
   - 2b GD: the same routine on `[bias]` + all 13 standardized features for
     5000 epochs at `alpha=0.01`.
   - 2a NE: `normal_equation` solves for weights directly on `[bias, AGE,
     TAX]` (unscaled) via Gaussian elimination.
5. **Evaluate.** Each model predicts on the 50-row validation set
   (`mat_vec_mul`) and I compute validation MSE (`mse`) for all three runs.
6. **Report.** `write_output` writes all three weight vectors and their
   validation MSEs to `output.txt`, and `main()` prints a short summary to
   the console.

## 5. Outcome

Results from the recorded run (`output.txt`):

| Model | Features | Validation MSE |
|---|---|---|
| GD (2a) | AGE_z, TAX_z | **22.04601584** |
| GD (2b) | all 13 features, standardized | **10.94781194** |
| Normal Equation (2a) | AGE, TAX (unscaled) | **22.04601584** |

A few things stand out:

- **Gradient descent and the normal equation agree exactly** on the 2a
  problem (both land on MSE `22.04601584`), which is a good sanity check
  that my from-scratch GD implementation actually converges to the true
  OLS solution — even though one model trained on standardized features
  and the other on raw ones, the underlying linear fit (and its
  predictions) is the same, just expressed with different coefficients:
  `theta_gd_2a = [22.941, -1.524, -3.652]` (standardized) vs.
  `theta_ne_2a = [35.447, -0.0527, -0.0230]` (unscaled).
- **Using all 13 features roughly halves the validation error** compared
  to using just `AGE` and `TAX` (MSE drops from ~22.05 to ~10.95),
  confirming that home value depends on much more than housing age and tax
  rate — the 2b weights show `LSTAT` (% lower status population, w ≈
  -4.05), `RM`-adjacent effects, and `TAX`/`PTRATIO`/`DIS` all pulling
  noticeably on the prediction.
- Both models share the same bias term in the standardized runs (`22.941`),
  which makes sense: with z-scored features the intercept is just the mean
  `MEDV` of the training set, independent of which features are included.

**What this demonstrates:** I can build the full OLS pipeline — data
loading, train/validation splitting without leakage, standardization,
matrix algebra, gradient descent, and closed-form Gaussian elimination —
without leaning on a regression library, and get gradient descent and the
normal equation to agree numerically. That agreement is the part I'm
proudest of: it's easy to write gradient descent that "looks like it
works," but getting it to converge to the exact same solution as the
closed-form answer is a much stronger sanity check, and confirms I
understand the cost surface (and why standardizing features helps GD
converge cleanly) rather than just copying update-rule pseudocode.

## How to run

```bash
python3 boston_linreg_hw2.py
```

This expects `boston.txt` in the same directory. It writes results to
`output.txt` and prints a short summary of the three validation MSEs to
the console.
