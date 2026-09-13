# Multivariate Linear Regression on the Boston Housing Dataset

A from-scratch implementation of multivariate linear regression — no NumPy, no scikit-learn, just Python lists and hand-written linear algebra — trained on the classic Boston housing dataset to predict median home value (`MEDV`) from structural, socioeconomic, and geographic features.

## 1. Purpose

This project exists to answer a simple question the hard way: can a linear model, built entirely by hand, predict Boston-area home prices from a handful of neighborhood features — and does the *from-scratch* version actually behave the way the textbook math says it should?

Rather than reaching for `numpy.linalg` or `sklearn.linear_model.LinearRegression`, the goal was to implement every piece of the pipeline — data parsing, feature standardization, matrix operations, gradient descent, and the normal equation — using only Python's standard library. That constraint turns the project from "call a library function" into "prove you understand what the library function is doing," which is the real point of the exercise: building intuition for how linear regression actually computes its weights, not just how to invoke it.

## 2. Problem and approach

This was an assigned homework problem (`boston_linreg_hw2.py`, credited to Boyd Emmons and Carter Ward, dated Oct 7, 2025): given the Boston housing dataset, predict `MEDV` (median value of owner-occupied homes, in $1000s) from the other 13 recorded features, and compare two different ways of fitting the model.

The approach taken breaks the problem into two parallel investigations on the same train/validation split (the first `N - 50` rows for training, the last 50 rows held out for validation, matching the assignment spec):

- **Part 1 — Gradient Descent.** Two sub-cases are fit:
  - **Case 2a:** a small, interpretable model using only two z-scored features, `AGE` (proportion of owner-occupied units built before 1940) and `TAX` (property tax rate), to see how far a minimal feature set can go.
  - **Case 2b:** the full model, using all 13 remaining features (z-scored on the training set), to see how much predictive power is gained by using everything available.
- **Part 2 — Normal Equation.** The same `AGE`/`TAX` model from Case 2a is refit using the closed-form ordinary least squares solution, on the *unscaled* raw features, as a check against the gradient descent result.

Standardization statistics (mean/std) are always computed on the training split only and then applied to both train and validation data, to avoid leaking validation information into the model — a detail that's easy to get wrong and was clearly handled deliberately here.

## 3. Structure and methodologies

**Dependencies:** none beyond the Python standard library (`csv`-style manual parsing, plain file I/O). No NumPy, pandas, or scikit-learn — a deliberate constraint of the assignment, not an oversight.

**Data:** `boston.txt`, the raw whitespace-delimited Boston housing dataset (14 columns: `CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV`), parsed by a custom loader that scans past the header/description text, reflows the whitespace-wrapped numeric rows, and regroups every 14 values into one record.

**Core data structures:** plain Python `list`s and `list[list[float]]` used as row-major matrices throughout — there's no dedicated matrix/tensor class, so every linear-algebra routine (transpose, matrix multiply, matrix-vector multiply) is written against raw nested lists.

**Methodologies/algorithms implemented from scratch:**
- `mean` / `stdev` and a train-fit `fit_standardizer` / `apply_standardizer` pair for z-score normalization (with divide-by-zero guarding on zero-variance columns).
- `add_bias_column` to prepend the intercept term.
- A small linear algebra kernel: `mat_shape`, `mat_transpose`, `mat_mul`, `mat_vec_mul`.
- `gradient_descent_two_feature` (a specialized 2-feature/3-parameter update rule) and `gradient_descent_general` (a fully vectorized-by-hand batch gradient descent using `w -= alpha * (2/n) * Xᵀ(Xw - y)`).
- `solve_linear_system`, a Gaussian elimination solver with partial pivoting and an automatic ridge-regularization fallback (`1e-8`) if the system is near-singular, used by `normal_equation` to solve the OLS normal equations `(XᵀX)w = Xᵀy` directly.
- `mse` for evaluation.
- `write_output`, which serializes every learned weight vector and validation MSE to `output.txt`.

## 4. Process

Reading the code, the build looks like it followed a fairly natural, spec-driven progression:

1. **Scaffolding first.** The file is laid out as a full end-to-end regression pipeline template — sections for data loading, EDA, preprocessing, linear algebra, modeling, prediction/loss, training workflow, evaluation, k-fold CV, diagnostics, and error handling — each clearly commented with `# TODO:` blocks describing what a complete version of that stage would do. This suggests the assignment (or the authors) started from a full checklist of "real" ML pipeline components before deciding what the actual homework required.
2. **Implementing only what the assignment needed.** Rather than filling in every TODO, the authors implemented exactly the pieces needed to answer the two specific homework parts — the data loader, standardizer, bias-column helper, matrix ops, both gradient descent variants, the Gaussian-elimination-based normal equation solver, and MSE — leaving the optional sections (EDA printing, k-fold cross-validation, polynomial expansion, residual diagnostics) as unfinished stubs. That's a pragmatic, scope-controlled way to work under a deadline (dated Oct 7 2025 in the file header).
3. **Building the 2-feature case first.** `gradient_descent_two_feature` and `predict_rows_two_feature` are hard-coded to exactly two inputs (`AGE`, `TAX`) with an explicit 3-parameter update rule — evidence this was written first, likely reusing update equations from an earlier single-feature regression assignment, before being generalized.
4. **Generalizing to arbitrary feature counts.** `gradient_descent_general` and `normal_equation` were then added as fully general, dimension-agnostic routines (operating on any `X_b` matrix), so the same machinery could handle both the 2-feature and 13-feature cases without rewriting the update rule.
5. **Wiring it together in `main()`.** The final `main()` function loads the data once, builds the train/validation split, runs Case 2a (GD), Case 2b (GD), and Case 2a (Normal Equation) back to back, and writes every result to `output.txt`, with a short console summary of the three validation MSEs — a clear sign the last step was assembling the pieces into the specific deliverable the assignment asked for, rather than a general-purpose CLI tool (the unused `--data`/`--alpha`/`--epochs`-style CLI arguments sketched in the TODO comments were never actually implemented).
6. **Numerical safety net.** The normal equation path includes an automatic ridge fallback in `solve_linear_system` if Gaussian elimination hits a near-zero pivot — a sign the authors ran into (or anticipated) a singular/ill-conditioned `XᵀX` matrix and hardened the solver rather than letting it crash.

## 5. Outcome

Running the pipeline produces `output.txt` with concrete, reproducible validation results on the held-out last 50 rows:

| Model | Features | MSE (validation) |
|---|---|---|
| Gradient Descent | `AGE`, `TAX` (z-scored) | **22.046** |
| Gradient Descent | all 13 features (z-scored) | **10.948** |
| Normal Equation | `AGE`, `TAX` (unscaled) | **22.046** |

Two results stand out as validation that the implementation is mathematically correct rather than just "runs without errors":

- **Gradient descent and the normal equation agree.** The two-feature gradient descent model and the closed-form normal equation solution converge to essentially the same validation MSE (22.04601584 vs. 22.04601584), even though one is fit on standardized features and the other on raw, unscaled features, and one is an iterative approximation while the other is an exact linear-algebra solution. That agreement is strong evidence the gradient descent update rule, the standardization/bias-column plumbing, and the Gaussian-elimination solver are all implemented correctly — a bug in any of them would very likely have broken this agreement.
- **More features meaningfully improved the fit.** Expanding from 2 features to the full 13-feature set roughly halved the validation MSE (22.05 → 10.95), which is the expected, sane outcome for adding genuinely predictive housing features (`RM`, `LSTAT`, `PTRATIO`, etc.) rather than noise.

Beyond the numbers, building this project meant implementing — and personally debugging — the full mechanics that are normally hidden behind a single `.fit()` call: manual matrix transpose/multiply routines, a Gaussian elimination solver with pivoting and a ridge-regularization safety net for near-singular systems, and batch gradient descent derived directly from the MSE loss gradient. It demonstrates the ability to translate the underlying linear algebra and calculus of OLS regression into working, numerically stable code, to reason about *why* two very different solution methods (iterative vs. closed-form) should converge to the same answer, and to use that cross-check as a form of self-validation instead of just trusting the output. The biggest takeaway was seeing firsthand how sensitive the normal equation is to conditioning (hence the ridge fallback) and how directly the number of informative features affects predictive error — lessons that are easy to state abstractly but much more memorable after implementing and debugging them from raw arithmetic.
