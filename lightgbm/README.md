# LightGBM From Scratch

Learn how LightGBM works under the hood.
- Classification with `lgb.train(..., objective='binary')` 
- Regression with `lgb.train(..., objective='regression')`

## What's new?

**LightGBM** is another [gradient boosting](../gradient_boosting/README.md) / [XGBoost](../xgboost/README.md) implementation, engineered for speed on large data.  

What sets LightGBM apart is how it searches for splits and how it grows trees.

### 1. Histogram-Based Split Finding

Instead of evaluating every distinct value (exact greedy) or a weighted-quantile sketch of values (XGBoost's approximate algorithm), LightGBM discretizes each feature into at most `max_bin` (255) histogram bins **once, up front**. Split search then only ever evaluates thresholds at bin edges, never at every distinct value. Building the histograms for a node is a single pass over the samples that reach it — this is what makes LightGBM fast.

### 2. Leaf-Wise (Best-First) Tree Growth

Most GBDT implementations (including scikit-learn) grow trees **level-wise** that split every node at the current depth before moving deeper. 

LightGBM grows **leaf-wise**: at each step it finds the **single existing leaf with the largest split gain and splits that one** , repeating until `num_leaves` is reached. This best-first growth tends to reach lower loss for a given leaf count, at the cost of potentially deeper, unbalanced trees — which is why `num_leaves` (not `max_depth`) is the primary complexity control in LightGBM.

## Advanced Topics
- [Categorical Feature Support](categorical_split.md)
- [Histogram-Based Algorithm](histrogram.md)
- [L1 objective](l1_objective.md)
- [Missing Value Handle](missing_value_handle.md)
