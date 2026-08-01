# XGBoost From Scratch
Learn how Xgboost works under the hood
- Classification with the `XGBClassifier` class.
- Regression with the `XGBRegressor` class.

## What's new?
**XGBoost (eXtreme Gradient Boosting)** is an implementation of [gradient boosting](../gradient_boosting/README.md), but it is not just an implementation — it introduces improvements over classic Gradient Boosting Machines (GBM). This section summarizes the key changes.

### 1. Regularization
Xgboost introduce the regularized objective.

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)
$$

where $\Omega(f_t)$ is a regularization term penalizing tree complexity.

$$
\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 \;(\,+\, \alpha \sum_{j=1}^T |w_j|\text{ optionally}\,)
$$

- $\gamma$ (gamma) penalizes the number of leaves $T$, discouraging overly complex trees. (default=0)
- $\lambda$ (L2 - default=1) and optionally $\alpha$ (L1 - default=0) shrink leaf weights toward zero. 

Because this regularization is part of the loss being minimized, it directly shapes both the leaf weights *and* which splits are considered worthwhile — a split is only kept if the gain outweighs $\gamma$. This tends to produce more robust models with less manual tuning than classic GBM.

### 2. Scoring Function for XGBoost Tree 
XGBoost uses new criterion or scoring function to decide which splits are worth making while growing the tree. 

Specifically, to decide whether to split a leaf into a left and right child, compare the score **before** the split (one leaf, $I = I_L \cup I_R$) against the score **after** the split (two leaves). The **gain** of a split is defined as the reduction in loss it produces, minus the complexity cost of adding one more leaf:

$$
\text{Gain} = \frac{1}{2}\left[ \underbrace{\frac{G_L^2}{H_L + \lambda}}_{\text{left leaf score}} + \underbrace{\frac{G_R^2}{H_R + \lambda}}_{\text{right leaf score}} - \underbrace{\frac{(G_L+G_R)^2}{H_L + H_R + \lambda}}_{\text{score if not split}} \right] - \gamma
$$

where $G_L, H_L$ and $G_R, H_R$ are the summed gradients and hessians of samples routed left and right by the candidate split, respectively.

If $\text{Gain} \le 0$ for every candidate split at a node, that node remains a leaf.

**How Gain Is Used During Tree Construction**

At each node, XGBoost:

1. Enumerates candidate split points for each feature (exact scan, or approximate candidates).
2. For each candidate split, computes $G_L, H_L, G_R, H_R$ by summing gradients and hessians of samples on each side.
3. Computes the Gain formula above for every candidate.
4. Picks the feature and threshold with the **highest Gain**.
5. Recurses on the resulting child nodes

### 3. Sparsity-Aware Split Finding (Handle Missing Values)
Traditional GBM implementations generally require imputing missing values before training. XGBoost instead learns a default direction for missing values at each split: during training, it tries sending missing values left or right at each node and picks whichever direction improves the objective the most. This means XGBoost handles missing data natively and often more effectively than manual imputation.

### 4. Approximate Split Finding via Weighted Quantile Sketch
For exact greedy split finding, every possible split point on every feature must be evaluated — expensive on large datasets. XGBoost introduces **a weighted quantile sketch algorithm** that proposes a small set of candidate split points (percentiles) per feature

### 5. Early stopping based on a validation metric, integrated directly into the training loop.


For more details, please go to [Xgboost Math](math.md).