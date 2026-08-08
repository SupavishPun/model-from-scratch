# SHAP From Scratch

Learn how SHAP (SHapley Additive exPlanations) works under the hood.
- Per-prediction feature attribution with `shap.TreeExplainer`.

## The intuition: a cooperative game

SHAP borrows **Shapley values** from cooperative game theory. Treat each feature as a *player* and the model's prediction as the *payout*. The Shapley value of a feature is its fair share of the payout — the average marginal contribution it makes when added to every possible coalition of other features.

A prediction is then decomposed additively (**local accuracy / efficiency**):

$$
f(x) = \underbrace{v_\varnothing}_{\text{base value (expected prediction)}} \;+\; \sum_{i=1}^{M} \phi_i
$$

where 
- $f(x)$ is the model's output for the sample. 
- $v_\varnothing$ is the base value is the model's average prediction over a background distribution
- $\phi_i$ is the SHAP value of feature $i$ 
- $M$ is the number of features 

So the SHAP values together explain exactly *how the sample deviates from the average*.

## The Shapley value formula

For feature $i$:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(M-|S|-1)!}{M!}\Big[\, f(S \cup \{i\}) - f(S) \,\Big]
$$

where $F$ is the full feature set and $S$ ranges over all subsets of the *other* features. The fraction is the Shapley weight (a factorial combinatorial factor); $|S|=0$ and $|S|=M-1$ get the largest weight, intermediate coalitions get less.

The whole difficulty is the **value function** $f(S)$: the model's expected prediction when only the features in $S$ are known (the rest are "missing"). Computing this expectation is what the two TreeSHAP variants differ on.

## TreeSHAP: exact Shapley values for trees in polynomial time

Naively, $f(S)$ requires marginalizing over the unknown features, and summing over $2^M$ coalitions is exponential. **TreeSHAP** exploits tree structure to compute *exact* Shapley values for tree ensembles in polynomial time, by recursing over the tree and tracking which features the path depends on.

The key choice is **how "a feature is missing" is interpreted**, controlled by whether a background dataset is supplied:

### 1. `tree_path_dependent` (no background data)
Conditional expectations are taken straight from the tree's own **cover statistics** — at each internal node, the unknown feature's effect is averaged using the proportion of training samples that went left vs. right. No background dataset is needed; it is fast, but the attributions are *path-dependent* rather than strictly marginal.

### 2. `interventional` (with background data)
Conditional expectations are computed by **intervention**: the unknown features are literally *substituted* with values from a background dataset and the result is averaged over all background rows. This gives a stricter Shapley guarantee (true marginal expectations) at higher computational cost.

> Both variants are reproduced from scratch in [tree_shap.ipynb](notebooks/tree_shap.ipynb) by walking a single LightGBM tree by hand, and the manual numbers match `shap` to floating-point precision.

## What output is being explained?

`TreeExplainer`'s `model_output` chooses the scale of the attribution:

- `"raw"` — the raw margin (sum of leaf outputs before any link transform). For LightGBM binary classification this is a **logit**.
- `"probability"` — the final predicted probability.

For `"probability"`, the sigmoid link is **non-linear**, so the value function becomes $f(S) = \mathbb{E}_z\!\big[\sigma(\text{raw}(x_S; z_{\sim S}))\big]$. The notebook shows that `shap` actually implements this as a **secant-rescaling** of the raw Shapley contributions per background reference:

$$
\phi^{\text{prob}}_i = \mathbb{E}_z\!\left[\, \phi^{\text{raw}}_i \cdot \frac{\sigma(\text{leaf}_x) - \sigma(\text{leaf}_z)}{\text{leaf}_x - \text{leaf}_z}\right]
$$

This splits each per-reference probability change $\sigma(\text{leaf}_x) - \sigma(\text{leaf}_z)$ among features in proportion to their raw contributions — and reproduces `shap`'s `model_output="probability"` values to ~1e-9.
