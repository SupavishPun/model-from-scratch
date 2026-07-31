# Gradient boosting

In gradient boosting, we build an additive model:

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$

where $f_t$ is the new tree added at boosting round $t$, and $\hat{y}_i^{(t-1)}$ is the prediction from all previous trees combined. 

We want to choose $f_t$ to minimize the objective:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i,\ \hat{y}_i^{(t)}\right) = \sum_{i=1}^n l\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(x_i)\right)
$$

where $l$ is a differentiable convex loss function e.g. squared error, log loss

## Taylor Expansion
For general loss functions (not just squared error), $l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))$ has no simple closed form in terms of $f_t$. To make the optimization tractable for **any** differentiable loss, we approximate it using a **second-order Taylor expansion** around the current prediction $\hat{y}_i^{(t-1)}$.

Recall the Taylor expansion of $f(x + \Delta x)$ around $x$:

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2
$$

Here, treat $\hat{y}_i^{(t-1)}$ as the fixed point $x$ and $f_t(x_i)$ as the small increment $\Delta x$. Applying this to the loss function:

$$
l\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(x_i)\right) \approx l\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

where:

$$
g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}} \quad \text{(first-order gradient)}
$$

$$
h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2} \quad \text{(second-order gradient, i.e. Hessian)}
$$

These $g_i$ and $h_i$ can be computed for **any** twice-differentiable loss function — this generality is exactly why the Taylor expansion is used.

## Simplifying the Objective

Since $l(y_i, \hat{y}_i^{(t-1)})$ is a constant with respect to $f_t$ (it doesn't depend on the new tree), we can drop it. The objective to minimize becomes:

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right]
$$

## Tree Structure

A regression tree $f_t$ can be written as a function that assigns each sample $x_i$ to a leaf, and each leaf $j$ has output value $w_j$:

$$
f_t(x_i) = w_{q(x_i)}
$$

where $q(x_i)$ maps a sample to its leaf index.

Let $I_j = \{ i \mid q(x_i) = j \}$ be the set of sample indices falling into leaf $j$. We can regroup the objective **by leaf** instead of by sample:

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i\right) w_j^2 \right]
$$

Define the aggregated gradient and Hessian for leaf $j$:

$$
G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i
$$

So the objective simplifies to a sum of independent per-leaf quadratics:

$$
\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[ G_j w_j + \frac{1}{2}H_j w_j^2 \right]
$$

## Optimal Leaf Output

Taking the derivative with respect to $w_j$ and setting it to zero:

$$
\frac{\partial}{\partial w_j}\left[ G_j w_j + \frac{1}{2}H_j w_j^2 \right] = G_j + H_j w_j = 0
$$

Solving gives the **optimal leaf output**:

$$
\boxed{w_j^* = -\frac{G_j}{H_j}}
$$

### Intuition

- $G_j$ (sum of gradients in the leaf) tells you the direction and magnitude of "how wrong" the current predictions are on average for samples in that leaf — the leaf output moves opposite to this direction (note the minus sign).
- $H_j$ (sum of Hessians) tells you how confident/curved the loss is — larger $H_j$ means predictions should move less per unit of gradient (more caution), acting like an adaptive learning rate per leaf.

## Squared Error Loss (Regression)

For $l(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2$:

$$
g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}} = \frac{\partial (\frac{1}{2}(y_i - \hat{y}_i^{(t-1)})^2)}{\partial \hat{y}_i^{(t-1)}} = \hat{y}_i^{(t-1)} - y_i
$$

$$
h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2} = \frac{\partial (\hat{y}_i^{(t-1)} - y_i)}{\partial \hat{y}_i^{(t-1)}} = 1
$$

Substituting into the formula:

$$
w_j^* = -\frac{\sum_{i \in I_j} (\hat{y}_i^{(t-1)} - y_i)}{|I_j|} = \frac{\sum_{i \in I_j}(y_i - \hat{y}_i^{(t-1)})}{|I_j|}
$$

## Log Loss (Binary Classification)

For binary classification, $y_i \in \{0, 1\}$, and the raw model output $\hat{y}_i^{(t-1)}$ is a **logit** (pre-sigmoid score). Predictions are converted to probabilities via the sigmoid function:

$$
p_i = \sigma(\hat{y}_i^{(t-1)}) = \frac{1}{1 + e^{-\hat{y}_i^{(t-1)}}}
$$

The log loss (binary cross-entropy) is:

$$
l(y_i, \hat{y}_i^{(t-1)}) = -\Big[ y_i \log p_i + (1-y_i)\log(1-p_i) \Big]
$$

**First derivative (gradient).** Differentiating with respect to $\hat{y}_i^{(t-1)}$ and using $\frac{d\sigma}{dz} = \sigma(z)(1-\sigma(z))$, the log terms simplify nicely to:

$$
g_i = \frac{\partial l}{\partial \hat{y}_i^{(t-1)}} = p_i - y_i
$$

**Second derivative (Hessian).** Differentiating again:

$$
h_i = \frac{\partial^2 l}{\partial (\hat{y}_i^{(t-1)})^2} = p_i (1 - p_i)
$$

**Optimal leaf weight.** Plugging $g_i$ and $h_i$ into the general formula:

$$
w_j^* = -\frac{G_j}{H_j} = \frac{\sum_{i \in I_j} (y_i - p_i)}{\sum_{i \in I_j} p_i(1-p_i)}
$$

Unlike squared error, $w_j^*$ here is a **logit adjustment**, not a direct probability. It gets added to the running score $\hat{y}_i^{(t-1)}$, and the sigmoid is reapplied afterward to get an updated probability. 

### Intuitively
- If most samples in the leaf are underpredicted ($y_i=1$ but $p_i$ small), $G_j$ is negative, so $w_j^*$ is positive — pushing the logit up, increasing predicted probability.
- If most samples are overpredicted ($y_i=0$ but $p_i$ large), $G_j$ is positive, so $w_j^*$ is negative — pushing the logit down.
- Leaves where the model is already confident and correct contribute small $|g_i|$ and small $h_i$, so they get small, cautious updates relative to leaves where the model is uncertain.


## Learner
The derivation above tells us the *optimal output value* $w_j^*$ for a leaf, **given that we already know which samples fall into that leaf**. But the tree itself — the split points that decide $q(x_i)$, i.e. which leaf each $x_i$ lands in — still needs to be learned.

Put another way, $w_j^*$ is only computed *after* a candidate tree structure exists; it is not directly predicted from $x_i$. The role of the learning algorithm is to search over possible tree structures (splits on features of $x_i$) so that samples with similar $(g_i, h_i)$ — and therefore similar optimal output — get grouped into the same leaf.