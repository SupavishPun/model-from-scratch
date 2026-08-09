# L1 Objective

## Loss Function

For LightGBM's L1 objective (mean absolute error), the loss for a single sample is:

$$L(y, F) = |y - F|$$

## Gradient

The gradient (first derivative with respect to the prediction $F$) is the sign function:

$$g = \frac{\partial L}{\partial F} = \text{sign}(F - y) = \begin{cases} +1 & \text{if } F > y \\ -1 & \text{if } F < y \\ 0 & \text{if } F = y \end{cases}$$

## Hessian

The true second derivative of $|F - y|$ is 0 everywhere except at $F = y$, where it's undefined (a discontinuity in the gradient). Since a hessian of 0 would break LightGBM's tree-building formulas (which divide by the hessian sum when computing leaf values and split gains), LightGBM instead uses a constant substitute:

$$h = 1$$

## Practical Implications

- Because the hessian is constant, leaf values under L1 loss aren't computed the usual Newton-step way (gradient sum divided by hessian sum). Instead, LightGBM computes the optimal leaf value as the **weighted median** of the residuals in that leaf, which is the closed-form minimizer of MAE.
- Split finding still uses the gradient/hessian statistics as usual, but because the hessian carries no real curvature information (it's just 1 for every sample), gain calculations are effectively driven by the sign-based gradients alone, so splits are chosen based on how well they separate points with different residual signs/magnitudes rather than true second-order information.
- This is also why L1 loss tends to train less smoothly than L2: the constant-magnitude gradient means every misprediction pushes with equal force regardless of how far off it is, so LightGBM often benefits from more boosting rounds or a smaller learning rate when using `objective: mae` or `regression_l1`.
