# Gradient Boosting From Scratch
Learn how gradient boosting works under the hood
- Classification with the `GradientBoostingClassifier` class.
- Regression with the `GradientBoostingRegressor` class.

## How Gradient Boosting Works

1. **Initialize with a Baseline Prediction ($F_0$)**  
   Start with a constant prediction that minimizes the overall loss function.
   For **squared-error loss**, this constant is the **mean of the target values**:

   $$F_0 = \frac{1}{n}\sum_{i=1}^{n} y_i$$

2. **Iteratively Train Decision Trees**  
   At each iteration:

   * **Calculate Pseudo-Residuals:** Compute the negative gradient of the loss function with respect to the current predictions.
   $$ PseudoResiduals_i = - (predict_i - observe_i) = observe_i - predict_i $$
   
   * **Fit a New Tree:** Train a decision tree to predict these pseudo-residuals.
   * **Optimize Leaf Outputs:** Calculate the optimal output value for each leaf. For a second-order optimization, the leaf value is typically calculated as:

     $$w_j = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i}$$
     $$w_j = \frac{\sum_{i \in I_j} -g_i}{\sum_{i \in I_j} h_i}$$


     where $g_i$ is the gradient, $h_i$ is the Hessian, and $I_j$ is the set of samples belonging to leaf $j$.
   * **Update Predictions:** Add the new tree's scaled leaf outputs to the current predictions:

     $$F_m(x) = F_{m-1}(x) + \eta f_m(x)$$

     where $\eta$ is the **learning rate** and $f_m(x)$ is the newly trained tree.

3. **Repeat Until the Stopping Condition Is Met**  
   Continue adding trees until reaching the maximum number of iterations, or until another stopping condition such as early stopping is triggered.
