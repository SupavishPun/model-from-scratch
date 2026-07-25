# Decision Tree From Scratch

A Decision Tree is a machine learning model that makes decisions by asking a sequence of simple **yes/no questions** such as whether a customer is likely to have a product or not. Each question is chosen because it best separates customers with different outcomes based on historical data. The process continues until the model reaches a final prediction

In this repository, we will learn how to build a decision tree from Scikit-learn
- Classification with the `DecisionTreeClassifier` class.
- Regression with the `DecisionTreeRegressor` class.

## How a Decision Tree Is Built (High-Level Overview)
The learning process is a recursive process that picks the best split at each level for building the tree until the tree is exhausted or a user-defined criterion (such as maximum tree depth) is reached.

1. Identify all possible splits for each feature.
2. For each split, evaluate it using a criterion such as Gini impurity (for classification) or Mean Squared Error (MSE) (for regression).
3. Select the split with the best score.
4. Recursively repeat the process for each child node until a stopping criterion is met.