# Categorical Feature Support

LightGBM offers good accuracy with integer-encoded categorical features. LightGBM applies Fisher (1958) to find the optimal split over categories. This often performs better than one-hot encoding.

To get the categorical feature support, use `categorical_feature` to specify the categorical features at `lgb.Dataset(..., categorical_feature=['column_name'])`

For a categorical feature, at each tree node, LightGBM derives the optimal split by computing fresh gradient and Hessian statistics for each category using only the samples that reach that node. This process is repeated for every node in a tree and for every new tree in the boosting process.

## Algorithm
1. **Collect the samples reaching the current node**  
    Only the samples that arrive at this node are considered.
2. **Compute a score for each category**.  
    $$ Score = \frac{\sum_{i=1}^n gradient}{\sum_{i=1}^n hessian}$$
    where $n$ is the number of samples in the node

3. **Sort categories by the score**.  
    For example
    |category|sum_of_gradient|sum_of_hessian|Score|
    |:-------|--------------:|-------------:|----:|
    |Red     |            -20|            15|-1.33|
    |Green   |              8|            12|-1.50|
    |Blue    |            -15|            10| 0.67|
    |Yellow  |            -10|            14| 0.71|

4. **Find candidate splits between all adjacent categories**.  
    Instead of evaluating every possible split, it only considers contiguous partitions.
    ```
    Green | Red Blue Yellow
    Green Red | Blue Yellow
    Green Red Blue | Yellow
    ```

5. **Evaluate all candidate splits**   
    For each candidate split, computes the total gradients and Hessians of the left and right groups and then compute the gain for every candidate split

    $$
    \text{Gain} = \frac{1}{2}\left[ \underbrace{\frac{G_L^2}{H_L + \lambda}}_{\text{left leaf score}} + \underbrace{\frac{G_R^2}{H_R + \lambda}}_{\text{right leaf score}} - \underbrace{\frac{(G_L+G_R)^2}{H_L + H_R + \lambda}}_{\text{score if not split}} \right] - \gamma
    $$

    where 
    - $G_L, H_L$ and $G_R, H_R$ are the summed gradients and hessians of samples routed left and right by the candidate split, respectively.
    - $\lambda$ is the L2 regularization.
    - $\gamma$ is the minimum split penalty.

    |Split| gain|
    |---|---|
    |{Green}, {Red Blue Yellow}|2.81|
    |{Green Red}, {Blue Yellow}|5.46|
    |{Green Red Blue}, {Yellow}|3.12|

6. **Selects the split with the maximum information gain**      
    Since the second candidate has the highest gain, LightGBM chooses
    ```
    Split = {Green, Red}, {Blue, Yellow}
    ```
    as the split for this node.



## Summary
- It does not enumerate all $2^{k−1} −1$ possible subsets of k categories.
- It first derives a data-driven ordering of categories using the current node's gradient and Hessian statistics.
- It then evaluates only k−1 adjacent partitions.
- Finally, it selects the partition with the maximum information gain, using the same gain function as for numerical features.

## Notes
- When using `pandas.DataFrame` inputs with columns of dtype `category`, LightGBM will align categories to those observed during training before converting them to integer values. This ensures consistent encoding between training and prediction without additional preprocessing.
- At `predict()` time, categories not seen during training will be treated as missing values.
- Use `min_data_per_group`, `cat_smooth` to deal with over-fitting (when #data is small or #category is large).
- For a categorical feature with high cardinality (`#category` is large), it often works best to treat the feature as numeric, either by simply ignoring the categorical interpretation of the integers or by embedding the categories in a low-dimensional numeric space.