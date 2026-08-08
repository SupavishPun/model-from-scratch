# Missing Value Handle

Unlike many ML libraries, we don't have to fill the missing values. LightGBM will automatically determine whether missing values should go left or right for every split independently.


LightGBM enables the missing value handle by default with `use_missing=true` at `lgb.Dataset(..., use_missing=true)`

## Algorithm
```
For each node:  
    For each feature:  
        
        Build histogram and treat missing values as a bin
        hist <- histogram_bins()
        
        For each threshold:              
            1. Send missing values to LEFT node and compute gain
                left_gain <- gain(threshold, missing→left)

            2. Send missing values to RIGHT node and compute gain
                right_gain <- gain(threshold, missing→right)

            3. Compare gain and keep the direction with higher gain

        Keep the threshold and missing direction with the highest gain for this feature

    Compare the best split from every feature
    Choose the feature, threshold, and missing direction with the highest gain
```

## Note
- A missing value (NaN) is never used as a threshold for a split. Instead, it is treated as a special case that is routed to either the left or right child.
- A missing value may be informative if the value is Missing Not at Random (MNAR). The fact that a value is missing can be related to the outcome. For example, it can be related to the behavior of refusing to disclose sensitive information.