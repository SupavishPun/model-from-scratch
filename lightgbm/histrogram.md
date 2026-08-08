# Histogram-Based Algorithm

The main cost in GBDT lies in learning the decision trees, and the most time-consuming part in
learning a decision tree is to find the best split points.

One of popular solutions is the **histogram-based algorithm**. it is efficient in both memory consumption and training speed.

## How it work
Histogram-based algorithm buckets continuous feature values into discrete bins and uses these
bins to construct feature histograms during training

### Step 1. Bin upfront (once, at dataset loading time)
For each feature, it is discretized into at most max_bin=(default=255) bins and computes a fixed set of bin edges (candidate thresholds)

### Step 2. Histogram Building (per node during tree growth)
For the subset of samples that reach that node, the algorithm tallies gradients and Hessians into buckets according to each sample's already-computed bin index from step1. (the bin edges are not recomputed) This produces a histogram of shape [num_bins] per feature, per node.


## Histogram-building parameters

These are the LightGBM parameters that govern histogram construction (defaults shown):

| parameter | default | role in histogram building |
|---|---|---|
| `max_bin` | `255` | max number of bins per feature; the histogram for one (feature, node) has at most this many slots |
| `min_data_in_bin` | `3` | while discretizing, a bin must collect at least this many samples before a bin edge is opened |
| `min_data_in_leaf` | `20` | a candidate split is rejected unless both leaves keep $\ge$ this many samples (checked during the histogram scan) |
| `min_sum_hessian_in_leaf` | `1e-3` | same constraint, in units of summed Hessian |
| `histogram_pool_size` | `-1` | max MB used to cache node histograms (`-1` $\approx$ half of system memory); histograms for every leaf of every tree are large, so they are pooled |
| `num_threads` | `0` | OpenMP threads used to build histograms (`0` = all cores) |
| `force_col_wise` | `false` | force the column-major histogram construction strategy |
| `force_row_wise` | `false` | force the row-major histogram construction strategy |
| `use_missing` | `true` | reserve a bin so missing values get their own histogram slot |


## force_col_wise vs force_row_wise

Both flags control the **order** in which the per-node histogram is filled, so they only affect speed and memory, never the tree. 
The histogram itself is **identical** either way.

- **col-wise** (column-major): build **one feature at a time**. For a given feature, walk that
  feature's column of bin indices and add each sample's $(g_i,h_i)$ into that feature's histogram.
  Threads are split **across features**, so each thread owns whole histograms and never contends
  with the others.
  
- **row-wise** (row-major): build **all features at once**. Walk the data row by row and add each
  sample's $(g_i,h_i)$ into **every** feature's histogram simultaneously, using that row's bin index
  per feature. Threads are split **across rows**, so they write to shared histograms — each thread
  keeps a private histogram copy that is summed (reduced) at the end.

When neither flag is set, LightGBM times both strategies on a small sample and **keeps the faster one.**

## Notes
- There is no public Python API method that returns the full array of bin edge values for each feature.