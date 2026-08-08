# Optuna From Scratch

Learn how Optuna tunes hyperparameters under the hood.
- Hyperparameter tuning of a [LightGBM](../lightgbm/README.md) regressor with `optuna.samplers.TPESampler`.

**Optuna** is a framework for that search, and **TPE (Tree-structured Parzen Estimator)** is its default sampler.

## The optimization loop

Optuna frames tuning as a black-box optimization. You write an **objective** function that takes a `trial`, samples hyperparameters from a search space, trains/evaluates, and returns a score:

```python
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    num_leaves   = trial.suggest_int("num_leaves", 31, 127)
    ...                                  # train LightGBM, return validation score
    return score
```

Then `study.optimize(objective, n_trials=20)` is, at its core, a loop:

1. **`study.ask()`** — the *sampler* proposes the next hyperparameters, using every trial that has finished so far.
2. Run the **objective** with those parameters to get a score.
3. **`study.tell(trial, score)`** — store the result so the sampler sees one more completed trial next iteration.

All the intelligence lives in the **sampler** (`TPESampler`), which decides the parameters at `ask()` time. The objective is a black box that returns a number.

## How TPE proposes the next hyperparameters

Because `multivariate=False` (the default), TPE samples **each parameter independently**. For one parameter it runs two phases.

### Phase 1 — pure random search (trials 0 … `n_startup_trials`−1)

TPE cannot fit any probability model until it has `n_startup_trials` (= 10) finished trials to look at, so it delegates to a plain `RandomSampler`. To draw an integer on a grid `[low, high]` with step `s`, it widens the interval by half a step on each side, draws one uniform number, then snaps it back onto the grid:

$$x = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{u - \mathrm{low}}{s}\right)\!\cdot s + \mathrm{low},\ \mathrm{low},\ \mathrm{high}\right),\qquad u \sim \mathrm{Uniform}\!\left(\mathrm{low}-\tfrac{s}{2},\ \mathrm{high}+\tfrac{s}{2}\right)$$

Widening by `s/2` matters: drawing from `[low, high]` and rounding would give the two endpoints only half a bin's width each, so they'd be sampled at ~50% the rate of interior points. Shifting the bounds out gives every grid value an equal-width bin → genuinely uniform over the discrete set. These startup parameters depend only on the RNG seed, **not** on the objective's values, so trials 0–9 are identical on every run.

### Phase 2 — TPE (trials `n_startup_trials`+)

From trial 10 on, TPE takes over. It uses **all** past completed trials, split into two groups:

1. **Split** the completed trials by score into **below** (the `gamma(n)` best — the "good" set) and **above** (every other trial — the "bad" set), where

$$\gamma(n) = \min\!\big(\lceil 0.1\,n \rceil,\ 25\big)$$

   So `l(x)` is fed at most the 25 best trials, while `g(x)` absorbs **all** the rest (no trial is ever dropped — there is no sliding window).

2. **Build $l(x)$** — a Gaussian-mixture (Parzen / KDE) density over the *good* values — and **$g(x)$** over the *bad* values. Each mixture also gets one extra "prior" component spread across the whole range for stability.

3. **Sample `n_ei_candidates` (= 24)** candidate values from $l(x)$.

4. **Score** each candidate by Expected Improvement,

$$\mathrm{EI}(x) = \log l(x) - \log g(x) = \log\!\frac{l(x)}{g(x)}$$

   large where the candidate is likely under the good model *and* unlikely under the bad model.

5. **Return** the candidate with the largest EI.

The whole point of the ratio $l(x)/g(x)$ (rather than just $l(x)$) is to steer toward regions that are good **and** under-explored: a value that is likely under $l$ *but also likely under $g$* (already well represented among bad trials) is penalised.

## How many trials should you run?

TPE has no fixed "right" number of trials — the best value improves quickly and then **plateaus**, and where the plateau sits is problem-dependent. Reference points from the mechanics above:

- **Trials 1–10** are pure random search — TPE isn't running yet.
- **~250 trials** is where `l(x)` reaches its full size of 25 (the `gamma` cap) and stops growing.
- For a **small** search space (2 parameters, like here), the optimum is tiny and is hit long before TPE is fully mature.

So instead of guessing a budget, run slightly more than you think you need, plot `best_value` over trials, and **stop when it flattens**. A small callback that calls `study.stop()` after `patience` consecutive non-improvements makes this automatic:

```python
class StopWhenNoImprovement:
    def __init__(self, patience): self.patience = patience; self._best = float("inf"); self._since = 0
    def __call__(self, study, trial):
        if study.best_value < self._best: self._best, self._since = study.best_value, 0
        else: self._since += 1
        if self._since >= self.patience: study.stop()

study.optimize(objective, n_trials=1000, callbacks=[StopWhenNoImprovement(20)])
```

> ⚠️ More trials can also **hurt**: each extra trial is another chance to exploit noise in the validation set (selection overfitting). The larger the search budget, the more you need a separate hold-out set to trust the reported best.