"""Single source of truth for settings shared by every script.

Changing a value here changes it in every table, so the tables cannot drift
apart. Values are exactly those used for the numbers in the manuscript.
"""

SEED = 42
TRAIN_RATIO = 0.75          # chronological train fraction of valid origins

# XGBoost hyperparameters used for every XGBoost result in the paper.
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, verbosity=0)


def xgb_params(**overrides):
    """Copy of XGB_PARAMS with overrides (e.g. n_jobs=4, random_state=7).
    n_jobs only affects speed, not results."""
    return {**XGB_PARAMS, **overrides}
