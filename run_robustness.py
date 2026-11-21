"""Robustness of the Table 3 bias to XGBoost's randomness and hyperparameters.

Re-estimates the headline quantity (naive skill C minus gap-aware skill A, in
percentage points, calendar-aligned windows) three ways:

  1. SEEDS : the base XGBoost configuration refit with N_SEEDS different
             random_state values (subsample / colsample are stochastic).
  2. GRID  : a small fixed hyperparameter grid, base seed.
  3. TUNED : hyperparameters chosen on a chronological validation block taken
             from the END of the gap-aware training set (with the usual
             seq_len + H embargo), then applied unchanged to BOTH arms, so
             tuning cannot favour either protocol.

Outputs robustness_seeds.csv, robustness_grid.csv, robustness_summary.csv.
Set ROB_SEEDS (default 10) to change the number of seeds.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
import run_artifact_aligned as RA
from config import xgb_params

N_SEEDS = int(os.environ.get('ROB_SEEDS', 10))
GRID = [dict(max_depth=d, n_estimators=n, learning_rate=lr)
        for d, n, lr in itertools.product((3, 6, 9), (150, 300), (0.05, 0.1))
        if not (n == 300 and lr == 0.1)]       # drop the slowest corner
VAL_FRAC = 0.2


def fit_pred(params, Xtr, ytr, Xte):
    m = XGBRegressor(**xgb_params(n_jobs=4, **params))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def arms(series, fgap, fnav, gcols, ncols, H, seq_len, params):
    """Return (skill_A, skill_C) for one parameter set, or None."""
    sp = RA.gap_aware_split(series, fgap, H, seq_len)
    if sp is None:
        return None
    _, tr, te = sp
    y_te = series.values[te + H]
    y_now = fgap.iloc[te]['lag_0'].values
    pa = fit_pred(params, fgap.iloc[tr][gcols].values, series.values[tr + H],
                  fgap.iloc[te][gcols].values)
    sk_a = RA.skill_from_err(y_te - pa, y_te - y_now)
    tr2, te2, fc = RA.naive_aligned_split(series, H, tr, te, fnav)
    pc = fit_pred(params, tr2[fc].values, tr2['target'].values, te2[fc].values)
    yt2, yn2 = te2['target'].values, te2['lag_0'].values
    sk_c = RA.skill_from_err(yt2 - pc, yt2 - yn2)
    return sk_a, sk_c


def tune(series, fgap, gcols, H, seq_len):
    """Pick the grid entry with the lowest honest validation RMSE."""
    _, tr, _ = RA.gap_aware_split(series, fgap, H, seq_len)
    cut = int(len(tr) * (1 - VAL_FRAC))
    fit_rows, val_rows = tr[:cut], tr[cut + seq_len + H:]
    best, best_r = None, np.inf
    for p in GRID:
        pred = fit_pred(p, fgap.iloc[fit_rows][gcols].values,
                        series.values[fit_rows + H],
                        fgap.iloc[val_rows][gcols].values)
        r = float(np.sqrt(np.mean((series.values[val_rows + H] - pred) ** 2)))
        if r < best_r:
            best, best_r = p, r
    return best


def runner(name, series, step, seq_len, horizons, out):
    print(f"\n=== {name}", flush=True)
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = FP.naive_features(series)
    ncols = list(fnav.columns)
    for H in horizons:
        base = arms(series, fgap, fnav, gcols, ncols, H, seq_len, {})
        if base is None:
            continue
        rows = []
        for seed in range(N_SEEDS):
            a, c = arms(series, fgap, fnav, gcols, ncols, H, seq_len,
                        dict(random_state=seed))
            out['seeds'].append(dict(dataset=name, horizon=H, seed=seed,
                                     skill_honest=a, skill_naive=c, bias=c - a))
            rows.append(c - a)
        for p in GRID:
            a, c = arms(series, fgap, fnav, gcols, ncols, H, seq_len, p)
            out['grid'].append(dict(dataset=name, horizon=H, **p,
                                    skill_honest=a, skill_naive=c, bias=c - a))
        tp = tune(series, fgap, gcols, H, seq_len)
        a, c = arms(series, fgap, fnav, gcols, ncols, H, seq_len, tp)
        g = [r for r in out['grid'] if r['dataset'] == name and r['horizon'] == H]
        gb = np.array([r['bias'] for r in g])
        out['summary'].append(dict(
            dataset=name, horizon=H,
            bias_base=round(base[1] - base[0], 2),
            seed_mean=round(np.mean(rows), 2), seed_sd=round(np.std(rows, ddof=1), 2),
            seed_min=round(min(rows), 2), seed_max=round(max(rows), 2),
            seeds_same_sign=bool(np.all(np.sign(rows) == np.sign(rows[0]))),
            grid_min=round(gb.min(), 2), grid_max=round(gb.max(), 2),
            tuned_bias=round(c - a, 2), tuned_skill_honest=round(a, 2),
            tuned_skill_naive=round(c, 2),
            tuned_params=str({k: tp[k] for k in ('max_depth', 'n_estimators', 'learning_rate')})))
        s = out['summary'][-1]
        print(f"  H={H}: base {s['bias_base']:+.2f} | seeds {s['seed_mean']:+.2f} "
              f"+/- {s['seed_sd']:.2f} [{s['seed_min']:+.2f},{s['seed_max']:+.2f}] | "
              f"grid [{s['grid_min']:+.2f},{s['grid_max']:+.2f}] | "
              f"tuned {s['tuned_bias']:+.2f}", flush=True)


if __name__ == '__main__':
    out = dict(seeds=[], grid=[], summary=[])
    RA.run_all(runner, out)
    for k, fn in (('seeds', 'robustness_seeds.csv'), ('grid', 'robustness_grid.csv'),
                  ('summary', 'robustness_summary.csv')):
        pd.DataFrame(out[k]).round(4).to_csv(os.path.join(HERE, fn), index=False)
    print("\nSaved robustness_seeds.csv, robustness_grid.csv, robustness_summary.csv")
