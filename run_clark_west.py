import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
import run_artifact_aligned as RA


def clark_west(y, f_null, f_alt, h=1):
    """One-sided Clark-West test. Returns (statistic, p-value, mean adjusted loss)."""
    y, f1, f2 = (np.asarray(v, float) for v in (y, f_null, f_alt))
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    T = len(adj)
    mu = adj.mean()
    dc = adj - mu
    L = min(FP.dm_lag(T, h), T - 1)
    v = dc @ dc / T
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / T
    v = max(v, 1e-12)
    stat = mu / np.sqrt(v / T)
    return float(stat), float(stats.norm.sf(stat)), float(mu)


def run(name, series, step, seq_len, horizons, out):
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = FP.naive_features(series)
    for H in horizons:
        sp = RA.gap_aware_split(series, fgap, H, seq_len)
        if sp is None:
            print(f"  H={H}: too few gap-valid rows, skipped")
            continue
        rows, tr, te = sp
        # A: gap-aware arm (same code as run_artifact_aligned.run)
        y_te = series.values[te + H]
        y_now = fgap.iloc[te]['lag_0'].values
        m = XGBRegressor(**RA.XGB)
        m.fit(fgap.iloc[tr][gcols].values, series.values[tr + H])
        pred_a = m.predict(fgap.iloc[te][gcols].values)
        cw_a = clark_west(y_te, y_now, pred_a, h=H)
        # C: naive arm on A's calendar window
        tr2, te2, fc = RA.naive_aligned_split(series, H, tr, te, fnav)
        m = XGBRegressor(**RA.XGB)
        m.fit(tr2[fc].values, tr2['target'].values)
        pred_c = m.predict(te2[fc].values)
        cw_c = clark_west(te2['target'].values, te2['lag_0'].values, pred_c, h=H)
        print(f"  H={H:>2} | A CW stat {cw_a[0]:+6.2f} p={cw_a[1]:.4f} | "
              f"C CW stat {cw_c[0]:+6.2f} p={cw_c[1]:.4f}")
        out.append(dict(dataset=name, horizon=H,
                        cw_stat_honest=round(cw_a[0], 3), cw_p_honest=float(f'{cw_a[1]:.3g}'),
                        cw_stat_naive=round(cw_c[0], 3), cw_p_naive=float(f'{cw_c[1]:.3g}'),
                        n_test_honest=len(te), n_test_naive=len(te2)))


if __name__ == '__main__':
    out = []
    RA.run_all(run, out)
    if not out:
        print('No results.')
        sys.exit(0)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(HERE, 'clark_west.csv'), index=False)
    print('\nSaved clark_west.csv')
    print(df.to_string(index=False))
