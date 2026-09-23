"""
=============================================================================
  run_artifact_study.py — Evaluation-artifact experiment
=============================================================================
  Quantifies how much reported forecast skill changes when gappy wind data
  is handled naively (the common practice) versus gap-aware (correct).

  Three configurations per dataset and horizon:
    A  honest    : gap-aware features, gap-aware valid rows
    B  features  : naive features, SAME gap-aware rows
                   -> isolates the FEATURE-CONTAMINATION effect
    C  naive     : naive features, naive rows (what a naive study reports)
                   -> C - B isolates the EVALUATION-SET COMPOSITION effect

  Reports the decomposition so the mechanism is explicit, not asserted.

  Requires: final_pipeline.py, dataset_adapters.py in the same folder.
  Run:      python run_artifact_study.py
  Outputs:  artifact_results.csv, artifact_fig.png
=============================================================================
"""
import warnings, os, sys, glob
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75


def naive_features(series):
    """Feature construction that IGNORES gaps — the common practice.
    Lags and rolling windows are taken by row position, so a window may
    silently span a multi-day outage."""
    f = pd.DataFrame(index=series.index)
    f['lag_0'] = series
    for lag in (1, 2, 3, 5, 10, 30, 60):
        f[f'lag_{lag}'] = series.shift(lag)
    for w in (5, 15, 60):
        f[f'roll_mean_{w}'] = series.rolling(w).mean()
        f[f'roll_std_{w}'] = series.rolling(w).std()
    f['wind_shear'] = series.diff()
    f['turbulence'] = series.rolling(30).std() / (series.rolling(30).mean() + 1e-9)
    f['kalman'] = FP.kalman_causal(series.values)
    f['hour_sin'] = np.sin(2 * np.pi * f.index.hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f.index.hour / 24)
    return f


def gap_profile(series, step):
    """Descriptive statistics of the gap structure."""
    d = series.index.to_series().diff()
    big = d[d > step]
    return dict(
        n_segments=int(FP.segment_ids(series.index, step).max() + 1),
        n_gaps=int(len(big)),
        pct_rows_after_gap=float(100 * len(big) / len(series)),
        median_gap=str(big.median()) if len(big) else '-',
        max_gap=str(big.max()) if len(big) else '-')


def run_one(name, series, step, seq_len, horizons, rows_out):
    prof = gap_profile(series, step)
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    print(f"  samples {len(series)} | segments {prof['n_segments']} | "
          f"gaps {prof['n_gaps']} ({prof['pct_rows_after_gap']:.1f}% of rows) | "
          f"median gap {prof['median_gap']} | max {prof['max_gap']}")

    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = naive_features(series)
    ncols = list(fnav.columns)

    print(f"  {'H':>4} {'run':>4} {'n_test':>8} {'Pers':>8} {'XGB':>8} {'skill':>9}")
    print("  " + "-" * 48)

    for H in horizons:
        mask = FP.valid_rows_for_horizon(series, fgap, H, seq_len)
        rows = np.where(mask)[0]
        if len(rows) < 800:
            continue
        split = int(len(rows) * TRAIN_RATIO)
        EMB = seq_len + H
        tr, te = rows[:split], rows[split + EMB:]
        if len(te) < 300:
            continue

        y_te = series.values[te + H]
        y_now = fgap.iloc[te]['lag_0'].values
        rp = float(np.sqrt(mean_squared_error(y_te, y_now)))

        # --- A: honest
        m = XGBRegressor(**XGB)
        m.fit(fgap.iloc[tr][gcols].values, series.values[tr + H])
        ra = float(np.sqrt(mean_squared_error(
            y_te, m.predict(fgap.iloc[te][gcols].values))))
        sk_a = 100 * (1 - ra / rp)

        # --- B: naive features, same rows (feature effect only)
        m = XGBRegressor(**XGB)
        m.fit(fnav.iloc[tr][ncols].values, series.values[tr + H])
        rb = float(np.sqrt(mean_squared_error(
            y_te, m.predict(fnav.iloc[te][ncols].values))))
        sk_b = 100 * (1 - rb / rp)

        # --- C: fully naive
        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2, te2 = dn.iloc[:s2], dn.iloc[s2:]
        m = XGBRegressor(**XGB)
        m.fit(tr2[fc].values, tr2['target'].values)
        yt2 = te2['target'].values
        yn2 = te2['lag_0'].values
        rpc = float(np.sqrt(mean_squared_error(yt2, yn2)))
        rc = float(np.sqrt(mean_squared_error(yt2, m.predict(te2[fc].values))))
        sk_c = 100 * (1 - rc / rpc)

        for lab, n, p_, x_, sk in (('A', len(te), rp, ra, sk_a),
                                   ('B', len(te), rp, rb, sk_b),
                                   ('C', len(te2), rpc, rc, sk_c)):
            print(f"  {H:>4} {lab:>4} {n:>8} {p_:>8.3f} {x_:>8.3f} {sk:>8.1f}%")
        print(f"       total bias C-A = {sk_c-sk_a:+.1f} pts | "
              f"feature {sk_b-sk_a:+.1f} | composition {sk_c-sk_b:+.1f}")

        rows_out.append(dict(
            dataset=name, segments=prof['n_segments'],
            pct_rows_after_gap=round(prof['pct_rows_after_gap'], 2),
            horizon=H,
            skill_honest=round(sk_a, 2), skill_naive_features=round(sk_b, 2),
            skill_naive=round(sk_c, 2),
            bias_total=round(sk_c - sk_a, 2),
            bias_feature=round(sk_b - sk_a, 2),
            bias_composition=round(sk_c - sk_b, 2),
            sign_flip=bool((sk_a < 0) != (sk_c < 0)),
            n_test_honest=len(te), n_test_naive=len(te2)))


if __name__ == '__main__':
    out = []

    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        run_one('ZephIR300 (contiguous)', s, pd.Timedelta('1min'), 60,
                (1, 10, 30), out)
    except FileNotFoundError:
        print('[skip] ZephIR CSVs not found')

    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        run_one('WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'), 36,
                (1, 3, 6), out)
    except FileNotFoundError:
        print('[skip] buoy_data/*.nc not found')

    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f)
        d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run_one('Onshore turbine (many short gaps)', s, pd.Timedelta('10min'),
                36, (1, 3, 6), out)
    except (IndexError, FileNotFoundError):
        print('[skip] onshore CSV not found')

    if out:
        df = pd.DataFrame(out)
        df.to_csv('artifact_results.csv', index=False)
        print(f"\nSaved artifact_results.csv ({len(df)} rows)")
        print(f"Sign reversals: {int(df['sign_flip'].sum())} of {len(df)} cases")
        print(f"Largest absolute bias: {df['bias_total'].abs().max():.1f} points")

        fig, axes = plt.subplots(1, df['dataset'].nunique(),
                                 figsize=(5.2 * df['dataset'].nunique(), 4.2),
                                 squeeze=False)
        for ax, (ds, g) in zip(axes[0], df.groupby('dataset', sort=False)):
            x = np.arange(len(g))
            ax.bar(x - 0.2, g['skill_honest'], 0.4, label='gap-aware (honest)',
                   color='#1D9E75')
            ax.bar(x + 0.2, g['skill_naive'], 0.4, label='naive',
                   color='#D85A30')
            ax.axhline(0, color='k', lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"H={h}" for h in g['horizon']])
            ax.set_ylabel('Skill vs persistence (%)')
            ax.set_title(f"{ds}\n({g['segments'].iloc[0]} segments)", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis='y')
        fig.suptitle('Reported skill under naive vs gap-aware evaluation',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig('artifact_fig.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved artifact_fig.png')
