import warnings, os, sys, glob
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
MAX_DIST    = 12          # how many samples after a gap to profile


def naive_features(series):
    """Gap-ignoring feature construction (the common practice)."""
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


# part (a)
def describe(name, series, step):
    d = series.index.to_series().diff()
    gaps = d[d > step]
    seg = FP.segment_ids(series.index, step)
    span = (series.index[-1] - series.index[0])
    expected = int(span / step) + 1
    gl = (gaps / step).astype(int)      # gap length in samples
    return dict(
        dataset=name,
        n_samples=len(series),
        span_days=round(span.total_seconds() / 86400, 1),
        completeness_pct=round(100 * len(series) / expected, 2),
        mean_ms=round(series.mean(), 2),
        sd_ms=round(series.std(), 2),
        min_ms=round(series.min(), 2),
        max_ms=round(series.max(), 2),
        n_segments=int(seg.max() + 1),
        n_gaps=int(len(gaps)),
        rows_after_gap_pct=round(100 * len(gaps) / len(series), 2),
        gap_median_samples=int(gl.median()) if len(gl) else 0,
        gap_p95_samples=int(gl.quantile(0.95)) if len(gl) else 0,
        gap_max_samples=int(gl.max()) if len(gl) else 0,
        gap_median=str(gaps.median()) if len(gaps) else '-',
        gap_p95=str(gaps.quantile(0.95)) if len(gaps) else '-',
        gap_max=str(gaps.max()) if len(gaps) else '-')


# part (b)
def target_staleness(series, step, H):
    """For each row p, how much MORE time than H*step separates row p from
    row p+H. Zero means the target really is H steps ahead. A positive value
    means a gap falls in between, so the naive protocol is asking the
    persistence baseline to forecast far further ahead than it appears."""
    t = series.index.values.astype('datetime64[ns]').astype(np.int64)
    n = len(t)
    extra = np.full(n, -1, dtype=np.int64)
    expected = np.int64(H) * np.int64(step.value)
    extra[:n - H] = (t[H:] - t[:n - H]) - expected
    return extra                      # nanoseconds of excess; 0 = clean


def error_profile(name, series, step, seq_len, H, out):
    """Profile persistence vs model error by how stale the target is.

    Uses NAIVE features throughout, because these are exactly the rows the
    naive protocol retains and the gap-aware protocol discards. Gap-aware
    features are undefined near a gap by construction, so they cannot be
    used to study what happens there.
    """
    fnav = naive_features(series)
    fcols = list(fnav.columns)
    ok = fnav.notna().all(axis=1).to_numpy()

    extra = target_staleness(series, step, H)
    n = len(series)
    p = np.arange(n)
    cand = p[ok & (p + H < n) & (extra >= 0)]
    if len(cand) < 300:
        return

    # train on the earlier portion, profile on the later portion
    split_pos = cand[int(len(cand) * TRAIN_RATIO)]
    tr = cand[cand < split_pos]
    te = cand[cand >= split_pos]
    if len(tr) < 200 or len(te) < 200:
        return

    m = XGBRegressor(**XGB)
    m.fit(fnav.iloc[tr][fcols].values, series.values[tr + H])
    y = series.values[te + H]
    e_pers = y - series.values[te]
    e_mdl = y - m.predict(fnav.iloc[te][fcols].values)

    ex = extra[te] / np.int64(step.value)          # excess, in samples
    bins = [(0, 0, 'clean'), (1, 3, '1-3'), (4, 12, '4-12'),
            (13, 72, '13-72'), (73, 10**9, '>72')]
    for lo, hi, lab in bins:
        sel = (ex >= lo) & (ex <= hi)
        if sel.sum() < 15:
            continue
        rp = float(np.sqrt(np.mean(e_pers[sel] ** 2)))
        rm = float(np.sqrt(np.mean(e_mdl[sel] ** 2)))
        out.append(dict(dataset=name, horizon=H, staleness=lab,
                        n=int(sel.sum()),
                        rmse_persistence=round(rp, 4),
                        rmse_model=round(rm, 4),
                        skill=round(100 * (1 - rm / rp), 2)))


def load_all():
    ds = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        ds.append(('D1 ZephIR (contiguous)', s, pd.Timedelta('1min'), 60, (1, 10)))
    except Exception as e:
        print('[skip D1]', e)
    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        ds.append(('D2 WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'), 36, (1, 3)))
    except Exception as e:
        print('[skip D2]', e)
    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        ds.append(('D3 Onshore (many short gaps)', s, pd.Timedelta('10min'), 36, (1, 3)))
    except Exception as e:
        print('[skip D3]', e)
    return ds


if __name__ == '__main__':
    datasets = load_all()
    if not datasets:
        print('No datasets found.'); sys.exit(0)

    # ---------- (a) descriptive statistics ----------
    stats_rows = [describe(nm, s, st) for nm, s, st, _, _ in datasets]
    sdf = pd.DataFrame(stats_rows)
    sdf.to_csv('dataset_statistics.csv', index=False)
    print("\n" + "=" * 74)
    print("  DATASET CHARACTERISTICS")
    print("=" * 74)
    show = ['dataset', 'n_samples', 'span_days', 'completeness_pct', 'mean_ms',
            'sd_ms', 'min_ms', 'max_ms', 'n_segments', 'n_gaps',
            'rows_after_gap_pct', 'gap_median', 'gap_p95', 'gap_max']
    print(sdf[show].to_string(index=False))
    print('\nSaved dataset_statistics.csv')

    # overview figure: a representative week from each dataset
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(11, 2.5 * len(datasets)), squeeze=False)
    for ax, (nm, s, st, _, _) in zip(axes[:, 0], datasets):
        n = int(pd.Timedelta('7D') / st)
        mid = len(s) // 2
        seg = s.iloc[mid:mid + n]
        ax.plot(seg.index, seg.values, lw=0.7, color='#2E6DA4')
        # mark gaps
        d = seg.index.to_series().diff()
        for t in seg.index[(d > st).values]:
            ax.axvline(t, color='#B23A48', lw=0.8, alpha=0.7)
        ax.set_title(f'{nm} \u2014 one representative week '
                     f'(red lines mark gaps)', fontsize=9)
        ax.set_ylabel('m/s'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('dataset_overview_fig.png', dpi=150, bbox_inches='tight')
    print('Saved dataset_overview_fig.png')

    # ---------- (b) error vs distance from gap ----------
    prof = []
    print("\n" + "=" * 74)
    print("  ERROR VERSUS DISTANCE FROM GAP")
    print("=" * 74)
    for nm, s, st, sl, hz in datasets:
        for H in hz:
            error_profile(nm, s, st, sl, H, prof)
    if not prof:
        print('No profile rows produced.'); sys.exit(0)
    pdf = pd.DataFrame(prof)
    pdf.to_csv('error_vs_distance.csv', index=False)
    for (ds, H), g in pdf.groupby(['dataset', 'horizon']):
        print(f"\n  {ds}  H={H}")
        print(f"    {'stale':>7}{'n':>8}{'RMSE pers':>11}{'RMSE model':>12}{'skill':>9}")
        for _, r in g.iterrows():
            print(f"    {r['staleness']:>7}{int(r['n']):>8}{r['rmse_persistence']:>11.3f}"
                  f"{r['rmse_model']:>12.3f}{r['skill']:>8.2f}%")
    print('\nSaved error_vs_distance.csv')

    fig, axes = plt.subplots(1, pdf['dataset'].nunique(),
                             figsize=(5.6 * pdf['dataset'].nunique(), 4.3),
                             squeeze=False)
    for ax, (ds, g) in zip(axes[0], pdf.groupby('dataset', sort=False)):
        gg = g[g['horizon'] == g['horizon'].min()]
        x = np.arange(len(gg))
        ax.plot(x, gg['rmse_persistence'], marker='o',
                label='Persistence', color='#8d99ae')
        ax.plot(x, gg['rmse_model'], marker='s',
                label='XGBoost', color='#2E6DA4')
        ax.set_xticks(x); ax.set_xticklabels(gg['staleness'])
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel('Extra samples between row and its target')
        ax.set_ylabel('RMSE (m/s)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Forecast error by target staleness',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('error_vs_distance_fig.png', dpi=150, bbox_inches='tight')
    print('Saved error_vs_distance_fig.png')
