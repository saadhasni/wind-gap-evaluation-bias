import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import final_pipeline as FP
import run_artifact_aligned as RA

WINDOWS = (15, 30, 60)
LAGS = (1, 2, 3, 5, 10, 30, 60)
ROLLS = (5, 15, 60)
EMBARGO_LEN = 36            # as in the paper for D2 and D3
RA.N_BOOT = int(os.environ.get('WD_NBOOT', 1000))
OUT_CSV = os.path.join(HERE, 'window_depth_sensitivity.csv')


def lag_set(W):
    return sorted({k for k in LAGS if k <= W} | {W})


def build_segmented(series, step, W):
    seg = FP.segment_ids(series.index, step)
    df = pd.DataFrame(index=series.index)
    df['_seg'] = seg
    g = series.groupby(seg)
    df['lag_0'] = series
    for lag in lag_set(W):
        df[f'lag_{lag}'] = g.shift(lag)
    for w in [w for w in ROLLS if w <= W]:
        df[f'roll_mean_{w}'] = g.transform(lambda s: s.rolling(w).mean())
        df[f'roll_std_{w}'] = g.transform(lambda s: s.rolling(w).std())
    df['wind_shear'] = g.diff()
    t = min(30, W)
    df['turbulence'] = (g.transform(lambda s: s.rolling(t).std())
                        / (g.transform(lambda s: s.rolling(t).mean()) + 1e-9))
    kv = np.empty(len(series))
    for _, idx in series.groupby(seg).groups.items():
        loc = series.index.get_indexer(idx)
        kv[loc] = FP.kalman_causal(series.loc[idx].values)
    df['kalman'] = kv
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    return df


def build_naive(series, W):
    f = pd.DataFrame(index=series.index)
    f['lag_0'] = series
    for lag in lag_set(W):
        f[f'lag_{lag}'] = series.shift(lag)
    for w in [w for w in ROLLS if w <= W]:
        f[f'roll_mean_{w}'] = series.rolling(w).mean()
        f[f'roll_std_{w}'] = series.rolling(w).std()
    f['wind_shear'] = series.diff()
    t = min(30, W)
    f['turbulence'] = series.rolling(t).std() / (series.rolling(t).mean() + 1e-9)
    f['kalman'] = FP.kalman_causal(series.values)
    f['hour_sin'] = np.sin(2 * np.pi * f.index.hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f.index.hour / 24)
    return f


def make_split(W):
    """Gap-aware split with a fixed embargo (36 + H) whatever W is; the
    validity mask uses a window that never binds beyond the features."""
    mask_len = min(EMBARGO_LEN, W + 1)

    def split(series, fgap, H, seq_len):
        mask = FP.valid_rows_for_horizon(series, fgap, H, mask_len)
        rows = np.where(mask)[0]
        if len(rows) < 800:
            return None
        s = int(len(rows) * RA.TRAIN_RATIO)
        tr, te = rows[:s], rows[s + EMBARGO_LEN + H:]
        if len(te) < 300:
            return None
        return rows, tr, te
    return split


def run_for_window(W, keys):
    FP.build_features_segmented = lambda series, step: build_segmented(series, step, W)
    FP.naive_features = lambda series: build_naive(series, W)
    RA.gap_aware_split = make_split(W)
    rows = []
    for name, step, seq_len, horizons in RA.DATASETS:
        if name[:2] not in keys:
            continue
        s = RA.load_dataset(name[:2])
        out = []
        print(f'\n##### W = {W}   {name}')
        RA.run(name, s, step, seq_len, horizons, out)
        for r in out:
            r['W'] = W
        rows += out
    return rows


def draw(df, stem):
    d = df[df.dataset.str.startswith('D3')] if df.dataset.str.startswith('D3').any() else df
    d = d.copy()
    d['f'] = 1 - d.n_test_honest / d.n_test_naive
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cols = {1: '#4A7BB7', 3: '#1D9E75', 6: '#D85A30'}
    ax = axes[0]
    for H, g in d.groupby('horizon'):
        g = g.sort_values('W')
        ax.errorbar(g.W, g.bias_total, yerr=[g.bias_total - g.bias_ci_low,
                    g.bias_ci_high - g.bias_total], marker='o', ms=3.5, lw=1.2,
                    capsize=2, color=cols.get(H, 'k'), label=f'H = {H}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(WINDOWS)
    ax.set_xlabel('Feature-window depth W (samples)', fontsize=8)
    ax.set_ylabel('Bias (percentage points)', fontsize=8)
    ax.set_title('(a) Bias vs window depth', fontsize=8.5, fontweight='bold', pad=3)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for H, g in d.groupby('horizon'):
        ax.scatter(100 * g.f, g.bias_total, s=22, color=cols.get(H, 'k'), zorder=3)
        for _, r in g.iterrows():
            ax.annotate(f'W={int(r.W)}', (100 * r.f, r.bias_total), fontsize=5.5,
                        xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Exposure f (% of scored rows displaced)', fontsize=8)
    ax.set_ylabel('Bias (percentage points)', fontsize=8)
    ax.set_title('(b) Bias vs exposure', fontsize=8.5, fontweight='bold', pad=3)
    for a in axes:
        a.tick_params(labelsize=7, length=2)
        a.grid(alpha=0.3, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    keys = set(os.environ.get('WD_DATASETS', 'D3').replace(' ', '').split(','))
    allrows = []
    for W in WINDOWS:
        allrows += run_for_window(W, keys)
    df = pd.DataFrame(allrows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV}')
    show = ['dataset', 'W', 'horizon', 'skill_honest', 'skill_naive',
            'bias_total', 'bias_ci_low', 'bias_ci_high', 'ci_excludes_zero',
            'n_test_honest', 'n_test_naive']
    print(df[show].to_string(index=False))

    ref = os.path.join(HERE, 'artifact_aligned.csv')
    if os.path.exists(ref):
        r0 = pd.read_csv(ref)
        m = df[df.W == 60].merge(r0, on=['dataset', 'horizon'],
                                 suffixes=('_wd', '_tbl3'))
        if len(m):
            diff = (m.bias_total_wd - m.bias_total_tbl3).abs().max()
            print(f'\nCheck: W = 60 reproduces Table III bias to within '
                  f'{diff:.2f} points (0.00 expected; small differences can '
                  'come from the bootstrap count N_BOOT only in the CI).')
    d3 = df.copy()
    d3['f'] = 1 - d3.n_test_honest / d3.n_test_naive
    if len(d3) > 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(d3.f, d3.bias_total)
        print(f'Spearman correlation of bias with exposure f: {rho:.2f} (p = {p:.3g})')
    draw(df, os.path.join(HERE, 'Figure_window_depth'))
    print('Saved Figure_window_depth.pdf and .png')
