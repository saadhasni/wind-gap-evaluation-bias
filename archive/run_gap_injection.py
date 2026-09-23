
import warnings, os, sys
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
from dataset_adapters import load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
SEQ_LEN = 12          # 2 h lookback: keeps short-gap segments usable
STEP = pd.Timedelta('10min')
HORIZONS = (1, 3)
MISSING_FRACS = (0.05, 0.10, 0.20)
SEEDS = tuple(range(30))


def naive_features(series):
    """Gap-ignoring feature construction (common practice)."""
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


def longest_contiguous(series, step):
    idx = series.index
    best_s, best_e, best_n, start = 0, len(series), 0, 0
    for i in range(1, len(series)):
        if (idx[i] - idx[i - 1]) != step:
            if i - start > best_n:
                best_n, best_s, best_e = i - start, start, i
            start = i
    if len(series) - start > best_n:
        best_s, best_e = start, len(series)
    return series.iloc[best_s:best_e]


def inject_gaps(series, n_gaps, gap_len, seed, margin=200):
    """Delete n_gaps blocks of gap_len samples, non-overlapping, away from
    the edges. Returns the punctured series (index preserved, so real gaps)."""
    rng = np.random.default_rng(seed)
    n = len(series)
    forbidden = np.zeros(n, dtype=bool)
    forbidden[:margin] = True
    forbidden[-margin:] = True
    starts = []
    attempts = 0
    while len(starts) < n_gaps and attempts < n_gaps * 200:
        attempts += 1
        s0 = int(rng.integers(margin, n - margin - gap_len))
        # keep a 1-sample buffer so gaps never merge into one longer gap
        if forbidden[max(0, s0 - 1):s0 + gap_len + 1].any():
            continue
        forbidden[s0:s0 + gap_len] = True
        starts.append(s0)
    drop = np.zeros(n, dtype=bool)
    for s0 in starts:
        drop[s0:s0 + gap_len] = True
    return series[~drop], len(starts)


def evaluate(series, H):
    """Return (honest_skill, naive_skill) on the given (possibly gappy) series."""
    # ---- honest: gap-aware
    fg = FP.build_features_segmented(series, STEP)
    gc = [c for c in fg.columns if c != '_seg']
    mask = FP.valid_rows_for_horizon(series, fg, H, SEQ_LEN)
    rows = np.where(mask)[0]
    if len(rows) < 400:
        return None, None, 0
    split = int(len(rows) * TRAIN_RATIO)
    EMB = SEQ_LEN + H
    tr, te = rows[:split], rows[split + EMB:]
    if len(te) < 150:
        return None, None, 0
    y_te = series.values[te + H]
    y_now = fg.iloc[te]['lag_0'].values
    rp = np.sqrt(mean_squared_error(y_te, y_now))
    m = XGBRegressor(**XGB)
    m.fit(fg.iloc[tr][gc].values, series.values[tr + H])
    ra = np.sqrt(mean_squared_error(y_te, m.predict(fg.iloc[te][gc].values)))
    sk_honest = 100 * (1 - ra / rp)

    # ---- naive: gap-ignoring
    fn = naive_features(series)
    dn = fn.copy()
    dn['target'] = series.shift(-H)
    dn = dn.dropna()
    fc = [c for c in dn.columns if c != 'target']
    s2 = int(len(dn) * TRAIN_RATIO)
    tr2, te2 = dn.iloc[:s2], dn.iloc[s2:]
    m = XGBRegressor(**XGB)
    m.fit(tr2[fc].values, tr2['target'].values)
    yt2 = te2['target'].values
    yn2 = te2['lag_0'].values
    rpc = np.sqrt(mean_squared_error(yt2, yn2))
    rc = np.sqrt(mean_squared_error(yt2, m.predict(te2[fc].values)))
    sk_naive = 100 * (1 - rc / rpc)
    return sk_honest, sk_naive, len(te)


if __name__ == '__main__':
    print('Loading buoy data and extracting the longest contiguous run...')
    s_all, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
    base = longest_contiguous(s_all, STEP)
    n = len(base)
    print(f'  contiguous base: {n} samples = {n*10/1440:.1f} days '
          f'({base.index[0]} -> {base.index[-1]})')

    # sanity: the base must be perfectly contiguous
    assert (base.index.to_series().diff().dropna() == STEP).all(), \
        'base run is not contiguous'
    print('  contiguity verified\n')

    rows_out = []
    print(f"{'frac':>6}{'condition':>13}{'H':>3}{'gaps':>6}{'len':>5}"
          f"{'honest':>9}{'naive':>9}{'bias':>8}")
    print('-' * 60)

    for frac in MISSING_FRACS:
        n_drop = int(frac * n)
        # matched total missing, opposite structure
        conditions = {
            'MANY-SHORT': dict(gap_len=3,   n_gaps=n_drop // 3),
            'MEDIUM':     dict(gap_len=18,  n_gaps=max(1, n_drop // 18)),
            'FEW-LONG':   dict(gap_len=144, n_gaps=max(1, n_drop // 144)),
        }
        for cname, cfg in conditions.items():
            for H in HORIZONS:
                biases = []
                for seed in SEEDS:
                    gapped, got = inject_gaps(base, cfg['n_gaps'],
                                              cfg['gap_len'], seed)
                    sk_h, sk_n, ntest = evaluate(gapped, H)
                    if sk_h is None:
                        continue
                    biases.append(sk_n - sk_h)
                    rows_out.append(dict(
                        missing_frac=frac, condition=cname, horizon=H,
                        seed=seed, n_gaps=got, gap_len=cfg['gap_len'],
                        actual_missing_pct=round(100*(1-len(gapped)/n), 2),
                        skill_honest=round(sk_h, 3), skill_naive=round(sk_n, 3),
                        bias=round(sk_n - sk_h, 3), n_test=ntest))
                if biases:
                    b = np.array(biases)
                    print(f"{frac:>6.0%}{cname:>13}{H:>3}{cfg['n_gaps']:>6}"
                          f"{cfg['gap_len']:>5}"
                          f"{rows_out[-1]['skill_honest']:>9.2f}"
                          f"{rows_out[-1]['skill_naive']:>9.2f}"
                          f"{b.mean():>+7.2f}\u00b1{b.std():.2f}")

    if not rows_out:
        print('No results produced.')
        sys.exit(0)

    df = pd.DataFrame(rows_out)
    df.to_csv('gap_injection_results.csv', index=False)
    print(f"\nSaved gap_injection_results.csv ({len(df)} runs)")

    print('\n' + '=' * 60)
    print('  CONTROLLED RESULT: bias by gap structure (mean \u00b1 sd over seeds)')
    print('=' * 60)
    summ = (df.groupby(['missing_frac', 'condition', 'horizon'])['bias']
              .agg(['mean', 'std', 'count']).reset_index())
    print(summ.to_string(index=False))

    # headline comparison at matched missing fraction
    print('\n  Direction test (does structure flip the sign?)')
    for frac in MISSING_FRACS:
        for H in HORIZONS:
            a = df[(df.missing_frac == frac) & (df.horizon == H) &
                   (df.condition == 'MANY-SHORT')]['bias']
            b = df[(df.missing_frac == frac) & (df.horizon == H) &
                   (df.condition == 'FEW-LONG')]['bias']
            if len(a) and len(b):
                flip = 'YES' if (a.mean() > 0) != (b.mean() > 0) else 'no'
                print(f"    missing {frac:.0%}, H={H}: "
                      f"many-short {a.mean():+.2f}, few-long {b.mean():+.2f} "
                      f"-> sign differs: {flip}")

    # figure
    fig, axes = plt.subplots(1, len(HORIZONS),
                             figsize=(6 * len(HORIZONS), 4.4), squeeze=False)
    for ax, H in zip(axes[0], HORIZONS):
        g = df[df.horizon == H]
        x = np.arange(len(MISSING_FRACS))
        for i, (cname, colr) in enumerate((('MANY-SHORT', '#1D9E75'),
                                           ('MEDIUM', '#4A7BB7'),
                                           ('FEW-LONG', '#D85A30'))):
            means = [g[(g.missing_frac == f) & (g.condition == cname)]['bias'].mean()
                     for f in MISSING_FRACS]
            errs = [g[(g.missing_frac == f) & (g.condition == cname)]['bias'].std()
                    for f in MISSING_FRACS]
            ax.bar(x + (i - 1) * 0.27, means, 0.27, yerr=errs, capsize=3,
                   label=cname, color=colr)
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f:.0%}' for f in MISSING_FRACS])
        ax.set_xlabel('Total missing fraction')
        ax.set_ylabel('Bias: naive \u2212 honest (percentage points)')
        ax.set_title(f'H = {H} step(s)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Controlled gap injection on one contiguous record: '
                 'bias direction depends on gap structure',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('gap_injection_fig.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved gap_injection_fig.png')
