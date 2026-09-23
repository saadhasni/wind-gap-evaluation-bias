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
from knmi_adapter import load_knmi_platform, longest_clean_block

# config
XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
SEQ_LEN = 12
STEP = pd.Timedelta('10min')
HORIZONS = (1, 3)
MISSING_FRACS = (0.05, 0.10, 0.20)
SEEDS = tuple(range(15))

# Records that passed the intact-record sanity check in diag_records.py
KNMI_RECORDS = [
    ('HKZA', 'knmi_hkza'),
    ('HKWB', 'knmi_hkwb'),
    ('HKWA', 'knmi_hkwa'),
    ('HKZB', 'knmi_hkzb'),
]
INCLUDE_BUOY = True
BUOY_DIR = 'buoy_data'

# A record whose intact honest skill lies outside this band cannot support
# a meaningful skill ratio. Checked before the sweep, not after.
INTACT_SKILL_LIMIT = 25.0
MIN_RECORD_LEN = 3000

OUT_CSV = 'gap_injection_multi_results.csv'


# unchanged core
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


def inject_gaps(series, n_gaps, gap_len, seed, margin=200):
    """Delete n_gaps blocks of gap_len samples, non-overlapping, away from
    the edges. Returns the punctured series (index preserved, so real gaps)."""
    rng = np.random.default_rng(seed)
    n = len(series)
    forbidden = np.zeros(n, dtype=bool)
    forbidden[:margin] = True
    forbidden[-margin:] = True
    starts, attempts = [], 0
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
    """Return (honest_skill, naive_skill, n_test) on a possibly gappy series."""
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


# record loading
def buoy_longest_contiguous(series, step):
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


def prepare_records():
    """Full clean blocks, no truncation. Contiguity asserted."""
    out = []
    for name, folder in KNMI_RECORDS:
        path = os.path.join(HERE, folder)
        if not os.path.isdir(path):
            print(f"  {name}: folder {folder} not found, skipping")
            continue
        df = load_knmi_platform(path, verbose=False)
        s = longest_clean_block(df)['wind_speed']
        if len(s) < MIN_RECORD_LEN:
            print(f"  {name}: clean block {len(s)} samples "
                  f"({len(s) / 144:.1f} d) below minimum, skipping")
            continue
        out.append((name, s))

    if INCLUDE_BUOY:
        bpath = os.path.join(HERE, BUOY_DIR)
        if os.path.isdir(bpath):
            s_all, _ = load_wfip3_buoy(bpath, height_m=38, resample='10min')
            s = buoy_longest_contiguous(s_all, STEP)
            if len(s) >= MIN_RECORD_LEN:
                out.append(('BUOY130', s))
            else:
                print(f"  BUOY130: only {len(s)} samples, skipping")
        else:
            print(f"  BUOY130: {BUOY_DIR} not found, skipping")

    for name, s in out:
        diffs = s.index.to_series().diff().dropna()
        assert (diffs == STEP).all(), f"{name} base run is not contiguous"
    return out


#  run
def main():
    print('Preparing records ...')
    records = prepare_records()
    if len(records) < 2:
        print('\nNeed at least two records. Stopping.')
        return

    print(f'\n{len(records)} records at full clean-block length, '
          f'contiguity verified:')
    for name, s in records:
        print(f"   {name:<9} {len(s):>6} samples  {len(s) / 144:>5.1f} d   "
              f"{s.index[0]} -> {s.index[-1]}   mean {s.mean():.2f} m/s")

    #  pre-flight: reject records that cannot support a skill ratio
    print('\nPre-flight: honest skill on each intact record '
          f'(must be within +/-{INTACT_SKILL_LIMIT:.0f}%)')
    keep = []
    for name, s in records:
        vals = []
        for H in HORIZONS:
            sk, _, _ = evaluate(s, H)
            vals.append(sk)
        ok = all(v is not None and abs(v) <= INTACT_SKILL_LIMIT for v in vals)
        shown = '  '.join('n/a' if v is None else f'{v:+.2f}' for v in vals)
        print(f"   {name:<9} {shown:<20} {'ok' if ok else 'EXCLUDED'}")
        if ok:
            keep.append((name, s))
    records = keep
    if len(records) < 2:
        print('\nToo few usable records after the sanity check. Stopping.')
        return

    rows_out = []
    for rec_i, (rec_name, base) in enumerate(records, 1):
        n = len(base)
        print(f"\n{'=' * 76}")
        print(f"  RECORD {rec_i}/{len(records)}: {rec_name}   "
              f"{n} samples ({n / 144:.1f} days)")
        print('=' * 76)
        print(f"{'frac':>6}{'condition':>13}{'H':>3}{'gaps':>6}{'len':>5}"
              f"{'honest':>9}{'naive':>9}{'bias':>10}{'n':>5}")
        print('-' * 76)

        for frac in MISSING_FRACS:
            n_drop = int(frac * n)
            conditions = {
                'MANY-SHORT': dict(gap_len=3,   n_gaps=n_drop // 3),
                'MEDIUM':     dict(gap_len=18,  n_gaps=max(1, n_drop // 18)),
                'FEW-LONG':   dict(gap_len=144, n_gaps=max(1, n_drop // 144)),
            }
            for cname, cfg in conditions.items():
                for H in HORIZONS:
                    biases, last_h, last_n, placed = [], None, None, []
                    for seed in SEEDS:
                        gapped, got = inject_gaps(base, cfg['n_gaps'],
                                                  cfg['gap_len'], seed)
                        sk_h, sk_n, ntest = evaluate(gapped, H)
                        if sk_h is None:
                            continue
                        placed.append(got)
                        biases.append(sk_n - sk_h)
                        last_h, last_n = sk_h, sk_n
                        rows_out.append(dict(
                            record=rec_name, record_len=n,
                            record_days=round(n / 144, 1),
                            missing_frac=frac, condition=cname, horizon=H,
                            seed=seed, n_gaps_requested=cfg['n_gaps'],
                            n_gaps_placed=got, gap_len=cfg['gap_len'],
                            actual_missing_pct=round(100 * (1 - len(gapped) / n), 2),
                            skill_honest=round(sk_h, 3),
                            skill_naive=round(sk_n, 3),
                            bias=round(sk_n - sk_h, 3), n_test=ntest))
                    if biases:
                        b = np.array(biases)
                        short = ''
                        if placed and min(placed) < cfg['n_gaps']:
                            short = f"  (placed {min(placed)}-{max(placed)})"
                        print(f"{frac:>6.0%}{cname:>13}{H:>3}{cfg['n_gaps']:>6}"
                              f"{cfg['gap_len']:>5}{last_h:>9.2f}{last_n:>9.2f}"
                              f"{b.mean():>+9.2f}\u00b1{b.std():.1f}"
                              f"{len(b):>5}{short}")
                    else:
                        print(f"{frac:>6.0%}{cname:>13}{H:>3}{cfg['n_gaps']:>6}"
                              f"{cfg['gap_len']:>5}{'--':>9}{'--':>9}"
                              f"{'not evaluable':>15}")

        if rows_out:
            pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint written: {OUT_CSV}]")

    if not rows_out:
        print('\nNo results produced.')
        return

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(df)} runs)")

    order = ['MANY-SHORT', 'MEDIUM', 'FEW-LONG']

    print('\n' + '=' * 76)
    print('  MONOTONICITY: is MANY-SHORT >= MEDIUM >= FEW-LONG?')
    print('  Only complete comparisons are counted.')
    print('=' * 76)
    per_record = {}
    for rec_name, _ in records:
        g = df[df.record == rec_name]
        if not len(g):
            continue
        ok = tot = 0
        details = []
        for frac in MISSING_FRACS:
            for H in HORIZONS:
                means = []
                for c in order:
                    v = g[(g.missing_frac == frac) & (g.horizon == H) &
                          (g.condition == c)]['bias']
                    means.append(v.mean() if len(v) else np.nan)
                if any(np.isnan(m) for m in means):
                    details.append((frac, H, means, 'incomplete'))
                    continue
                tot += 1
                mono = means[0] >= means[1] >= means[2]
                ok += int(mono)
                details.append((frac, H, means, 'yes' if mono else 'NO'))
        per_record[rec_name] = (ok, tot)
        print(f"\n  {rec_name}: {ok}/{tot} complete comparisons monotonic")
        print(f"    {'frac':>5} {'H':>2}   {'MANY-SHORT':>11}"
              f"{'MEDIUM':>10}{'FEW-LONG':>11}   verdict")
        for frac, H, means, verdict in details:
            ms = ''.join('       n/a' if np.isnan(m) else f'{m:>+10.2f}'
                         for m in means)
            print(f"    {frac:>4.0%} {H:>2}  {ms}   {verdict}")

    tot_ok = sum(v[0] for v in per_record.values())
    tot_all = sum(v[1] for v in per_record.values())
    print(f"\n  ACROSS ALL RECORDS: {tot_ok}/{tot_all} monotonic")

    print('\n' + '=' * 76)
    print('  MEAN BIAS BY CONDITION, PER RECORD (percentage points)')
    print('=' * 76)
    piv = (df.groupby(['record', 'condition'])['bias']
             .agg(['mean', 'std', 'count']).reset_index())
    piv['condition'] = pd.Categorical(piv['condition'], order, ordered=True)
    print(piv.sort_values(['record', 'condition']).to_string(index=False))

    print('\n  MANY-SHORT vs FEW-LONG separation, per record:')
    consistent = 0
    for rec_name in per_record:
        g = df[df.record == rec_name]
        a = g[g.condition == 'MANY-SHORT']['bias']
        c = g[g.condition == 'FEW-LONG']['bias']
        if len(a) and len(c):
            sep = a.mean() - c.mean()
            consistent += int(sep > 0)
            print(f"    {rec_name:<9} many-short {a.mean():+7.2f}   "
                  f"few-long {c.mean():+7.2f}   separation {sep:+7.2f}")
    print(f"\n    separation positive in {consistent}/{len(per_record)} records")

    # figure
    recs = list(per_record.keys())
    fig, axes = plt.subplots(len(HORIZONS), len(recs),
                             figsize=(3.3 * len(recs), 3.6 * len(HORIZONS)),
                             squeeze=False, sharey='row')
    colours = {'MANY-SHORT': '#1D9E75', 'MEDIUM': '#4A7BB7',
               'FEW-LONG': '#D85A30'}
    for r_i, H in enumerate(HORIZONS):
        for c_i, rec_name in enumerate(recs):
            ax = axes[r_i][c_i]
            g = df[(df.horizon == H) & (df.record == rec_name)]
            x = np.arange(len(MISSING_FRACS))
            for i, cname in enumerate(order):
                means = [g[(g.missing_frac == f) &
                           (g.condition == cname)]['bias'].mean()
                         for f in MISSING_FRACS]
                errs = [g[(g.missing_frac == f) &
                          (g.condition == cname)]['bias'].std()
                        for f in MISSING_FRACS]
                ax.bar(x + (i - 1) * 0.27, means, 0.27, yerr=errs, capsize=2,
                       label=cname if (r_i == 0 and c_i == 0) else None,
                       color=colours[cname])
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{f:.0%}' for f in MISSING_FRACS], fontsize=8)
            ax.grid(alpha=0.3, axis='y')
            if r_i == 0:
                days = df[df.record == rec_name]['record_days'].iloc[0]
                ax.set_title(f'{rec_name}\n{days:.0f} days', fontsize=9,
                             fontweight='bold')
            if c_i == 0:
                ax.set_ylabel(f'H = {H}\nBias (percentage points)', fontsize=9)
            if r_i == len(HORIZONS) - 1:
                ax.set_xlabel('Total missing fraction', fontsize=9)
    fig.legend(loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Gap injection replicated across independent contiguous '
                 'records', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig('gap_injection_multi_fig.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved gap_injection_multi_fig.png')


if __name__ == '__main__':
    main()
