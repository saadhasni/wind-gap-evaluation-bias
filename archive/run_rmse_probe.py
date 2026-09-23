import warnings, os, sys, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_wfip3_buoy
from knmi_adapter import load_knmi_platform, longest_clean_block

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
SEQ_LEN = 12
STEP = pd.Timedelta('10min')
HORIZONS = (1, 3)
MISSING_FRACS = (0.05, 0.10, 0.20)
SEEDS = tuple(range(15))

RECORDS = [('HKWA', 'knmi_hkwa'), ('HKWB', 'knmi_hkwb')]
BUOY_DIR = 'buoy_data'

OUT_CSV = 'rmse_probe_results.csv'


def inject_gaps(series, n_gaps, gap_len, seed, margin=200):
    """Identical to the experiment's injection."""
    rng = np.random.default_rng(seed)
    n = len(series)
    forbidden = np.zeros(n, dtype=bool)
    forbidden[:margin] = True
    forbidden[-margin:] = True
    starts, attempts = [], 0
    while len(starts) < n_gaps and attempts < n_gaps * 200:
        attempts += 1
        s0 = int(rng.integers(margin, n - margin - gap_len))
        if forbidden[max(0, s0 - 1):s0 + gap_len + 1].any():
            continue
        forbidden[s0:s0 + gap_len] = True
        starts.append(s0)
    drop = np.zeros(n, dtype=bool)
    for s0 in starts:
        drop[s0:s0 + gap_len] = True
    return series[~drop], len(starts)


def naive_features(series):
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


def evaluate_instrumented(series, H):
    fg = FP.build_features_segmented(series, STEP)
    gc_cols = [c for c in fg.columns if c != '_seg']
    mask = FP.valid_rows_for_horizon(series, fg, H, SEQ_LEN)
    rows = np.where(mask)[0]
    if len(rows) < 400:
        return None
    split = int(len(rows) * TRAIN_RATIO)
    EMB = SEQ_LEN + H
    tr, te = rows[:split], rows[split + EMB:]
    if len(te) < 150:
        return None

    y_te = series.values[te + H]
    y_now = fg.iloc[te]['lag_0'].values
    rp_h = np.sqrt(mean_squared_error(y_te, y_now))
    m = XGBRegressor(**XGB)
    m.fit(fg.iloc[tr][gc_cols].values, series.values[tr + H])
    pred_h = m.predict(fg.iloc[te][gc_cols].values)
    ra_h = np.sqrt(mean_squared_error(y_te, pred_h))

    # test-set character, to spot calm windows
    te_idx = series.index[te]
    d_te = np.diff(y_te)

    fn = naive_features(series)
    dn = fn.copy()
    dn['target'] = series.shift(-H)
    dn = dn.dropna()
    fc = [c for c in dn.columns if c != 'target']
    s2 = int(len(dn) * TRAIN_RATIO)
    tr2, te2 = dn.iloc[:s2], dn.iloc[s2:]
    m2 = XGBRegressor(**XGB)
    m2.fit(tr2[fc].values, tr2['target'].values)
    yt2 = te2['target'].values
    yn2 = te2['lag_0'].values
    rp_n = np.sqrt(mean_squared_error(yt2, yn2))
    ra_n = np.sqrt(mean_squared_error(yt2, m2.predict(te2[fc].values)))

    out = dict(
        rmse_pers_honest=rp_h, rmse_model_honest=ra_h,
        rmse_pers_naive=rp_n, rmse_model_naive=ra_n,
        ratio_honest=ra_h / rp_h, ratio_naive=ra_n / rp_n,
        skill_honest=100 * (1 - ra_h / rp_h),
        skill_naive=100 * (1 - ra_n / rp_n),
        excess_honest=ra_h - rp_h, excess_naive=ra_n - rp_n,
        n_test_honest=len(te), n_test_naive=len(te2),
        n_train_honest=len(tr),
        test_target_mean=float(np.mean(y_te)),
        test_target_std=float(np.std(y_te)),
        test_target_sddiff=float(np.std(d_te)),
        test_start=str(te_idx[0]), test_end=str(te_idx[-1]),
        pred_std=float(np.std(pred_h)),
        pred_min=float(np.min(pred_h)), pred_max=float(np.max(pred_h)),
    )
    out['rmse_bias'] = out['excess_naive'] - out['excess_honest']
    out['bias'] = out['skill_naive'] - out['skill_honest']
    del m, m2, fg, fn, dn, tr2, te2
    return out


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


def load_records():
    out = []
    for name, folder in RECORDS:
        if name == 'BUOY130':
            bp = os.path.join(HERE, BUOY_DIR)
            if os.path.isdir(bp):
                s_all, _ = load_wfip3_buoy(bp, height_m=38, resample='10min')
                out.append((name, buoy_longest_contiguous(s_all, STEP)))
            continue
        p = os.path.join(HERE, folder)
        if not os.path.isdir(p):
            print(f"  {name}: {folder} not found, skipping")
            continue
        df = load_knmi_platform(p, verbose=False)
        out.append((name, longest_clean_block(df)['wind_speed']))
    for name, s in out:
        assert (s.index.to_series().diff().dropna() == STEP).all(), \
            f"{name} not contiguous"
    return out


def main():
    records = load_records()
    if not records:
        print('No records loaded.')
        return

    print('Records:')
    for name, s in records:
        print(f"   {name:<9} {len(s):>6} samples  {len(s) / 144:>5.1f} d   "
              f"mean {s.mean():.2f} m/s   sd_diff {s.diff().std():.3f}")

    # intact baseline, for reference
    print('\nIntact record (no gaps):')
    print(f"   {'record':<9}{'H':>3}{'rp':>9}{'ra':>9}{'ratio':>8}"
          f"{'skill':>9}{'excess':>9}")
    for name, s in records:
        for H in HORIZONS:
            r = evaluate_instrumented(s, H)
            if r:
                print(f"   {name:<9}{H:>3}{r['rmse_pers_honest']:>9.4f}"
                      f"{r['rmse_model_honest']:>9.4f}{r['ratio_honest']:>8.3f}"
                      f"{r['skill_honest']:>9.2f}{r['excess_honest']:>9.4f}")
        gc.collect()

    rows = []
    conditions = {'MANY-SHORT': 3, 'MEDIUM': 18, 'FEW-LONG': 144}

    for rec_name, base in records:
        n = len(base)
        print(f"\n{'=' * 78}")
        print(f"  {rec_name}   {n} samples ({n / 144:.1f} days)")
        print('=' * 78)
        print(f"{'frac':>6}{'cond':>12}{'H':>3}{'rp_h':>8}{'ra_h':>8}"
              f"{'rp_n':>8}{'ra_n':>8}{'skill_bias':>12}{'rmse_bias':>11}")
        print('-' * 78)

        for frac in MISSING_FRACS:
            n_drop = int(frac * n)
            for cname, glen in conditions.items():
                n_gaps = max(1, n_drop // glen)
                for H in HORIZONS:
                    got_rows = []
                    for seed in SEEDS:
                        gapped, placed = inject_gaps(base, n_gaps, glen, seed)
                        r = evaluate_instrumented(gapped, H)
                        if r is None:
                            continue
                        r.update(record=rec_name, record_len=n,
                                 missing_frac=frac, condition=cname,
                                 horizon=H, seed=seed, gap_len=glen,
                                 n_gaps_requested=n_gaps, n_gaps_placed=placed)
                        rows.append(r)
                        got_rows.append(r)
                    if got_rows:
                        g = pd.DataFrame(got_rows)
                        print(f"{frac:>6.0%}{cname:>12}{H:>3}"
                              f"{g.rmse_pers_honest.median():>8.4f}"
                              f"{g.rmse_model_honest.median():>8.4f}"
                              f"{g.rmse_pers_naive.median():>8.4f}"
                              f"{g.rmse_model_naive.median():>8.4f}"
                              f"{g.bias.median():>+12.2f}"
                              f"{g.rmse_bias.median():>+11.4f}")
                    else:
                        print(f"{frac:>6.0%}{cname:>12}{H:>3}"
                              f"{'not evaluable':>55}")
                    sys.stdout.flush()
                    gc.collect()
        if rows:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint: {OUT_CSV}]")

    if not rows:
        print('\nNo results.')
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(df)} runs)")

    print('\n' + '=' * 78)
    print('  THE TEST: does the effect survive without a denominator?')
    print('  skill_bias is in percentage points; rmse_bias is in m/s.')
    print('  If rmse_bias is ~0 where skill_bias is large, the percentage')
    print('  figure is a small-denominator artifact.')
    print('=' * 78)
    summ = (df.groupby(['record', 'condition'])
              .agg(skill_bias_med=('bias', 'median'),
                   skill_bias_mean=('bias', 'mean'),
                   rmse_bias_med=('rmse_bias', 'median'),
                   rp_honest_med=('rmse_pers_honest', 'median'),
                   ra_honest_med=('rmse_model_honest', 'median'),
                   n=('bias', 'size')).round(4).reset_index())
    print(summ.to_string(index=False))

    print('\n  Correlation between skill_bias and rmse_bias, per record:')
    for rec in df.record.unique():
        g = df[df.record == rec]
        if len(g) > 3:
            print(f"    {rec:<9} pearson {g.bias.corr(g.rmse_bias):+.3f}   "
                  f"spearman {g.bias.corr(g.rmse_bias, method='spearman'):+.3f}")

    print('\n  Runs with |skill_bias| > 100 — what drives them?')
    ext = df[df.bias.abs() > 100]
    if len(ext):
        cols = ['record', 'condition', 'missing_frac', 'horizon', 'seed',
                'rmse_pers_honest', 'rmse_model_honest', 'ratio_honest',
                'rmse_bias', 'bias', 'n_test_honest', 'test_target_std',
                'test_target_sddiff']
        print(ext[cols].round(4).head(20).to_string(index=False))
        print(f"\n    {len(ext)} extreme runs of {len(df)}")
        print(f"    median rp_honest in extreme runs: "
              f"{ext.rmse_pers_honest.median():.4f}")
        print(f"    median rp_honest elsewhere:       "
              f"{df[df.bias.abs() <= 100].rmse_pers_honest.median():.4f}")
        print(f"    median ra_honest in extreme runs: "
              f"{ext.rmse_model_honest.median():.4f}")
        print(f"    median ra_honest elsewhere:       "
              f"{df[df.bias.abs() <= 100].rmse_model_honest.median():.4f}")
    else:
        print('    none')

    print('\ndone.')


if __name__ == '__main__':
    main()
