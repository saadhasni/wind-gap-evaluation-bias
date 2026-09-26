import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from scipy import stats
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

N_ORIGINS   = 6      # number of expanding origins
FIRST_TRAIN = 0.50   # first origin trains on this fraction
LAST_TRAIN  = 0.80   # last origin trains on this fraction (overlapping design)
MIN_TEST    = 200
ZERO_TOL    = 0.005  # |bias| below this counts as zero for sign stability
# False: origins at 0.50 + k/12 (k=0..5), each tested on the next 1/12 of the
# gap-aware rows, so the six test blocks are disjoint (each starts EMB rows
# after its origin; the embargo rows between blocks are not scored).
# True: the original design (origins linspace(0.50, 0.80, 6), each tested on
# the next 20% of rows; test windows overlap ~70-75%).
OVERLAPPING = False


def skill(y, pred, persist):
    return 100.0 * (1.0 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, persist)))


def origin_windows(n, overlapping):
    """(train_frac, test_end_row) per origin, in gap-aware row units."""
    if overlapping:
        fracs = np.linspace(FIRST_TRAIN, LAST_TRAIN, N_ORIGINS)
        return [(fr, min(int(n * (fr + (1 - LAST_TRAIN))), n)) for fr in fracs]
    w = (1 - FIRST_TRAIN) / N_ORIGINS
    fracs = FIRST_TRAIN + w * np.arange(N_ORIGINS)
    return [(fr, n if k == N_ORIGINS - 1 else min(int(n * (fr + w)), n))
            for k, fr in enumerate(fracs)]


def run_dataset(name, series, step, seq_len, horizons, out, overlapping):
    print(f"\n{'='*74}\n  {name}   "
          f"[{'overlapping' if overlapping else 'non-overlapping'} test "
          f"windows]\n{'='*74}")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = FP.naive_features(series)

    for H in horizons:
        mask = FP.valid_rows_for_horizon(series, fgap, H, seq_len)
        rows = np.where(mask)[0]
        if len(rows) < 1200:
            continue
        EMB = seq_len + H

        dn = FP.naive_frame(series, H, fnav)
        fc = [c for c in dn.columns if c not in ('target', '_target_ts')]

        print(f"  H={H:>2}  valid rows A={len(rows)}  naive rows C={len(dn)}")

        for k, (fr, teA_end) in enumerate(origin_windows(len(rows),
                                                         overlapping), 1):
            # ---------- honest (A) ----------
            sA = int(len(rows) * fr)
            trA, teA = rows[:sA], rows[sA + EMB:teA_end]
            if len(teA) < MIN_TEST:
                continue
            yA = series.values[teA + H]
            pA = fgap.iloc[teA]['lag_0'].values
            m = XGBRegressor(**XGB)
            m.fit(fgap.iloc[trA][gcols].values, series.values[trA + H])
            sk_A = skill(yA, m.predict(fgap.iloc[teA][gcols].values), pA)

            # A's calendar windows
            a_train_end = series.index[trA[-1]]
            a_start = series.index[teA[0]]
            a_end = series.index[teA[-1]]

            # ---------- naive (C): calendar split on A's windows ----------
            trC, teC = FP.naive_calendar_split(dn, a_train_end, a_start, a_end)
            # size of the old row-fraction naive slice, for reference only
            n_before = (int(len(dn) * teA_end / len(rows))
                        - int(len(dn) * fr))
            if len(teC) < MIN_TEST:
                print(f"      origin {k}/{N_ORIGINS}: aligned naive test "
                      f"only {len(teC)} rows — skipped")
                continue
            covers = (teC.index[0] <= a_start and teC.index[-1] >= a_end
                      and series.index[teA].isin(teC.index).all())
            assert covers, f"{name} H={H} origin {k}: C misses A's window"
            assert trC['_target_ts'].max() < a_start

            m = XGBRegressor(**XGB)
            m.fit(trC[fc].values, trC['target'].values)
            sk_C = skill(teC['target'].values, m.predict(teC[fc].values),
                         teC['lag_0'].values)

            bias = sk_C - sk_A
            print(f"      origin {k}/{N_ORIGINS} (train {fr:.1%})  "
                  f"honest {sk_A:+7.2f}%  naive {sk_C:+7.2f}%  "
                  f"bias {bias:+7.2f}   rows A={len(teA)} C={len(teC)}  "
                  f"train A={len(trA)} C={len(trC)}  "
                  f"C covers A's window: yes")
            out.append(dict(dataset=name, horizon=H, origin=k,
                            train_frac=round(fr, 3),
                            skill_honest=round(sk_A, 2),
                            skill_naive=round(sk_C, 2),
                            bias=round(bias, 2),
                            n_test_honest=len(teA), n_test_naive=len(teC),
                            n_test_naive_unaligned=n_before,
                            row_expansion=round(len(teC) / len(teA), 2),
                            test_start=str(a_start), test_end=str(a_end)))


def summarise(df):
    rows = []
    for (ds, H), g in df.groupby(['dataset', 'horizon']):
        b = g['bias'].values
        n = len(b)
        m = b.mean()
        if n > 1:
            sd = b.std(ddof=1)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        else:
            sd = half = np.nan
        sg = np.where(np.abs(b) < ZERO_TOL, 0, np.sign(b))
        same = ('zero' if np.all(sg == 0) else
                'yes' if np.all(sg == sg[0]) else 'NO')
        excl = bool(np.isfinite(half) and ((m - half > 0) or (m + half < 0)))
        print(f"  {ds:<32} H={H:<3} n={n}  {m:+7.2f} +/- {sd:5.2f}   "
              f"95% CI [{m-half:+7.2f}, {m+half:+7.2f}]   sign stable: {same}")
        rows.append(dict(dataset=ds, horizon=H, n_origins=n,
                         mean_bias=round(m, 2), sd_bias=round(sd, 2),
                         ci_low=round(m - half, 2), ci_high=round(m + half, 2),
                         sign_stable=same, excludes_zero=excl))
    summ = pd.DataFrame(rows)
    print(f"Cases whose CI excludes zero: "
          f"{int(summ['excludes_zero'].sum())} of {len(summ)}")
    return summ


def run_all(overlapping):
    out = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
    except FileNotFoundError as e:
        print('[skip D1]', e)
    else:
        print(f"G1 causality check: PASSED "
              f"({FP.causality_self_check(s, pd.Timedelta('1min'))} points)")
        run_dataset('D1 ZephIR (contiguous)', s, pd.Timedelta('1min'), 60,
                    (1, 10, 30), out, overlapping)
    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
    except FileNotFoundError as e:
        print('[skip D2]', e)
    else:
        print(f"G1 causality check: PASSED "
              f"({FP.causality_self_check(s, pd.Timedelta('10min'))} points)")
        run_dataset('D2 WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'),
                    36, (1, 3, 6), out, overlapping)
    try:
        s = FP.load_onshore(HERE)
    except FileNotFoundError as e:
        print('[skip D3]', e)
    else:
        print(f"G1 causality check: PASSED "
              f"({FP.causality_self_check(s, pd.Timedelta('10min'))} points)")
        run_dataset('D3 Onshore (many short gaps)', s, pd.Timedelta('10min'),
                    36, (1, 3, 6), out, overlapping)
    return pd.DataFrame(out)


if __name__ == '__main__':
    df = run_all(OVERLAPPING)
    if df.empty:
        print('No results produced.'); sys.exit(0)
    df.to_csv(os.path.join(HERE, 'rolling_origin_aligned.csv'), index=False)
    print(f"\nSaved rolling_origin_aligned.csv ({len(df)} rows)")

    print("\n" + "=" * 74)
    print(f"  BIAS ACROSS ORIGINS  (mean +/- sd, 95% CI of the mean; "
          f"{'overlapping' if OVERLAPPING else 'non-overlapping'} test blocks)")
    print("=" * 74)
    summ = summarise(df)
    summ.to_csv(os.path.join(HERE, 'rolling_origin_aligned_summary.csv'),
                index=False)
    print("Saved rolling_origin_aligned_summary.csv")

    fig, axes = plt.subplots(1, df['dataset'].nunique(),
                             figsize=(5.6 * df['dataset'].nunique(), 4.3),
                             squeeze=False)
    for ax, (ds, g) in zip(axes[0], df.groupby('dataset', sort=False)):
        for H, gg in g.groupby('horizon'):
            ax.plot(gg['origin'], gg['bias'], marker='o', label=f'H={H}')
        ax.axhline(0, color='k', lw=1)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel('Expanding origin'
                      + ('' if OVERLAPPING else ' (disjoint test blocks)'))
        ax.set_ylabel('Bias: naive − honest (pts)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Gap-handling bias across rolling origins '
                 '(calendar-aligned windows)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(HERE, 'rolling_origin_aligned_fig.png'), dpi=150,
                bbox_inches='tight')
    print('Saved rolling_origin_aligned_fig.png')

    if not OVERLAPPING:
        # sensitivity: the original overlapping windows (same calendar-split C)
        print("\n" + "=" * 74)
        print("  SENSITIVITY: original overlapping test windows")
        print("=" * 74)
        dfo = run_all(True)
        dfo.to_csv(os.path.join(HERE, 'rolling_origin_overlapping.csv'),
                   index=False)
        print("\n  BIAS ACROSS ORIGINS (overlapping test windows)")
        summarise(dfo).to_csv(
            os.path.join(HERE, 'rolling_origin_overlapping_summary.csv'),
            index=False)
        print("Saved rolling_origin_overlapping.csv, "
              "rolling_origin_overlapping_summary.csv")
