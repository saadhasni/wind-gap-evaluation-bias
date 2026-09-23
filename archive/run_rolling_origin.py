import warnings, os, sys, glob
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
LAST_TRAIN  = 0.80   # last origin trains on this fraction


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


def skill(y, pred, persist):
    return 100.0 * (1.0 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, persist)))


def run_dataset(name, series, step, seq_len, horizons, out):
    print(f"\n{'='*74}\n  {name}\n{'='*74}")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = naive_features(series)

    for H in horizons:
        mask = FP.valid_rows_for_horizon(series, fgap, H, seq_len)
        rows = np.where(mask)[0]
        if len(rows) < 1200:
            continue
        EMB = seq_len + H

        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']

        print(f"  H={H:>2}  valid rows A={len(rows)}  naive rows C={len(dn)}")
        fracs = np.linspace(FIRST_TRAIN, LAST_TRAIN, N_ORIGINS)

        for k, fr in enumerate(fracs, 1):
            # ---------- honest (A) ----------
            sA = int(len(rows) * fr)
            teA_end = int(len(rows) * (fr + (1 - LAST_TRAIN) / 1.0))
            teA_end = min(teA_end, len(rows))
            trA, teA = rows[:sA], rows[sA + EMB:teA_end]
            if len(teA) < 200:
                continue
            yA = series.values[teA + H]
            pA = fgap.iloc[teA]['lag_0'].values
            m = XGBRegressor(**XGB)
            m.fit(fgap.iloc[trA][gcols].values, series.values[trA + H])
            sk_A = skill(yA, m.predict(fgap.iloc[teA][gcols].values), pA)

            # ---------- naive (C) ----------
            sC = int(len(dn) * fr)
            teC_end = min(int(len(dn) * (fr + (1 - LAST_TRAIN))), len(dn))
            trC, teC = dn.iloc[:sC], dn.iloc[sC:teC_end]
            if len(teC) < 200:
                continue
            m = XGBRegressor(**XGB)
            m.fit(trC[fc].values, trC['target'].values)
            sk_C = skill(teC['target'].values, m.predict(teC[fc].values),
                         teC['lag_0'].values)

            bias = sk_C - sk_A
            print(f"      origin {k}/{N_ORIGINS} (train {fr:.0%})  "
                  f"honest {sk_A:+7.2f}%  naive {sk_C:+7.2f}%  bias {bias:+7.2f}")
            out.append(dict(dataset=name, horizon=H, origin=k,
                            train_frac=round(fr, 3),
                            skill_honest=round(sk_A, 2),
                            skill_naive=round(sk_C, 2),
                            bias=round(bias, 2),
                            n_test_honest=len(teA), n_test_naive=len(teC)))


if __name__ == '__main__':
    out = []
    for loader, args, nm, step, sl, hz in [
        (load_zephir, (HERE,), 'D1 ZephIR (contiguous)',
         pd.Timedelta('1min'), 60, (1, 10, 30)),
        (load_wfip3_buoy, (os.path.join(HERE, 'buoy_data'),),
         'D2 WFIP3 Buoy (few long gaps)', pd.Timedelta('10min'), 36, (1, 3, 6)),
    ]:
        try:
            s, _ = loader(*args, height_m=38,
                          resample='1min' if 'ZephIR' in nm else '10min')
            run_dataset(nm, s, step, sl, hz, out)
        except Exception as e:
            print(f'[skip {nm}]', e)

    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run_dataset('D3 Onshore (many short gaps)', s, pd.Timedelta('10min'),
                    36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D3]', e)

    if not out:
        print('No results produced.'); sys.exit(0)

    df = pd.DataFrame(out)
    df.to_csv('rolling_origin.csv', index=False)
    print(f"\nSaved rolling_origin.csv ({len(df)} rows)")

    print("\n" + "=" * 74)
    print("  BIAS ACROSS ORIGINS  (mean +/- sd, 95% CI of the mean)")
    print("=" * 74)
    rows = []
    for (ds, H), g in df.groupby(['dataset', 'horizon']):
        b = g['bias'].values
        n = len(b)
        m, sd = b.mean(), b.std(ddof=1) if n > 1 else 0.0
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        same = 'yes' if np.all(np.sign(b) == np.sign(m)) else 'NO'
        print(f"  {ds:<32} H={H:<3} n={n}  {m:+7.2f} +/- {sd:5.2f}   "
              f"95% CI [{m-half:+7.2f}, {m+half:+7.2f}]   sign stable: {same}")
        rows.append(dict(dataset=ds, horizon=H, n_origins=n,
                         mean_bias=round(m, 2), sd_bias=round(sd, 2),
                         ci_low=round(m - half, 2), ci_high=round(m + half, 2),
                         sign_stable=same, excludes_zero=bool((m-half>0) or (m+half<0))))
    summ = pd.DataFrame(rows)
    summ.to_csv('rolling_origin_summary.csv', index=False)
    print(f"\nSaved rolling_origin_summary.csv")
    print(f"Cases whose CI excludes zero: "
          f"{int(summ['excludes_zero'].sum())} of {len(summ)}")

    fig, axes = plt.subplots(1, df['dataset'].nunique(),
                             figsize=(5.6 * df['dataset'].nunique(), 4.3),
                             squeeze=False)
    for ax, (ds, g) in zip(axes[0], df.groupby('dataset', sort=False)):
        for H, gg in g.groupby('horizon'):
            ax.plot(gg['origin'], gg['bias'], marker='o', label=f'H={H}')
        ax.axhline(0, color='k', lw=1)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel('Expanding origin')
        ax.set_ylabel('Bias: naive \u2212 honest (pts)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Gap-handling bias across rolling origins',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('rolling_origin_fig.png', dpi=150, bbox_inches='tight')
    print('Saved rolling_origin_fig.png')
