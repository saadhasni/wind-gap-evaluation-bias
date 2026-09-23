"""
=============================================================================
  add_statistics.py — compute significance tests for the artifact study
=============================================================================
  Produces, for every dataset and horizon:
    * Diebold-Mariano test, honest XGBoost vs honest persistence
    * Diebold-Mariano test, naive  XGBoost vs naive  persistence
    * Circular block-bootstrap 95% CI on the BIAS itself (naive - honest skill)

  The bias CI is the important one: it is what tells a reviewer whether the
  reported bias is distinguishable from zero.

  Place beside final_pipeline.py, dataset_adapters.py and run_artifact_study.py
  with the same data folders, then:      python add_statistics.py

  Output: artifact_statistics.csv
=============================================================================
"""
import warnings, os, sys, glob
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
N_BOOT      = 2000
BLOCK       = 50
SEED        = 42


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


def diebold_mariano(e1, e2, lmax=10):
    """DM test on squared-error loss, Newey-West (Bartlett) variance."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    mu = d.mean()
    v = np.var(d, ddof=1)
    for L in range(1, lmax + 1):
        if L < T:
            v += 2 * (1 - L / (lmax + 1)) * np.cov(d[:-L], d[L:])[0, 1]
    v = max(v, 1e-12)
    dm = mu / np.sqrt(v / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def skill(err_model, err_pers):
    """Skill score from two error arrays."""
    rm = np.sqrt(np.mean(np.asarray(err_model) ** 2))
    rp = np.sqrt(np.mean(np.asarray(err_pers) ** 2))
    return 100.0 * (1.0 - rm / rp)


def bias_ci(eh_m, eh_p, en_m, en_p, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Circular block-bootstrap 95% CI for (naive skill - honest skill).

    The honest and naive runs have different lengths, so each is resampled
    on its own circular block grid using a shared random stream.
    """
    rng = np.random.default_rng(seed)
    eh_m, eh_p = np.asarray(eh_m), np.asarray(eh_p)
    en_m, en_p = np.asarray(en_m), np.asarray(en_p)
    Th, Tn = len(eh_m), len(en_m)
    bh = min(block, max(5, Th // 4))
    bn = min(block, max(5, Tn // 4))
    out = np.empty(n_boot)
    for b in range(n_boot):
        ih = (rng.integers(0, Th, int(np.ceil(Th / bh)))[:, None]
              + np.arange(bh)[None, :]).ravel()[:Th] % Th
        idn = (rng.integers(0, Tn, int(np.ceil(Tn / bn)))[:, None]
               + np.arange(bn)[None, :]).ravel()[:Tn] % Tn
        out[b] = skill(en_m[idn], en_p[idn]) - skill(eh_m[ih], eh_p[ih])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(name, series, step, seq_len, horizons, rows_out):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = naive_features(series)
    ncols = list(fnav.columns)

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

        # ---------- honest ----------
        y_te = series.values[te + H]
        y_now = fgap.iloc[te]['lag_0'].values
        m = XGBRegressor(**XGB)
        m.fit(fgap.iloc[tr][gcols].values, series.values[tr + H])
        pred_h = m.predict(fgap.iloc[te][gcols].values)
        eh_m = y_te - pred_h
        eh_p = y_te - y_now
        sk_h = skill(eh_m, eh_p)
        dm_h, p_h = diebold_mariano(eh_m, eh_p)

        # ---------- naive ----------
        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2, te2 = dn.iloc[:s2], dn.iloc[s2:]
        m = XGBRegressor(**XGB)
        m.fit(tr2[fc].values, tr2['target'].values)
        pred_n = m.predict(te2[fc].values)
        yt2 = te2['target'].values
        yn2 = te2['lag_0'].values
        en_m = yt2 - pred_n
        en_p = yt2 - yn2
        sk_n = skill(en_m, en_p)
        dm_n, p_n = diebold_mariano(en_m, en_p)

        lo, hi = bias_ci(eh_m, eh_p, en_m, en_p)
        bias = sk_n - sk_h
        excl = (lo > 0) or (hi < 0)

        print(f"  H={H:>2} | honest {sk_h:+6.2f}% (DM p={p_h:.4f}) | "
              f"naive {sk_n:+6.2f}% (DM p={p_n:.4f}) | "
              f"bias {bias:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"{'excludes 0' if excl else 'includes 0'}")

        rows_out.append(dict(
            dataset=name, horizon=H,
            skill_honest=round(sk_h, 2), dm_p_honest=round(p_h, 4),
            skill_naive=round(sk_n, 2), dm_p_naive=round(p_n, 4),
            bias=round(bias, 2),
            bias_ci_low=round(lo, 2), bias_ci_high=round(hi, 2),
            ci_excludes_zero=bool(excl),
            n_test_honest=len(te), n_test_naive=len(te2)))


if __name__ == '__main__':
    out = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        run('D1 ZephIR300 (contiguous)', s, pd.Timedelta('1min'), 60, (1, 10, 30), out)
    except Exception as e:
        print('[skip D1]', e)

    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        run('D2 WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'), 36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D2]', e)

    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f)
        d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run('D3 Onshore turbine (many short gaps)', s, pd.Timedelta('10min'), 36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D3]', e)

    if out:
        df = pd.DataFrame(out)
        df.to_csv('artifact_statistics.csv', index=False)
        print('\nSaved artifact_statistics.csv')
        print(df.to_string(index=False))
        n_excl = int(df['ci_excludes_zero'].sum())
        print(f"\nBias CIs excluding zero: {n_excl} of {len(df)}")
