import warnings, os, sys, glob
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
OUT_CSV = 'artifact_audit.csv'


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


def skill(y, pred, pers):
    return 100.0 * (1 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, pers)))


# ---------------------------------------------------------------- (3)
def interpolation_report():
    """How many samples in D1 and D2 are interpolated rather than observed?"""
    print('=' * 78)
    print('  (3) INTERPOLATION IN THE ADAPTERS')
    print('  .interpolate() uses the observation AFTER the gap, so filled')
    print('  values embed future information. G1 cannot detect this.')
    print('=' * 78)

    # D1: replicate load_zephir without the interpolate step
    try:
        files = sorted(set(glob.glob(os.path.join(
            HERE, 'ZephIR_windlidar_*.[Cc][Ss][Vv]'))))
        col = 'Horizontal Wind Speed (m/s) at 38m'
        fr = [pd.read_csv(f, skiprows=1, usecols=['Time and Date', col],
                          na_values=['#N/A', '9999', '']) for f in files]
        raw = pd.concat(fr, ignore_index=True)
        raw['dt'] = pd.to_datetime(raw['Time and Date'],
                                   format='%d/%m/%Y %H:%M:%S')
        s0 = pd.to_numeric(raw.set_index('dt')[col], errors='coerce')
        s0 = s0.where(s0.between(0, 45)).resample('1min').mean()
        raw_n = int(s0.notna().sum())
        filled = int(s0.interpolate(limit=5).dropna().shape[0])
        print(f"  D1 ZephIR : observed {raw_n}, after interpolate(limit=5) "
              f"{filled}  -> {filled - raw_n} filled "
              f"({100 * (filled - raw_n) / max(filled, 1):.2f}% of series)")
    except Exception as e:
        print(f'  D1: {e}')

    # D2: replicate load_wfip3_buoy without the interpolate step
    try:
        import xarray as xr
        files = sorted(glob.glob(os.path.join(HERE, 'buoy_data', '*.nc')))
        parts = []
        for f in files:
            ds = xr.open_dataset(f)
            h = ds['height'].values
            j = int(np.argmin(np.abs(h - 38)))
            hws = ds['horizontal_wind_speed'].isel(height=j).values
            qc = ds['qc_horizontal_wind_speed'].isel(height=j).values
            parts.append(pd.Series(np.where(qc == 0, hws, np.nan),
                                   index=pd.to_datetime(ds['time'].values)))
            ds.close()
        s0 = pd.concat(parts).sort_index()
        s0 = s0[~s0.index.duplicated(keep='first')]
        s0 = s0.where(s0.between(0, 45)).resample('10min').mean()
        raw_n = int(s0.notna().sum())
        filled = int(s0.interpolate(limit=3).dropna().shape[0])
        print(f"  D2 Buoy   : observed {raw_n}, after interpolate(limit=3) "
              f"{filled}  -> {filled - raw_n} filled "
              f"({100 * (filled - raw_n) / max(filled, 1):.2f}% of series)")
    except Exception as e:
        print(f'  D2: {e}')
    print('  D3 Onshore: no interpolation')
    print()


# ---------------------------------------------------------------- (1)(2)
def audit_one(name, series, step, seq_len, horizons, out):
    print('=' * 78)
    print(f'  {name}')
    print('=' * 78)
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

        # ---- A honest
        y_te = series.values[te + H]
        y_now = fgap.iloc[te]['lag_0'].values
        rp = np.sqrt(mean_squared_error(y_te, y_now))
        m = XGBRegressor(**XGB)
        m.fit(fgap.iloc[tr][gcols].values, series.values[tr + H])
        ra = np.sqrt(mean_squared_error(
            y_te, m.predict(fgap.iloc[te][gcols].values)))
        sk_a = 100 * (1 - ra / rp)

        # ---- B naive features, same rows
        m = XGBRegressor(**XGB)
        m.fit(fnav.iloc[tr][ncols].values, series.values[tr + H])
        rb = np.sqrt(mean_squared_error(
            y_te, m.predict(fnav.iloc[te][ncols].values)))
        sk_b = 100 * (1 - rb / rp)

        # ---- C naive, as published
        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2, te2 = dn.iloc[:s2], dn.iloc[s2:]
        m = XGBRegressor(**XGB)
        m.fit(tr2[fc].values, tr2['target'].values)
        sk_c_old = skill(te2['target'].values, m.predict(te2[fc].values),
                         te2['lag_0'].values)

        # ---- window drift
        a_start, a_end = series.index[te[0]], series.index[te[-1]]
        c_start, c_end = te2.index[0], te2.index[-1]
        drift_d = (a_start - c_start).total_seconds() / 86400

        # ---- C aligned: same model, test restricted to A's calendar window
        te2a = te2[(te2.index >= a_start) & (te2.index <= a_end)]
        if len(te2a) < 200:
            sk_c_new = np.nan
        else:
            sk_c_new = skill(te2a['target'].values,
                             m.predict(te2a[fc].values),
                             te2a['lag_0'].values)

        bias_old = sk_c_old - sk_a
        bias_new = sk_c_new - sk_a
        print(f"  H={H:>2}  valid {len(rows):>6}/{len(dn):>6} "
              f"({100*len(rows)/len(dn):>4.1f}%)  drift {drift_d:+6.2f} d")
        print(f"        A {sk_a:+7.2f}   B {sk_b:+7.2f}   "
              f"C_old {sk_c_old:+7.2f}   C_aligned {sk_c_new:+7.2f}")
        print(f"        bias published {bias_old:+7.2f}   "
              f"bias aligned {bias_new:+7.2f}   "
              f"window artifact {bias_old - bias_new:+7.2f}")
        print(f"        test rows: A {len(te)}  C_old {len(te2)}  "
              f"C_aligned {len(te2a)}")

        out.append(dict(dataset=name, horizon=H,
                        n_valid=len(rows), n_naive=len(dn),
                        pct_valid=round(100*len(rows)/len(dn), 1),
                        drift_days=round(drift_d, 2),
                        skill_A=round(sk_a, 2), skill_B=round(sk_b, 2),
                        skill_C_published=round(sk_c_old, 2),
                        skill_C_aligned=round(sk_c_new, 2),
                        bias_published=round(bias_old, 2),
                        bias_aligned=round(bias_new, 2),
                        window_artifact=round(bias_old - bias_new, 2),
                        bias_feature=round(sk_b - sk_a, 2),
                        n_test_A=len(te), n_test_C_published=len(te2),
                        n_test_C_aligned=len(te2a),
                        A_test_start=str(a_start), C_test_start=str(c_start)))


if __name__ == '__main__':
    interpolation_report()
    out = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        audit_one('D1 ZephIR', s, pd.Timedelta('1min'), 60, (1, 10, 30), out)
    except Exception as e:
        print('[skip D1]', e)
    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        audit_one('D2 Buoy', s, pd.Timedelta('10min'), 36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D2]', e)
    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        audit_one('D3 Onshore', s, pd.Timedelta('10min'), 36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D3]', e)

    if not out:
        print('No results.'); sys.exit(0)
    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV}')

    print('\n' + '=' * 78)
    print('  VERDICT')
    print('=' * 78)
    print(df[['dataset', 'horizon', 'pct_valid', 'drift_days',
              'bias_published', 'bias_aligned', 'window_artifact']]
          .to_string(index=False))
    worst = df.window_artifact.abs().max()
    print(f'\n  Largest window artifact: {worst:.2f} percentage points')
    if worst < 1.0:
        print('  -> The published observational result is window-clean.')
        print('     Report the aligned figures and state that the check was made.')
    else:
        print('  -> The window artifact is material. Table 4 must be')
        print('     regenerated with aligned windows, and the headline')
        print('     D3 H=6 case re-examined.')
    hd = df[(df.dataset.str.contains('D3')) & (df.horizon == 6)]
    if len(hd):
        r = hd.iloc[0]
        print(f"\n  HEADLINE CASE (D3, H=6):")
        print(f"    published bias {r.bias_published:+.2f}  ->  "
              f"aligned bias {r.bias_aligned:+.2f}")
        print(f"    honest skill {r.skill_A:+.2f}%, "
              f"aligned naive skill {r.skill_C_aligned:+.2f}%")
