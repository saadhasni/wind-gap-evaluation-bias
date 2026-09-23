import warnings, os, sys, glob
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0)
TRAIN_RATIO = 0.75
MODELS = ('XGBoost', 'Ridge')
OUT_CSV = 'artifact_decomposition.csv'


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


def fit_predict(name, Xtr, ytr, Xte):
    if name == 'XGBoost':
        m = XGBRegressor(**XGB)
    else:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    del m
    return p


def skill(y, pred, pers):
    return 100.0 * (1 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, pers)))


def run(name, series, step, seq_len, horizons, out):
    print(f"\n{'=' * 82}\n  {name}\n{'=' * 82}")
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

        a_start, a_end = series.index[te[0]], series.index[te[-1]]

        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2 = dn.iloc[:s2]
        teC = dn.iloc[s2:]
        teC = teC[(teC.index >= a_start) & (teC.index <= a_end)]
        if len(teC) < 300:
            continue

        teD_idx = series.index[te]

        y_A = series.values[te + H]
        pers_A = fgap.iloc[te]['lag_0'].values

        print(f"  H={H:>2}  test rows: A/B/D {len(te)}   C {len(teC)}  "
              f"(expansion x{len(teC)/len(te):.2f})   "
              f"train rows: A/B {len(tr)}  C/D {len(tr2)} "
              f"(x{len(tr2)/len(tr):.2f})")

        for mname in MODELS:
            # A
            pa = fit_predict(mname, fgap.iloc[tr][gcols].values,
                             series.values[tr + H],
                             fgap.iloc[te][gcols].values)
            sk_a = skill(y_A, pa, pers_A)

            # B: naive features, gap-valid train, gap-valid test
            pb = fit_predict(mname, fnav.iloc[tr][ncols].values,
                             series.values[tr + H],
                             fnav.iloc[te][ncols].values)
            sk_b = skill(y_A, pb, pers_A)

            # C and D share one model, trained on naive rows
            Xtr2, ytr2 = tr2[fc].values, tr2['target'].values
            # D: evaluate on A's exact rows
            XD = dn.reindex(teD_idx)[fc]
            okD = XD.notna().all(axis=1).values
            if okD.sum() < 300:
                sk_d = np.nan
            else:
                pd_ = fit_predict(mname, Xtr2, ytr2, XD[okD].values)
                sk_d = skill(y_A[okD], pd_, pers_A[okD])
            # C: evaluate on every row in the window
            pc = fit_predict(mname, Xtr2, ytr2, teC[fc].values)
            sk_c = skill(teC['target'].values, pc, teC['lag_0'].values)

            feat = sk_b - sk_a
            train = sk_d - sk_b if sk_d == sk_d else np.nan
            comp = sk_c - sk_d if sk_d == sk_d else np.nan
            total = sk_c - sk_a

            print(f"        {mname:<8} A {sk_a:+7.2f}  B {sk_b:+7.2f}  "
                  f"D {sk_d:+7.2f}  C {sk_c:+7.2f}  ||  "
                  f"feature {feat:+6.2f}  training {train:+6.2f}  "
                  f"composition {comp:+6.2f}  total {total:+6.2f}")

            out.append(dict(
                dataset=name, horizon=H, model=mname,
                skill_A=round(sk_a, 2), skill_B=round(sk_b, 2),
                skill_D=round(sk_d, 2) if sk_d == sk_d else np.nan,
                skill_C=round(sk_c, 2),
                bias_feature=round(feat, 2),
                bias_training=round(train, 2) if train == train else np.nan,
                bias_composition=round(comp, 2) if comp == comp else np.nan,
                bias_total=round(total, 2),
                n_test_ABD=len(te), n_test_C=len(teC),
                n_train_AB=len(tr), n_train_CD=len(tr2),
                row_expansion=round(len(teC) / len(te), 2)))


if __name__ == '__main__':
    out = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        run('D1 ZephIR (contiguous)', s, pd.Timedelta('1min'), 60,
            (1, 10, 30), out)
    except Exception as e:
        print('[skip D1]', e)
    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        run('D2 WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'), 36,
            (1, 3, 6), out)
    except Exception as e:
        print('[skip D2]', e)
    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run('D3 Onshore (many short gaps)', s, pd.Timedelta('10min'), 36,
            (1, 3, 6), out)
    except Exception as e:
        print('[skip D3]', e)

    if not out:
        print('No results.'); sys.exit(0)
    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print('\n' + '=' * 82)
    print('  DECOMPOSITION (percentage points)')
    print('  feature = B-A   training = D-B   composition = C-D   '
          'total = C-A')
    print('=' * 82)
    print(df[['dataset', 'horizon', 'model', 'bias_feature', 'bias_training',
              'bias_composition', 'bias_total', 'row_expansion']]
          .to_string(index=False))

    print('\n' + '=' * 82)
    print('  WHICH CHANNEL DOMINATES, PER DATASET (mean |contribution|)')
    print('=' * 82)
    for ds, g in df.groupby('dataset', sort=False):
        f_ = g.bias_feature.abs().mean()
        t_ = g.bias_training.abs().mean()
        c_ = g.bias_composition.abs().mean()
        tot = f_ + t_ + c_
        if tot == 0:
            print(f"  {ds:<32} no bias to attribute")
            continue
        print(f"  {ds:<32} feature {100*f_/tot:5.1f}%   "
              f"training {100*t_/tot:5.1f}%   composition {100*c_/tot:5.1f}%"
              f"   (mean |total| {g.bias_total.abs().mean():.2f} pts)")
    print('\n  Expectation: on D2 the test sets are identical after')
    print('  alignment, so composition should be ~0 and training should')
    print('  carry the effect. On D3 both channels should be present.')
