import warnings, os, sys
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

KNMI_RECORDS = [('HKZA', 'knmi_hkza'), ('HKWB', 'knmi_hkwb'),
                ('HKWA', 'knmi_hkwa'), ('HKZB', 'knmi_hkzb'),
                ('BSB', 'knmi_bsb'), ('HKN', 'knmi_hkn')]
BUOY_DIR = 'buoy_data'


def inject_gaps(series, n_gaps, gap_len, seed, margin=200):
    """Identical to the injection used in the experiment."""
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


def honest_skill_intact(series, H):
    """Honest skill on a gap-free record. Returns (skill, rmse_p, rmse_m)."""
    fg = FP.build_features_segmented(series, STEP)
    gc = [c for c in fg.columns if c != '_seg']
    mask = FP.valid_rows_for_horizon(series, fg, H, SEQ_LEN)
    rows = np.where(mask)[0]
    if len(rows) < 400:
        return None, None, None
    split = int(len(rows) * TRAIN_RATIO)
    EMB = SEQ_LEN + H
    tr, te = rows[:split], rows[split + EMB:]
    if len(te) < 150:
        return None, None, None
    y_te = series.values[te + H]
    rp = np.sqrt(mean_squared_error(y_te, fg.iloc[te]['lag_0'].values))
    m = XGBRegressor(**XGB)
    m.fit(fg.iloc[tr][gc].values, series.values[tr + H])
    ra = np.sqrt(mean_squared_error(y_te, m.predict(fg.iloc[te][gc].values)))
    return 100 * (1 - ra / rp), rp, ra


def load_all():
    out = []
    for name, folder in KNMI_RECORDS:
        p = os.path.join(HERE, folder)
        if not os.path.isdir(p):
            continue
        df = load_knmi_platform(p, verbose=False)
        s = longest_clean_block(df)['wind_speed']
        if len(s) >= 1500:
            out.append((name, s))
    bp = os.path.join(HERE, BUOY_DIR)
    if os.path.isdir(bp):
        s_all, _ = load_wfip3_buoy(bp, height_m=38, resample='10min')
        s = buoy_longest_contiguous(s_all, STEP)
        if len(s) >= 1500:
            out.append(('BUOY130', s))
    return out


def main():
    records = load_all()
    if not records:
        print('No records found.')
        return

    print('=' * 78)
    print('  RECORD CHARACTER (no gaps injected)')
    print('=' * 78)
    print(f"{'record':<9}{'samples':>8}{'days':>7}{'mean':>7}{'std':>7}"
          f"{'ac1':>7}{'sd_diff':>9}")
    print('-' * 78)
    for name, s in records:
        d = s.diff().dropna()
        print(f"{name:<9}{len(s):>8}{len(s) / 144:>7.1f}{s.mean():>7.2f}"
              f"{s.std():>7.2f}{s.autocorr(1):>7.3f}{d.std():>9.3f}")

    print('\n' + '=' * 78)
    print('  TRAIN vs TEST under the 75/25 chronological split')
    print('  (large differences mean XGBoost trains on one regime and is '
          'tested on another)')
    print('=' * 78)
    print(f"{'record':<9}{'tr_mean':>9}{'te_mean':>9}{'shift':>8}"
          f"{'tr_std':>8}{'te_std':>8}{'ratio':>7}")
    print('-' * 78)
    for name, s in records:
        k = int(len(s) * TRAIN_RATIO)
        tr, te = s.iloc[:k], s.iloc[k:]
        shift = te.mean() - tr.mean()
        ratio = te.std() / tr.std() if tr.std() else np.nan
        flag = '   <-- large' if abs(shift) > 2 or ratio > 1.6 or ratio < 0.6 else ''
        print(f"{name:<9}{tr.mean():>9.2f}{te.mean():>9.2f}{shift:>+8.2f}"
              f"{tr.std():>8.2f}{te.std():>8.2f}{ratio:>7.2f}{flag}")

    print('\n' + '=' * 78)
    print('  HONEST SKILL ON THE INTACT RECORD (this is the sanity check)')
    print('  A healthy record sits within a few points of zero.')
    print('=' * 78)
    print(f"{'record':<9}{'H':>3}{'rmse_pers':>11}{'rmse_xgb':>10}"
          f"{'skill %':>10}   verdict")
    print('-' * 78)
    for name, s in records:
        for H in HORIZONS:
            sk, rp, ra = honest_skill_intact(s, H)
            if sk is None:
                print(f"{name:<9}{H:>3}{'--':>11}{'--':>10}"
                      f"{'too few rows':>10}")
                continue
            verdict = 'ok' if abs(sk) < 25 else 'UNSTABLE'
            print(f"{name:<9}{H:>3}{rp:>11.4f}{ra:>10.4f}{sk:>10.2f}   "
                  f"{verdict}")

    print('\n' + '=' * 78)
    print('  VALID ROWS UNDER INJECTION (measured, not estimated)')
    print(f'  evaluate() needs >= 400 candidate rows and >= 150 test rows.')
    print('  This is why conditions return "no valid runs". No models are '
          'fitted here.')
    print('=' * 78)
    print(f"{'record':<9}{'frac':>6}{'condition':>12}{'H':>3}{'placed':>8}"
          f"{'segs':>7}{'med_seg':>9}{'rows':>7}{'test':>7}   status")
    print('-' * 78)
    conditions = {'MANY-SHORT': 3, 'MEDIUM': 18, 'FEW-LONG': 144}
    for name, s in records:
        n = len(s)
        for frac in MISSING_FRACS:
            n_drop = int(frac * n)
            for cname, glen in conditions.items():
                n_gaps = max(1, n_drop // glen)
                gapped, got = inject_gaps(s, n_gaps, glen, seed=0)
                for H in HORIZONS:
                    fg = FP.build_features_segmented(gapped, STEP)
                    mask = FP.valid_rows_for_horizon(gapped, fg, H, SEQ_LEN)
                    rows = int(np.sum(mask))
                    n_test = max(0, rows - int(rows * TRAIN_RATIO)
                                 - (SEQ_LEN + H))
                    segs = int(fg['_seg'].nunique()) if '_seg' in fg else -1
                    seglens = (fg['_seg'].value_counts()
                               if '_seg' in fg else pd.Series([len(fg)]))
                    med = float(seglens.median())
                    if rows < 400:
                        status = 'FAIL: too few rows'
                    elif n_test < 150:
                        status = 'FAIL: too few test'
                    else:
                        status = 'ok'
                    print(f"{name:<9}{frac:>6.0%}{cname:>12}{H:>3}{got:>8}"
                          f"{segs:>7}{med:>9.1f}{rows:>7}{n_test:>7}   "
                          f"{status}")
        print('-' * 78)

    print('\ndone')


if __name__ == '__main__':
    main()
