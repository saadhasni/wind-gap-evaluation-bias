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
from run_artifact_aligned import (XGB, TRAIN_RATIO, gap_aware_split,
                                  naive_aligned_split, load_dataset,
                                  run_all)

OUT_CSV = os.path.join(HERE, 'artifact_audit.csv')


def skill(y, pred, pers):
    return 100.0 * (1 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, pers)))


# ---------------------------------------------------------------- (3)
def interpolation_report():
    """How many samples in D1 and D2 are interpolated rather than observed?"""
    print('=' * 78)
    print('  (3) INTERPOLATION IN THE ADAPTERS')
    print('  Short bounded gaps are filled linearly, using the observation')
    print('  AFTER the gap, so filled values embed future information.')
    print('  G1 cannot detect this; see run_interpolation_sensitivity.py.')
    print('=' * 78)
    for key, lim in (('D1', 5), ('D2', 3)):
        try:
            s_raw = load_dataset(key, interpolate=False)
            s_fil = load_dataset(key, interpolate=True)
        except FileNotFoundError as e:
            print(f'  {key}: {e}')
            continue
        n_f = len(s_fil) - len(s_raw)
        print(f"  {key}: observed {len(s_raw)}, after filling gaps <= {lim} "
              f"samples {len(s_fil)}  -> {n_f} filled "
              f"({100 * n_f / max(len(s_fil), 1):.2f}% of series)")
    print('  D3 Onshore: no interpolation')
    print()


# ---------------------------------------------------------------- (1)(2)
def audit_one(name, series, step, seq_len, horizons, out):
    print('=' * 78)
    print(f'  {name}')
    print('=' * 78)
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    print(f"  G1 causality check: PASSED "
          f"({FP.causality_self_check(series, step)} points)")
    fnav = FP.naive_features(series)
    ncols = list(fnav.columns)

    for H in horizons:
        sp = gap_aware_split(series, fgap, H, seq_len)
        if sp is None:
            print(f"  H={H}: too few gap-valid rows — skipped")
            continue
        rows, tr, te = sp

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

        # ---- C naive, as published: first 75% of naive ROWS, no embargo
        dn = FP.naive_frame(series, H, fnav)
        fc = [c for c in dn.columns if c not in ('target', '_target_ts')]
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

        # ---- C aligned: calendar split on A's window (as in Table III)
        tr2a, te2a, _ = naive_aligned_split(series, H, tr, te, fnav)
        m = XGBRegressor(**XGB)
        m.fit(tr2a[fc].values, tr2a['target'].values)
        sk_c_new = skill(te2a['target'].values, m.predict(te2a[fc].values),
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
              f"C_aligned {len(te2a)}   train rows: A {len(tr)}  "
              f"C_old {len(tr2)}  C_aligned {len(tr2a)}")

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
                        n_train_A=len(tr), n_train_C_published=len(tr2),
                        n_train_C_aligned=len(tr2a),
                        A_test_start=str(a_start), C_test_start=str(c_start)))


if __name__ == '__main__':
    interpolation_report()
    out = []
    run_all(audit_one, out)

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
        print('  -> The window artifact is material. Table III must be')
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
