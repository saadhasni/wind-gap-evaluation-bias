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
N_BOOT = 2000
BLOCK = 50
SEED = 42
OUT_CSV = 'artifact_aligned.csv'


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
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    mu = d.mean()
    v = np.var(d, ddof=1)
    for L in range(1, lmax + 1):
        if L < T:
            v += 2 * (1 - L / (lmax + 1)) * np.cov(d[:-L], d[L:])[0, 1]
    v = max(v, 1e-12)
    dm = mu / np.sqrt(v / T)
    return float(dm), float(2 * (1 - stats.norm.cdf(abs(dm))))


def skill_from_err(err_model, err_pers):
    rm = np.sqrt(np.mean(np.asarray(err_model) ** 2))
    rp = np.sqrt(np.mean(np.asarray(err_pers) ** 2))
    return 100.0 * (1.0 - rm / rp)


def _segment_blocks(idx, seg_series, block):
    """Block id per row of `idx`. Blocks are consecutive runs of at most
    `block` rows and never cross a segment boundary."""
    seg = seg_series.reindex(idx).to_numpy()
    if np.isnan(seg.astype(float)).any():
        raise ValueError("some scored timestamps are absent from the "
                         "segment index; pass the full series index")
    b = np.empty(len(idx), dtype=np.int64)
    bid, count, prev = -1, block, None
    for i, s in enumerate(seg):
        if s != prev or count >= block:
            bid += 1
            count = 0
            prev = s
        b[i] = bid
        count += 1
    return b


def bias_ci(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg_series,
            n_boot=2000, block=50, seed=42, min_frac=0.5):
    """Block-bootstrap 95% CI for (naive skill - gap-aware skill).

    eh_m, eh_p, idx_h : gap-aware model errors, persistence errors, timestamps
    en_m, en_p, idx_n : naive model errors, persistence errors, timestamps
    seg_series        : pd.Series of segment id indexed by the full series index
    min_frac          : a replicate is discarded if either arm retains fewer
                        than this fraction of its rows, which guards the skill
                        denominator against a degenerate draw
    """
    eh_m, eh_p = np.asarray(eh_m, float), np.asarray(eh_p, float)
    en_m, en_p = np.asarray(en_m, float), np.asarray(en_p, float)
    idx_h, idx_n = pd.DatetimeIndex(idx_h), pd.DatetimeIndex(idx_n)

    # master grid: the union of both arms' scored timestamps, in time order
    master = idx_h.union(idx_n).sort_values()
    blk = _segment_blocks(master, seg_series, block)
    bmap = pd.Series(blk, index=master)

    bh = bmap.reindex(idx_h).to_numpy()
    bn = bmap.reindex(idx_n).to_numpy()
    n_blk = int(blk.max()) + 1

    # row positions belonging to each block, for each arm
    pos_h = [np.where(bh == k)[0] for k in range(n_blk)]
    pos_n = [np.where(bn == k)[0] for k in range(n_blk)]

    rng = np.random.default_rng(seed)
    out, tries, need_h, need_n = [], 0, min_frac * len(eh_m), min_frac * len(en_m)

    while len(out) < n_boot and tries < 5 * n_boot:
        tries += 1
        draw = rng.integers(0, n_blk, n_blk)
        ph = np.concatenate([pos_h[k] for k in draw if pos_h[k].size])
        pn = np.concatenate([pos_n[k] for k in draw if pos_n[k].size])
        if ph.size < need_h or pn.size < need_n:
            continue
        out.append(skill_from_err(en_m[pn], en_p[pn])
                   - skill_from_err(eh_m[ph], eh_p[ph]))

    if len(out) < n_boot // 2:
        raise RuntimeError(f"only {len(out)} usable replicates; "
                           f"lower `block` or `min_frac`")
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def run(name, series, step, seq_len, horizons, out):
    print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}")
    seg = FP.segment_ids(series.index, step)
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = naive_features(series)
    ncols = list(fnav.columns)
    n_seg = int(seg.max() + 1)

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

        # A: honest
        y_te = series.values[te + H]
        y_now = fgap.iloc[te]['lag_0'].values
        m = XGBRegressor(**XGB)
        m.fit(fgap.iloc[tr][gcols].values, series.values[tr + H])
        pred_a = m.predict(fgap.iloc[te][gcols].values)
        eh_m, eh_p = y_te - pred_a, y_te - y_now
        sk_a = skill_from_err(eh_m, eh_p)
        _, p_a = diebold_mariano(eh_m, eh_p)

        # B: naive features, SAME rows
        m = XGBRegressor(**XGB)
        m.fit(fnav.iloc[tr][ncols].values, series.values[tr + H])
        pred_b = m.predict(fnav.iloc[te][ncols].values)
        sk_b = skill_from_err(y_te - pred_b, eh_p)

        # C: naive features, naive rows, ALIGNED window
        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2, te2_all = dn.iloc[:s2], dn.iloc[s2:]
        m = XGBRegressor(**XGB)
        m.fit(tr2[fc].values, tr2['target'].values)

        a_start, a_end = series.index[te[0]], series.index[te[-1]]
        te2 = te2_all[(te2_all.index >= a_start) & (te2_all.index <= a_end)]
        if len(te2) < 300:
            print(f"  H={H}: aligned naive test too small — skipped")
            continue
        pred_c = m.predict(te2[fc].values)
        yt2, yn2 = te2['target'].values, te2['lag_0'].values
        en_m, en_p = yt2 - pred_c, yt2 - yn2
        sk_c = skill_from_err(en_m, en_p)
        _, p_c = diebold_mariano(en_m, en_p)

        seg_s = pd.Series(seg, index=series.index)
        lo, hi = bias_ci(eh_m, eh_p, series.index[te], en_m, en_p, te2.index, seg_s)
        sens = {50: (lo, hi)}
        for B in (25, 100):
          sens[B] = bias_ci(eh_m, eh_p, series.index[te],
                      en_m, en_p, te2.index, seg_s, block=B)
        print("        block sensitivity   " + "   ".join(
        f"B={B}: [{sens[B][0]:+.2f}, {sens[B][1]:+.2f}]" for B in (25, 50, 100)))
        bias = sk_c - sk_a
        excl = (lo > 0) or (hi < 0)

        rp_h = float(np.sqrt(np.mean(eh_p ** 2)))
        rp_n = float(np.sqrt(np.mean(en_p ** 2)))

        print(f"  H={H:>2} | A {sk_a:+6.2f}% (DM p={p_a:.3f}) | "
              f"B {sk_b:+6.2f}% | C {sk_c:+6.2f}% (DM p={p_c:.3f})")
        print(f"        bias {bias:+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"{'excludes 0' if excl else 'includes 0'}   "
              f"feature {sk_b - sk_a:+.2f}   composition {sk_c - sk_b:+.2f}")
        print(f"        rows A {len(te)} -> C {len(te2)} "
              f"(x{len(te2)/len(te):.2f})   "
              f"persistence RMSE {rp_h:.3f} -> {rp_n:.3f}")

        out.append(dict(
            dataset=name, segments=n_seg, horizon=H,
            skill_honest=round(sk_a, 2), skill_naive_features=round(sk_b, 2),
            skill_naive=round(sk_c, 2),
            dm_p_honest=round(p_a, 4), dm_p_naive=round(p_c, 4),
            bias_total=round(bias, 2),
            bias_feature=round(sk_b - sk_a, 2),
            bias_composition=round(sk_c - sk_b, 2),
            bias_ci_low=round(lo, 2), bias_ci_high=round(hi, 2),
            ci_low_b25=round(sens[25][0], 2),
            ci_high_b25=round(sens[25][1], 2),
            ci_low_b100=round(sens[100][0], 2),
            ci_high_b100=round(sens[100][1], 2),
            ci_excludes_zero=bool(excl),
            sign_flip=bool((sk_a < 0) != (sk_c < 0)),
            n_test_honest=len(te), n_test_naive=len(te2),
            row_expansion=round(len(te2) / len(te), 2),
            rmse_pers_honest=round(rp_h, 4),
            rmse_pers_naive=round(rp_n, 4),
            test_start=str(a_start), test_end=str(a_end)))


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

    print('\n' + '=' * 78)
    print('  TABLE 4 (aligned windows)')
    print('=' * 78)
    cols = ['dataset', 'segments', 'horizon', 'skill_honest', 'skill_naive',
            'bias_total', 'bias_ci_low', 'bias_ci_high', 'ci_excludes_zero',
            'dm_p_honest', 'dm_p_naive']
    print(df[cols].to_string(index=False))
    print(f"\n  Sign reversals: {int(df.sign_flip.sum())} of {len(df)}")
    print(f"  Largest absolute bias: {df.bias_total.abs().max():.2f} points")
    print(f"  Bias CIs excluding zero: {int(df.ci_excludes_zero.sum())} "
          f"of {len(df)}")
    print(f"  Feature contamination: max |B-A| = "
          f"{df.bias_feature.abs().max():.2f} points")
    d1 = df[df.dataset.str.startswith('D1')]
    if len(d1):
        print(f"  Contiguous control D1: max |bias| = "
              f"{d1.bias_total.abs().max():.2f} points")
