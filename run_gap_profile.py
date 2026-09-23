import warnings, os, sys, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from knmi_adapter import load_knmi_platform


HORIZONS = (1, 3, 6)
MAX_D = 12                    # profile distances d = 1 .. MAX_D
TRAIN_RATIO = 0.75            # retained for reference; walk-forward is used
N_FOLDS = 6                   # block 0 trains only; blocks 1-5 are scored
MIN_ROWS_PER_D = 15           # do not report a distance with fewer rows
N_BOOT = 2000                 # bootstrap replicates for the d=1 ratio
PREGAP_WINDOW = 18            # samples before a gap for the context test
MODEL = 'Ridge'               # 'Ridge' or 'XGBoost'; Ridge is far stabler
                              # on highly autocorrelated wind records

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=4)

# KNMI platform folders with REAL gaps (full records, not clean blocks)
KNMI_FOLDERS = [('BSB', 'knmi_bsb'), ('HKN', 'knmi_hkn'),
                ('HKZB', 'knmi_hkzb'), ('HKWA', 'knmi_hkwa'),
                ('HKWB', 'knmi_hkwb'), ('HKZA', 'knmi_hkza')]

# D1 / D2 / D3 from the paper.
# dataset_adapters.py provides only load_zephir and load_wfip3_buoy. D3 has
# no adapter, so load_d3_onshore() below reproduces the inline read used in
# run_artifact_aligned.py, plausibility mask included.
INCLUDE_D1 = True
D1_DIR = '.'                  # the ZephIR_windlidar_BSA_1s_*.CSV files
BUOY_DIR = 'buoy_data'

OUT_CSV = 'gap_profile_rows.csv'
OUT_PROFILE_CSV = 'gap_profile_summary.csv'
OUT_FIG = 'gap_profile_fig.png'
OUT_FIG_CTX = 'gap_context_fig.png'


def naive_features(series):
    """Row-position features, exactly as a naive study would build them."""
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


def fit_predict(Xtr, ytr, Xte):
    if MODEL == 'XGBoost':
        m = XGBRegressor(**XGB)
    else:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    del m
    return p


def grid_and_runs(series, step):
  
    grid = pd.date_range(series.index.min(), series.index.max(), freq=step)
    s = series.reindex(grid)
    present = s.notna().values
    n = len(s)
    d = np.zeros(n, dtype=int)
    gb = np.zeros(n, dtype=int)
    run, cur_gap, last_gap = 0, 0, 0
    for i in range(n):
        if present[i]:
            if run == 0:
                last_gap = cur_gap
                cur_gap = 0
            run += 1
            d[i] = run
            gb[i] = last_gap
        else:
            run = 0
            cur_gap += 1
    return s, present, d, gb


def profile_record(name, series, H, step):
  
    s_grid, present, d_arr, gb_arr = grid_and_runs(series, step)
    meta = pd.DataFrame({'d': d_arr, 'gap_before': gb_arr},
                        index=s_grid.index)[present]

    comp = s_grid.dropna()                    # what a naive study works with
    if len(comp) < 1000:
        return None

    f = naive_features(comp)
    f['target'] = comp.shift(-H)
    # real elapsed time to the target, to detect target displacement
    tt = pd.Series(comp.index, index=comp.index).shift(-H)
    f['target_lag_min'] = (tt - comp.index.to_series()).dt.total_seconds() / 60
    f = f.dropna()
    if len(f) < 500:
        return None

    cols = [c for c in f.columns if c not in ('target', 'target_lag_min')]
    nominal = H * step.total_seconds() / 60
    block = len(f) // N_FOLDS
    if block < 150:
        return None

    pieces = []
    for k in range(1, N_FOLDS):
        tr = f.iloc[:k * block]
        te = f.iloc[k * block:(k + 1) * block] if k < N_FOLDS - 1 \
            else f.iloc[k * block:]
        if len(tr) < 300 or len(te) < 100:
            continue
        pred = fit_predict(tr[cols].values, tr['target'].values,
                           te[cols].values)
        y = te['target'].values
        pers = te['lag_0'].values
        pieces.append(pd.DataFrame({
            'record': name, 'horizon': H, 'fold': k,
            'timestamp': te.index,
            'err_pers': np.abs(y - pers), 'sq_pers': (y - pers) ** 2,
            'err_model': np.abs(y - pred), 'sq_model': (y - pred) ** 2,
            'target_lag_min': te['target_lag_min'].values,
        }))
        gc.collect()

    if not pieces:
        return None
    out = pd.concat(pieces, ignore_index=True)
    out = out.join(meta, on='timestamp')
    out['d'] = out['d'].fillna(0).astype(int)
    out['gap_before'] = out['gap_before'].fillna(0).astype(int)
    out['target_displaced'] = out['target_lag_min'] > nominal + 1e-6
    return out


def gap_context(name, series, step):
    s_grid, present, d_arr, _ = grid_and_runs(series, step)
    v = s_grid.values
    n = len(v)
    # gap start positions
    starts = [i for i in range(1, n) if present[i - 1] and not present[i]]
    W = PREGAP_WINDOW

    pre_sd, pre_ramp = [], []
    for i in starts:
        seg = v[max(0, i - W):i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < max(5, W // 2):
            continue
        dd = np.diff(seg)
        pre_sd.append(np.std(dd))
        pre_ramp.append(np.max(np.abs(dd)) if len(dd) else np.nan)

    # baseline: every fully present window of the same length
    base_sd, base_ramp = [], []
    for i in range(W, n, max(1, W // 2)):
        seg = v[i - W:i]
        if np.isnan(seg).any():
            continue
        dd = np.diff(seg)
        base_sd.append(np.std(dd))
        base_ramp.append(np.max(np.abs(dd)))

    if len(pre_sd) < 10 or len(base_sd) < 30:
        return None
    u, p = mannwhitneyu(pre_sd, base_sd, alternative='two-sided')
    return dict(record=name, n_gaps=len(pre_sd),
                pre_gap_sd=float(np.median(pre_sd)),
                baseline_sd=float(np.median(base_sd)),
                ratio=float(np.median(pre_sd) / np.median(base_sd)),
                pre_gap_ramp=float(np.nanmedian(pre_ramp)),
                baseline_ramp=float(np.median(base_ramp)),
                mannwhitney_p=float(p))


def load_d3_onshore():

    import glob
    cand = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
            if '10min' in x]
    if not cand:
        raise FileNotFoundError("no onshore/*10min*.csv found")
    d = pd.read_csv(cand[0])
    d['Time'] = pd.to_datetime(d['Time'])
    s = d.set_index('Time')['WindSpeed'].sort_index()
    s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
    return s


def load_all():
 
    recs = []
    for name, folder in KNMI_FOLDERS:
        p = os.path.join(HERE, folder)
        if not os.path.isdir(p):
            continue
        df = load_knmi_platform(p, verbose=False)
        s = df['wind_speed'].dropna()
        if len(s) > 1000:
            recs.append((name, s, pd.Timedelta('10min')))

    try:
        import dataset_adapters as DA
    except Exception as e:
        print(f"  dataset_adapters not importable ({e}); KNMI only")
        return recs

    avail = [a for a in dir(DA) if a.startswith('load_')]

  
    if INCLUDE_D1 and hasattr(DA, 'load_zephir'):
        try:
            r = DA.load_zephir(D1_DIR, height_m=38, resample='1min')
            s = r[0] if isinstance(r, tuple) else r
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            recs.append(('D1_ZEPHIR', s.dropna(), pd.Timedelta('1min')))
        except Exception as e:
            print(f"  D1 load failed: {e}")

    bp = os.path.join(HERE, BUOY_DIR)
    if hasattr(DA, 'load_wfip3_buoy') and os.path.isdir(bp):
        try:
            r = DA.load_wfip3_buoy(bp, height_m=38, resample='10min')
            s = r[0] if isinstance(r, tuple) else r
            recs.append(('D2_BUOY', s.dropna(), pd.Timedelta('10min')))
        except Exception as e:
            print(f"  D2 load failed: {e}")

    # D3 — onshore turbine
    try:
        s = load_d3_onshore()
        recs.append(('D3_ONSHORE', s, pd.Timedelta('10min')))
    except Exception as e:
        print(f"  D3 not loaded: {e}")
    return recs


def main():
    print(f'  model {MODEL}   horizons {HORIZONS}   max distance {MAX_D}\n')

    records = load_all()
    if not records:
        print('No records loaded.')
        return

    print('Records with real gaps:')
    for name, s, step in records:
        grid = pd.date_range(s.index.min(), s.index.max(), freq=step)
        miss = len(grid) - len(s)
        per_day = int(pd.Timedelta('1D') / step)
        print(f"   {name:<12}{len(s):>8} samples  {100 * miss / len(grid):>6.2f}% "
              f"missing  {len(grid) / per_day:>6.1f} days  "
              f"step {int(step.total_seconds() / 60)} min")

    # PART A
    all_rows = []
    for name, s, step in records:
        for H in HORIZONS:
            r = profile_record(name, s, H, step)
            if r is not None:
                all_rows.append(r)
            gc.collect()
    if not all_rows:
        print('\nNo profiles produced.')
        return
    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(OUT_CSV, index=False)

    print('\nCoverage after walk-forward scoring '
          f'({N_FOLDS - 1} scored blocks per record):')
    print(f"  {'record':<12}{'scored':>9}{'near-gap (d<=12)':>18}"
          f"{'d=1 rows':>10}{'min d':>8}")
    for name in sorted(rows.record.unique()):
        g = rows[rows.record == name]
        print(f"  {name:<12}{len(g):>9}{int((g.d <= MAX_D).sum()):>18}"
              f"{int((g.d == 1).sum()):>10}{int(g.d.min()):>8}")

    prof = []
    for (name, H), g in rows.groupby(['record', 'horizon']):
        far = g[g.d > MAX_D]
        base_p = np.sqrt(far.sq_pers.mean()) if len(far) > 50 else np.nan
        base_m = np.sqrt(far.sq_model.mean()) if len(far) > 50 else np.nan
        for dd in range(1, MAX_D + 1):
            sub = g[g.d == dd]
            if len(sub) < MIN_ROWS_PER_D:
                continue
            prof.append(dict(
                record=name, horizon=H, d=dd, n=len(sub),
                rmse_pers=np.sqrt(sub.sq_pers.mean()),
                rmse_model=np.sqrt(sub.sq_model.mean()),
                rmse_pers_far=base_p, rmse_model_far=base_m,
                pers_inflation=np.sqrt(sub.sq_pers.mean()) / base_p
                if base_p == base_p else np.nan,
                model_inflation=np.sqrt(sub.sq_model.mean()) / base_m
                if base_m == base_m else np.nan,
                frac_displaced=float(sub.target_displaced.mean()),
                median_gap_before=float(sub.gap_before.median())))
    prof = pd.DataFrame(prof)
    prof.to_csv(OUT_PROFILE_CSV, index=False)

    print('\n' + '=' * 88)
    print('  PART A — ERROR BY DISTANCE FROM GAP  (d = 1 is the first')
    print('  observation after a gap; "far" means d > %d)' % MAX_D)
    print('  Inflation is RMSE at this distance divided by RMSE far from '
          'any gap.')
    print('=' * 88)
    for (name, H), g in prof.groupby(['record', 'horizon']):
        print(f"\n  {name}   H={H}   "
              f"(far-from-gap RMSE: persistence "
              f"{g.rmse_pers_far.iloc[0]:.4f}, model "
              f"{g.rmse_model_far.iloc[0]:.4f})")
        print(f"    {'d':>3}{'n':>7}{'RMSE pers':>11}{'RMSE model':>12}"
              f"{'pers infl':>11}{'model infl':>12}{'displaced':>11}")
        for _, r in g.iterrows():
            print(f"    {int(r.d):>3}{int(r.n):>7}{r.rmse_pers:>11.4f}"
                  f"{r.rmse_model:>12.4f}{r.pers_inflation:>11.2f}"
                  f"{r.model_inflation:>12.2f}{r.frac_displaced:>11.2f}")

    print('\n' + '=' * 88)
    print('  WHICH DEGRADES MORE AT d = 1? This is the sign question.')
    print('  persistence inflated MORE than the model  -> naive flatters')
    print('     the model -> POSITIVE bias')
    print('  model inflated more -> NEGATIVE bias')
    print('  CI is a bootstrap over the d=1 rows; it excludes 1.0 when the')
    print('  direction is statistically supported.')
    print('=' * 88)
    print(f"  {'record':<12}{'H':>3}{'n':>6}{'pers infl':>11}{'model infl':>12}"
          f"{'ratio':>8}{'95% CI':>20}   implied sign")
    for (name, H), g in prof.groupby(['record', 'horizon']):
        d1 = g[g.d == 1]
        if not len(d1):
            continue
        pi, mi = d1.pers_inflation.iloc[0], d1.model_inflation.iloc[0]
        if pi != pi or mi != mi:
            continue
        ratio = pi / mi
        sub = rows[(rows.record == name) & (rows.horizon == H) &
                   (rows.d == 1)]
        far = rows[(rows.record == name) & (rows.horizon == H) &
                   (rows.d > MAX_D)]
        lo = hi = np.nan
        if len(sub) >= 10 and len(far) > 50:
            fp = np.sqrt(far.sq_pers.mean())
            fm = np.sqrt(far.sq_model.mean())
            rng = np.random.default_rng(0)
            sp = sub.sq_pers.values
            sm = sub.sq_model.values
            idx = rng.integers(0, len(sp), size=(N_BOOT, len(sp)))
            bp = np.sqrt(sp[idx].mean(axis=1)) / fp
            bm = np.sqrt(sm[idx].mean(axis=1)) / fm
            br = bp / bm
            lo, hi = np.percentile(br, [2.5, 97.5])
        sign = 'POSITIVE' if lo == lo and lo > 1.0 else (
            'NEGATIVE' if hi == hi and hi < 1.0 else 'not resolved')
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else 'n/a'
        print(f"  {name:<12}{H:>3}{int(d1.n.iloc[0]):>6}{pi:>11.3f}"
              f"{mi:>12.3f}{ratio:>8.3f}{ci:>20}   {sign}")

    print('\n  Contribution of target displacement (rows whose target lies')
    print('  across a gap) versus rows with a stale input only:')
    print(f"  {'record':<12}{'H':>3}{'displaced n':>13}{'RMSE pers':>11}"
          f"{'not displ n':>13}{'RMSE pers':>11}")
    for (name, H), g in rows.groupby(['record', 'horizon']):
        a = g[g.target_displaced]
        b = g[~g.target_displaced]
        if len(a) < 20 or len(b) < 20:
            continue
        print(f"  {name:<12}{H:>3}{len(a):>13}"
              f"{np.sqrt(a.sq_pers.mean()):>11.4f}{len(b):>13}"
              f"{np.sqrt(b.sq_pers.mean()):>11.4f}")

    # ---------------------------------------------------------- PART B
    print('\n' + '=' * 88)
    print('  PART B — DO GAPS FALL DURING DIFFICULT PERIODS?')
    print(f'  Variability is the sd of first differences over the {PREGAP_WINDOW}')
    print('  samples immediately before each gap, against all clean windows.')
    print('=' * 88)
    ctx = [c for c in (gap_context(n, s, st) for n, s, st in records) if c]
    if ctx:
        cdf = pd.DataFrame(ctx)
        print(cdf.round(4).to_string(index=False))
        print('\n  ratio > 1 means conditions just before a gap are MORE')
        print('  variable than typical, so gaps cluster in harder periods.')
        n_hard = int((cdf.ratio > 1).sum())
        print(f'  ratio > 1 in {n_hard}/{len(cdf)} records; '
              f'significant (p<0.05) in '
              f'{int((cdf.mannwhitney_p < 0.05).sum())}/{len(cdf)}')
    else:
        cdf = pd.DataFrame()
        print('  Not enough gaps for the context test.')

    #  FIGURES
    recs = sorted(prof.record.unique())
    fig, axes = plt.subplots(len(HORIZONS), len(recs), squeeze=False,
                             figsize=(3.1 * len(recs), 3.4 * len(HORIZONS)),
                             sharex=True)
    for r_i, H in enumerate(HORIZONS):
        for c_i, name in enumerate(recs):
            ax = axes[r_i][c_i]
            g = prof[(prof.record == name) & (prof.horizon == H)]
            if len(g):
                ax.plot(g.d, g.rmse_pers, 'o-', color='#D85A30',
                        label='persistence' if (r_i == 0 and c_i == 0) else None)
                ax.plot(g.d, g.rmse_model, 's-', color='#1D9E75',
                        label=f'{MODEL}' if (r_i == 0 and c_i == 0) else None)
                if g.rmse_pers_far.iloc[0] == g.rmse_pers_far.iloc[0]:
                    ax.axhline(g.rmse_pers_far.iloc[0], color='#D85A30',
                               ls=':', lw=1)
                    ax.axhline(g.rmse_model_far.iloc[0], color='#1D9E75',
                               ls=':', lw=1)
            ax.grid(alpha=0.3)
            if r_i == 0:
                ax.set_title(name, fontsize=9, fontweight='bold')
            if c_i == 0:
                ax.set_ylabel(f'H = {H}\nRMSE (m/s)', fontsize=9)
            if r_i == len(HORIZONS) - 1:
                ax.set_xlabel('observations since gap (d)', fontsize=9)
    fig.legend(loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Forecast error by distance from a gap. Dotted lines are '
                 'the far-from-gap level.', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_FIG}')

    if len(cdf):
        fig2, ax = plt.subplots(figsize=(1.5 + 1.1 * len(cdf), 4))
        x = np.arange(len(cdf))
        ax.bar(x - 0.2, cdf.baseline_sd, 0.4, label='typical window',
               color='#4A7BB7')
        ax.bar(x + 0.2, cdf.pre_gap_sd, 0.4, label='window before a gap',
               color='#D85A30')
        for i, p in enumerate(cdf.mannwhitney_p):
            if p < 0.05:
                ax.text(i, max(cdf.baseline_sd.iloc[i],
                               cdf.pre_gap_sd.iloc[i]) * 1.03, '*',
                        ha='center', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(cdf.record, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('sd of first differences (m/s)', fontsize=9)
        ax.set_title('Conditions immediately before a gap vs typical\n'
                     '(* Mann-Whitney p < 0.05)', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3, axis='y')
        fig2.tight_layout()
        fig2.savefig(OUT_FIG_CTX, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {OUT_FIG_CTX}')

    print(f'\nSaved {OUT_CSV} and {OUT_PROFILE_CSV}')


if __name__ == '__main__':
    main()
