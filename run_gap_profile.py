import warnings, os, sys, gc, glob, platform, datetime
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import mannwhitneyu, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
import dataset_adapters as DA
from knmi_adapter import load_knmi_platform

try:
    sys.stdout.reconfigure(encoding='utf-8')   # logs are UTF-8 on any OS
except AttributeError:
    pass


HORIZONS = (1, 3, 6)
MAX_D = 12                    # profile distances d = 1 .. MAX_D
FAR_MIN_D = 60                # "far": beyond the deepest feature window
                              # (lag_60, roll_*_60); see far_mask()
N_FOLDS = 6                   # block 0 trains only; blocks 1-5 are scored
MIN_ROWS_PER_D = 15           # do not report a distance with fewer rows
N_BOOT = 2000                 # bootstrap replicates for the d=1 ratio
PREGAP_WINDOW = 18            # samples before a gap for the context test
MODEL = 'Ridge'               # 'Ridge' or 'XGBoost'; Ridge is far stabler
                              # on highly autocorrelated wind records
# Part B figure / headline use the corrected context test; pass
# --legacy-context to use the original computation instead (both are printed).
LEGACY_CONTEXT = '--legacy-context' in sys.argv

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=4)

# KNMI platform folders with REAL gaps (full records, not clean blocks)
KNMI_FOLDERS = [('BSB', 'knmi_bsb'), ('HKN', 'knmi_hkn'),
                ('HKZB', 'knmi_hkzb'), ('HKWA', 'knmi_hkwa'),
                ('HKWB', 'knmi_hkwb'), ('HKZA', 'knmi_hkza')]

INCLUDE_D1 = True
D1_DIR = HERE                 # the ZephIR_windlidar_BSA_1s_*.CSV files
BUOY_DIR = 'buoy_data'

OUT_CSV = os.path.join(HERE, 'gap_profile_rows.csv')
OUT_PROFILE_CSV = os.path.join(HERE, 'gap_profile_summary.csv')
OUT_TABLE7_CSV = os.path.join(HERE, 'gap_profile_table7.csv')
OUT_CTX_CSV = os.path.join(HERE, 'gap_context_summary.csv')
OUT_NUMBERS_CSV = os.path.join(HERE, 'gap_profile_numbers.csv')
OUT_FIG = os.path.join(HERE, 'gap_profile_fig.png')
OUT_FIG_PAPER = os.path.join(HERE, 'gap_profile_fig_paper.png')
OUT_FIG_CTX = os.path.join(HERE, 'gap_context_fig.png')
DECOMP_CSV = os.path.join(HERE, 'artifact_decomposition.csv')
# composition channel (XGBoost) quoted in the paper, used only as fallback
PAPER_COMPOSITION_D3 = {1: 1.21, 3: 2.96, 6: 4.16}


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
    """d = position in the contiguous run, gap_before = length of the gap
    before the run, to_gap = grid steps to the next missing sample."""
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
    # no gap ahead before the record end: treat as infinitely far
    tg = np.zeros(n, dtype=np.int64)
    nxt = 10 ** 9
    for i in range(n - 1, -1, -1):
        if not present[i]:
            nxt = i
        tg[i] = nxt - i
    return s, present, d, gb, tg


def far_mask(g, H, legacy=False):
    """Rows used as the far-from-gap reference.
    legacy: d > MAX_D (original; still contains rows whose 60-step features
    span a gap, and target-displaced rows at the end of runs)."""
    if legacy:
        return g.d > MAX_D
    return (g.d > FAR_MIN_D) & (~g.target_displaced) & (g.to_gap > H)


def profile_record(name, series, H, step):

    s_grid, present, d_arr, gb_arr, tg_arr = grid_and_runs(series, step)
    meta = pd.DataFrame({'d': d_arr, 'gap_before': gb_arr, 'to_gap': tg_arr},
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
    out['to_gap'] = out['to_gap'].fillna(0).astype(np.int64)
    out['target_displaced'] = out['target_lag_min'] > nominal + 1e-6
    return out


def gap_context(name, series, step, legacy=False):
    """Part B. Corrected version (default):
      - pre-gap window = the W samples before a gap start, used only if all
        W are present (so every difference is a one-step difference);
      - baseline = non-overlapping (stride W) fully present windows, excluding
        any window whose end lies within W samples of a gap start.
    legacy=True reproduces the original computation (NaNs dropped inside the
    pre-gap window, so differences can span earlier gaps; baseline stride
    W//2, pre-gap windows not excluded)."""
    s_grid, present, d_arr, _, _ = grid_and_runs(series, step)
    v = s_grid.values
    n = len(v)
    starts = np.array([i for i in range(1, n)
                       if present[i - 1] and not present[i]], dtype=int)
    W = PREGAP_WINDOW

    pre_sd, pre_ramp = [], []
    for i in starts:
        seg = v[max(0, i - W):i]
        if legacy:
            seg = seg[~np.isnan(seg)]
            if len(seg) < max(5, W // 2):
                continue
        elif i < W or np.isnan(seg).any():
            continue
        dd = np.diff(seg)
        pre_sd.append(np.std(dd))
        pre_ramp.append(np.max(np.abs(dd)) if len(dd) else np.nan)

    base_sd, base_ramp = [], []
    stride = max(1, W // 2) if legacy else W
    for j in range(W, n, stride):
        seg = v[j - W:j]
        if np.isnan(seg).any():
            continue
        if not legacy and len(starts):
            k = np.searchsorted(starts, j)      # nearest gap start >= j
            near = False
            for kk in (k - 1, k):
                if 0 <= kk < len(starts) and abs(starts[kk] - j) <= W:
                    near = True
            if near:
                continue
        dd = np.diff(seg)
        base_sd.append(np.std(dd))
        base_ramp.append(np.max(np.abs(dd)))

    if len(pre_sd) < 10 or len(base_sd) < 30:
        print(f"  Part B {'legacy' if legacy else 'clean'}: {name} skipped "
              f"({len(pre_sd)} usable of {len(starts)} gaps, "
              f"{len(base_sd)} baseline windows)")
        return None
    _, p = mannwhitneyu(pre_sd, base_sd, alternative='two-sided')
    _, p_gt = mannwhitneyu(pre_sd, base_sd, alternative='greater')
    return dict(record=name, method='legacy' if legacy else 'clean',
                n_gap_starts=len(starts), n_gaps=len(pre_sd),
                n_baseline=len(base_sd),
                pre_gap_sd=float(np.median(pre_sd)),
                baseline_sd=float(np.median(base_sd)),
                ratio=float(np.median(pre_sd) / np.median(base_sd)),
                pre_gap_ramp=float(np.nanmedian(pre_ramp)),
                baseline_ramp=float(np.median(base_ramp)),
                mannwhitney_p=float(p), mannwhitney_p_greater=float(p_gt))


def load_all():

    recs = []
    for name, folder in KNMI_FOLDERS:
        p = os.path.join(HERE, folder)
        if not os.path.isdir(p):
            print(f"  {name}: folder {folder} not found, skipped")
            continue
        df = load_knmi_platform(p, verbose=False)
        s = df['wind_speed'].dropna()
        if len(s) > 1000:
            recs.append((name, s, pd.Timedelta('10min')))

    if INCLUDE_D1:
        try:
            r = DA.load_zephir(D1_DIR, height_m=38, resample='1min')
        except FileNotFoundError as e:
            print(f"  D1 not loaded: {e}")
        else:
            s = r[0] if isinstance(r, tuple) else r
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            recs.append(('D1_ZEPHIR', s.dropna(), pd.Timedelta('1min')))

    bp = os.path.join(HERE, BUOY_DIR)
    try:
        r = DA.load_wfip3_buoy(bp, height_m=38, resample='10min')
    except FileNotFoundError as e:
        print(f"  D2 not loaded: {e}")
    else:
        s = r[0] if isinstance(r, tuple) else r
        recs.append(('D2_BUOY', s.dropna(), pd.Timedelta('10min')))

    try:
        s = FP.load_onshore(HERE)
    except FileNotFoundError as e:
        print(f"  D3 not loaded: {e}")
    else:
        recs.append(('D3_ONSHORE', s, pd.Timedelta('10min')))
    return recs


def far_rmse(g, H, legacy):
    far = g[far_mask(g, H, legacy)]
    if len(far) <= 50:
        return np.nan, np.nan, len(far)
    return (np.sqrt(far.sq_pers.mean()), np.sqrt(far.sq_model.mean()),
            len(far))


def d1_ratio_ci(sub, fp, fm):
    """Paired iid bootstrap over the d=1 rows; far RMSEs held fixed."""
    if len(sub) < 10 or fp != fp or fm != fm:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    sp = sub.sq_pers.values
    sm = sub.sq_model.values
    idx = rng.integers(0, len(sp), size=(N_BOOT, len(sp)))
    bp = np.sqrt(sp[idx].mean(axis=1)) / fp
    bm = np.sqrt(sm[idx].mean(axis=1)) / fm
    lo, hi = np.percentile(bp / bm, [2.5, 97.5])
    return lo, hi


def sign_of(lo, hi):
    return 'POSITIVE' if lo == lo and lo > 1.0 else (
        'NEGATIVE' if hi == hi and hi < 1.0 else 'not resolved')


def annotate_n(ax, g):
    """Small n labels at the top of the panel, one per plotted d."""
    for _, r in g.iterrows():
        ax.annotate(f'{int(r.n)}', (r.d, 0.97), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=5.5, color='0.35')


def draw_profile(ax, g, legend=False):
    ax.plot(g.d, g.rmse_pers, 'o-', color='#D85A30', ms=4,
            label='persistence' if legend else None)
    ax.plot(g.d, g.rmse_model, 's-', color='#1D9E75', ms=4,
            label=f'{MODEL}' if legend else None)
    if g.rmse_pers_far.iloc[0] == g.rmse_pers_far.iloc[0]:
        ax.axhline(g.rmse_pers_far.iloc[0], color='#D85A30', ls=':', lw=1)
        ax.axhline(g.rmse_model_far.iloc[0], color='#1D9E75', ls=':', lw=1)
    lo = min(g.rmse_pers.min(), g.rmse_model.min(), g.rmse_pers_far.iloc[0],
             g.rmse_model_far.iloc[0])
    hi = max(g.rmse_pers.max(), g.rmse_model.max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.18 * (hi - lo))  # room for n
    annotate_n(ax, g)
    ax.set_xlim(0.5, MAX_D + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def main():
    print(f"PROVENANCE run_gap_profile.py  run {datetime.datetime.now():%Y-%m-%d %H:%M}"
          f"  script mtime {datetime.datetime.fromtimestamp(os.path.getmtime(__file__)):%Y-%m-%d %H:%M}"
          f"  {platform.system()} python {platform.python_version()}"
          f"  numpy {np.__version__} pandas {pd.__version__} sklearn {sklearn.__version__}"
          f"  scipy {scipy.__version__}  context={'legacy' if LEGACY_CONTEXT else 'clean'}")
    print(f'  model {MODEL}   horizons {HORIZONS}   max distance {MAX_D}   '
          f'far = d > {FAR_MIN_D}, target not displaced, next gap > H ahead '
          f'(legacy far = d > {MAX_D})\n')

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
          f"{'d=1 rows':>10}{'min d':>8}{'far rows':>10}{'legacy far':>12}")
    for name in sorted(rows.record.unique()):
        g = rows[rows.record == name]
        nf = sum(int(far_mask(g[g.horizon == H], H).sum()) for H in HORIZONS)
        print(f"  {name:<12}{len(g):>9}{int((g.d <= MAX_D).sum()):>18}"
              f"{int((g.d == 1).sum()):>10}{int(g.d.min()):>8}{nf:>10}"
              f"{int((g.d > MAX_D).sum()):>12}")

    prof = []
    for (name, H), g in rows.groupby(['record', 'horizon']):
        base_p, base_m, _ = far_rmse(g, H, legacy=False)
        old_p, old_m, _ = far_rmse(g, H, legacy=True)
        for dd in range(1, MAX_D + 1):
            sub = g[g.d == dd]
            if len(sub) < MIN_ROWS_PER_D:
                continue
            rp = np.sqrt(sub.sq_pers.mean())
            rm = np.sqrt(sub.sq_model.mean())
            prof.append(dict(
                record=name, horizon=H, d=dd, n=len(sub),
                rmse_pers=rp, rmse_model=rm,
                rmse_pers_far=base_p, rmse_model_far=base_m,
                pers_inflation=rp / base_p, model_inflation=rm / base_m,
                frac_displaced=float(sub.target_displaced.mean()),
                median_gap_before=float(sub.gap_before.median()),
                rmse_pers_far_legacy=old_p, rmse_model_far_legacy=old_m,
                pers_inflation_legacy=rp / old_p,
                model_inflation_legacy=rm / old_m))
    prof = pd.DataFrame(prof)
    prof.to_csv(OUT_PROFILE_CSV, index=False)

    print('\n' + '=' * 88)
    print('  PART A — ERROR BY DISTANCE FROM GAP  (d = 1 is the first')
    print(f'  observation after a gap; "far" means d > {FAR_MIN_D}, target not')
    print('  displaced, next gap more than H samples ahead)')
    print('  Inflation is RMSE at this distance divided by RMSE far from '
          'any gap.')
    print('=' * 88)
    for (name, H), g in prof.groupby(['record', 'horizon']):
        print(f"\n  {name}   H={H}   "
              f"(far-from-gap RMSE: persistence "
              f"{g.rmse_pers_far.iloc[0]:.4f}, model "
              f"{g.rmse_model_far.iloc[0]:.4f};  legacy d>{MAX_D}: "
              f"{g.rmse_pers_far_legacy.iloc[0]:.4f}, "
              f"{g.rmse_model_far_legacy.iloc[0]:.4f})")
        print(f"    {'d':>3}{'n':>7}{'RMSE pers':>11}{'RMSE model':>12}"
              f"{'pers infl':>11}{'model infl':>12}{'displaced':>11}")
        for _, r in g.iterrows():
            print(f"    {int(r.d):>3}{int(r.n):>7}{r.rmse_pers:>11.4f}"
                  f"{r.rmse_model:>12.4f}{r.pers_inflation:>11.2f}"
                  f"{r.model_inflation:>12.2f}{r.frac_displaced:>11.2f}")

    print('\n' + '=' * 88)
    print('  WHICH DEGRADES MORE AT d = 1? This is the sign question (Table VII).')
    print('  persistence inflated MORE than the model  -> naive flatters')
    print('     the model -> POSITIVE bias')
    print('  model inflated more -> NEGATIVE bias')
    print('  CI is a bootstrap over the d=1 rows; it excludes 1.0 when the')
    print('  direction is statistically supported. Left: new far definition;')
    print(f'  right: legacy far (d > {MAX_D}).')
    print('=' * 88)
    print(f"  {'record':<12}{'H':>3}{'n':>6}{'n far':>7}{'pers infl':>10}"
          f"{'model infl':>11}{'ratio':>7}{'95% CI':>17}  {'sign':<13}|"
          f"{'pers':>7}{'model':>7}{'ratio':>7}{'95% CI':>17}  sign")
    t7 = []
    for (name, H), g in rows.groupby(['record', 'horizon']):
        sub = g[g.d == 1]
        if len(sub) < MIN_ROWS_PER_D:
            continue
        rp = np.sqrt(sub.sq_pers.mean())
        rm = np.sqrt(sub.sq_model.mean())
        fp, fm, nfar = far_rmse(g, H, legacy=False)
        op, om, nfar_old = far_rmse(g, H, legacy=True)
        if fp != fp or op != op:
            continue
        lo, hi = d1_ratio_ci(sub, fp, fm)
        olo, ohi = d1_ratio_ci(sub, op, om)
        rec = dict(record=name, horizon=H, n=len(sub), n_far=nfar,
                   pers_inflation=rp / fp, model_inflation=rm / fm,
                   ratio=(rp / fp) / (rm / fm), ci_lo=lo, ci_hi=hi,
                   sign=sign_of(lo, hi), n_far_legacy=nfar_old,
                   pers_inflation_legacy=rp / op,
                   model_inflation_legacy=rm / om,
                   ratio_legacy=(rp / op) / (rm / om), ci_lo_legacy=olo,
                   ci_hi_legacy=ohi, sign_legacy=sign_of(olo, ohi))
        t7.append(rec)
        print(f"  {name:<12}{H:>3}{len(sub):>6}{nfar:>7}"
              f"{rec['pers_inflation']:>10.3f}{rec['model_inflation']:>11.3f}"
              f"{rec['ratio']:>7.3f}{f'[{lo:.3f}, {hi:.3f}]':>17}  "
              f"{rec['sign']:<13}|{rec['pers_inflation_legacy']:>7.3f}"
              f"{rec['model_inflation_legacy']:>7.3f}{rec['ratio_legacy']:>7.3f}"
              f"{f'[{olo:.3f}, {ohi:.3f}]':>17}  {rec['sign_legacy']}")
    t7 = pd.DataFrame(t7)
    t7.to_csv(OUT_TABLE7_CSV, index=False)

    print('\n  Contribution of target displacement (rows whose target lies')
    print('  across a gap) versus rows with a stale input only:')
    print(f"  {'record':<12}{'H':>3}{'displaced n':>13}{'RMSE pers':>11}"
          f"{'not displ n':>13}{'RMSE pers':>11}{'all n':>8}{'RMSE pers':>11}"
          f"{'% displ':>9}")
    disp = {}
    for (name, H), g in rows.groupby(['record', 'horizon']):
        a = g[g.target_displaced]
        b = g[~g.target_displaced]
        disp[(name, H)] = (len(a), np.sqrt(a.sq_pers.mean()), len(b),
                           np.sqrt(b.sq_pers.mean()), len(g),
                           np.sqrt(g.sq_pers.mean()))
        if len(a) < 20 or len(b) < 20:
            continue
        print(f"  {name:<12}{H:>3}{len(a):>13}"
              f"{np.sqrt(a.sq_pers.mean()):>11.4f}{len(b):>13}"
              f"{np.sqrt(b.sq_pers.mean()):>11.4f}{len(g):>8}"
              f"{np.sqrt(g.sq_pers.mean()):>11.4f}{100 * len(a) / len(g):>9.1f}")

    # ---------------------------------------------------------- PART B
    print('\n' + '=' * 88)
    print('  PART B — DO GAPS FALL DURING DIFFICULT PERIODS?')
    print(f'  Variability is the sd of first differences over the {PREGAP_WINDOW}')
    print('  samples immediately before each gap.')
    print('  clean : pre-gap window fully present; baseline = non-overlapping')
    print(f'          fully present windows (stride {PREGAP_WINDOW}) not ending '
          f'within {PREGAP_WINDOW} samples of a gap start')
    print('  legacy: original (NaNs dropped inside the pre-gap window, so')
    print('          differences can span earlier gaps; baseline stride '
          f'{PREGAP_WINDOW // 2})')
    print('  "supports" = ratio > 1 AND two-sided Mann-Whitney p < 0.05')
    print('=' * 88)
    ctx_all = {}
    for legacy in (False, True):
        ctx = [c for c in (gap_context(n, s, st, legacy=legacy)
                           for n, s, st in records) if c]
        cdf = pd.DataFrame(ctx)
        ctx_all['legacy' if legacy else 'clean'] = cdf
        print(f"\n  [{'legacy' if legacy else 'clean'}]")
        if not len(cdf):
            print('  Not enough gaps for the context test.')
            continue
        print(cdf.drop(columns='method').round(4).to_string(index=False))
        n_sup = int(((cdf.ratio > 1) & (cdf.mannwhitney_p < 0.05)).sum())
        n_opp = int(((cdf.ratio < 1) & (cdf.mannwhitney_p < 0.05)).sum())
        print(f'  ratio > 1 in {int((cdf.ratio > 1).sum())}/{len(cdf)} records; '
              f'significant AND ratio > 1 (supports harder conditions) in '
              f'{n_sup}/{len(cdf)}; significant with ratio < 1 (calmer) in '
              f'{n_opp}/{len(cdf)}; Holm-adjusted supports in '
              f'{holm_support(cdf)}/{len(cdf)}')
    pd.concat(ctx_all.values(), ignore_index=True).to_csv(OUT_CTX_CSV,
                                                          index=False)
    cdf = ctx_all['legacy' if LEGACY_CONTEXT else 'clean']

    # ------------------------------------------- hand-computed paper numbers
    nums = []

    def put(item, key, value, note=''):
        nums.append(dict(item=item, key=key, value=value, note=note))
        print(f"  {item:<4}{key:<58}{value:>10.4f}  {note}"
              if isinstance(value, float) else
              f"  {item:<4}{key:<58}{value!s:>10}  {note}")

    print('\n' + '=' * 88)
    print('  NUMBERS QUOTED IN SECTION IV-E (saved to gap_profile_numbers.csv)')
    print('=' * 88)
    # (a) D3 d=1 ratio vs composition channel
    comp = {}
    src = 'paper values (+1.21, +2.96, +4.16; artifact_decomposition.csv absent)'
    if os.path.exists(DECOMP_CSV):
        dc = pd.read_csv(DECOMP_CSV)
        dc = dc[dc.dataset.str.startswith('D3')]
        for mdl in ('XGBoost', 'Ridge'):
            sel = dc[dc.model == mdl].set_index('horizon')['bias_composition']
            comp[mdl] = {H: float(sel[H]) for H in HORIZONS if H in sel.index}
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(DECOMP_CSV))
        src = f'artifact_decomposition.csv (mtime {mt:%Y-%m-%d %H:%M})'
    else:
        comp['XGBoost'] = dict(PAPER_COMPOSITION_D3)
    print(f'  composition channel source: {src}')
    d3 = t7[t7.record == 'D3_ONSHORE'].set_index('horizon')
    for mdl, cv in comp.items():
        hs = [H for H in HORIZONS if H in cv and H in d3.index]
        if len(hs) < 3:
            continue
        for H in hs:
            put('a', f'D3 composition channel {mdl} H={H} (pp)', cv[H], src)
        for col, lab in (('ratio', 'new far'), ('ratio_legacy', 'legacy far')):
            r = float(np.corrcoef([d3.loc[H, col] for H in hs],
                                  [cv[H] for H in hs])[0, 1])
            put('a', f'Pearson r D3 d=1 ratio ({lab}) vs {mdl} composition',
                r, 'n=3, 1 dof')
    # (b) cross-record: Part B ratio vs H=1 d=1 ratio
    t1 = t7[t7.horizon == 1].set_index('record')
    for meth, cd in ctx_all.items():
        if not len(cd):
            continue
        cd = cd.set_index('record')
        common = [r for r in cd.index if r in t1.index]
        for col, lab in (('ratio', 'new far'), ('ratio_legacy', 'legacy far')):
            if len(common) < 3:
                continue
            r, p = pearsonr(cd.loc[common, 'ratio'], t1.loc[common, col])
            put('b', f'Pearson PartB({meth}) vs H=1 d=1 ratio ({lab}) r',
                float(r), f'n={len(common)} records')
            put('b', f'Pearson PartB({meth}) vs H=1 d=1 ratio ({lab}) p',
                float(p), f'n={len(common)} records')
    # (c) displaced rows, D3
    for H in HORIZONS:
        if ('D3_ONSHORE', H) not in disp:
            continue
        na, ra, nb, rb, nt, rt = disp[('D3_ONSHORE', H)]
        put('c', f'D3 H={H} displaced rows n', na)
        put('c', f'D3 H={H} displaced rows % of scored origins',
            float(100 * na / nt))
        put('c', f'D3 H={H} persistence RMSE displaced', float(ra))
        put('c', f'D3 H={H} persistence RMSE not displaced', float(rb),
            'paper "0.668" is this' if H == 1 else '')
        put('c', f'D3 H={H} persistence RMSE all rows', float(rt))
        put('c', f'D3 H={H} penalty displaced/not displaced', float(ra / rb))
    pd.DataFrame(nums).to_csv(OUT_NUMBERS_CSV, index=False)

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
                draw_profile(ax, g, legend=(r_i == 0 and c_i == 0))
            if r_i == 0:
                ax.set_title(name, fontsize=9, fontweight='bold')
            if c_i == 0:
                ax.set_ylabel(f'H = {H}\nRMSE (m/s)', fontsize=9)
            if r_i == len(HORIZONS) - 1:
                ax.set_xlabel('observations since gap (d)', fontsize=9)
    fig.legend(loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Forecast error by distance from a gap. Dotted lines are '
                 f'the far-from-gap level (d > {FAR_MIN_D}, target not '
                 'displaced); grey numbers are rows per point.',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_FIG}')

    # paper Fig. 2: D3 top, D2 bottom, H = 1, 3, 6
    paper_recs = [('D3_ONSHORE', 'D3 onshore'), ('D2_BUOY', 'D2 buoy')]
    fig, axes = plt.subplots(2, len(HORIZONS), squeeze=False,
                             figsize=(3.3 * len(HORIZONS), 5.6), sharex=True)
    for r_i, (name, lab) in enumerate(paper_recs):
        for c_i, H in enumerate(HORIZONS):
            ax = axes[r_i][c_i]
            g = prof[(prof.record == name) & (prof.horizon == H)]
            if len(g):
                draw_profile(ax, g, legend=(r_i == 0 and c_i == 0))
            ax.set_title(f'{lab}, H = {H}', fontsize=9, fontweight='bold')
            if c_i == 0:
                ax.set_ylabel('RMSE (m/s)', fontsize=9)
            if r_i == 1:
                ax.set_xlabel('observations since gap (d)', fontsize=9)
    fig.legend(loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Forecast error by distance from a gap. Dotted lines are '
                 'the far-from-gap level; grey numbers are rows per point.',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(OUT_FIG_PAPER, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_FIG_PAPER}')

    if len(cdf):
        fig2, ax = plt.subplots(figsize=(1.5 + 1.1 * len(cdf), 4))
        x = np.arange(len(cdf))
        ax.bar(x - 0.2, cdf.baseline_sd, 0.4, label='typical window',
               color='#4A7BB7')
        ax.bar(x + 0.2, cdf.pre_gap_sd, 0.4, label='window before a gap',
               color='#D85A30')
        for i, (p, rt) in enumerate(zip(cdf.mannwhitney_p, cdf.ratio)):
            if p < 0.05:
                ax.text(i, max(cdf.baseline_sd.iloc[i],
                               cdf.pre_gap_sd.iloc[i]) * 1.03,
                        '▲' if rt > 1 else '▼', ha='center', fontsize=10,
                        color='#D85A30' if rt > 1 else '#4A7BB7')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r}\n(n={k})' for r, k in
                            zip(cdf.record, cdf.n_gaps)],
                           rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('sd of first differences (m/s)', fontsize=9)
        ax.set_ylim(0, max(cdf.baseline_sd.max(), cdf.pre_gap_sd.max()) * 1.15)
        ax.set_title('Conditions immediately before a gap vs typical\n'
                     '(▲ more / ▼ less variable, Mann-Whitney p < 0.05)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3, axis='y')
        fig2.tight_layout()
        fig2.savefig(OUT_FIG_CTX, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {OUT_FIG_CTX}')

    print(f'\nSaved {OUT_CSV}, {OUT_PROFILE_CSV}, {OUT_TABLE7_CSV}, '
          f'{OUT_CTX_CSV} and {OUT_NUMBERS_CSV}')


def holm_support(cdf, alpha=0.05):
    """Records with ratio > 1 that stay significant after Holm correction
    over all records tested."""
    p = cdf.mannwhitney_p.values
    order = np.argsort(p)
    m = len(p)
    ok = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (m - rank):
            break
        ok[i] = True
    return int((ok & (cdf.ratio.values > 1)).sum())


if __name__ == '__main__':
    main()
