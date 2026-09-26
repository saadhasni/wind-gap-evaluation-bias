import warnings, os, sys, gc, time
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_wfip3_buoy
from knmi_adapter import load_knmi_platform, longest_clean_block

# CONFIG
# <-- set False for the full run (or run with INJECTION_QUICK=1)
QUICK = os.environ.get('INJECTION_QUICK', '0') == '1'

STEP = pd.Timedelta('10min')
SEQ_LEN = 12
HORIZONS = (1, 3)
MISSING_FRACS = (0.05, 0.10, 0.20)

# nominal gap lengths in 10-minute samples
CONDITIONS = {'MANY-SHORT': 3,      # 30 min
              'MEDIUM': 18,         # 3 h
              'FEW-LONG': 144}      # 24 h

N_REPLICATES = 30             # distinct gap placements
N_ORIGINS = 5                 # rolling split boundaries, cycled
ORIGIN_FRACS = (0.60, 0.65, 0.70, 0.75, 0.80)
TEST_SPAN = 0.25              # test window length as a fraction of record


MIN_TRAIN_HONEST = 350
MIN_TEST_HONEST = 120

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbosity=0,
           n_jobs=int(os.environ.get('INJECTION_NJOBS', 4)))  # threads only
RIDGE_ALPHA = 1.0
MODELS = ('XGBoost', 'Ridge')

RECORDS = [('HKZA', 'knmi_hkza'), ('HKWB', 'knmi_hkwb'),
           ('HKWA', 'knmi_hkwa'), ('HKZB', 'knmi_hkzb'),
           ('BUOY130', None)]
BUOY_DIR = 'buoy_data'

OUT_CSV = 'gap_injection_v3_results.csv'
OUT_FIG = 'gap_injection_v3_fig.png'          # Ridge (paper Fig. 3)
OUT_FIG_XGB = 'gap_injection_v3_fig_xgb.png'  # XGBoost

# A run is flagged `reliable` when the honest arm has not collapsed and its
# persistence denominator is representative of the intact record.
RELIABLE_MIN_SKILL = -50.0
RELIABLE_RP_RANGE = (0.7, 1.4)
OLD_BUOY_BASE = 6120          # interpolated base used for the shipped results

if QUICK:
    RECORDS = RECORDS[:2]
    MISSING_FRACS = (0.05, 0.10)
    HORIZONS = (1,)
    N_REPLICATES = 4
    OUT_CSV = 'gap_injection_v3_QUICK.csv'
    OUT_FIG = 'gap_injection_v3_QUICK.png'
    OUT_FIG_XGB = 'gap_injection_v3_QUICK_xgb.png'

OUT_CSV = os.path.join(HERE, OUT_CSV)
OUT_FIG = os.path.join(HERE, OUT_FIG)
OUT_FIG_XGB = os.path.join(HERE, OUT_FIG_XGB)


# gap injection
def inject_gaps_exact(series, target_drop, nominal_len, seed, margin=200):
    """
    Delete EXACTLY target_drop samples in gaps of approximately
    nominal_len, non-overlapping, away from the edges.

    n_gaps is rounded rather than floored, and the lengths are
    n_drop // n_gaps with the remainder spread one sample at a time, so
    the achieved missing fraction matches the target exactly. The actual
    mean gap length is returned so it can be reported rather than assumed.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    n_gaps = max(1, int(round(target_drop / nominal_len)))
    base_len = target_drop // n_gaps
    rem = target_drop - base_len * n_gaps
    lengths = [base_len + (1 if i < rem else 0) for i in range(n_gaps)]
    if base_len < 1:
        return None, 0, 0.0
    rng.shuffle(lengths)

    forbidden = np.zeros(n, dtype=bool)
    forbidden[:margin] = True
    forbidden[-margin:] = True
    drop = np.zeros(n, dtype=bool)
    placed, placed_lens = 0, []

    for glen in lengths:
        ok = False
        for _ in range(400):
            s0 = int(rng.integers(margin, n - margin - glen))
            # 1-sample buffer so gaps never merge into one longer gap
            if forbidden[max(0, s0 - 1):s0 + glen + 1].any():
                continue
            forbidden[s0:s0 + glen] = True
            drop[s0:s0 + glen] = True
            placed += 1
            placed_lens.append(glen)
            ok = True
            break
        if not ok:
            break

    if not placed_lens:
        return None, 0, 0.0
    return series[~drop], placed, float(np.mean(placed_lens))


#  features
def naive_features(series):
    """Gap-ignoring feature construction (common practice). Unchanged."""
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
        m = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    del m
    return p


#  the evaluation
def window_bounds(base, H, origin_i):
    """
    Train/test boundaries as TIMESTAMPS on the original contiguous
    timeline. Both protocols use these, so both score the same calendar
    interval and differ only in which rows inside it they retain.
    """
    n = len(base)
    f_o = ORIGIN_FRACS[origin_i % len(ORIGIN_FRACS)]
    train_end = int(f_o * n)
    test_start = train_end + (SEQ_LEN + H)          # embargo
    test_end = min(n - 1, test_start + int(TEST_SPAN * n))
    if test_start >= test_end:
        return None
    return (base.index[train_end], base.index[test_start],
            base.index[test_end])


def evaluate_arms(gapped, H, bounds, model_name):
    """
    Honest and naive evaluation over the SAME calendar window.
    Returns a dict of every component, or None if guards fail.
    """
    t_train_end, t_test_start, t_test_end = bounds

    #  honest: gap-aware features, gap-valid rows only
    fg = FP.build_features_segmented(gapped, STEP)
    cols = [c for c in fg.columns if c != '_seg']
    mask = FP.valid_rows_for_horizon(gapped, fg, H, SEQ_LEN)
    pos = np.where(mask)[0]
    if len(pos) < MIN_TRAIN_HONEST + MIN_TEST_HONEST:
        return None
    t = gapped.index[pos]
    tr = pos[t <= t_train_end]
    te = pos[(t >= t_test_start) & (t <= t_test_end)]
    if len(tr) < MIN_TRAIN_HONEST or len(te) < MIN_TEST_HONEST:
        return None

    y_te = gapped.values[te + H]
    rp_h = np.sqrt(mean_squared_error(y_te, fg.iloc[te]['lag_0'].values))
    pred = fit_predict(model_name, fg.iloc[tr][cols].values,
                       gapped.values[tr + H], fg.iloc[te][cols].values)
    ra_h = np.sqrt(mean_squared_error(y_te, pred))

    # naive: gap-ignoring features, every row in window
    # target is H ROWS ahead, so after a gap a training label can fall inside
    # the test window; naive_calendar_split drops those training rows.
    fn = naive_features(gapped)
    dn = FP.naive_frame(gapped, H, fnav=fn)
    fc = [c for c in dn.columns if c not in ('target', '_target_ts')]
    tr2, te2 = FP.naive_calendar_split(dn, t_train_end, t_test_start,
                                       t_test_end)
    if len(tr2) < MIN_TRAIN_HONEST or len(te2) < MIN_TEST_HONEST:
        return None

    yt2 = te2['target'].values
    rp_n = np.sqrt(mean_squared_error(yt2, te2['lag_0'].values))
    pred2 = fit_predict(model_name, tr2[fc].values, tr2['target'].values,
                        te2[fc].values)
    ra_n = np.sqrt(mean_squared_error(yt2, pred2))

    out = dict(
        model=model_name,
        rmse_pers_honest=rp_h, rmse_model_honest=ra_h,
        rmse_pers_naive=rp_n, rmse_model_naive=ra_n,
        skill_honest=100 * (1 - ra_h / rp_h),
        skill_naive=100 * (1 - ra_n / rp_n),
        excess_honest=ra_h - rp_h, excess_naive=ra_n - rp_n,
        n_train_honest=len(tr), n_test_honest=len(te),
        n_train_naive=len(tr2), n_test_naive=len(te2),
        test_start=str(t_test_start), test_end=str(t_test_end),
        test_target_std=float(np.std(y_te)),
    )
    out['bias'] = out['skill_naive'] - out['skill_honest']
    out['rmse_bias'] = out['excess_naive'] - out['excess_honest']
    del fg, fn, dn, tr2, te2
    return out


# record loading
def intact_model_skill(base, H, model_name):
    """No-gap reference: honest-arm skill on the intact record, per origin."""
    out = {}
    for i in range(len(ORIGIN_FRACS)):
        b = window_bounds(base, H, i)
        if b is None:
            continue
        r = evaluate_arms(base, H, b, model_name)
        if r is not None:
            out[i] = r['skill_honest']
    return out


def intact_persistence_rmse(base, H):
    """
    Persistence RMSE on the intact record over the same style of test
    window. Used as a REFERENCE denominator: if the honest arm's rp
    departs far from this, its test window is unrepresentative and the
    skill ratio should not be trusted.
    """
    vals = []
    for i in range(len(ORIGIN_FRACS)):
        b = window_bounds(base, H, i)
        if b is None:
            continue
        _, t0, t1 = b
        m = (base.index >= t0) & (base.index <= t1)
        pos = np.where(m)[0]
        pos = pos[pos + H < len(base)]
        if len(pos) < 50:
            continue
        vals.append(np.sqrt(mean_squared_error(base.values[pos + H],
                                               base.values[pos])))
    return float(np.median(vals)) if vals else np.nan


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


def load_records():
    out = []
    for name, folder in RECORDS:
        if folder is None:
            bp = os.path.join(HERE, BUOY_DIR)
            if not os.path.isdir(bp):
                print(f"  {name}: {BUOY_DIR} not found, skipping")
                continue
            # raw series: the interpolated one would put filled points
            # inside the base record
            s_all, _ = load_wfip3_buoy(bp, height_m=38, resample='10min',
                                       interpolate=False)
            s = buoy_longest_contiguous(s_all, STEP)
            print(f"  {name}: longest RAW contiguous block {len(s)} samples "
                  f"({len(s) / 144:.1f} d), {s.index[0]} -> {s.index[-1]}; "
                  f"old interpolated base {OLD_BUOY_BASE} samples "
                  f"({OLD_BUOY_BASE / 144:.1f} d)")
            out.append((name, s))
            continue
        p = os.path.join(HERE, folder)
        if not os.path.isdir(p):
            print(f"  {name}: {folder} not found, skipping")
            continue
        df = load_knmi_platform(p, verbose=False)
        out.append((name, longest_clean_block(df)['wind_speed']))
    for name, s in out:
        assert (s.index.to_series().diff().dropna() == STEP).all(), \
            f"{name} is not contiguous"
    return out


def boot_ci(x, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_cluster(x, groups, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean, resampling whole clusters
    (here split origins, whose test windows overlap heavily)."""
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    if len(x) < 3:
        return np.nan, np.nan
    ids = np.unique(groups)
    sums = np.array([x[groups == g].sum() for g in ids])
    cnts = np.array([(groups == g).sum() for g in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_figure(df, model_name, path):
    """Grouped bars of mean bias, one panel per record x horizon. Each bar is
    labelled with n (surviving placements); cells with < 3 are 'n/a'."""
    order = list(CONDITIONS.keys())
    recs = list(df.record.unique())
    fig, axes = plt.subplots(len(HORIZONS), len(recs),
                             figsize=(3.4 * len(recs), 3.6 * len(HORIZONS)),
                             squeeze=False)
    colours = {'MANY-SHORT': '#1D9E75', 'MEDIUM': '#4A7BB7',
               'FEW-LONG': '#D85A30'}
    sub = df[df.model == model_name]
    w = 0.27
    for r_i, H in enumerate(HORIZONS):
        for c_i, rec in enumerate(recs):
            ax = axes[r_i][c_i]
            g = sub[(sub.horizon == H) & (sub.record == rec)]
            x = np.arange(len(MISSING_FRACS))
            labels = []
            for i, c in enumerate(order):
                means, los, his = [], [], []
                for j, f in enumerate(MISSING_FRACS):
                    gv = g[(g.missing_frac == f) & (g.condition == c)]
                    v = gv['bias']
                    xb = x[j] + (i - 1) * w
                    if len(v) < 3:
                        means.append(np.nan); los.append(0); his.append(0)
                        labels.append((xb, None, None, len(v)))
                    else:
                        lo, hi = boot_ci_cluster(v, gv['origin'])
                        means.append(v.mean())
                        los.append(max(0, v.mean() - lo))
                        his.append(max(0, hi - v.mean()))
                        labels.append((xb, v.mean(), hi, len(v)))
                ax.bar(x + (i - 1) * w, means, w,
                       yerr=[los, his], capsize=2,
                       label=c if (r_i == 0 and c_i == 0) else None,
                       color=colours[c])
            ax.axhline(0, color='k', lw=1)
            lo_y, hi_y = ax.get_ylim()
            span = hi_y - lo_y
            for xb, m, hi, n in labels:
                if m is None:
                    ax.text(xb, 0.01 * span, 'n/a', ha='center',
                            va='bottom', fontsize=6,
                            rotation=90, color='0.4')
                else:
                    top = max(m, hi if hi == hi else m, 0)
                    ax.text(xb, top + 0.01 * span, f'{n}', ha='center',
                            va='bottom', fontsize=5.5)
            ax.set_ylim(lo_y, hi_y + 0.08 * span)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{f:.0%}' for f in MISSING_FRACS], fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(alpha=0.3, axis='y')
            if r_i == 0:
                d = df[df.record == rec]['record_days'].iloc[0]
                ax.set_title(f'{rec}\n{d:.0f} days', fontsize=9,
                             fontweight='bold')
            if c_i == 0:
                ax.set_ylabel(f'H = {H}\nBias (percentage points)', fontsize=9)
            if r_i == len(HORIZONS) - 1:
                ax.set_xlabel('Total missing fraction', fontsize=9)
    fig.legend(loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    mlabel = 'ridge regression' if model_name == 'Ridge' else model_name
    fig.suptitle('Controlled gap injection across independent contiguous '
                 f'records ({mlabel})', fontsize=11, fontweight='bold',
                 y=1.03)
    fig.text(0.5, 0.945, 'bar = mean bias; number = surviving placements n; '
             'error bar = 95% CI, bootstrap over split origins; n/a = fewer '
             'than 3 placements survive; y-axes differ by panel',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {os.path.basename(path)}')


# main
def main():
    t0 = time.time()
    mode = 'QUICK SUBSET' if QUICK else 'FULL RUN'
    print(f"Gap injection v3 — {mode}")
    print(f"  records {len(RECORDS)}  fracs {MISSING_FRACS}  "
          f"horizons {HORIZONS}  replicates {N_REPLICATES}  "
          f"origins {N_ORIGINS}  models {MODELS}\n")

    records = load_records()
    if not records:
        print('No records loaded.')
        return

    print('Records (full clean blocks, contiguity verified):')
    for name, s in records:
        print(f"   {name:<9}{len(s):>7} samples{len(s) / 144:>7.1f} d   "
              f"mean {s.mean():>5.2f}   sd_diff {s.diff().std():.3f}")

    for name, s in records:
        g, _, _ = inject_gaps_exact(s, int(0.05 * len(s)), 3, seed=0)
        k = FP.causality_self_check(g, STEP)
        print(f"   {name:<9}G1 causality check (5% many-short example): "
              f"PASSED ({k} points)")

    rows, skipped = [], 0
    intact_rp, intact_sk = {}, {}
    for rec_name, base in records:
        for H in HORIZONS:
            intact_rp[(rec_name, H)] = intact_persistence_rmse(base, H)
            for mname in MODELS:
                intact_sk[(rec_name, H, mname)] = intact_model_skill(
                    base, H, mname)

    print('\nNO-GAP REFERENCE (intact record, same windows).')
    print('  rp = intact persistence RMSE (median over origins, the reference')
    print('  denominator). Model skill = median over origins; a model already')
    print('  far below zero here is failing on the record, not on the gaps.')
    print(f"   {'record':<9}{'H':>3}{'rp':>9}"
          + ''.join(f"{m + ' skill':>16}" for m in MODELS)
          + '   per-origin skill ' + '/'.join(MODELS))
    for rec_name, _ in records:
        for H in HORIZONS:
            line = f"   {rec_name:<9}{H:>3}{intact_rp[(rec_name, H)]:>9.4f}"
            per = []
            for mname in MODELS:
                d = intact_sk[(rec_name, H, mname)]
                v = float(np.median(list(d.values()))) if d else np.nan
                line += f"{v:>16.2f}" if v == v else f"{'n/a':>16}"
                per.append(' '.join(f"{d[k]:.1f}" for k in sorted(d)))
            print(line + '   ' + ' | '.join(per))

    for rec_name, base in records:
        n = len(base)
        print(f"\n{'=' * 84}")
        print(f"  {rec_name}   {n} samples ({n / 144:.1f} days)")
        print('=' * 84)
        print('  All numeric columns except n are MEDIANS over the surviving '
              'placements.')
        print(f"{'frac':>6}{'condition':>12}{'model':>9}{'H':>3}"
              f"{'gaps':>6}{'glen':>6}{'miss%':>7}"
              f"{'med_sk_h':>9}{'med_sk_n':>9}{'med_bias':>9}"
              f"{'med_rmse_b':>11}{'n':>4}")
        print('-' * 84)

        for frac in MISSING_FRACS:
            target_drop = int(frac * n)
            for cname, nominal in CONDITIONS.items():
                for H in HORIZONS:
                    for model_name in MODELS:
                        acc = []
                        for i in range(N_REPLICATES):
                            bounds = window_bounds(base, H, i)
                            if bounds is None:
                                continue
                            gapped, placed, glen = inject_gaps_exact(
                                base, target_drop, nominal, seed=i)
                            if gapped is None:
                                continue
                            r = evaluate_arms(gapped, H, bounds, model_name)
                            if r is None:
                                skipped += 1
                                continue
                            r.update(record=rec_name, record_len=n,
                                     record_days=round(n / 144, 1),
                                     missing_frac=frac, condition=cname,
                                     horizon=H, replicate=i,
                                     origin=i % N_ORIGINS,
                                     nominal_gap_len=nominal,
                                     n_gaps_placed=placed,
                                     mean_gap_len=round(glen, 1),
                                     achieved_missing_pct=round(
                                         100 * (1 - len(gapped) / n), 3))
                            ref = intact_rp.get((rec_name, H), np.nan)
                            r['intact_rp'] = round(ref, 4)
                            r['rp_h_over_intact'] = round(
                                r['rmse_pers_honest'] / ref, 3) if ref else np.nan
                            sk0 = intact_sk.get((rec_name, H, model_name),
                                                {}).get(i % N_ORIGINS, np.nan)
                            r['intact_skill'] = round(sk0, 3) \
                                if sk0 == sk0 else np.nan
                            # the gaps' effect on the honest arm, as opposed
                            # to the model's baseline skill on this window
                            r['honest_vs_intact'] = round(
                                r['skill_honest'] - sk0, 3) \
                                if sk0 == sk0 else np.nan
                            r['reliable'] = bool(
                                r['skill_honest'] > RELIABLE_MIN_SKILL and
                                RELIABLE_RP_RANGE[0] <= r['rp_h_over_intact']
                                <= RELIABLE_RP_RANGE[1])
                            rows.append(r)
                            acc.append(r)
                        if acc:
                            g = pd.DataFrame(acc)
                            print(f"{frac:>6.0%}{cname:>12}{model_name:>9}"
                                  f"{H:>3}{g.n_gaps_placed.median():>6.0f}"
                                  f"{g.mean_gap_len.median():>6.1f}"
                                  f"{g.achieved_missing_pct.median():>7.2f}"
                                  f"{g.skill_honest.median():>9.2f}"
                                  f"{g.skill_naive.median():>9.2f}"
                                  f"{g.bias.median():>+9.2f}"
                                  f"{g.rmse_bias.median():>+11.4f}"
                                  f"{len(g):>4}")
                        else:
                            print(f"{frac:>6.0%}{cname:>12}{model_name:>9}"
                                  f"{H:>3}{'not evaluable':>52}")
                        sys.stdout.flush()
                        gc.collect()
        if rows:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint: {os.path.basename(OUT_CSV)}]")

    if not rows:
        print('\nNo results produced.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    mins = (time.time() - t0) / 60
    print(f"\nSaved {os.path.basename(OUT_CSV)}  ({len(df)} runs, {skipped} skipped by guards, "
          f"{mins:.0f} min)")

    # -------- defect 3 check: are fractions actually matched now?
    print('\n' + '=' * 84)
    print('  CHECK: achieved missing fraction by condition '
          '(these must now agree)')
    print('=' * 84)
    print(df.groupby(['missing_frac', 'condition'])['achieved_missing_pct']
            .agg(['min', 'median', 'max']).round(3).to_string())
    print('\n  achieved mean gap length (samples, 1 = 10 min):')
    print(df.groupby(['missing_frac', 'condition'])['mean_gap_len']
            .median().round(1).to_string())

    # mechanism check: is the persistence baseline handicapped?
    print('\n' + '=' * 84)
    print('  CHECK: the mechanism. Naive evaluation should HANDICAP the')
    print('  persistence baseline by feeding it stale observations, so')
    print('  rp_naive / rp_honest should exceed 1, and the test set should')
    print('  grow. Bias should track how far these depart from 1.')
    print('=' * 84)
    df['rp_ratio'] = df.rmse_pers_naive / df.rmse_pers_honest
    df['test_growth'] = df.n_test_naive / df.n_test_honest
    m = (df.groupby(['record', 'condition'])
           .agg(rp_ratio=('rp_ratio', 'median'),
                test_growth=('test_growth', 'median'),
                rp_h_over_intact=('rp_h_over_intact', 'median'),
                bias=('bias', 'median'),
                n_test_h=('n_test_honest', 'median'),
                n=('bias', 'size')).round(3).reset_index())
    print(m.to_string(index=False))
    print('\n  Correlation of bias with the mechanism quantities:')
    print(f"    bias vs rp_ratio     pearson "
          f"{df.bias.corr(df.rp_ratio):+.3f}")
    print(f"    bias vs test_growth  pearson "
          f"{df.bias.corr(df.test_growth):+.3f}")

    # denominator sanity
    print('\n  Denominator sanity (rp_honest / intact rp; 1.0 is ideal):')
    bad = df[(df.rp_h_over_intact < 0.7) | (df.rp_h_over_intact > 1.4)]
    print(f"    median {df.rp_h_over_intact.median():.3f}   "
          f"range {df.rp_h_over_intact.min():.3f} to "
          f"{df.rp_h_over_intact.max():.3f}")
    print(f"    runs outside [0.7, 1.4]: {len(bad)} of {len(df)}")
    print(f"\n  honest skill range: {df.skill_honest.min():.1f} to "
          f"{df.skill_honest.max():.1f}   "
          f"(runs below -100: {(df.skill_honest < -100).sum()})")

    # the result
    order = list(CONDITIONS.keys())
    print('\n' + '=' * 84)
    print('  MEAN BIAS BY CONDITION, WITH BOOTSTRAP 95% CI')
    print('=' * 84)
    print(f"{'record':<9}{'model':>9}{'frac':>7}{'H':>3}"
          f"{'MANY-SHORT':>22}{'MEDIUM':>22}{'FEW-LONG':>22}   mono")
    mono_ok = mono_tot = 0
    for rec in df.record.unique():
        for model_name in MODELS:
            for frac in MISSING_FRACS:
                for H in HORIZONS:
                    cells, means = [], []
                    for c in order:
                        v = df[(df.record == rec) & (df.model == model_name) &
                               (df.missing_frac == frac) &
                               (df.horizon == H) & (df.condition == c)]['bias']
                        if len(v) < 3:
                            cells.append(f"{'n/a':>22}")
                            means.append(np.nan)
                        else:
                            lo, hi = boot_ci(v)
                            cells.append(
                                f"{v.mean():>+8.2f} [{lo:+6.2f},{hi:+6.2f}]")
                            means.append(v.mean())
                    if any(np.isnan(m) for m in means):
                        verdict = '-'
                    else:
                        mono_tot += 1
                        good = means[0] >= means[1] >= means[2]
                        mono_ok += int(good)
                        verdict = 'yes' if good else 'NO'
                    print(f"{rec:<9}{model_name:>9}{frac:>7.0%}{H:>3}"
                          + ''.join(cells) + f"   {verdict}")
    print(f"\n  MONOTONIC IN {mono_ok}/{mono_tot} COMPLETE COMPARISONS")

    print('\n' + '=' * 84)
    print('  MODEL RELIABILITY. A model whose intact (no-gap) skill is far')
    print('  below zero, or whose honest arm collapses, cannot support a')
    print('  skill-ratio comparison.')
    print(f"  Rule: reliable = skill_honest > {RELIABLE_MIN_SKILL:.0f}% AND "
          f"{RELIABLE_RP_RANGE[0]} <= rp_h_over_intact <= "
          f"{RELIABLE_RP_RANGE[1]}")
    print('=' * 84)
    print(f"{'model':>9}{'runs':>6}{'med intact sk':>15}{'med honest sk':>15}"
          f"{'bias sd':>9}{'|bias|>50':>10}{'sk_h<-100':>10}"
          f"{'rp out':>8}{'reliable':>9}")
    for mname in MODELS:
        g = df[df.model == mname]
        if not len(g):
            continue
        rp_out = ((g.rp_h_over_intact < RELIABLE_RP_RANGE[0]) |
                  (g.rp_h_over_intact > RELIABLE_RP_RANGE[1])).sum()
        print(f"{mname:>9}{len(g):>6}{g.intact_skill.median():>15.2f}"
              f"{g.skill_honest.median():>15.2f}{g.bias.std():>9.2f}"
              f"{int((g.bias.abs() > 50).sum()):>10}"
              f"{int((g.skill_honest < -100).sum()):>10}"
              f"{int(rp_out):>8}{int(g.reliable.sum()):>9}")
    print('\n  Reliable runs by model x condition (of surviving runs):')
    print(df.groupby(['model', 'condition'])['reliable']
            .agg(['sum', 'size']).to_string())
    print('\n  Monotonicity by model (cell means, >= 3 runs per condition):')
    for mname in MODELS:
        ok = tot = 0
        for rec in df.record.unique():
            for frac in MISSING_FRACS:
                for H in HORIZONS:
                    means = []
                    for c in order:
                        v = df[(df.record == rec) & (df.model == mname) &
                               (df.missing_frac == frac) &
                               (df.horizon == H) &
                               (df.condition == c)]['bias']
                        means.append(v.mean() if len(v) >= 3 else np.nan)
                    if any(m != m for m in means):
                        continue
                    tot += 1
                    ok += int(means[0] >= means[1] >= means[2])
        print(f"    {mname:<9} {ok}/{tot}")

    print('\n  Variance attributable to split origin (bias sd within vs '
          'across origins):')
    for rec in df.record.unique():
        g = df[df.record == rec]
        within = g.groupby(['condition', 'missing_frac', 'horizon',
                            'model', 'origin'])['bias'].std().mean()
        overall = g.groupby(['condition', 'missing_frac', 'horizon',
                             'model'])['bias'].std().mean()
        print(f"    {rec:<9} within-origin sd {within:6.2f}   "
              f"overall sd {overall:6.2f}")

    # figures: Ridge is the paper's Fig. 3; XGBoost kept for reference
    print()
    make_figure(df, 'Ridge', OUT_FIG)
    make_figure(df, 'XGBoost', OUT_FIG_XGB)
    if QUICK:
        print('\nThis was the QUICK subset. If the two CHECK blocks above '
              'look right,\nset QUICK = False and run the full experiment.')


if __name__ == '__main__':
    main()
