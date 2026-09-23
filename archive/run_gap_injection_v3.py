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
QUICK = False                 # <-- set False for the full run

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
           colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=4)
RIDGE_ALPHA = 1.0
MODELS = ('XGBoost', 'Ridge')

RECORDS = [('HKZA', 'knmi_hkza'), ('HKWB', 'knmi_hkwb'),
           ('HKWA', 'knmi_hkwa'), ('HKZB', 'knmi_hkzb'),
           ('BUOY130', None)]
BUOY_DIR = 'buoy_data'

OUT_CSV = 'gap_injection_v3_results.csv'
OUT_FIG = 'gap_injection_v3_fig.png'

if QUICK:
    RECORDS = RECORDS[:2]
    MISSING_FRACS = (0.05, 0.10)
    HORIZONS = (1,)
    N_REPLICATES = 4
    OUT_CSV = 'gap_injection_v3_QUICK.csv'
    OUT_FIG = 'gap_injection_v3_QUICK.png'


# gap injection
def inject_gaps_exact(series, target_drop, nominal_len, seed, margin=200):
  
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


# features
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
    
    t_train_end, t_test_start, t_test_end = bounds

    # honest: gap-aware features, gap-valid rows only
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
    fn = naive_features(gapped)
    dn = fn.copy()
    dn['target'] = gapped.shift(-H)
    dn = dn.dropna()
    fc = [c for c in dn.columns if c != 'target']
    tr2 = dn[dn.index <= t_train_end]
    te2 = dn[(dn.index >= t_test_start) & (dn.index <= t_test_end)]
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

    vals = []
    for i in range(len(ORIGIN_FRACS)):
        b = window_bounds(base, H, i)
        if b is None:
            continue
        r = evaluate_arms(base, H, b, model_name)
        if r is not None:
            vals.append(r['skill_honest'])
    return float(np.median(vals)) if vals else np.nan


def intact_persistence_rmse(base, H):
  
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
            s_all, _ = load_wfip3_buoy(bp, height_m=38, resample='10min')
            out.append((name, buoy_longest_contiguous(s_all, STEP)))
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


#  main
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

    rows, skipped = [], 0
    intact_rp, intact_sk = {}, {}
    for rec_name, base in records:
        for H in HORIZONS:
            intact_rp[(rec_name, H)] = intact_persistence_rmse(base, H)
            for mname in MODELS:
                intact_sk[(rec_name, H, mname)] = intact_model_skill(
                    base, H, mname)

    print('\nNO-GAP REFERENCE (intact record, same windows).')
    print('  Read every injected result against these. A model already far')
    print('  below zero here is failing on the record, not on the gaps.')
    print(f"   {'record':<9}{'H':>3}{'rp':>9}"
          + ''.join(f"{m + ' skill':>16}" for m in MODELS))
    for rec_name, _ in records:
        for H in HORIZONS:
            line = f"   {rec_name:<9}{H:>3}{intact_rp[(rec_name, H)]:>9.4f}"
            for mname in MODELS:
                v = intact_sk[(rec_name, H, mname)]
                line += f"{v:>16.2f}" if v == v else f"{'n/a':>16}"
            print(line)

    for rec_name, base in records:
        n = len(base)
        print(f"\n{'=' * 84}")
        print(f"  {rec_name}   {n} samples ({n / 144:.1f} days)")
        print('=' * 84)
        print(f"{'frac':>6}{'condition':>12}{'model':>9}{'H':>3}"
              f"{'gaps':>6}{'glen':>6}{'miss%':>7}"
              f"{'skill_h':>9}{'skill_n':>9}{'bias':>9}{'rmse_bias':>11}{'n':>4}")
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
                                                np.nan)
                            r['intact_skill'] = round(sk0, 3) if sk0 == sk0 \
                                else np.nan
                            # how much of the honest arm's position is the
                            # gaps, as opposed to the model's baseline
                            # performance on this record
                            r['honest_vs_intact'] = round(
                                r['skill_honest'] - sk0, 3) if sk0 == sk0 \
                                else np.nan
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
            print(f"  [checkpoint: {OUT_CSV}]")

    if not rows:
        print('\nNo results produced.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    mins = (time.time() - t0) / 60
    print(f"\nSaved {OUT_CSV}  ({len(df)} runs, {skipped} skipped by guards, "
          f"{mins:.0f} min)")

    #  defect 3 check: are fractions actually matched now?
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

    # -------- denominator sanity
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
    print('  below zero cannot support a skill-ratio comparison: its bias')
    print('  estimates carry very wide intervals and can break monotonicity')
    print('  for reasons unrelated to gap handling.')
    print('=' * 84)
    print(f"{'model':>9}{'intact skill':>14}{'honest skill':>14}"
          f"{'bias sd':>10}{'|bias|>50':>11}{'runs<-100':>11}")
    for mname in MODELS:
        g = df[df.model == mname]
        if not len(g):
            continue
        print(f"{mname:>9}{g.intact_skill.median():>14.2f}"
              f"{g.skill_honest.median():>14.2f}{g.bias.std():>10.2f}"
              f"{int((g.bias.abs() > 50).sum()):>11}"
              f"{int((g.skill_honest < -100).sum()):>11}")
    print('\n  Monotonicity by model:')
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

    # ------------------------------------------------------------ figure
    recs = list(df.record.unique())
    fig, axes = plt.subplots(len(HORIZONS), len(recs),
                             figsize=(3.4 * len(recs), 3.6 * len(HORIZONS)),
                             squeeze=False, sharey='row')
    colours = {'MANY-SHORT': '#1D9E75', 'MEDIUM': '#4A7BB7',
               'FEW-LONG': '#D85A30'}
    sub = df[df.model == 'XGBoost']
    for r_i, H in enumerate(HORIZONS):
        for c_i, rec in enumerate(recs):
            ax = axes[r_i][c_i]
            g = sub[(sub.horizon == H) & (sub.record == rec)]
            x = np.arange(len(MISSING_FRACS))
            for i, c in enumerate(order):
                means, los, his = [], [], []
                for f in MISSING_FRACS:
                    v = g[(g.missing_frac == f) & (g.condition == c)]['bias']
                    if len(v) < 3:
                        means.append(np.nan); los.append(0); his.append(0)
                    else:
                        lo, hi = boot_ci(v)
                        means.append(v.mean())
                        los.append(max(0, v.mean() - lo))
                        his.append(max(0, hi - v.mean()))
                ax.bar(x + (i - 1) * 0.27, means, 0.27,
                       yerr=[los, his], capsize=2,
                       label=c if (r_i == 0 and c_i == 0) else None,
                       color=colours[c])
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{f:.0%}' for f in MISSING_FRACS], fontsize=8)
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
    fig.suptitle('Controlled gap injection across independent contiguous '
                 'records (XGBoost; error bars are bootstrap 95% CI)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_FIG}')
    if QUICK:
        print('\nThis was the QUICK subset. If the two CHECK blocks above '
              'look right,\nset QUICK = False and run the full experiment.')


if __name__ == '__main__':
    main()
