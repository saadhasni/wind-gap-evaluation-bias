import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'paper_numbers.csv')
rows = []


def add(section, key, value, source):
    if isinstance(value, (float, np.floating)):
        value = round(float(value), 4)
    rows.append(dict(section=section, key=key, value=value, source=source))


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print(f"[missing] {name} (run the script that produces it)")
        return None
    return pd.read_csv(p)


def ds(df, prefix):
    return df[df['dataset'].str.startswith(prefix)]


# Table 1
t1 = load('data_section_statistics.csv')
if t1 is not None:
    for _, r in t1.iterrows():
        d = r['dataset'].split()[0]
        for c in ('n', 'span_days', 'completeness', 'mean', 'sd', 'segments',
                  'n_gaps', 'pct_after_gap', 'gap_p95', 'gap_max'):
            add('Table 1', f'{d} {c}', r[c], 'data_section_statistics.csv')

# Sect. 2: KNMI records
kn = load('knmi_record_statistics.csv')
if kn is not None:
    pl = kn[kn['kind'] == 'platform']
    add('Sect. 2', 'KNMI missing % range',
        f"{pl['missing_pct'].min():.2f}-{pl['missing_pct'].max():.2f}",
        'knmi_record_statistics.csv')
    add('Sect. 2', 'KNMI segment range',
        f"{int(pl['segments'].min())}-{int(pl['segments'].max())}",
        'knmi_record_statistics.csv')
    for rec in ('BSB', 'HKZB'):
        r = pl[pl['record'] == rec].iloc[0]
        add('Sect. 2', f'{rec} missing % / segments',
            f"{r['missing_pct']:.2f} / {int(r['segments'])}",
            'knmi_record_statistics.csv')

# Table 3 (A, C, C - A) / abstract / Sect. 4.1
t3 = load('artifact_aligned.csv')
if t3 is not None:
    src = 'artifact_aligned.csv'
    for _, r in t3.iterrows():
        d = r['dataset'].split()[0]
        add('Table 3', f"{d} H={r['horizon']} A / C / bias [CI]",
            f"{r['skill_honest']:+.2f} / {r['skill_naive']:+.2f} / "
            f"{r['bias_total']:+.2f} [{r['bias_ci_low']:+.2f}, "
            f"{r['bias_ci_high']:+.2f}]", src)
        add('Table 3', f"{d} H={r['horizon']} DM p A / C",
            f"{r['dm_p_honest']:.4f} / {r['dm_p_naive']:.4f}", src)
    add('Sect. 4.1', 'D1 max |bias|', t3.loc[t3.dataset.str.startswith('D1'),
                                        'bias_total'].abs().max(), src)
    add('Sect. 4.1', 'CIs excluding zero', int(t3['ci_excludes_zero'].sum()), src)
    add('Sect. 4.1', 'sign reversals', int(t3['sign_flip'].sum()), src)
    add('Sect. 4.1', 'max |B-A|', t3['bias_feature'].abs().max(), src)
    w = []
    for B in (25, 100):
        for _, r in t3.iterrows():
            base = r['bias_ci_high'] - r['bias_ci_low']
            if base > 0:
                w.append(abs((r[f'ci_high_b{B}'] - r[f'ci_low_b{B}']) / base - 1))
    add('Sect. 3.5', 'max CI width change at block 25/100 (%)', 100 * max(w), src)
    d3 = ds(t3, 'D3')
    add('Sect. 3.2', 'D3 test rows naive vs gap-aware (H=1)',
        f"{int(d3.iloc[0]['n_test_naive'])} vs {int(d3.iloc[0]['n_test_honest'])}",
        src)

# Supplementary S2: calendar-alignment audit
au = load('artifact_audit.csv')
if au is not None:
    src = 'artifact_audit.csv'
    for p in ('D1', 'D2', 'D3'):
        a = ds(au, p)
        add('Suppl. S2', f'{p} drift days',
            f"{a['drift_days'].min():.2f}-{a['drift_days'].max():.2f}", src)
        add('Suppl. S2', f'{p} max |window artifact|',
            a['window_artifact'].abs().max(), src)
        add('Table 1', f'{p} gap-valid retention % by H',
            ' / '.join(f'{v:.1f}' for v in a['pct_valid']), src)

# Table 3 decomposition columns / Sect. 4.2
t4 = load('artifact_decomposition.csv')
if t4 is not None:
    src = 'artifact_decomposition.csv'
    for _, r in t4[t4.model == 'XGBoost'].iterrows():
        d = r['dataset'].split()[0]
        add('Table 3', f"{d} H={r['horizon']} D / B-A / D-B / C-D / C-A",
            f"{r['skill_D']:+.2f} / {r['bias_feature']:+.2f} / {r['bias_training']:+.2f} / "
            f"{r['bias_composition']:+.2f} / {r['bias_total']:+.2f}", src)
    for _, r in t4[(t4.model == 'Ridge') & t4.dataset.str.startswith('D3')].iterrows():
        add('Sect. 4.2', f"D3 ridge H={r['horizon']} A / D / C",
            f"{r['skill_A']:+.2f} / {r['skill_D']:+.2f} / {r['skill_C']:+.2f}", src)
    d3 = ds(t4, 'D3')
    gb = d3[d3.model == 'XGBoost'].set_index('horizon')['bias_composition']
    rd = d3[d3.model == 'Ridge'].set_index('horizon')['bias_composition']
    add('Sect. 4.2', 'D3 composition GB', ' / '.join(f'{v:+.2f}' for v in gb), src)
    add('Sect. 4.2', 'D3 composition Ridge', ' / '.join(f'{v:+.2f}' for v in rd), src)
    add('Sect. 4.2', 'D3 composition GB-Ridge max |diff|', (gb - rd).abs().max(), src)
    add('Sect. 4.2', 'D3 composition GB-Ridge mean |diff|', (gb - rd).abs().mean(), src)
    add('Sect. 4.2', 'max |training| GB',
        t4.loc[t4.model == 'XGBoost', 'bias_training'].abs().max(), src)
    add('Sect. 4.2', 'max |training| Ridge',
        t4.loc[t4.model == 'Ridge', 'bias_training'].abs().max(), src)
    for p in ('D2', 'D3'):
        a = ds(t4, p)
        ratio = a['n_train_CD'] / a['n_train_AB']
        add('Sect. 4.2', f'{p} naive/gap-aware training rows',
            f"{ratio.min():.2f}-{ratio.max():.2f}", src)
sh = load('artifact_channel_shares.csv')
if sh is not None:
    for _, r in sh.dropna(subset=['share_training']).iterrows():
        d = r['dataset'].split()[0]
        add('Sect. 4.2', f"{d} shares {r['model']} {r['method']} (feat/train/comp %)",
            f"{r['share_feature']:.1f} / {r['share_training']:.1f} / "
            f"{r['share_composition']:.1f}", 'artifact_channel_shares.csv')

# Table S1 / Sects. 4.1 and 4.6
t5 = load('model_generalisation_aligned.csv')
if t5 is not None:
    src = 'model_generalisation_aligned.csv'
    t5 = t5.assign(d=t5['dataset'].str.split().str[0])
    piv = t5.pivot_table(index='d', columns='model', values='bias', aggfunc='mean')
    for d, r in piv.iterrows():
        for m, v in r.items():
            add('Table S1', f'{d} {m} mean bias', v, src)
    stable = t5[t5.model != 'LSTM']
    agree = 0
    for _, g in stable.groupby(['d', 'horizon']):
        s = np.sign(np.where(g['bias'].abs() < 0.005, 0, g['bias']))
        agree += int(len(set(s)) == 1)
    add('Sect. 4.6 / Suppl. S4', 'stable families agree in sign (cases of 9)', agree, src)
    l1 = t5[(t5.d == 'D1') & (t5.model == 'LSTM')]
    add('Sect. 4.6 / Suppl. S4', 'D1 LSTM bias by H', ' / '.join(f'{v:+.2f}' for v in l1['bias']), src)
    add('Sect. 4.6 / Suppl. S4', 'D1 LSTM gap-aware skill by H',
        ' / '.join(f'{v:+.2f}' for v in l1['skill_honest']), src)

# Table S2
t6 = load('rolling_origin_aligned_summary.csv')
if t6 is not None:
    for _, r in t6.iterrows():
        d = r['dataset'].split()[0]
        add('Table S2', f"{d} H={r['horizon']} mean [CI] sign/excl",
            f"{r['mean_bias']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
            f"{r['sign_stable']}/{r['excludes_zero']}",
            'rolling_origin_aligned_summary.csv')

# Table S3 / Sect. 4.3
t7 = load('gap_profile_table7.csv')
if t7 is not None:
    src = 'gap_profile_table7.csv'
    for rec in ('D3_ONSHORE', 'D2_BUOY'):
        for _, r in t7[t7.record == rec].iterrows():
            add('Table S3', f"{rec} H={r['horizon']} n / pers / model / ratio [CI]",
                f"{int(r['n'])} / {r['pers_inflation']:.3f} / "
                f"{r['model_inflation']:.3f} / {r['ratio']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", src)
    kn7 = t7[~t7.record.isin(['D3_ONSHORE', 'D2_BUOY', 'D1_ZEPHIR'])]
    add('Table S3', 'KNMI ratio range', f"{kn7['ratio'].min():.2f}-"
        f"{kn7['ratio'].max():.2f}", src)
    add('Table S3', 'KNMI d=1 n range', f"{int(kn7['n'].min())}-"
        f"{int(kn7['n'].max())}", src)
gc = load('gap_context_summary.csv')
if gc is not None:
    c = gc[gc.method == 'clean']
    for _, r in c.iterrows():
        add('Sect. 4.3 / Suppl. S6', f"{r['record']} pre-gap / baseline sd, ratio, p",
            f"{r['pre_gap_sd']:.3f} / {r['baseline_sd']:.3f}, "
            f"{r['ratio']:.3f}, p={r['mannwhitney_p']:.3g} (n={int(r['n_gaps'])})",
            'gap_context_summary.csv')
    add('Sect. 4.3 / Suppl. S6', 'records supporting harder conditions (ratio>1, p<0.05)',
        int(((c.ratio > 1) & (c.mannwhitney_p < 0.05)).sum()),
        'gap_context_summary.csv')
gn = load('gap_profile_numbers.csv')
if gn is not None:
    for _, r in gn.iterrows():
        add('Sect. 4.3 / Suppl. S6', r['key'], r['value'], 'gap_profile_numbers.csv')

# Table 5 / Supplementary S5
inj = load('injection_summary.csv')
if inj is not None:
    for _, r in inj.iterrows():
        vals = {k: v for k, v in r.items() if pd.notna(v)}
        key = ' | '.join(str(vals.pop(k)) for k in list(vals)[:4])
        add('Table 5 / Suppl. S5', key, '; '.join(f'{k}={v}' for k, v in vals.items()),
            'injection_summary.csv')

# Supplementary S2: interpolation sensitivity
isn = load('interpolation_sensitivity.csv')
if isn is not None:
    for _, r in isn[~isn['interpolate']].iterrows():
        d = r['dataset'].split()[0]
        add('Suppl. S2', f"{d} no-interpolation H={r['horizon']} bias [CI] (segments)",
            f"{r['bias_total']:+.2f} [{r['bias_ci_low']:+.2f}, "
            f"{r['bias_ci_high']:+.2f}] ({int(r['segments'])})",
            'interpolation_sensitivity.csv')

# Sect. 5.2: operator conversion (2 MW, 90 m rotor, Cp 0.45, rho 1.225)
if t3 is not None and t1 is not None:
    v = float(ds(t1, 'D3').iloc[0]['mean'])
    r6 = ds(t3, 'D3').query('horizon == 6').iloc[0]
    P = 0.5 * 1.225 * np.pi * 45 ** 2 * 0.45 * v ** 3 / 1e3        # kW
    slope = 3 * P / v
    rp = float(r6['rmse_pers_honest'])
    dv = r6['skill_naive'] / 100 * rp
    add('Sect. 5.2', 'dP/dv at D3 mean speed (kW per m/s)', slope, 'derived')
    add('Sect. 5.2', 'gap-aware persistence RMSE H=6 (m/s), kW, % rated',
        f"{rp:.3f}, {rp * slope:.0f}, {100 * rp * slope / 2000:.1f}", 'derived')
    add('Sect. 5.2', 'claimed reduction (m/s), kW, % rated',
        f"{dv:.3f}, {dv * slope:.0f}, {100 * dv * slope / 2000:.1f}", 'derived')
    if t4 is not None:
        g6 = ds(t4, 'D3').query("model == 'XGBoost' and horizon == 6").iloc[0]
        over = (g6['skill_C'] - g6['skill_D']) / 100 * rp
        add('Sect. 5.2', 'overstatement by the scoring bias alone (m/s), kW, % rated',
            f"{over:.3f}, {over * slope:.1f}, {100 * over * slope / 2000:.2f}", 'derived')

# Sects. 3.4 and 4.3 to 4.5: exposure account of the bias
ex = load('exposure_check.csv')
if ex is not None:
    same = bool(((ex.valid_formula == ex.valid_pipeline)
                 & (ex.naive_formula == ex.naive_pipeline)).all())
    add('Table 1 / Suppl. S8', 'Eq. (1) reproduces the pipeline origin counts exactly', same,
        'exposure_check.csv')
    for _, r in ex.iterrows():
        add('Table 1 / Suppl. S8', f"{r['dataset']} H={r['H']} exposure % at W=60",
            r['exposure_pct'], 'exposure_check.csv')
mx = load('mixture_table.csv')
if mx is not None:
    live = mx[(mx.f > 0) & (mx.dataset == 'D3')]
    add('Sect. 3.4', 'first-order / exact scoring bias on D3, range',
        f"{(live.formula_first_order / live.formula_exact).min():.2f}-"
        f"{(live.formula_first_order / live.formula_exact).max():.2f}", 'mixture_table.csv')
    for _, r in mx[mx.f > 0].iterrows():
        add('Table 4',
            f"{r['dataset']} {r['model']} H={r['H']} f / rho_p / rho_m / C-D",
            f"{r['f']:.3f} / {r['rho_p']:.3f} / {r['rho_m']:.3f} / "
            f"{r['measured_bias_C_minus_D']:+.2f}", 'mixture_table.csv')
wd = load('window_depth_sensitivity.csv')
if wd is not None:
    wd = wd.assign(f=1 - wd['n_test_honest'] / wd['n_test_naive'])
    for _, r in wd.iterrows():
        add('Table S5',
            f"W={int(r['W'])} H={int(r['horizon'])} skill A / C / bias [CI] / f%",
            f"{r['skill_honest']:+.2f} / {r['skill_naive']:+.2f} / "
            f"{r['bias_total']:+.2f} [{r['bias_ci_low']:+.2f}, "
            f"{r['bias_ci_high']:+.2f}] / {100 * r['f']:.1f}",
            'window_depth_sensitivity.csv')
    rho, p = stats.spearmanr(wd['f'], wd['bias_total'])
    add('Suppl. S9', 'Spearman correlation of bias with exposure (9 cases)',
        f"{rho:.2f} (p={p:.3f})", 'window_depth_sensitivity.csv')
im = load('injection_mixture_table.csv')
if im is not None:
    add('Sect. 4.4 / Fig. 5', 'ridge runs with displaced rows', len(im),
        'injection_mixture_table.csv')
    cuts = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0]
    med = im.groupby(pd.cut(im['f'], cuts), observed=True)['bias'].median()
    for k, v in med.items():
        add('Sect. 4.4 / Fig. 5', f'median bias, exposure {k}', v, 'injection_mixture_table.csv')
    for c, g2 in im.groupby('condition'):
        add('Sect. 4.4 / Fig. 5', f'{c} median f / f_design / rho_p / rho_m / bias',
            ' / '.join(f"{g2[col].median():.3f}"
                       for col in ('f', 'f_design', 'rho_p', 'rho_m', 'bias')),
            'injection_mixture_table.csv')
    add('Sect. 4.4 / Fig. 5', 'Spearman correlation of f_design with f',
        f"{stats.spearmanr(im['f_design'], im['f'])[0]:.2f}",
        'injection_mixture_table.csv')

# Table S4: Clark-West tests for the A and C arms of Table 3
cw = load('clark_west.csv')
if cw is not None:
    add('Sect. 3.5 / Table S4', 'Clark-West p-value, maximum over the 18 cases',
        float(cw[['cw_p_honest', 'cw_p_naive']].max().max()), 'clark_west.csv')
    for _, r in cw.iterrows():
        add('Sect. 3.5 / Table S4', f"{r['dataset']} H={r['horizon']} CW stat / p, gap-aware | naive",
            f"{r['cw_stat_honest']:.2f} / {r['cw_p_honest']:.3g} | "
            f"{r['cw_stat_naive']:.2f} / {r['cw_p_naive']:.3g}", 'clark_west.csv')

# Sect. 5.2: exposure for other window depths (Eq. 1)
ew = load('exposure_by_window.csv')
if ew is not None:
    for _, r in ew.iterrows():
        add('Sect. 5.2', f"{r['dataset']} W={int(r['W'])} H={int(r['H'])} exposure %",
            r['exposure_pct'], 'exposure_by_window.csv')

# Table 4 (last column) / Table S7: scoring bias split by lead time
lt = load('composition_by_lead_time.csv')
if lt is not None:
    for _, r in lt.iterrows():
        add('Table 4 / Table S7',
            f"D3 {r['model']} H={r['horizon']} n input-only / target-crossing; "
            f"rho_p, rho_m input-only; rho_p, rho_m target-crossing; C-D = input + target",
            f"{int(r['n_input_only'])} / {int(r['n_target_crossing'])}; "
            f"{r['rho_p_input_only']:.3f}, {r['rho_m_input_only']:.3f}; "
            f"{r['rho_p_target_crossing']:.3f}, {r['rho_m_target_crossing']:.3f}; "
            f"{r['comp_total']:+.2f} = {r['comp_input_only']:+.2f} (input-only) "
            f"{r['comp_target_crossing']:+.2f} (target-crossing)",
            'composition_by_lead_time.csv')

# Sect. 4.5 / Table S6: decomposition at W = 15, 30, 60
wc = load('window_depth_channels.csv')
if wc is not None:
    for _, r in wc.iterrows():
        add('Sect. 4.5 / Table S6',
            f"D3 {r['model']} W={int(r['W'])} H={int(r['horizon'])} D-B / C-D / train rows A, C",
            f"{r['bias_training']:+.2f} / {r['bias_composition']:+.2f} / "
            f"{int(r['n_train_AB'])}, {int(r['n_train_CD'])}", 'window_depth_channels.csv')

# Sect. 4.1 / Table S8: robustness to seeds and hyperparameters
rb = load('robustness_summary.csv')
if rb is not None:
    for _, r in rb.iterrows():
        d = r['dataset'].split()[0]
        add('Sect. 4.1 / Table S8',
            f"{d} H={r['horizon']} default; seeds mean (sd) [min, max]; grid range; tuned (A / C); tuned settings",
            f"{r['bias_base']:+.2f}; {r['seed_mean']:+.2f} ({r['seed_sd']:.2f}) "
            f"[{r['seed_min']:+.2f}, {r['seed_max']:+.2f}]; {r['grid_min']:+.2f} to {r['grid_max']:+.2f}; "
            f"{r['tuned_bias']:+.2f} ({r['tuned_skill_honest']:+.2f} / {r['tuned_skill_naive']:+.2f}); "
            f"{r['tuned_params']}", 'robustness_summary.csv')

out = pd.DataFrame(rows)

out.to_csv(OUT, index=False)
w = max(len(k) for k in out['key']) if len(out) else 10
for sec, g in out.groupby('section', sort=False):
    print(f"\n[{sec}]")
    for _, r in g.iterrows():
        print(f"  {r['key']:<{w}}  {r['value']}")
print(f"\nSaved {OUT} ({len(out)} numbers)")
