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


# Table I / Section II-A
t1 = load('data_section_statistics.csv')
if t1 is not None:
    for _, r in t1.iterrows():
        d = r['dataset'].split()[0]
        for c in ('n', 'span_days', 'completeness', 'mean', 'sd', 'segments',
                  'n_gaps', 'pct_after_gap', 'gap_p95', 'gap_max'):
            add('Table I', f'{d} {c}', r[c], 'data_section_statistics.csv')

# Section II-B: KNMI records
kn = load('knmi_record_statistics.csv')
if kn is not None:
    pl = kn[kn['kind'] == 'platform']
    add('II-B', 'KNMI missing % range',
        f"{pl['missing_pct'].min():.2f}-{pl['missing_pct'].max():.2f}",
        'knmi_record_statistics.csv')
    add('II-B', 'KNMI segment range',
        f"{int(pl['segments'].min())}-{int(pl['segments'].max())}",
        'knmi_record_statistics.csv')
    for rec in ('BSB', 'HKZB'):
        r = pl[pl['record'] == rec].iloc[0]
        add('II-B', f'{rec} missing % / segments',
            f"{r['missing_pct']:.2f} / {int(r['segments'])}",
            'knmi_record_statistics.csv')

# Table III / abstract / IV-A
t3 = load('artifact_aligned.csv')
if t3 is not None:
    src = 'artifact_aligned.csv'
    for _, r in t3.iterrows():
        d = r['dataset'].split()[0]
        add('Table III', f"{d} H={r['horizon']} A / C / bias [CI]",
            f"{r['skill_honest']:+.2f} / {r['skill_naive']:+.2f} / "
            f"{r['bias_total']:+.2f} [{r['bias_ci_low']:+.2f}, "
            f"{r['bias_ci_high']:+.2f}]", src)
        add('Table III', f"{d} H={r['horizon']} DM p A / C",
            f"{r['dm_p_honest']:.4f} / {r['dm_p_naive']:.4f}", src)
    add('IV-A', 'D1 max |bias|', t3.loc[t3.dataset.str.startswith('D1'),
                                        'bias_total'].abs().max(), src)
    add('IV-A', 'CIs excluding zero', int(t3['ci_excludes_zero'].sum()), src)
    add('IV-A', 'sign reversals', int(t3['sign_flip'].sum()), src)
    add('IV-A', 'max |B-A|', t3['bias_feature'].abs().max(), src)
    w = []
    for B in (25, 100):
        for _, r in t3.iterrows():
            base = r['bias_ci_high'] - r['bias_ci_low']
            if base > 0:
                w.append(abs((r[f'ci_high_b{B}'] - r[f'ci_low_b{B}']) / base - 1))
    add('III-E', 'max CI width change at block 25/100 (%)', 100 * max(w), src)
    d3 = ds(t3, 'D3')
    add('III-D', 'D3 test rows naive vs gap-aware (H=1)',
        f"{int(d3.iloc[0]['n_test_naive'])} vs {int(d3.iloc[0]['n_test_honest'])}",
        src)

# Section III-D audit
au = load('artifact_audit.csv')
if au is not None:
    src = 'artifact_audit.csv'
    for p in ('D1', 'D2', 'D3'):
        a = ds(au, p)
        add('III-D', f'{p} drift days',
            f"{a['drift_days'].min():.2f}-{a['drift_days'].max():.2f}", src)
        add('III-D', f'{p} max |window artifact|',
            a['window_artifact'].abs().max(), src)
        add('II-A', f'{p} gap-valid retention % by H',
            ' / '.join(f'{v:.1f}' for v in a['pct_valid']), src)

# Table IV / Section IV-B
t4 = load('artifact_decomposition.csv')
if t4 is not None:
    src = 'artifact_decomposition.csv'
    for _, r in t4[t4.model == 'XGBoost'].iterrows():
        d = r['dataset'].split()[0]
        add('Table IV', f"{d} H={r['horizon']} B-A / D-B / C-D / C-A",
            f"{r['bias_feature']:+.2f} / {r['bias_training']:+.2f} / "
            f"{r['bias_composition']:+.2f} / {r['bias_total']:+.2f}", src)
    d3 = ds(t4, 'D3')
    gb = d3[d3.model == 'XGBoost'].set_index('horizon')['bias_composition']
    rd = d3[d3.model == 'Ridge'].set_index('horizon')['bias_composition']
    add('IV-B', 'D3 composition GB', ' / '.join(f'{v:+.2f}' for v in gb), src)
    add('IV-B', 'D3 composition Ridge', ' / '.join(f'{v:+.2f}' for v in rd), src)
    add('IV-B', 'D3 composition GB-Ridge max |diff|', (gb - rd).abs().max(), src)
    add('IV-B', 'D3 composition GB-Ridge mean |diff|', (gb - rd).abs().mean(), src)
    add('IV-B', 'max |training| GB',
        t4.loc[t4.model == 'XGBoost', 'bias_training'].abs().max(), src)
    add('IV-B', 'max |training| Ridge',
        t4.loc[t4.model == 'Ridge', 'bias_training'].abs().max(), src)
    for p in ('D2', 'D3'):
        a = ds(t4, p)
        ratio = a['n_train_CD'] / a['n_train_AB']
        add('IV-B', f'{p} naive/gap-aware training rows',
            f"{ratio.min():.2f}-{ratio.max():.2f}", src)
sh = load('artifact_channel_shares.csv')
if sh is not None:
    for _, r in sh.dropna(subset=['share_training']).iterrows():
        d = r['dataset'].split()[0]
        add('IV-B', f"{d} shares {r['model']} {r['method']} (feat/train/comp %)",
            f"{r['share_feature']:.1f} / {r['share_training']:.1f} / "
            f"{r['share_composition']:.1f}", 'artifact_channel_shares.csv')

# Table V / Section IV-D
t5 = load('model_generalisation_aligned.csv')
if t5 is not None:
    src = 'model_generalisation_aligned.csv'
    t5 = t5.assign(d=t5['dataset'].str.split().str[0])
    piv = t5.pivot_table(index='d', columns='model', values='bias', aggfunc='mean')
    for d, r in piv.iterrows():
        for m, v in r.items():
            add('Table V', f'{d} {m} mean bias', v, src)
    stable = t5[t5.model != 'LSTM']
    agree = 0
    for _, g in stable.groupby(['d', 'horizon']):
        s = np.sign(np.where(g['bias'].abs() < 0.005, 0, g['bias']))
        agree += int(len(set(s)) == 1)
    add('IV-D', 'stable families agree in sign (cases of 9)', agree, src)
    l1 = t5[(t5.d == 'D1') & (t5.model == 'LSTM')]
    add('IV-D', 'D1 LSTM bias by H', ' / '.join(f'{v:+.2f}' for v in l1['bias']), src)
    add('IV-D', 'D1 LSTM gap-aware skill by H',
        ' / '.join(f'{v:+.2f}' for v in l1['skill_honest']), src)

# Table VI
t6 = load('rolling_origin_aligned_summary.csv')
if t6 is not None:
    for _, r in t6.iterrows():
        d = r['dataset'].split()[0]
        add('Table VI', f"{d} H={r['horizon']} mean [CI] sign/excl",
            f"{r['mean_bias']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
            f"{r['sign_stable']}/{r['excludes_zero']}",
            'rolling_origin_aligned_summary.csv')

# Table VII / Section IV-E
t7 = load('gap_profile_table7.csv')
if t7 is not None:
    src = 'gap_profile_table7.csv'
    for rec in ('D3_ONSHORE', 'D2_BUOY'):
        for _, r in t7[t7.record == rec].iterrows():
            add('Table VII', f"{rec} H={r['horizon']} n / pers / model / ratio [CI]",
                f"{int(r['n'])} / {r['pers_inflation']:.3f} / "
                f"{r['model_inflation']:.3f} / {r['ratio']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", src)
    kn7 = t7[~t7.record.isin(['D3_ONSHORE', 'D2_BUOY', 'D1_ZEPHIR'])]
    add('Table VII', 'KNMI ratio range', f"{kn7['ratio'].min():.2f}-"
        f"{kn7['ratio'].max():.2f}", src)
    add('Table VII', 'KNMI d=1 n range', f"{int(kn7['n'].min())}-"
        f"{int(kn7['n'].max())}", src)
gc = load('gap_context_summary.csv')
if gc is not None:
    c = gc[gc.method == 'clean']
    for _, r in c.iterrows():
        add('IV-E', f"{r['record']} pre-gap / baseline sd, ratio, p",
            f"{r['pre_gap_sd']:.3f} / {r['baseline_sd']:.3f}, "
            f"{r['ratio']:.3f}, p={r['mannwhitney_p']:.3g} (n={int(r['n_gaps'])})",
            'gap_context_summary.csv')
    add('IV-E', 'records supporting harder conditions (ratio>1, p<0.05)',
        int(((c.ratio > 1) & (c.mannwhitney_p < 0.05)).sum()),
        'gap_context_summary.csv')
gn = load('gap_profile_numbers.csv')
if gn is not None:
    for _, r in gn.iterrows():
        add('IV-E', r['key'], r['value'], 'gap_profile_numbers.csv')

# Section IV-F / Table VIII
inj = load('injection_summary.csv')
if inj is not None:
    for _, r in inj.iterrows():
        vals = {k: v for k, v in r.items() if pd.notna(v)}
        key = ' | '.join(str(vals.pop(k)) for k in list(vals)[:4])
        add('IV-F', key, '; '.join(f'{k}={v}' for k, v in vals.items()),
            'injection_summary.csv')

# Interpolation sensitivity
isn = load('interpolation_sensitivity.csv')
if isn is not None:
    for _, r in isn[~isn['interpolate']].iterrows():
        d = r['dataset'].split()[0]
        add('II-A', f"{d} no-interpolation H={r['horizon']} bias [CI] (segments)",
            f"{r['bias_total']:+.2f} [{r['bias_ci_low']:+.2f}, "
            f"{r['bias_ci_high']:+.2f}] ({int(r['segments'])})",
            'interpolation_sensitivity.csv')

# Section V-A: operator conversion (2 MW, 90 m rotor, Cp 0.45, rho 1.225)
if t3 is not None and t1 is not None:
    v = float(ds(t1, 'D3').iloc[0]['mean'])
    r6 = ds(t3, 'D3').query('horizon == 6').iloc[0]
    P = 0.5 * 1.225 * np.pi * 45 ** 2 * 0.45 * v ** 3 / 1e3        # kW
    slope = 3 * P / v
    rp = float(r6['rmse_pers_honest'])
    dv = r6['skill_naive'] / 100 * rp
    add('V-A', 'dP/dv at D3 mean speed (kW per m/s)', slope, 'derived')
    add('V-A', 'gap-aware persistence RMSE H=6 (m/s), kW, % rated',
        f"{rp:.3f}, {rp * slope:.0f}, {100 * rp * slope / 2000:.1f}", 'derived')
    add('V-A', 'claimed reduction (m/s), kW, % rated',
        f"{dv:.3f}, {dv * slope:.0f}, {100 * dv * slope / 2000:.1f}", 'derived')

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
w = max(len(k) for k in out['key']) if len(out) else 10
for sec, g in out.groupby('section', sort=False):
    print(f"\n[{sec}]")
    for _, r in g.iterrows():
        print(f"  {r['key']:<{w}}  {r['value']}")
print(f"\nSaved {OUT} ({len(out)} numbers)")
