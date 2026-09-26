import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['MANY-SHORT', 'MEDIUM', 'FEW-LONG']
MODELS = ['Ridge', 'XGBoost']
N_REPLICATES = 30
MIN_CELL = 3
N_BOOT = 2000
RELIABLE_MIN_SKILL = -50.0
RELIABLE_RP_RANGE = (0.7, 1.4)

rows_out = []
lines = []


def say(s=''):
    lines.append(s)
    print(s)


def rec(section, model, condition, fractions, stat, value, lo=np.nan,
        hi=np.nan, n=np.nan, ci='', **extra):
    d = dict(section=section, model=model, condition=condition,
             fractions=fractions, stat=stat, value=value, ci_lo=lo, ci_hi=hi,
             n=n, ci_method=ci)
    d.update(extra)
    rows_out.append(d)


def iid_ci(x, seed=0):
    """Percentile bootstrap of the mean over runs (as run_gap_injection.py)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


class ClusterBoot:
    """Bootstrap weights over (record, origin) clusters, shared by every
    statistic so that differences between conditions are paired.
    stratified=False resamples the clusters freely; stratified=True resamples
    origins within each record (record set fixed)."""

    def __init__(self, df, stratified=False, seed=1):
        self.key = list(zip(df.record, df.origin))
        self.ids = sorted(set(self.key))
        self.idx = {k: i for i, k in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        C = len(self.ids)
        W = np.zeros((N_BOOT, C))
        if not stratified:
            pick = rng.integers(0, C, size=(N_BOOT, C))
            for b in range(N_BOOT):
                W[b] = np.bincount(pick[b], minlength=C)
        else:
            recs = sorted(set(r for r, _ in self.ids))
            for r in recs:
                cols = [self.idx[k] for k in self.ids if k[0] == r]
                pick = rng.integers(0, len(cols), size=(N_BOOT, len(cols)))
                for b in range(N_BOOT):
                    np.add.at(W[b], np.array(cols)[pick[b]], 1)
        self.W = W

    def sums(self, sub, col='bias'):
        S = np.zeros(len(self.ids))
        N = np.zeros(len(self.ids))
        for (r, o), g in sub.groupby(['record', 'origin']):
            S[self.idx[(r, o)]] = g[col].sum()
            N[self.idx[(r, o)]] = len(g)
        return S, N

    def mean_draws(self, sub, col='bias'):
        S, N = self.sums(sub, col)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (self.W @ S) / (self.W @ N)

    def ci(self, draws):
        d = draws[np.isfinite(draws)]
        if len(d) < 10:
            return np.nan, np.nan
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cell_means(df, model):
    """Mean bias per record x fraction x horizon x condition (>= MIN_CELL)."""
    g = df[df.model == model].groupby(
        ['record', 'missing_frac', 'horizon', 'condition'])['bias']
    m = g.agg(['mean', 'size']).reset_index()
    m = m[m['size'] >= MIN_CELL]
    return m.pivot_table(index=['record', 'missing_frac', 'horizon'],
                         columns='condition', values='mean')


def complete_cells(df, model):
    cm = cell_means(df, model)
    cm = cm.reindex(columns=ORDER)
    return cm.dropna()


def monotonic(df, model):
    cc = complete_cells(df, model)
    ok = ((cc['MANY-SHORT'] >= cc['MEDIUM']) &
          (cc['MEDIUM'] >= cc['FEW-LONG']))
    return int(ok.sum()), len(cc), cc[~ok]


def fmt(m, lo, hi):
    if lo != lo:
        return f"{m:+8.2f}{'':>19}"
    return f"{m:+8.2f} [{lo:+7.2f},{hi:+7.2f}]"


def pooled_table(df, section, fr_label, boot=None, boot_s=None, mods=MODELS):
    """Mean [CI], median and n per model x condition. iid CI when boot is
    None, else cluster CIs (free and stratified)."""
    hdr = f"{'model':<9}" + ''.join(f"{c:>30}" for c in ORDER)
    say(hdr)
    for mname in mods:
        line, line_med = f"{mname:<9}", f"{'  median':<9}"
        line_s = f"{'  strat':<9}"
        for c in ORDER:
            v = df[(df.model == mname) & (df.condition == c)]
            if not len(v):
                line += f"{'n/a':>30}"; line_med += f"{'':>30}"
                line_s += f"{'':>30}"
                continue
            m = v.bias.mean()
            if boot is None:
                lo, hi = iid_ci(v.bias)
                ci = 'iid bootstrap over runs'
            else:
                lo, hi = boot.ci(boot.mean_draws(v))
                ci = 'cluster bootstrap over (record, origin)'
            line += f"{fmt(m, lo, hi):>30}"
            line_med += f"{'med ' + format(v.bias.median(), '+.2f') + '  n=' + str(len(v)):>30}"
            rec(section, mname, c, fr_label, 'mean', m, lo, hi, len(v), ci)
            rec(section, mname, c, fr_label, 'median', v.bias.median(),
                n=len(v))
            if boot_s is not None:
                slo, shi = boot_s.ci(boot_s.mean_draws(v))
                line_s += f"{'[' + format(slo, '+.2f') + ',' + format(shi, '+.2f') + ']':>30}"
                rec(section, mname, c, fr_label, 'mean', m, slo, shi, len(v),
                    'origin bootstrap stratified by record')
        say(line)
        say(line_med)
        if boot_s is not None:
            say(line_s)


def diff_line(df, section, fr_label, boot, mname, a='MANY-SHORT',
              b='FEW-LONG'):
    va = df[(df.model == mname) & (df.condition == a)]
    vb = df[(df.model == mname) & (df.condition == b)]
    if not len(va) or not len(vb):
        return
    d = va.bias.mean() - vb.bias.mean()
    draws = boot.mean_draws(va) - boot.mean_draws(vb)
    lo, hi = boot.ci(draws)
    say(f"   {mname:<8} {a} - {b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
        f"(paired cluster bootstrap)")
    rec(section, mname, f'{a} - {b}', fr_label, 'mean difference', d, lo, hi,
        len(va) + len(vb), 'paired cluster bootstrap over (record, origin)')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, 'gap_injection_v3_results.csv')
    out_dir = sys.argv[2] if len(sys.argv) > 2 else HERE
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path)
    fracs = sorted(df.missing_frac.unique())
    hors = sorted(df.horizon.unique())
    recs = list(df.record.unique())
    n_slot = len(recs) * len(hors) * N_REPLICATES

    say(f"Injection summary for {os.path.abspath(path)}")
    say(f"  {len(df)} runs; records {recs}; fractions {fracs}; horizons "
        f"{hors}; models {sorted(df.model.unique())}")
    have = {m: {c: sorted(df[(df.model == m) & (df.condition == c)]
                          .missing_frac.unique()) for c in ORDER}
            for m in MODELS}
    for m in MODELS:
        say(f"  fractions present, {m}: " +
            '; '.join(f"{c} {[f'{f:.0%}' for f in have[m][c]]}"
                      for c in ORDER))

    # ================================================================ PART A
    say('\n' + '=' * 96)
    say('PART A. TABLE VIII AS THE PAPER COMPUTED IT (all surviving runs, '
        'each condition over the fractions it has)')
    say('=' * 96)
    say('\nA1. Pooled mean bias (pp) [iid bootstrap over runs, 2000, '
        'seed 0], with median and n')
    pooled_table(df, 'A1 paper pooled (unmatched fractions)', 'all')

    say('\nA2. Monotonic complete comparisons (cell mean needs >= 3 runs in '
        'every condition)')
    tot_ok = tot_n = 0
    for m in MODELS:
        ok, n, bad = monotonic(df, m)
        tot_ok += ok; tot_n += n
        say(f"   {m:<8} {ok}/{n}")
        rec('A2 monotonic', m, '', 'all', 'monotonic count', ok, n=n)
        for idx, r in bad.iterrows():
            say(f"      not monotonic: {idx}  " +
                '  '.join(f"{c} {r[c]:+.2f}" for c in ORDER))
    say(f"   pooled   {tot_ok}/{tot_n}")
    rec('A2 monotonic', 'pooled', '', 'all', 'monotonic count', tot_ok,
        n=tot_n)

    say('\nA3. Per-record mean bias, Ridge, pooled over fractions and '
        'horizons (median, n)')
    for r in recs:
        parts = []
        for c in ORDER:
            v = df[(df.model == 'Ridge') & (df.record == r) &
                   (df.condition == c)].bias
            parts.append(f"{c} {v.mean():+6.2f} (med {v.median():+6.2f}, "
                         f"n {len(v)})" if len(v) else f"{c} n/a")
            if len(v):
                rec('A3 per record', 'Ridge', c, 'all', 'mean', v.mean(),
                    n=len(v), record=r)
        say(f"   {r:<9}" + '   '.join(parts))

    say('\nA4. Per-fraction mean bias (pp), all records and horizons '
        '(median, n)')
    for m in MODELS:
        for c in ORDER:
            parts = []
            for f in fracs:
                v = df[(df.model == m) & (df.condition == c) &
                       (df.missing_frac == f)].bias
                if len(v):
                    parts.append(f"{f:.0%} {v.mean():+7.2f} "
                                 f"(med {v.median():+6.2f}, n {len(v)})")
                    rec('A4 per fraction', m, c, f'{f:.2f}', 'mean', v.mean(),
                        n=len(v))
                    rec('A4 per fraction', m, c, f'{f:.2f}', 'median',
                        v.median(), n=len(v))
                else:
                    parts.append(f"{f:.0%} n/a")
            say(f"   {m:<8}{c:<11}" + '   '.join(parts))

    say(f'\nA5. Survival: surviving placements of {n_slot} per model x '
        f'condition x fraction ({len(recs)} records x {len(hors)} H x '
        f'{N_REPLICATES})')
    for m in MODELS:
        for c in ORDER:
            parts = []
            for f in fracs:
                k = len(df[(df.model == m) & (df.condition == c) &
                           (df.missing_frac == f)])
                parts.append(f"{f:.0%} {k}/{n_slot}")
                rec('A5 survival', m, c, f'{f:.2f}', 'surviving runs', k,
                    n=n_slot)
            say(f"   {m:<8}{c:<11}" + '   '.join(parts))
    say(f'   by record, MANY-SHORT (of {len(hors) * N_REPLICATES} per '
        'fraction):')
    for m in MODELS:
        for f in fracs:
            s = df[(df.model == m) & (df.condition == 'MANY-SHORT') &
                   (df.missing_frac == f)].groupby('record').size()
            say(f"   {m:<8}{f:.0%}  " + '  '.join(
                f"{r} {int(s.get(r, 0))}/{len(hors) * N_REPLICATES}"
                for r in recs))

    say('\nA6. Common support: complete cells, equal weight per cell '
        '[iid bootstrap over cells]')
    for m in MODELS:
        cc = complete_cells(df, m)
        parts = []
        for c in ORDER:
            x = cc[c].values
            lo, hi = iid_ci(x)
            parts.append(f"{c} {fmt(x.mean(), lo, hi).strip()} "
                         f"(med {np.median(x):+.2f})")
            rec('A6 common support', m, c, 'complete cells', 'mean of cell '
                'means', x.mean(), lo, hi, len(x), 'iid bootstrap over cells')
        say(f"   {m:<8}({len(cc)} cells)  " + '   '.join(parts))

    say('\nA7. Test-set growth n_test_naive / n_test_honest, median by '
        'condition')
    g = df.assign(test_growth=df.n_test_naive / df.n_test_honest)
    for lab, sub in [('all runs', g), ('Ridge', g[g.model == 'Ridge']),
                     ('XGBoost', g[g.model == 'XGBoost'])]:
        say(f"   {lab:<9}" + '   '.join(
            f"{c} {sub[sub.condition == c].test_growth.median():.2f}"
            for c in ORDER))
        for c in ORDER:
            rec('A7 test growth', lab, c, 'all', 'median',
                sub[sub.condition == c].test_growth.median())

    say('\nA8. Medians of MANY-SHORT bias over all surviving runs')
    for m in MODELS:
        v = df[(df.model == m) & (df.condition == 'MANY-SHORT')].bias
        say(f"   {m:<8} median {v.median():+.2f}   mean {v.mean():+.2f}   "
            f"n {len(v)}")

    say('\nA9. Per-cell MANY-SHORT CIs excluding zero (complete cells; iid '
        'bootstrap over runs as in run_gap_injection.py, and bootstrap over '
        'the 5 origins)')
    for m in MODELS:
        cc = complete_cells(df, m)
        ex_iid = ex_or = 0
        for (r, f, h) in cc.index:
            v = df[(df.model == m) & (df.record == r) &
                   (df.missing_frac == f) & (df.horizon == h) &
                   (df.condition == 'MANY-SHORT')]
            lo, hi = iid_ci(v.bias)
            ex_iid += int(lo > 0 or hi < 0)
            lo2, hi2 = origin_ci(v)
            ex_or += int(lo2 > 0 or hi2 < 0)
        say(f"   {m:<8} iid: {ex_iid}/{len(cc)}   origin-cluster: "
            f"{ex_or}/{len(cc)}")
        rec('A9 cells excluding zero', m, 'MANY-SHORT', 'complete cells',
            'count iid', ex_iid, n=len(cc))
        rec('A9 cells excluding zero', m, 'MANY-SHORT', 'complete cells',
            'count origin-cluster', ex_or, n=len(cc))

    # ================================================================ PART B
    matched = [f for f in fracs
               if all(f in have[m][c] for m in MODELS for c in ORDER)]
    mlab = '+'.join(f'{f:.0%}' for f in matched)
    dm = df[df.missing_frac.isin(matched)]
    boot_all = ClusterBoot(df)
    boot_all_s = ClusterBoot(df, stratified=True)

    say('\n' + '=' * 96)
    say('PART B. CORRECTED VERSIONS')
    say('=' * 96)
    say(f"  Cluster bootstrap: {N_BOOT} draws of the {len(boot_all.ids)} "
        "(record, origin) clusters with replacement; the same draw is used "
        "for every condition and model, so differences are paired. 'strat' "
        "rows resample origins within each record instead.")

    say('\nB0. As Table VIII (all runs, unmatched fractions) but with '
        'cluster CIs')
    pooled_table(df, 'B0 unmatched, cluster CI', 'all', boot_all, boot_all_s)

    say(f'\nB1. MATCHED fractions ({mlab} only; every condition has these)')
    pooled_table(dm, f'B1 matched {mlab}', mlab, boot_all, boot_all_s)
    for m in MODELS:
        diff_line(dm, f'B1 matched {mlab}', mlab, boot_all, m)
    for f in matched:
        say(f'\n   B1.{f:.0%}  single fraction {f:.0%}')
        pooled_table(df[df.missing_frac == f], f'B1 fraction {f:.2f}',
                     f'{f:.2f}', boot_all)
        for m in MODELS:
            diff_line(df[df.missing_frac == f], f'B1 fraction {f:.2f}',
                      f'{f:.2f}', boot_all, m)

    say('\nB2. Common support (complete cells, equal weight) with cluster '
        'bootstrap')
    for m in MODELS:
        cc = complete_cells(df, m)
        keys = set(cc.index)
        in_cc = np.array([k in keys for k in zip(df.record, df.missing_frac,
                                                 df.horizon)])
        sub = df[(df.model == m).values & in_cc]
        parts = []
        for c in ORDER:
            s = sub[sub.condition == c]
            draws = []
            for cell, gc in s.groupby(['record', 'missing_frac', 'horizon']):
                draws.append(boot_all.mean_draws(gc))
            draws = np.nanmean(np.vstack(draws), axis=0)
            lo, hi = boot_all.ci(draws)
            x = cc[c].values
            parts.append(f"{c} {fmt(x.mean(), lo, hi).strip()}")
            rec('B2 common support', m, c, 'complete cells',
                'mean of cell means', x.mean(), lo, hi, len(x),
                'cluster bootstrap over (record, origin)')
        say(f"   {m:<8}({len(cc)} cells)  " + '   '.join(parts))

    # reliability
    if 'reliable' in df.columns:
        rel = df['reliable'].astype(str).str.lower().isin(['true', '1'])
        src = 'column `reliable` from the CSV'
    else:
        rel = ((df.skill_honest > RELIABLE_MIN_SKILL) &
               df.rp_h_over_intact.between(*RELIABLE_RP_RANGE))
        src = 'computed here (CSV has no `reliable` column)'
    say('\nB3. Reliability: reliable = skill_honest > '
        f'{RELIABLE_MIN_SKILL:.0f}% AND {RELIABLE_RP_RANGE[0]} <= '
        f'rp_h_over_intact <= {RELIABLE_RP_RANGE[1]}  ({src})')
    for m in MODELS:
        g = df[df.model == m]
        r = rel[df.model == m]
        rp_out = (~g.rp_h_over_intact.between(*RELIABLE_RP_RANGE)).sum()
        extra = ''
        if 'intact_skill' in g.columns:
            extra = f"   median intact skill {g.intact_skill.median():+.2f}"
        say(f"   {m:<8} runs {len(g)}   reliable {int(r.sum())}   "
            f"honest < -100: {int((g.skill_honest < -100).sum())}   "
            f"|bias| > 50: {int((g.bias.abs() > 50).sum())}   "
            f"rp_h/intact outside: {int(rp_out)}   "
            f"median honest skill {g.skill_honest.median():+.2f}{extra}")
        rec('B3 reliability', m, '', 'all', 'reliable runs', int(r.sum()),
            n=len(g))
        rec('B3 reliability', m, '', 'all', 'honest skill < -100',
            int((g.skill_honest < -100).sum()), n=len(g))
        rec('B3 reliability', m, '', 'all', '|bias| > 50',
            int((g.bias.abs() > 50).sum()), n=len(g))
        rec('B3 reliability', m, '', 'all', 'rp_h_over_intact outside',
            int(rp_out), n=len(g))
        for c in ORDER:
            say(f"      {c:<11} reliable " + '  '.join(
                f"{f:.0%} {int(rel[(df.model == m) & (df.condition == c) & (df.missing_frac == f)].sum())}"
                f"/{int(((df.model == m) & (df.condition == c) & (df.missing_frac == f)).sum())}"
                for f in fracs))

    dr = df[rel]
    boot_r = ClusterBoot(dr)
    drm = dr[dr.missing_frac.isin(matched)]
    say(f'\nB4. Reliable runs only, matched fractions ({mlab}), cluster CI')
    pooled_table(drm, f'B4 reliable matched {mlab}', mlab, boot_r)
    for m in MODELS:
        diff_line(drm, f'B4 reliable matched {mlab}', mlab, boot_r, m)
    say('   monotonic (reliable runs, all fractions): ' + '   '.join(
        '{} {}/{}'.format(m, *monotonic(dr, m)[:2]) for m in MODELS))
    for m in MODELS:
        ok, n, _ = monotonic(dr, m)
        rec('B4 monotonic reliable', m, '', 'all', 'monotonic count', ok, n=n)

    pd.DataFrame(rows_out).to_csv(os.path.join(out_dir,
                                               'injection_summary.csv'),
                                  index=False)
    with open(os.path.join(out_dir, 'injection_summary_report.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\nSaved injection_summary.csv and injection_summary_report.txt "
          f"in {out_dir}")


def origin_ci(v, seed=0):
    """Bootstrap of a cell mean over its split origins (the clusters)."""
    x = v.bias.values
    o = v.origin.values
    ids = np.unique(o)
    if len(x) < 3:
        return np.nan, np.nan
    sums = np.array([x[o == i].sum() for i in ids])
    cnts = np.array([(o == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    m = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


if __name__ == '__main__':
    main()
