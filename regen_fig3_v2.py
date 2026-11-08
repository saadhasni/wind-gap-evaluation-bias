import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42      # embed TrueType fonts in the PDF
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1
                   else 'gap_injection_v3_results.csv')
MODEL = 'Ridge'
HORIZONS = (1, 3)
MISSING_FRACS = (0.05, 0.10, 0.20)
ORDER = ['MANY-SHORT', 'MEDIUM', 'FEW-LONG']
COLOURS = {'MANY-SHORT': '#1D9E75', 'MEDIUM': '#4A7BB7', 'FEW-LONG': '#D85A30'}
LABELS = {'MANY-SHORT': 'Many-short', 'MEDIUM': 'Medium', 'FEW-LONG': 'Few-long'}

# text sizes in points at the drawn size (figure is ~7 in wide, placed at ~6.5 in)
F_TITLE, F_AXIS, F_TICK, F_COUNT, F_LEG = 8.5, 8, 7, 6.5, 8


def boot_ci_cluster(x, groups, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean, resampling whole clusters
    (split origins), identical to run_gap_injection.py."""
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


def make_figure(df, model_name, stem):
    recs = list(df.record.unique())
    sub = df[df.model == model_name]
    n_rows, n_cols = len(recs), len(HORIZONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 1.55 * n_rows + 0.7),
                             squeeze=False)
    w = 0.27
    x = np.arange(len(MISSING_FRACS))

    for r_i, rec in enumerate(recs):
        days = df[df.record == rec]['record_days'].iloc[0]
        for c_i, H in enumerate(HORIZONS):
            ax = axes[r_i][c_i]
            g = sub[(sub.horizon == H) & (sub.record == rec)]
            labels = []
            for i, cond in enumerate(ORDER):
                means, los, his = [], [], []
                for j, f in enumerate(MISSING_FRACS):
                    gv = g[(g.missing_frac == f) & (g.condition == cond)]
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
                ax.bar(x + (i - 1) * w, means, w, yerr=[los, his], capsize=1.5,
                       color=COLOURS[cond], error_kw=dict(lw=0.8),
                       label=LABELS[cond] if (r_i == 0 and c_i == 0) else None)
            ax.axhline(0, color='k', lw=0.8)
            lo_y, hi_y = ax.get_ylim()
            span = hi_y - lo_y
            for xb, m, hi, n in labels:
                if m is None:
                    ax.text(xb, 0.01 * span, 'n/a', ha='center', va='bottom',
                            fontsize=F_COUNT, rotation=90, color='0.35')
                else:
                    top = max(m, hi if hi == hi else m, 0)
                    ax.text(xb, top + 0.01 * span, f'{n}', ha='center',
                            va='bottom', fontsize=F_COUNT)
            ax.set_ylim(lo_y, hi_y + 0.10 * span)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{f:.0%}' for f in MISSING_FRACS],
                               fontsize=F_TICK)
            ax.tick_params(axis='y', labelsize=F_TICK, length=2)
            ax.tick_params(axis='x', length=2)
            ax.grid(alpha=0.3, axis='y', lw=0.5)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{rec} ({days:.0f} d), H = {H}', fontsize=F_TITLE,
                         fontweight='bold', pad=3)
            if c_i == 0:
                ax.set_ylabel('Bias (percentage points)', fontsize=F_AXIS)
            if r_i == n_rows - 1:
                ax.set_xlabel('Total missing fraction', fontsize=F_AXIS)

    fig.legend(loc='lower center', ncol=3, fontsize=F_LEG, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=0.9, w_pad=1.2)
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved', os.path.basename(stem) + '.pdf and .png')


if __name__ == '__main__':
    if not os.path.exists(CSV):
        sys.exit(f'Cannot find {CSV}. Put this script in the folder that '
                 'contains the results CSV, or pass its name as an argument.')
    df = pd.read_csv(CSV)
    need = {'record', 'record_days', 'model', 'horizon', 'missing_frac',
            'condition', 'bias', 'origin'}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f'CSV is missing columns: {sorted(missing)}')
    make_figure(df, MODEL, os.path.join(HERE, 'Figure_3'))
