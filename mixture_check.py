import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))


def analyse(dec, ali):
    ali = ali[['dataset', 'horizon', 'rmse_pers_honest', 'rmse_pers_naive']]
    d = dec.merge(ali, on=['dataset', 'horizon'], how='left')
    out = []
    for _, r in d.iterrows():
        f = 1 - r.n_test_ABD / r.n_test_C
        ap = r.rmse_pers_honest
        am = ap * (1 - r.skill_D / 100)
        rpc = r.rmse_pers_naive
        rmc = rpc * (1 - r.skill_C / 100)
        row = dict(dataset=r.dataset.split()[0], model=r.model, H=r.horizon,
                   f=f, measured_bias_C_minus_D=r.skill_C - r.skill_D)
        if f > 1e-9:
            bp2 = (rpc ** 2 - (1 - f) * ap ** 2) / f
            bm2 = (rmc ** 2 - (1 - f) * am ** 2) / f
            rho_p, rho_m = np.sqrt(bp2) / ap, np.sqrt(bm2) / am
            exact = 100 * (am / ap) * (1 - np.sqrt(
                (1 + f * (rho_m ** 2 - 1)) / (1 + f * (rho_p ** 2 - 1))))
            first = 100 * (am / ap) * f * (rho_p ** 2 - rho_m ** 2) / 2
            row.update(rho_p=rho_p, rho_m=rho_m, formula_exact=exact,
                       formula_first_order=first)
        out.append(row)
    return pd.DataFrame(out).round(3)


def draw(t, stem):
    t = t[(t.f > 1e-9) & t.rho_p.notna()].copy()
    t['label'] = t.model + ' H=' + t.H.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axes[0]
    x = np.arange(len(t))
    w = 0.38
    ax.bar(x - w / 2, t.rho_p, w, color='#D85A30', label='persistence')
    ax.bar(x + w / 2, t.rho_m, w, color='#1D9E75', label='model')
    ax.axhline(1, color='k', lw=0.8)
    ax.set_ylim(0, max(t.rho_p.max(), t.rho_m.max()) * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(t.label, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('Error ratio, displaced / valid rows', fontsize=7.5)
    ax.set_title('(a) Error inflation', fontsize=8, fontweight='bold', pad=3)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    ax.bar(x, 100 * t.f, 0.6, color='0.55')
    ax.set_xticks(x)
    ax.set_xticklabels(t.label, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('Displaced share f of scored rows (%)', fontsize=7.5)
    ax.set_title('(b) Exposure', fontsize=8, fontweight='bold', pad=3)
    ax = axes[2]
    ax.scatter(t.measured_bias_C_minus_D, t.formula_exact, s=18, c='#4A7BB7',
               label='exact formula', zorder=3)
    ax.scatter(t.measured_bias_C_minus_D, t.formula_first_order, s=18,
               marker='x', c='#D85A30', label='first order', zorder=3)
    allv = pd.concat([t.measured_bias_C_minus_D, t.formula_exact,
                      t.formula_first_order])
    lim = [min(0, allv.min()) - 0.3, max(allv.max(), 0.5) + 0.3]
    ax.plot(lim, lim, color='k', lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Measured C - D (points)', fontsize=7.5)
    ax.set_ylabel('Formula (points)', fontsize=7.5)
    ax.set_title('(c) Formula vs measured', fontsize=8, fontweight='bold', pad=3)
    ax.legend(fontsize=6.5, frameon=False)
    for a in axes:
        a.tick_params(labelsize=6.5, length=2)
        a.grid(alpha=0.3, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    dec = pd.read_csv(os.path.join(HERE, 'artifact_decomposition.csv'))
    ali = pd.read_csv(os.path.join(HERE, 'artifact_aligned.csv'))
    t = analyse(dec, ali)
    t.to_csv(os.path.join(HERE, 'mixture_table.csv'), index=False)
    print(t.to_string(index=False))
    live = t[t.f > 1e-9]
    if len(live):
        err = (live.formula_exact - live.measured_bias_C_minus_D).abs().max()
        print(f'\nLargest |exact formula - measured|: {err:.3f} points '
              '(it should be near the CSV rounding, about 0.01-0.05)')
    draw(t, os.path.join(HERE, 'Figure_mixture'))
    print('Saved mixture_table.csv and Figure_mixture.pdf / .png')
