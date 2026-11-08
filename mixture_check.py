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
    """Figure 5: error inflation on displaced rows and the resulting composition
    bias, D3 only (D1 and D2 have exposure 0 in the aligned window)."""
    t = t[(t.f > 1e-9) & t.rho_p.notna()].copy()
    ORG, GRN = '#D85A30', '#1D9E75'
    H = sorted(t.H.unique())
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    a = ax[0]
    g = t[t.model == 'XGBoost'].sort_values('H')
    a.plot(H, g.rho_p, 'o-', color=ORG, ms=4.5, lw=1.3, label='persistence')
    for mod, fill in (('XGBoost', GRN), ('Ridge', 'white')):
        g = t[t.model == mod].sort_values('H')
        a.plot(H, g.rho_m, marker='s', ms=4.5, lw=1.3, color=GRN, mfc=fill, label=mod)
    a.axhline(1, color='k', lw=0.8)
    a.set_xticks(H)
    a.set_xlabel('Horizon H (samples)', fontsize=8)
    a.set_ylabel('Error ratio, displaced / valid rows', fontsize=8)
    a.set_ylim(0.95, 1.4)
    a.set_title('(a) Error inflation', fontsize=8.5, fontweight='bold', pad=3)
    a.legend(fontsize=7, frameon=False, loc='upper right')
    b = ax[1]
    w = 0.36
    x = np.arange(len(H))
    for i, (mod, hat) in enumerate((('XGBoost', None), ('Ridge', '//'))):
        g = t[t.model == mod].sort_values('H')
        b.bar(x + (i - 0.5) * w, g.measured_bias_C_minus_D, w, color=GRN, hatch=hat,
              edgecolor='white', label=mod, alpha=1 if hat is None else 0.65)
    b.set_xticks(x)
    b.set_xticklabels([f'H = {h}' for h in H], fontsize=7)
    b.axhline(0, color='k', lw=0.8)
    b.set_ylabel('Composition bias, C - D (points)', fontsize=8)
    b.set_title('(b) Resulting composition bias', fontsize=8.5, fontweight='bold', pad=3)
    b.legend(fontsize=7, frameon=False, loc='upper left')
    for A in ax:
        A.tick_params(labelsize=7, length=2)
        A.grid(alpha=0.3, lw=0.5)
        A.spines[['top', 'right']].set_visible(False)
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
              '(an identity check of the implementation; it should be near CSV rounding)')
    draw(t, os.path.join(HERE, 'Figure_5'))
    print('Saved mixture_table.csv and Figure_5.pdf / .png')
