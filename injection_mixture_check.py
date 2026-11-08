import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'gap_injection_v3_results.csv')
W = 60
COLOURS = {'MANY-SHORT': '#1D9E75', 'MEDIUM': '#4A7BB7', 'FEW-LONG': '#D85A30'}


def analyse(df):
    d = df[df.model == 'Ridge'].copy()
    d['f'] = 1 - d.n_test_honest / d.n_test_naive
    d = d[d.f > 1e-6].copy()
    ap, am = d.rmse_pers_honest, d.rmse_model_honest
    f = d.f
    bp2 = (d.rmse_pers_naive ** 2 - (1 - f) * ap ** 2) / f
    bm2 = (d.rmse_model_naive ** 2 - (1 - f) * am ** 2) / f
    ok = (bp2 > 0) & (bm2 > 0)
    d = d[ok].copy()
    ap, am, f, bp2, bm2 = ap[ok], am[ok], f[ok], bp2[ok], bm2[ok]
    d['rho_p'] = np.sqrt(bp2) / ap
    d['rho_m'] = np.sqrt(bm2) / am
    d['bias_hat'] = 100 * (am / ap) * (1 - np.sqrt(
        (1 + f * (d.rho_m ** 2 - 1)) / (1 + f * (d.rho_p ** 2 - 1))))
    p = d.achieved_missing_pct / 100
    g = d.mean_gap_len
    d['f_design'] = np.minimum(1, p * (W + d.horizon) / ((1 - p) * g))
    return d


def draw(d, stem):
    """Figure 7: (a) bias by exposure bin, (b) measured vs design exposure,
    (c) median error inflation by gap structure."""
    from scipy.stats import spearmanr
    ORG, GRN, BLU = '#D85A30', '#1D9E75', '#4A7BB7'
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0]
    d = d.copy()
    d['fbin'] = pd.cut(d.f, bins)
    g = d.groupby('fbin', observed=True).bias.agg(
        ['count', 'median', lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    g.columns = ['n', 'med', 'q1', 'q3']
    mid = [(a + b) / 2 * 100 for a, b in zip(bins[:-1], bins[1:])]
    a = ax[0]
    a.errorbar(mid, g.med, yerr=[g.med - g.q1, g.q3 - g.med], fmt='o-', color='k',
               ms=3.5, lw=1.1, capsize=2)
    a.axhline(0, color='k', lw=0.8)
    a.set_xlabel('Exposure f (%, bin centre)', fontsize=8)
    a.set_ylabel('Bias (points), median and IQR', fontsize=8)
    a.set_title('(a) Bias vs exposure', fontsize=8.5, fontweight='bold', pad=3)
    a.set_ylim(-1.5, 10.5)
    b = ax[1]
    for c, col in (('MANY-SHORT', GRN), ('MEDIUM', BLU), ('FEW-LONG', ORG)):
        s = d[d.condition == c]
        b.scatter(s.f_design, s.f, s=5, alpha=0.35, color=col, edgecolor='none',
                  label=c.capitalize())
    b.plot([0, 1], [0, 1], color='k', lw=0.8)
    b.text(0.03, 0.93, f'Spearman {spearmanr(d.f_design, d.f)[0]:.2f}', fontsize=7,
           transform=b.transAxes)
    b.set_xlabel('Design exposure, Eq. (6)', fontsize=8)
    b.set_ylabel('Measured exposure f', fontsize=8)
    b.set_title('(b) Exposure from gap design', fontsize=8.5, fontweight='bold', pad=3)
    b.legend(fontsize=6.5, frameon=False, loc='lower right', markerscale=2)
    c = ax[2]
    conds = ['FEW-LONG', 'MEDIUM', 'MANY-SHORT']
    med = d.groupby('condition')[['rho_p', 'rho_m']].median().loc[conds]
    x = np.arange(3)
    c.plot(x, med.rho_p, 'o', color=ORG, ms=6, label='persistence')
    c.plot(x, med.rho_m, 's', color=GRN, ms=6, label='ridge')
    for i in range(3):
        c.plot([i, i], [med.rho_m.iloc[i], med.rho_p.iloc[i]], color='0.5', lw=1)
    c.axhline(1, color='k', lw=0.8)
    c.set_xticks(x)
    c.set_xticklabels(['Few-long', 'Medium', 'Many-short'], fontsize=7)
    c.set_ylabel('Median error ratio, displaced / valid', fontsize=8)
    c.set_title('(c) Error inflation', fontsize=8.5, fontweight='bold', pad=3)
    c.legend(fontsize=7, frameon=False, loc='upper right')
    c.set_xlim(-0.5, 2.5)
    c.set_ylim(0.93, 1.27)
    for A in ax:
        A.tick_params(labelsize=7, length=2)
        A.grid(alpha=0.3, lw=0.5)
        A.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('\nBias by exposure bin:')
    print(g.round(2).to_string())


if __name__ == '__main__':
    df = pd.read_csv(CSV)
    d = analyse(df)
    cols = ['record', 'condition', 'missing_frac', 'horizon', 'origin', 'f',
            'f_design', 'rho_p', 'rho_m', 'bias', 'bias_hat']
    d[cols].round(4).to_csv(os.path.join(HERE, 'injection_mixture_table.csv'),
                            index=False)
    print(f'{len(d)} ridge runs with displaced rows')
    print('\nMedians by structure:')
    print(d.groupby('condition')[['f', 'f_design', 'rho_p', 'rho_m', 'bias',
                                  'bias_hat']].median().round(3).to_string())
    err = (d.bias_hat - d.bias).abs()
    print(f'\n|formula - measured|: median {err.median():.3f}, '
          f'95th percentile {err.quantile(0.95):.3f} points')
    r = np.corrcoef(np.log(d.f_design.clip(1e-4)), np.log(d.f.clip(1e-4)))[0, 1]
    print(f'Correlation of log f_design with log f: {r:.3f}')
    ratio = (d.f / d.f_design).groupby(d.condition).median().round(2)
    print('Median measured / design exposure by structure:')
    print(ratio.to_string())
    draw(d, os.path.join(HERE, 'Figure_7'))
    print('Saved injection_mixture_table.csv and Figure_7.pdf / .png')
