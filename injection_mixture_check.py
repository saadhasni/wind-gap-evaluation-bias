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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax = axes[0]
    for c, g in d.groupby('condition'):
        ax.scatter(g.bias, g.bias_hat, s=7, alpha=0.6, color=COLOURS[c],
                   label=c.capitalize(), edgecolor='none')
    lim = [min(d.bias.min(), d.bias_hat.min()), max(d.bias.max(), d.bias_hat.max())]
    ax.plot(lim, lim, color='k', lw=0.8)
    ax.set_xlabel('Measured bias (points)', fontsize=7.5)
    ax.set_ylabel('Mixture formula (points)', fontsize=7.5)
    ax.set_title('(a) Formula vs measured', fontsize=8, fontweight='bold', pad=3)
    ax.legend(fontsize=6.5, frameon=False, markerscale=1.5)
    ax = axes[1]
    for c, g in d.groupby('condition'):
        ax.scatter(100 * g.f, g.bias, s=7, alpha=0.6, color=COLOURS[c],
                   edgecolor='none')
    ax.set_xlabel('Exposure f (% of scored rows displaced)', fontsize=7.5)
    ax.set_ylabel('Measured bias (points)', fontsize=7.5)
    ax.set_title('(b) Bias vs exposure', fontsize=8, fontweight='bold', pad=3)
    ax = axes[2]
    for c, g in d.groupby('condition'):
        ax.scatter(g.f_design, g.f, s=7, alpha=0.6, color=COLOURS[c],
                   edgecolor='none')
    ax.plot([0, 1], [0, 1], color='k', lw=0.8)
    ax.set_xlabel('Design exposure p(W+H)/((1-p)g)', fontsize=7.5)
    ax.set_ylabel('Measured exposure f', fontsize=7.5)
    ax.set_title('(c) Exposure from gap design', fontsize=8, fontweight='bold',
                 pad=3)
    for a in axes:
        a.tick_params(labelsize=6.5, length=2)
        a.grid(alpha=0.3, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    draw(d, os.path.join(HERE, 'Figure_injection_mechanism'))
    print('Saved injection_mixture_table.csv and Figure_injection_mechanism.pdf / .png')
