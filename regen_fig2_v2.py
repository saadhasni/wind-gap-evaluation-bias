import os
import sys

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1
                   else 'gap_profile_summary.csv')
HORIZONS = (1, 3, 6)
MAX_D = 12
MODEL = 'Ridge'
PAPER_RECS = [('D3_ONSHORE', 'D3 onshore'), ('D2_BUOY', 'D2 buoy')]
PERS_C, MODEL_C = '#D85A30', '#1D9E75'

F_TITLE, F_AXIS, F_TICK, F_N, F_LEG = 8.5, 8, 7, 6, 8


def annotate_n(ax, g):
    """Small n labels at the top of the panel, one per plotted d."""
    for _, r in g.iterrows():
        ax.annotate(f'{int(r.n)}', (r.d, 0.97), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=F_N, color='0.35', rotation=90)


def draw_profile(ax, g, legend=False):
    ax.plot(g.d, g.rmse_pers, 'o-', color=PERS_C, ms=3.2, lw=1.1,
            label='persistence' if legend else None)
    ax.plot(g.d, g.rmse_model, 's-', color=MODEL_C, ms=3.2, lw=1.1,
            label=MODEL if legend else None)
    if g.rmse_pers_far.iloc[0] == g.rmse_pers_far.iloc[0]:
        ax.axhline(g.rmse_pers_far.iloc[0], color=PERS_C, ls=':', lw=1)
        ax.axhline(g.rmse_model_far.iloc[0], color=MODEL_C, ls=':', lw=1)
    lo = min(g.rmse_pers.min(), g.rmse_model.min(), g.rmse_pers_far.iloc[0],
             g.rmse_model_far.iloc[0])
    hi = max(g.rmse_pers.max(), g.rmse_model.max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.30 * (hi - lo))  # room for n
    annotate_n(ax, g)
    ax.set_xlim(0.5, MAX_D + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(labelsize=F_TICK, length=2)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)


def main():
    if not os.path.exists(CSV):
        sys.exit(f'Cannot find {CSV}. Put this script in the folder that '
                 'contains gap_profile_summary.csv, or pass its name.')
    prof = pd.read_csv(CSV)
    need = {'record', 'horizon', 'd', 'n', 'rmse_pers', 'rmse_model',
            'rmse_pers_far', 'rmse_model_far'}
    if need - set(prof.columns):
        sys.exit(f'CSV is missing columns: {sorted(need - set(prof.columns))}')

    fig, axes = plt.subplots(2, len(HORIZONS), squeeze=False,
                             figsize=(7.0, 4.9), sharex=True)
    for r_i, (name, lab) in enumerate(PAPER_RECS):
        for c_i, H in enumerate(HORIZONS):
            ax = axes[r_i][c_i]
            g = prof[(prof.record == name) & (prof.horizon == H)]
            if len(g):
                draw_profile(ax, g, legend=(r_i == 0 and c_i == 0))
            ax.set_title(f'{lab}, H = {H}', fontsize=F_TITLE,
                         fontweight='bold', pad=3)
            if c_i == 0:
                ax.set_ylabel('RMSE (m s$^{-1}$)', fontsize=F_AXIS)
            if r_i == 1:
                ax.set_xlabel('Observations since gap (d)', fontsize=F_AXIS)
            ax.spines[['top', 'right']].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=F_LEG, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.045, 1, 1], h_pad=1.0, w_pad=1.0)
    stem = os.path.join(HERE, 'Figure_2')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved Figure_2.pdf and .png')


if __name__ == '__main__':
    main()
