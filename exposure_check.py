import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W_PIPE = 60
STEPS = {'D1': pd.Timedelta('1min'), 'D2': pd.Timedelta('10min'),
         'D3': pd.Timedelta('10min')}
HORIZONS = {'D1': (1, 10, 30), 'D2': (1, 3, 6), 'D3': (1, 3, 6)}
SEQ = {'D1': 60, 'D2': 36, 'D3': 36}


def segment_lengths(index, step):
    """Lengths of the runs of consecutive samples on the regular grid."""
    d = index.to_series().diff()
    breaks = (d != step).to_numpy(copy=True)
    breaks[0] = False
    return np.bincount(np.cumsum(breaks))


def n_valid(lengths, W, H):
    return int(np.maximum(0, lengths - W - H).sum())


def retention(lengths, W, H):
    naive = int(lengths.sum()) - W - H
    return n_valid(lengths, W, H) / naive if naive > 0 else float('nan')


def pipeline_counts(series, step, H, seq_len):
    """(valid origins, naive origins) exactly as the pipeline builds them."""
    import final_pipeline as FP
    fgap = FP.build_features_segmented(series, step)
    fnav = FP.naive_features(series)
    valid = int(FP.valid_rows_for_horizon(series, fgap, H, seq_len).sum())
    naive = len(FP.naive_frame(series, H, fnav))
    return valid, naive


def draw(lengths_by_ds, path_stem):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cols = {'D1': '#4A7BB7', 'D2': '#1D9E75', 'D3': '#D85A30'}
    W = np.arange(1, 151)
    ax = axes[0]
    for k, L in lengths_by_ds.items():
        ax.plot(W, [100 * retention(L, w, 6) for w in W], color=cols[k], lw=1.4,
                label=f'{k} ({len(L):,} segment' + ('' if len(L) == 1 else 's') + ')')
    ax.axvline(W_PIPE, color='0.3', ls=':', lw=1)
    ax.text(W_PIPE + 2, 62, 'W = 60', fontsize=7, color='0.3')
    ax.set_xlabel('Feature-window depth W (samples)', fontsize=8)
    ax.set_ylabel('Origins kept by gap-aware protocol (%)', fontsize=8)
    ax.set_title('H = 6', fontsize=8.5, fontweight='bold', pad=3)
    ax.set_ylim(-3, 103)
    ax = axes[1]
    Hs = np.arange(1, 61)
    for k, L in lengths_by_ds.items():
        ax.plot(Hs, [100 * retention(L, W_PIPE, h) for h in Hs], color=cols[k],
                lw=1.4, label=f'{k}')
    ax.set_xlabel('Horizon H (samples)', fontsize=8)
    ax.set_title('W = 60', fontsize=8.5, fontweight='bold', pad=3)
    ax.set_ylim(-3, 103)
    for a in axes:
        a.tick_params(labelsize=7, length=2)
        a.grid(alpha=0.3, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(path_stem + '.pdf', bbox_inches='tight')
    fig.savefig(path_stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    import run_artifact_aligned as RA
    rows, lengths_by_ds = [], {}
    for key in ('D1', 'D2', 'D3'):
        try:
            s = RA.load_dataset(key)
        except FileNotFoundError as e:
            print(f'[skip {key}]', e)
            continue
        L = segment_lengths(s.index, STEPS[key])
        lengths_by_ds[key] = L
        for H in HORIZONS[key]:
            v_f, n_f = n_valid(L, W_PIPE, H), int(L.sum()) - W_PIPE - H
            v_p, n_p = pipeline_counts(s, STEPS[key], H, SEQ[key])
            rows.append(dict(dataset=key, segments=len(L), H=H,
                             valid_formula=v_f, valid_pipeline=v_p,
                             naive_formula=n_f, naive_pipeline=n_p,
                             retention_formula_pct=round(100 * v_f / n_f, 2),
                             retention_pipeline_pct=round(100 * v_p / n_p, 2),
                             exposure_pct=round(100 * (1 - v_f / n_f), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'exposure_check.csv'), index=False)
    print(df.to_string(index=False))
    bad = df[(df.valid_formula != df.valid_pipeline) |
             (df.naive_formula != df.naive_pipeline)]
    print('\nFormula reproduces the pipeline counts exactly:', bad.empty)
    if not bad.empty:
        print(bad.to_string(index=False))
    draw(lengths_by_ds, os.path.join(HERE, 'Figure_exposure'))
    print('Saved Figure_exposure.pdf and .png')


if __name__ == '__main__':
    main()
