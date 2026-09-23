import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

GREEN = '#2e9e78'      # rows retained by the gap-aware protocol
BLUE = '#2f6fb5'       # rows retained by the naive protocol
GREY = '#c3c7cc'       # rows discarded after a gap
ORANGE = '#e2622a'     # naive arm in Figure 3
DPI = 200

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'artifact_aligned.csv')


# Figure 2 - the four configurations (schematic, no data needed)

def figure2(path='fig2_configurations.png'):
    fig, ax = plt.subplots(figsize=(11.0, 4.1))

    # a schematic record: (start, width, kind) with kind in
    # 'clean' | 'post_gap' | 'gap'
    blocks = [(0.5, 2.2, 'clean'), (2.7, 0.5, 'gap'), (3.2, 0.6, 'post_gap'),
              (3.8, 1.4, 'clean'), (5.2, 0.4, 'gap'), (5.6, 0.6, 'post_gap'),
              (6.2, 2.0, 'clean'), (8.2, 0.4, 'gap'), (8.6, 0.6, 'post_gap'),
              (9.2, 1.9, 'clean'), (11.1, 0.4, 'gap'), (11.5, 0.6, 'post_gap'),
              (12.1, 1.4, 'clean')]
    split_x = 7.4                      # train / test boundary
    bar_h = 0.42

    rows = [
        ('A (gap-aware)', 'gap-aware features', 'gap-valid | gap-valid', True),
        ('B',             'naive features',     'gap-valid | gap-valid', True),
        ('D',             'naive features',     'naive | gap-valid',     False),
        ('C (naive)',     'naive features',     'naive | all rows',      False),
    ]

    for i, (name, sub, right, aware_train) in enumerate(rows):
        y = 3.3 - i * 1.0
        # split any block that straddles the train/test boundary so the
        # colour change lines up exactly with the dashed line
        pieces = []
        for x0, w, kind in blocks:
            if x0 < split_x < x0 + w:
                pieces.append((x0, split_x - x0, kind, True))
                pieces.append((split_x, x0 + w - split_x, kind, False))
            else:
                pieces.append((x0, w, kind, x0 + w <= split_x))

        for x0, w, kind, train_side in pieces:
            if kind == 'gap':
                ax.add_patch(mpatches.Rectangle((x0, y), w, bar_h,
                             facecolor='white', edgecolor='#9aa0a6',
                             linewidth=0.7, linestyle='-'))
                continue
            if name.startswith('C'):            
                colour = BLUE
            elif name == 'D':                   
                colour = BLUE if train_side else (GREY if kind == 'post_gap' else GREEN)
            else:                              
                colour = GREY if kind == 'post_gap' else GREEN
            ax.add_patch(mpatches.Rectangle((x0, y), w, bar_h,
                         facecolor=colour, edgecolor='none'))

        ax.text(0.35, y + bar_h / 2 + 0.10, name, ha='right', va='center',
                fontsize=10, fontweight='bold')
        ax.text(0.35, y + bar_h / 2 - 0.14, sub, ha='right', va='center',
                fontsize=8, color='#555555')
        ax.text(13.75, y + bar_h / 2, right, ha='left', va='center',
                fontsize=8, color='#333333')

    ax.axvline(split_x, ymin=0.22, ymax=0.88, color='#333333',
               linestyle='--', linewidth=1.0)
    ax.text(split_x - 0.15, 4.15, 'training period', ha='right', fontsize=9)
    ax.text(split_x + 0.15, 4.15,
            'test period (same calendar window for all runs)',
            ha='left', fontsize=9)

    ax.text(0.5, -0.42, 'B \u2212 A  =  feature contamination', fontsize=9)
    ax.text(4.9, -0.42, 'D \u2212 B  =  training contamination', fontsize=9)
    ax.text(9.3, -0.42, 'C \u2212 D  =  evaluation composition', fontsize=9)
    ax.text(0.5, -0.80, 'C \u2212 A  =  total bias', fontsize=9,
            fontweight='bold')

    handles = [mpatches.Patch(facecolor=GREEN, label='gap-valid rows retained'),
               mpatches.Patch(facecolor=BLUE, label='all rows retained (naive)'),
               mpatches.Patch(facecolor=GREY, label='rows discarded after a gap'),
               mpatches.Patch(facecolor='white', edgecolor='#9aa0a6',
                              label='gap in record')]
    ax.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, -0.14))

    ax.set_title('The four configurations. Only gap handling differs; '
                 'the test period is identical.',
                 fontsize=10, fontweight='bold', pad=16)
    ax.set_xlim(-3.0, 17.0)
    ax.set_ylim(-1.15, 4.6)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


# Figure 3 - reported skill under the two protocols

def figure3(path='fig3_skill.png', csv=CSV):
    df = pd.read_csv(csv)
    order = [d for d in ['D1 ZephIR (contiguous)',
                         'D2 WFIP3 Buoy (few long gaps)',
                         'D3 Onshore (many short gaps)'] if d in set(df.dataset)]

    fig, axes = plt.subplots(1, len(order), figsize=(12.5, 3.5))
    if len(order) == 1:
        axes = [axes]

    for ax, ds in zip(axes, order):
        g = df[df.dataset == ds].sort_values('horizon')
        x = np.arange(len(g))
        w = 0.38
        ax.bar(x - w / 2, g.skill_honest, w, color=GREEN, label='gap-aware')
        ax.bar(x + w / 2, g.skill_naive, w, color=ORANGE, label='naive')

        for xi, (a, c) in enumerate(zip(g.skill_honest, g.skill_naive)):
            if abs(c - a) < 0.01:      
                continue               
            ax.plot([xi - w / 2, xi + w / 2], [a, c], color='#333333',
                    linewidth=0.9, zorder=3)

        segs = int(g.segments.iloc[0]) if 'segments' in g else None
        if segs is None:
            title = ds
        else:
            word = 'segment' if segs == 1 else 'segments'
            title = f'{ds}\n({segs:,} {word})'
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'H={h}' for h in g.horizon], fontsize=9)
        ax.axhline(0, color='#666666', linewidth=0.8)
        ax.set_ylabel('Skill vs persistence (%)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Reported skill under naive versus gap-aware evaluation, '
                 'both scored on the same calendar window',
                 fontsize=10, fontweight='bold', y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    figure2()
    if os.path.exists(CSV):
        figure3()
    else:
        print(f'skipped Figure 3: {CSV} not found - run '
              'run_artifact_aligned.py first, or edit CSV at the top')
