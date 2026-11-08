import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['hatch.linewidth'] = 0.6

GREEN = '#2e9e78'      # rows retained by the gap-aware protocol
BLUE = '#2f6fb5'       # rows retained by the naive protocol
GREY = '#c3c7cc'       # rows discarded after a gap
HERE = os.path.dirname(os.path.abspath(__file__))

F_NAME, F_SUB, F_SIDE, F_HEAD, F_EQ, F_LEG = 8, 6.8, 6.8, 7.2, 7.2, 6.6


def main(stem=os.path.join(HERE, 'Figure_1')):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    # a schematic record: (start, width, kind), kind in clean | post_gap | gap
    blocks = [(0.5, 2.2, 'clean'), (2.7, 0.5, 'gap'), (3.2, 0.6, 'post_gap'),
              (3.8, 1.4, 'clean'), (5.2, 0.4, 'gap'), (5.6, 0.6, 'post_gap'),
              (6.2, 2.0, 'clean'), (8.2, 0.4, 'gap'), (8.6, 0.6, 'post_gap'),
              (9.2, 1.9, 'clean'), (11.1, 0.4, 'gap'), (11.5, 0.6, 'post_gap'),
              (12.1, 1.4, 'clean')]
    # training ends at train_x, testing starts at split_x; the embargo between
    # them is left out by every arm, so no training target lies in the test window
    train_x, split_x = 7.0, 7.4
    bar_h = 0.46

    rows = [
        ('A (gap-aware)', 'gap-aware features', 'gap-valid | gap-valid', True),
        ('B',             'naive features',     'gap-valid | gap-valid', True),
        ('D',             'naive features',     'naive | gap-valid',     False),
        ('C (naive)',     'naive features',     'naive | all rows',      False),
    ]

    for i, (name, sub, right, aware_train) in enumerate(rows):
        y = 3.3 - i * 1.0
        pieces = []
        for x0, w, kind in blocks:
            cuts = [x0] + [c for c in (train_x, split_x) if x0 < c < x0 + w]
            cuts.append(x0 + w)
            for a, b in zip(cuts[:-1], cuts[1:]):
                pieces.append((a, b - a, kind, b <= train_x,
                               a >= train_x and b <= split_x))

        for x0, w, kind, train_side, embargo in pieces:
            if embargo:
                ax.add_patch(mpatches.Rectangle((x0, y), w, bar_h,
                             facecolor='white', edgecolor='#555555',
                             hatch='////', linewidth=0.0))
                continue
            if kind == 'gap':
                ax.add_patch(mpatches.Rectangle((x0, y), w, bar_h,
                             facecolor='white', edgecolor='#9aa0a6',
                             linewidth=0.6, linestyle='-'))
                continue
            if name.startswith('C'):
                colour = BLUE
            elif name == 'D':
                colour = BLUE if train_side else (GREY if kind == 'post_gap' else GREEN)
            else:
                colour = GREY if kind == 'post_gap' else GREEN
            ax.add_patch(mpatches.Rectangle((x0, y), w, bar_h,
                         facecolor=colour, edgecolor='none'))

        ax.text(0.3, y + bar_h / 2 + 0.13, name, ha='right', va='center',
                fontsize=F_NAME, fontweight='bold')
        ax.text(0.3, y + bar_h / 2 - 0.17, sub, ha='right', va='center',
                fontsize=F_SUB, color='#555555')
        ax.text(13.7, y + bar_h / 2, right, ha='left', va='center',
                fontsize=F_SIDE, color='#333333')

    for xv in (train_x, split_x):      # dashed lines stop at the bars
        ax.plot([xv, xv], [0.2, 3.95], color='#333333', linestyle='--',
                linewidth=0.8)
    ax.text((train_x + split_x) / 2, 0.14, 'embargo', ha='center', va='top',
            fontsize=F_SIDE, color='#333333')
    ax.text(train_x - 0.15, 4.12, 'training period (same end for all runs)',
            ha='right', fontsize=F_HEAD)
    ax.text(split_x + 0.15, 4.12,
            'test period (same calendar window for all runs)',
            ha='left', fontsize=F_HEAD)

    ax.text(-1.9, -0.55, 'B \u2212 A  =  feature contamination', fontsize=F_EQ)
    ax.text(3.7, -0.55, 'D \u2212 B  =  training contamination', fontsize=F_EQ)
    ax.text(9.5, -0.55, 'C \u2212 D  =  evaluation composition', fontsize=F_EQ)
    ax.text(-1.9, -0.95, 'C \u2212 A  =  total bias', fontsize=F_EQ,
            fontweight='bold')

    handles = [mpatches.Patch(facecolor=GREEN, label='gap-valid rows retained'),
               mpatches.Patch(facecolor=BLUE, label='all rows retained (naive)'),
               mpatches.Patch(facecolor=GREY, label='rows discarded after a gap'),
               mpatches.Patch(facecolor='white', edgecolor='#9aa0a6',
                              label='gap in record'),
               mpatches.Patch(facecolor='white', edgecolor='#555555',
                              hatch='////', label='embargo (unused)')]
    ax.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
              fontsize=F_LEG, bbox_to_anchor=(0.5, -0.07), columnspacing=1.2,
              handlelength=1.4)

    ax.set_xlim(-2.2, 17.2)
    ax.set_ylim(-1.75, 4.5)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(stem + '.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', os.path.basename(stem) + '.pdf and .png')


if __name__ == '__main__':
    main()
