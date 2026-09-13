import warnings, os, sys, glob
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

NAVY = '#2F4F6F'
GAPC = '#D85A30'


def load_all():
    ds = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        ds.append(('D1 ZephIR', s, pd.Timedelta('1min')))
    except Exception as e:
        print('[skip D1]', e)
    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        ds.append(('D2 Buoy 130', s, pd.Timedelta('10min')))
    except Exception as e:
        print('[skip D2]', e)
    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        ds.append(('D3 Onshore', s, pd.Timedelta('10min')))
    except Exception as e:
        print('[skip D3]', e)
    return ds


def stats(name, s, step):
    d = s.index.to_series().diff()
    gaps = d[d > step]
    gl = (gaps / step).astype(int)
    span = s.index[-1] - s.index[0]
    expected = int(span / step) + 1
    seg = FP.segment_ids(s.index, step)
    return dict(
        dataset=name, n=len(s),
        span_days=round(span.total_seconds() / 86400, 1),
        completeness=round(100 * len(s) / expected, 2),
        mean=round(float(s.mean()), 2), sd=round(float(s.std()), 2),
        vmin=round(float(s.min()), 2), vmax=round(float(s.max()), 2),
        p05=round(float(s.quantile(0.05)), 2),
        p95=round(float(s.quantile(0.95)), 2),
        segments=int(seg.max() + 1), n_gaps=int(len(gaps)),
        pct_after_gap=round(100 * len(gaps) / len(s), 2),
        gap_median=str(gaps.median()) if len(gaps) else '-',
        gap_p95=str(gaps.quantile(0.95)) if len(gaps) else '-',
        gap_max=str(gaps.max()) if len(gaps) else '-',
        gap_median_samples=int(gl.median()) if len(gl) else 0,
        gap_max_samples=int(gl.max()) if len(gl) else 0)


def main():
    ds = load_all()
    if not ds:
        print('No datasets found.'); return
    rows = [stats(n, s, st) for n, s, st in ds]
    df = pd.DataFrame(rows)
    df.to_csv('data_section_statistics.csv', index=False)

    print('=' * 78)
    print('  PASTE THESE INTO TABLE 2 OF THE PAPER')
    print('=' * 78)
    order = [('Mean wind speed (m/s)', 'mean'),
             ('Standard deviation (m/s)', 'sd'),
             ('Minimum (m/s)', 'vmin'),
             ('Maximum (m/s)', 'vmax'),
             ('5th / 95th percentile (m/s)', None)]
    names = list(df.dataset)
    print(f"\n  {'Quantity':<30}" + ''.join(f"{n:>16}" for n in names))
    for label, key in order:
        if key is None:
            vals = [f"{r.p05} / {r.p95}" for r in df.itertuples()]
        else:
            vals = [f"{getattr(r, key)}" for r in df.itertuples()]
        print(f"  {label:<30}" + ''.join(f"{v:>16}" for v in vals))

    print('\n  Cross-check against the values in Table 2:')
    for label, key in [('Samples', 'n'), ('Calendar span (d)', 'span_days'),
                       ('Contiguous segments', 'segments'),
                       ('Number of gaps', 'n_gaps'),
                       ('Rows following a gap (%)', 'pct_after_gap')]:
        vals = [f"{getattr(r, key)}" for r in df.itertuples()]
        print(f"  {label:<30}" + ''.join(f"{v:>16}" for v in vals))
    print(f"  {'Median gap':<30}" + ''.join(f"{r.gap_median:>16}"
                                            for r in df.itertuples()))
    print(f"  {'Maximum gap':<30}" + ''.join(f"{r.gap_max:>16}"
                                             for r in df.itertuples()))
    print('\n  Saved data_section_statistics.csv')

    fig, axes = plt.subplots(len(ds), 1, figsize=(11, 2.35 * len(ds)),
                             squeeze=False)
    for ax, (name, s, st) in zip(axes[:, 0], ds):
        n = int(pd.Timedelta('7D') / st)
        mid = len(s) // 2
        seg = s.iloc[mid:mid + n]
        ax.plot(seg.index, seg.values, lw=0.7, color=NAVY)
        d = seg.index.to_series().diff()
        marks = seg.index[(d > st).values]
        for t in marks:
            ax.axvline(t, color=GAPC, lw=0.9, alpha=0.75)
        segs = FP.segment_ids(seg.index, st)
        ax.set_title(f'{name} \u2014 one representative week: '
                     f'{len(marks)} gaps, {int(segs.max()) + 1} segments',
                     fontsize=9.5, fontweight='bold')
        ax.set_ylabel('m/s', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle('Representative week from each dataset. '
                 'Vertical lines mark gaps.',
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('fig_data_overview.png', dpi=190, bbox_inches='tight')
    print('  Saved fig_data_overview.png  \u2014 this is Figure 2 of the paper')


if __name__ == '__main__':
    main()
