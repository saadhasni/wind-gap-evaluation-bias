import warnings, os, sys
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
    # No silent skipping: a missing dataset would silently change Table I.
    ds = []
    s, meta = load_zephir(HERE, height_m=38, resample='1min')
    print(f"  D1 loaded: height_m={meta['height_m']}, "
          f"n_filled={meta['n_filled']}")
    ds.append(('D1 ZephIR', s, pd.Timedelta('1min')))
    s, meta = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                              height_m=38, resample='10min')
    print(f"  D2 loaded: height_m={meta['height_m']}, "
          f"n_filled={meta['n_filled']}")
    ds.append(('D2 Buoy 130', s, pd.Timedelta('10min')))
    s = FP.load_onshore(HERE)
    print("  D3 loaded: no height metadata, n_filled=0 (not interpolated)")
    ds.append(('D3 Onshore', s, pd.Timedelta('10min')))
    return ds


def fmt_td(td):
    """Timedelta rounded to the minute, as 'X d Y.Y h' / 'Y.Y h' / 'N min'."""
    td = pd.Timedelta(td).round('1min')
    h = td.total_seconds() / 3600
    if h >= 24:
        return f"{int(h // 24)} d {h % 24:.1f} h"
    if h >= 1:
        return f"{h:.1f} h"
    return f"{int(round(h * 60))} min"


def stats(name, s, step):
    d = s.index.to_series().diff().iloc[1:]
    assert (d > pd.Timedelta(0)).all(), f'{name}: index not strictly increasing'
    gaps = d[d != step]
    assert ((gaps / step) % 1 == 0).all(), f'{name}: off-grid timestamps'
    # gap length = number of MISSING samples between two valid ones
    gl = (gaps / step).astype(int) - 1
    span = s.index[-1] - s.index[0]
    grid = pd.date_range(s.index[0], s.index[-1], freq=step)
    completeness = 100 * s.reindex(grid).notna().sum() / len(grid)
    seg = FP.segment_ids(s.index, step)
    n_seg = int(seg.max() + 1)
    assert n_seg == len(gaps) + 1, f'{name}: segments != gaps + 1'
    has = len(gl) > 0
    q = lambda p: float(gl.quantile(p)) if has else 0.0
    return dict(
        dataset=name, n=len(s),
        span_days=round(span.total_seconds() / 86400, 1),
        completeness=round(completeness, 2),
        mean=round(float(s.mean()), 2), sd=round(float(s.std()), 2),
        vmin=round(float(s.min()), 2), vmax=round(float(s.max()), 2),
        p05=round(float(s.quantile(0.05)), 2),
        p95=round(float(s.quantile(0.95)), 2),
        segments=n_seg, n_gaps=int(len(gaps)),
        pct_after_gap=round(100 * len(gaps) / len(s), 2),
        # durations of the missing stretch (missing samples x step)
        gap_median=fmt_td(q(0.5) * step) if has else '-',
        gap_p95=fmt_td(q(0.95) * step) if has else '-',
        gap_max=fmt_td(int(gl.max()) * step) if has else '-',
        gap_median_samples=q(0.5),
        gap_p95_samples=round(q(0.95), 1),
        gap_max_samples=int(gl.max()) if has else 0,
        missing_samples=int(gl.sum()) if has else 0)


def week_window(s, step):
    """Seven calendar days starting at midnight of the day of the record's
    middle sample, reindexed to the regular grid so gaps show as breaks.
    Records shorter than seven days are shown whole."""
    t0, t1 = s.index[0], s.index[-1]
    if t1 - t0 < pd.Timedelta('7D'):
        a, b = t0, t1
    else:
        a = s.index[len(s) // 2].floor('1D')
        a = min(max(a, t0.ceil('1D')), (t1 - pd.Timedelta('7D')).floor('1D'))
        b = a + pd.Timedelta('7D') - step
    grid = pd.date_range(a, b, freq=step)
    return s.reindex(grid)


def main():
    ds = load_all()
    rows = [stats(n, s, st) for n, s, st in ds]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'data_section_statistics.csv'), index=False)

    print('=' * 78)
    print('  VALUES FOR TABLE I OF THE PAPER')
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

    print('\n  Cross-check against the values in Table I:')
    for label, key in [('Samples', 'n'), ('Calendar span (d)', 'span_days'),
                       ('Completeness (%)', 'completeness'),
                       ('Contiguous segments', 'segments'),
                       ('Number of gaps', 'n_gaps'),
                       ('Rows following a gap (%)', 'pct_after_gap'),
                       ('Median gap (missing samples)', 'gap_median_samples'),
                       ('Median gap', 'gap_median'),
                       ('95th pct gap (samples)', 'gap_p95_samples'),
                       ('95th pct gap', 'gap_p95'),
                       ('Maximum gap (samples)', 'gap_max_samples'),
                       ('Maximum gap', 'gap_max')]:
        vals = [f"{getattr(r, key)}" for r in df.itertuples()]
        print(f"  {label:<30}" + ''.join(f"{v:>16}" for v in vals))
    print('  (gap length = samples missing between two valid samples)')
    print('\n  Saved data_section_statistics.csv')

    fig, axes = plt.subplots(len(ds), 1, figsize=(11, 2.35 * len(ds)),
                             squeeze=False)
    for ax, (name, s, st) in zip(axes[:, 0], ds):
        w = week_window(s, st)
        ax.plot(w.index, w.values, lw=0.7, color=NAVY)
        isna = w.isna().to_numpy()
        starts = w.index[isna & ~np.r_[False, isna[:-1]]]
        valid = ~isna
        n_seg = int((valid & ~np.r_[False, valid[:-1]]).sum())
        for t in starts:
            ax.axvline(t, color=GAPC, lw=0.9, alpha=0.75)
        ng = len(starts)
        days = (w.index[-1] - w.index[0] + st) / pd.Timedelta('1D')
        ax.set_title(f"{name} — {w.index[0]:%Y-%m-%d} + {days:.0f} d: "
                     f"{ng} gap{'' if ng == 1 else 's'}, "
                     f"{n_seg} segment{'' if n_seg == 1 else 's'}",
                     fontsize=9.5, fontweight='bold')
        print(f"  week panel {name}: {w.index[0]} -> {w.index[-1]}, "
              f"{ng} gaps, {n_seg} segments")
        ax.set_ylabel('m/s', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle('Seven days from each dataset (D1: whole 6-day record). '
                 'Vertical lines mark gap starts.',
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, 'fig_data_overview.png'), dpi=190,
                bbox_inches='tight')
    print('  Saved fig_data_overview.png  — this is Fig. S1 '
          '(supplementary material)')


if __name__ == '__main__':
    main()
