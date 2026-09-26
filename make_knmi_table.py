import os, re, sys, glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import knmi_adapter as KA
from dataset_adapters import load_wfip3_buoy

PLATFORMS = ['BSB', 'HKN', 'HKWA', 'HKWB', 'HKZA', 'HKZB']
INJECTION = ['HKZA', 'HKWB', 'HKWA', 'HKZB']     # as in run_gap_injection.py
STEP = pd.Timedelta('10min')


def runs(ok):
    """Lengths of consecutive True runs."""
    ok = np.asarray(ok, bool)
    if not ok.any():
        return np.array([0])
    d = np.diff(np.r_[0, ok.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def file_info(folder):
    """Rows present per timestamp, sentinel count and lidar unit per file."""
    stamps, n_sent, units = [], 0, []
    for f in sorted(set(glob.glob(os.path.join(folder, '*.[Cc][Ss][Vv]')))):
        with open(f, encoding='latin-1') as fh:
            m = re.search(r'Unit:?\s*(\d+)', fh.readline())
        day = re.search(r'_10min_(\d{8})', os.path.basename(f)).group(1)
        units.append((day, m.group(1) if m else '?'))
        _, s, _ = KA.load_knmi_day(f)
        n_sent += s
        t = pd.read_csv(f, skiprows=1, usecols=['Time and Date'])
        stamps.append(pd.to_datetime(t['Time and Date'],
                                     format='%d/%m/%Y %H:%M:%S',
                                     errors='coerce'))
    stamps = pd.DatetimeIndex(pd.concat(stamps).dropna()).round('10min')
    return stamps.unique(), n_sent, units


def unit_changes(units):
    out, prev = [], None
    for day, u in units:
        if u != prev:
            out.append(f"{u} from {day[:4]}-{day[4:6]}-{day[6:]}")
            prev = u
    return '; '.join(out)


def row(name, kind, v, **extra):
    ok = v.notna().to_numpy()
    r = runs(ok)
    n = len(v)
    return dict(record=name, kind=kind,
                start=v.index[0], end=v.index[-1],
                span_days=round(n * STEP / pd.Timedelta('1D'), 1),
                intervals=n, samples=int(ok.sum()),
                missing=int(n - ok.sum()),
                missing_pct=round(100 * (n - ok.sum()) / n, 2),
                segments=int((r > 0).sum()),
                longest_run_days=round(r.max() * STEP / pd.Timedelta('1D'), 1),
                **extra)


def main():
    rows = []
    for p in PLATFORMS:
        folder = os.path.join(HERE, f'knmi_{p.lower()}')
        df = KA.load_knmi_platform(folder, verbose=False)
        v = df['wind_speed']
        present, n_sent, units = file_info(folder)
        has_row = v.index.isin(present)
        bad = has_row & v.isna().to_numpy()          # row there, value bad
        apparent = runs(has_row).max() * STEP / pd.Timedelta('1D')
        rows.append(row(p, 'platform', v,
                        n_sentinel=n_sent, n_bad=int(bad.sum()),
                        n_absent=int((~has_row).sum()),
                        apparent_longest_run_days=round(apparent, 1),
                        units=unit_changes(units)))
        if p in INJECTION:
            b = KA.longest_clean_block(df)['wind_speed']
            rows.append(row(f'{p} base', 'injection base', b))

    bp = os.path.join(HERE, 'buoy_data')
    for interp in (False, True):
        s, _ = load_wfip3_buoy(bp, height_m=38, resample='10min',
                               interpolate=interp)
        g = pd.date_range(s.index[0], s.index[-1], freq=STEP)
        full = s.reindex(g)
        # longest contiguous run, as buoy_longest_contiguous() in
        # run_gap_injection.py
        ok = full.notna().to_numpy()
        d = np.diff(np.r_[0, ok.astype(int), 0])
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        k = int(np.argmax(en - st))
        base = full.iloc[st[k]:en[k]]
        tag = 'interpolated' if interp else 'raw'
        rows.append(row(f'D2 base ({tag})', 'injection base', base))

    out = pd.DataFrame(rows)
    for c in ('n_sentinel', 'n_bad', 'n_absent'):
        out[c] = out[c].astype('Int64')
    out.to_csv(os.path.join(HERE, 'knmi_record_statistics.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(out.drop(columns=['units']).to_string(index=False))
    plat = out[out.kind == 'platform']
    print(f"\n  missing {plat.missing_pct.min():.2f}%-{plat.missing_pct.max():.2f}%,"
          f" segments {plat.segments.min()}-{plat.segments.max()}")
    for r in plat.itertuples():
        print(f"  {r.record}: {r.n_bad} bad intervals; longest run "
              f"{r.apparent_longest_run_days} d apparent -> "
              f"{r.longest_run_days} d clean; units {r.units}")
    print('\n  Saved knmi_record_statistics.csv (supplementary Table S1)')


if __name__ == '__main__':
    main()
