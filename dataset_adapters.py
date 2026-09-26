import glob
import os
import numpy as np
import pandas as pd

def fill_short_gaps(s, limit):
    """Linearly fill ONLY runs of at most `limit` consecutive NaNs that have a
    valid value on both sides. Longer gaps are left untouched (pandas'
    interpolate(limit=N) would also fill the first N samples of every longer
    gap, using a value from after the gap). Returns (series, n_filled)."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform('sum')
    inside = s.interpolate(limit_area='inside')
    mask = isna & (run_len <= limit) & inside.notna()
    out = s.copy()
    out[mask] = inside[mask]
    return out, int(mask.sum())


# Adapter 1 — ZephIR 300 (CSV, ~17-s profile scans -> resample to chosen grid)
def load_zephir(folder, height_m=38, resample='1min', interpolate=True):
    """ZephIR 300 CSV files. Heights: 14,38,63,72,94,119,145,156,179,199,249."""
    files = sorted(set(glob.glob(os.path.join(
        folder, 'ZephIR_windlidar_*.[Cc][Ss][Vv]'))))
    if not files:
        raise FileNotFoundError(f"No ZephIR CSVs in {folder}")

    col = f'Horizontal Wind Speed (m/s) at {height_m}m'
    frames = []
    for f in files:
        d = pd.read_csv(f, skiprows=1, usecols=['Time and Date', col],
                        na_values=['#N/A', '9999', ''])
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw['dt'] = pd.to_datetime(raw['Time and Date'],
                               format='%d/%m/%Y %H:%M:%S')
    s = pd.to_numeric(raw.set_index('dt')[col], errors='coerce')
    s = s.where(s.between(0, 45))                       # physical QC
    s = s.resample(resample).mean()
    n_filled = 0
    if interpolate:
        s, n_filled = fill_short_gaps(s, limit=5)
    s = s.dropna()
    s.name = 'wind_speed'
    meta = dict(name='ZephIR300', height_m=height_m, resolution=resample,
                site='North Sea', license='provided', n_filled=n_filled,
                notes='~17-s profile scans averaged to grid')
    return s, meta

# Adapter 2 — WFIP3 DOE Buoy 130 lidar (netCDF, native 10-minute)
def load_wfip3_buoy(folder, height_m=38, resample='10min', interpolate=True):
    """WFIP3 DOE Buoy 130 lidar netCDF (.nc). Heights:
    10,38,57,87,107,137,157,177,197,247,277. Native 10-min, QC flags 0/1."""
    import xarray as xr
    files = sorted(glob.glob(os.path.join(folder, '*.nc')))
    if not files:
        raise FileNotFoundError(f"No .nc files in {folder}")

    parts, heights = [], set()
    for f in files:
        with xr.open_dataset(f) as ds:
            h = ds['height'].values
            j = int(np.argmin(np.abs(h - height_m)))     # nearest height
            actual_h = float(h[j])
            if abs(actual_h - height_m) > 2:
                raise ValueError(f"{os.path.basename(f)}: nearest height "
                                 f"{actual_h} m, requested {height_m} m")
            heights.add(actual_h)
            hws = ds['horizontal_wind_speed'].isel(height=j).values
            qc = ds['qc_horizontal_wind_speed'].isel(height=j).values
            hws = np.where(qc == 0, hws, np.nan)         # keep only good QC
            parts.append(pd.Series(hws,
                                   index=pd.to_datetime(ds['time'].values)))

    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep='first')]            # de-dup overlaps
    s = s.where(s.between(0, 45))
    # regular grid at native resolution; short gaps (<=30 min) interpolated
    s = s.resample(resample).mean()
    n_filled = 0
    if interpolate:
        s, n_filled = fill_short_gaps(s, limit=3)
    s = s.dropna()
    s.name = 'wind_speed'
    actual_h = sorted(heights)[0] if len(heights) == 1 else sorted(heights)
    meta = dict(name='WFIP3_Buoy130', height_m=actual_h, resolution=resample,
                n_filled=n_filled,
                site='Martha\'s Vineyard (offshore)', license='public domain',
                notes='native 10-min, QC-filtered (qc==0 only)')
    return s, meta

# Registry — pipeline calls these by name
ADAPTERS = {
    'zephir': load_zephir,
    'wfip3_buoy': load_wfip3_buoy,
}


if __name__ == '__main__':
    # Self-test on the buoy files next to this script
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'buoy_data')
    s, meta = load_wfip3_buoy(test_dir, height_m=38)
    print("META:", meta)
    print("Samples:", len(s))
    print("Range :", f"{s.min():.2f}-{s.max():.2f} m/s")
    print("Mean  :", f"{s.mean():.2f} +- {s.std():.2f}")
    print("Index contiguous:",
          (s.index.to_series().diff().dropna()
           == pd.Timedelta(meta['resolution'])).all())
    print("First 3:\n", s.head(3))
