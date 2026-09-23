import os
import glob
import numpy as np
import pandas as pd

SPEED = "Horizontal Wind Speed (m/s) at {h}m"
PACKETS = "Packets in Average at {h}m"

HEIGHTS = [252, 227, 202, 183, 158, 133, 108, 83, 58, 38, 16]

# 38 m matches dataset D1 in the paper, and is the ZephIR's fixed reference
# measurement, so it is present throughout the campaign.
DEFAULT_HEIGHT = 38

# Anything at or above this is a fill value, not a measurement. The highest
# 10-minute mean wind speed ever recorded anywhere is well under 60 m/s.
SENTINEL_MIN = 100.0


PLATFORM_OFFSET_M = {
    "BSA": 45, "BSB": 45, "HKZA": 45, "HKZB": 45,
    "HKN": 43, "HKWA": 43, "HKWB": 43,
}


def load_knmi_day(path, height=DEFAULT_HEIGHT, min_packets=None):
    """Load one daily CSV. Returns a DataFrame; invalid values become NaN."""
    df = pd.read_csv(path, skiprows=1, low_memory=False)

    speed_col = SPEED.format(h=height)
    if speed_col not in df.columns:
        raise KeyError(f"{speed_col!r} not present. Heights here: {HEIGHTS}")

    ts = pd.to_datetime(df["Time and Date"], dayfirst=True, errors="coerce")
    v = pd.to_numeric(df[speed_col], errors="coerce")

    n_sentinel = int(((v >= SENTINEL_MIN) | (v < 0)).sum())
    v = v.mask((v >= SENTINEL_MIN) | (v < 0))

    n_packets = 0
    if min_packets is not None:
        pcol = PACKETS.format(h=height)
        if pcol in df.columns:
            pk = pd.to_numeric(df[pcol], errors="coerce")
            bad = (pk < min_packets) & v.notna()
            n_packets = int(bad.sum())
            v = v.mask(bad)

    out = pd.DataFrame({"wind_speed": v.values}, index=ts)
    out = out[out.index.notna()]
    return out, n_sentinel, n_packets


def gap_report(df, col="wind_speed", step_min=10):
    """Segment count, gap lengths and longest clean run."""
    ok = df[col].notna().values
    n = len(ok)

    # contiguous runs of valid data
    runs, cur = [], 0
    for v in ok:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    # gap lengths in intervals
    gaps, cur = [], 0
    for v in ok:
        if not v:
            cur += 1
        elif cur:
            gaps.append(cur)
            cur = 0
    if cur:
        gaps.append(cur)

    per_day = 1440 // step_min
    return {
        "intervals": n,
        "valid": int(ok.sum()),
        "missing": int(n - ok.sum()),
        "missing_pct": 100.0 * (n - ok.sum()) / n if n else 0.0,
        "segments": len(runs),
        "longest_run_days": (max(runs) / per_day) if runs else 0.0,
        "n_gaps": len(gaps),
        "median_gap_min": float(np.median(gaps) * step_min) if gaps else 0.0,
        "max_gap_hours": (max(gaps) * step_min / 60) if gaps else 0.0,
    }


def load_knmi_platform(folder, height=DEFAULT_HEIGHT, min_packets=None,
                       verbose=True):
    """Concatenate a folder of daily CSVs onto a strict 10-minute grid."""

    seen, files = set(), []
    for pattern in ("*.CSV", "*.csv"):
        for f in glob.glob(os.path.join(folder, pattern)):
            key = os.path.normcase(os.path.abspath(f))
            if key not in seen:
                seen.add(key)
                files.append(f)
    files.sort()
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")

    frames, sent, packs, failed = [], 0, 0, []
    for f in files:
        try:
            d, s, p = load_knmi_day(f, height, min_packets)
            frames.append(d)
            sent += s
            packs += p
        except Exception as e:
            failed.append((os.path.basename(f), str(e)))

    if not frames:
        raise RuntimeError(f"Every file in {folder} failed to parse")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    grid = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(grid)
    df.index.name = "timestamp"

    if verbose:
        r = gap_report(df)
        name = os.path.basename(os.path.normpath(folder))
        print(f"\n{name}  (height {height} m)")
        print(f"  files              {len(files)}")
        print(f"  span               {r['intervals']} intervals, "
              f"{r['intervals'] / 144:.1f} days")
        print(f"  fill values (9999) {sent} masked")
        if min_packets is not None:
            print(f"  low packet count   {packs} masked "
                  f"(< {min_packets} packets)")
        print(f"  missing total      {r['missing']} "
              f"({r['missing_pct']:.2f}%)")
        print(f"  segments           {r['segments']}")
        print(f"  gaps               {r['n_gaps']}, median "
              f"{r['median_gap_min']:.0f} min, max "
              f"{r['max_gap_hours']:.1f} h")
        print(f"  LONGEST CLEAN RUN  {r['longest_run_days']:.1f} days")
        if r["longest_run_days"] >= 30:
            print("  -> usable for gap injection")
        else:
            print("  -> too short for gap injection (want 30+ days)")
        if failed:
            print(f"  {len(failed)} file(s) failed to parse:")
            for nm, err in failed[:5]:
                print(f"    {nm}: {err[:80]}")
        # sanity check on the physics
        v = df["wind_speed"].dropna()
        if len(v):
            print(f"  wind speed         mean {v.mean():.2f}, "
                  f"max {v.max():.2f} m/s")
            if v.max() > 60:
                print("  WARNING: implausible maximum, check for other "
                      "fill values")

    return df


def longest_clean_block(df, col="wind_speed"):
    """Return the longest NaN-free slice as its own DataFrame."""
    ok = df[col].notna().values
    best_len = best_end = cur = 0
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best_len:
            best_len, best_end = cur, i + 1
    if best_len == 0:
        return df.iloc[0:0]
    return df.iloc[best_end - best_len:best_end]


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "knmi_hkwa"
    height = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HEIGHT
    minpk = int(sys.argv[3]) if len(sys.argv) > 3 else None

    df = load_knmi_platform(folder, height=height, min_packets=minpk)
    block = longest_clean_block(df)
    if len(block):
        print(f"\n  longest clean block: {block.index.min()} -> "
              f"{block.index.max()}  ({len(block)} intervals)")
