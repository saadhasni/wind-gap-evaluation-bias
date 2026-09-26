# Data sources

None of the raw data is stored in this repository (git is a poor fit for large binary files,
and every source below is already openly archived under its own persistent identifier). This
file tells you exactly where to get each record and where to put it. All paths are relative
to the repository root (the folder containing the scripts).

---

## D1 — ZephIR 300 offshore lidar, Borssele Alpha (contiguous control)

- **Source:** Royal Netherlands Meteorological Institute (KNMI) Data Platform
- **Dataset:** the ZephIR wind lidar **1-second** product for the TenneT North Sea platforms
  (`windlidar_nz_wp_platform_1s`, version 1), platform **Borssele Alpha (BSA)**. Despite the
  "1s" name, each row is one full 11-height profile scan, roughly every 17 s; the loader
  averages these onto a 1-minute grid.
- **License:** CC-BY-4.0
- **Files used (6 days, 21–26 November 2019):**
  ```
  ZephIR_windlidar_BSA_1s_201911210000_201911212350_v1.CSV
  ZephIR_windlidar_BSA_1s_201911220000_201911222350_v1.CSV
  ZephIR_windlidar_BSA_1s_201911230000_201911232350_v1.CSV
  ZephIR_windlidar_BSA_1s_201911240000_201911242350_v1.CSV
  ZephIR_windlidar_BSA_1s_201911250000_201911252350_v1.CSV
  ZephIR_windlidar_BSA_1s_201911260000_201911262350_v1.CSV
  ```
- **How to get it:** download these six files **manually** from the KNMI Data Platform
  (https://dataplatform.knmi.nl/). They are **not** downloaded by `knmi_auto.py`, which
  fetches only the 10-minute product for the six other platforms below.
- **Where it goes:** the repository root, next to the scripts (not a subfolder). The
  loader matches `ZephIR_windlidar_*.CSV` case-insensitively.

## D2 — WFIP3 buoy Doppler lidar, "Buoy 130" in the paper (few long gaps)

- **Source:** U.S. Department of Energy Wind Data Hub, WFIP3 campaign, buoy lidar
  (`buoy.lidar.z01.a0`, netCDF, 10-minute)
- **License:** public domain
- **Dates used:** 2024-01-19 to 2024-07-07. 152 daily files; 19 days in this range have no
  file on the archive (2024-03-10 to 03-13, 03-19, 03-20, 03-23, 03-26, 03-27, 04-01,
  04-06 to 04-08, 04-13, 04-14, 04-20 to 04-23).
- **How to get it:** https://wdh.energy.gov/ (search "WFIP3 buoy lidar z01"). Download the
  `a0` netCDF files for the dates above.
- **Where it goes:** `buoy_data/`, one file per day, named `buoy.lidar.z01.a0.YYYYMMDD.HHMMSS.nc`.
- **Duplicate file note:** a browser re-download may create a copy such as
  `buoy.lidar.z01.a0.20240403.000000 (1).nc`. The author's folder contains exactly this
  byte-identical duplicate; the loader removes duplicate timestamps, so it is harmless, but it
  can be deleted.

## D3 — Onshore turbine anemometer (many short gaps)

- **Source:** Y. Ding, "Wind Time Series Dataset," Zenodo, 2021
- **DOI:** 10.5281/zenodo.5516539
- **License:** CC-BY-4.0
- **File used:** `Wind Time Series Dataset(10min).csv`, 2014-10-07 01:20 to 2015-10-06 23:50
  (39,195 rows, 10-minute). The hourly file in the same record is not used.
- **How to get it:** https://doi.org/10.5281/zenodo.5516539
- **Where it goes:** `onshore/` (exactly one `*10min*.csv` file).

## Additional records (six KNMI platforms, for replication and injection)

- **Source:** KNMI Data Platform, `windlidar_nz_wp_platform_10min`, version 1
- **License:** CC-BY-4.0
- **How to get it:** register for a free API key at https://developer.dataplatform.knmi.nl/,
  set it as the `KNMI_API_KEY` environment variable (see main README), then run
  `knmi_auto.py`. For each platform it takes the longest unbroken run of daily files, capped
  at 150 days.
- **Where it goes:** downloaded automatically into `knmi_bsb/`, `knmi_hkn/`, `knmi_hkwa/`,
  `knmi_hkwb/`, `knmi_hkza/`, `knmi_hkzb/` next to the script.
- **Date ranges used (150 daily files each):**

  | Platform | First day | Last day | Lidar unit |
  |---|---|---|---|
  | BSB  | 2020-06-13 | 2020-11-09 | 733 |
  | HKN  | 2023-08-09 | 2024-01-05 | 1150 |
  | HKWA | 2023-11-12 | 2024-04-09 | 1288 |
  | HKWB | 2025-07-21 | 2025-12-17 | 2266 |
  | HKZA | 2024-07-17 | 2024-12-13 | 934 |
  | HKZB | 2024-12-12 | 2025-05-10 | 973, then 1150 from 2025-03-18 |

  The HKZB record contains a lidar unit change on 2025-03-18 (that day's file has only 11
  rows). Per-record spans, missing fractions and segment counts (supplementary Table S1) are
  produced by `make_knmi_table.py`.

---

## Fill values and masking

Raw KNMI files encode missing readings as `9999` (or `9999.000`); `knmi_adapter.py` masks any
value >= 100 m/s or < 0 to `NaN` before processing. Height-specific instrument flags mean a given
measurement level can be missing while the instrument's overall status flag reports normal
operation. `make_knmi_table.py` reports, per record, the number of such bad intervals and the
longest clean run with and without them (Section II-B of the paper).

## Access dates

Data for this paper was accessed [ADD DATE — e.g. "between August and September 2026"].
KNMI, DOE and Zenodo archives may add data after this date; the date ranges above are the ones
used in the paper.

## Storage note

`gap_profile_rows.csv` (the full per-row error output behind Table VII and Fig. 2, roughly
60 MB) is not included in this git repository because it exceeds GitHub's soft size limit.
It is included in the archived Zenodo release of this code (see main README for the DOI) as
an additional file.
