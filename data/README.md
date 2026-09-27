# Data sources

No raw data are stored in this repository, owing to size and licensing terms. This file lists
where each record comes from, which files to download, and where to place them. All paths are
relative to the repository root.

## D1: offshore lidar, Borssele Alpha (contiguous control)

- **Source:** KNMI Data Platform, "Wind — Lidar wind profiles measured at North Sea wind farm
  TenneT platforms", 1-second product:
  https://dataplatform.knmi.nl/dataset/windlidar-nz-wp-platform-1s-1
- **Files:** the six daily files `ZephIR_windlidar_BSA_1s_*.CSV` for 21–26 November 2019.
- **Place in:** the repository root.
- **Level used:** 38 m. Download manually; `knmi_auto.py` does not fetch D1.
- **Licence:** CC BY 4.0.

## Additional KNMI platform records (Section II-B and the injection experiment)

- **Source:** the same KNMI dataset, 10-minute product:
  https://dataplatform.knmi.nl/dataset/windlidar-nz-wp-platform-10min-1
- **Platforms:** BSB, HKN, HKWA, HKWB, HKZA, HKZB (Borssele Alpha is excluded because it is
  the source of D1).
- **How:** run `knmi_auto.py` with `KNMI_API_KEY` set. For each platform it downloads the
  longest unbroken run of daily files, capped at 150 days, into `knmi_bsb/`, `knmi_hkn/`,
  `knmi_hkwa/`, `knmi_hkwb/`, `knmi_hkza/` and `knmi_hkzb/`. The periods used in the paper
  are listed in Table S1 of the supplementary material and in `knmi_record_statistics.csv`.
- **Licence:** CC BY 4.0.

## D2: WFIP3 Buoy 130 lidar (few long gaps)

- **Source:** U.S. DOE Wind Data Hub, WFIP3 Buoy 130 lidar, 10-minute netCDF files.
- **Place in:** `buoy_data/` (all `.nc` files).
- **Level used:** 38 m; only samples with QC flag 0 are kept.
- **Licence:** public domain.

## D3: onshore turbine anemometer (many short gaps)

- **Source:** Y. Ding, "Wind time series dataset," Zenodo, 2021,
  https://doi.org/10.5281/zenodo.5516539
- **File:** the 10-minute CSV (its file name contains `10min`).
- **Place in:** `onshore/`.
- **Licence:** CC BY 4.0.

## Large intermediate files

`gap_profile_rows.csv` (62 MB, the per-row error table behind Table VII) is not included in
the repository or its Zenodo archive; `run_gap_profile.py` regenerates it. The summary tables
it feeds (`gap_profile_table7.csv`, `gap_profile_summary.csv`) are included.
