# Data sources

None of the raw data is stored in this repository (git is a poor fit for large binary files,
and every source below is already openly archived under its own persistent identifier). This
file tells you exactly where to get each record and where to put it.

---

## D1 — ZephIR 300 offshore lidar (contiguous control)

- **Source:** Royal Netherlands Meteorological Institute (KNMI) Data Platform
- **Dataset:** `windlidar-nz-wp-platform-1s-1` (Borssele Alpha platform)
- **License:** CC-BY-4.0
- **How to get it:** register for a free API key at
  https://developer.dataplatform.knmi.nl/, set it as the `KNMI_API_KEY` environment
  variable (see main README), then run `knmi_auto.py`.
- **Where it goes:** downloaded automatically into `knmi_bsa/` by the script above.

## D2 — WFIP3 Buoy 130 Doppler lidar (few long gaps)

- **Source:** U.S. Department of Energy Wind Data Hub, WFIP3 campaign, Buoy 130, Martha's
  Vineyard
- **License:** public domain
- **How to get it:** https://wfip3-wdh.pnnl.gov/ (or the current DOE Wind Data Hub URL —
  search "WFIP3 Buoy 130"). Download the lidar wind-speed files for the 2024 deployment.
- **Where it goes:** place the downloaded files in `buoy_data/`.

## D3 — Onshore turbine anemometer (many short gaps)

- **Source:** Y. Ding, "Wind Time Series Dataset," Zenodo, 2021
- **DOI:** 10.5281/zenodo.5516539
- **License:** CC-BY-4.0
- **How to get it:** https://doi.org/10.5281/zenodo.5516539
- **Where it goes:** place the downloaded file(s) in `onshore/`.

## Additional records (six KNMI platforms, for replication and injection)

- **Source:** KNMI Data Platform, `windlidar-nz-wp-platform-10min-1`
- **Platforms:** HKZA, HKZB, HKWA, HKWB, HKN, BSB (see Table S1 in the supplementary
  material for spans and roles)
- **License:** CC-BY-4.0
- **How to get it:** same API key as D1; `knmi_auto.py` downloads all platforms listed in
  its `PLATFORMS` variable automatically.
- **Where it goes:** downloaded automatically into `knmi_hkza/`, `knmi_hkzb/`, `knmi_hkwa/`,
  `knmi_hkwb/`, `knmi_hkn/`, `knmi_bsb/`.

---

## Fill values and masking

Raw KNMI files encode missing readings as `9999`; these are masked to `NaN` before any
processing (see `final_pipeline.py`). Height-specific instrument flags mean a given
measurement level can be missing while the instrument's overall status flag reports normal
operation — see Section II-B of the paper for the one record where this reduced an apparent
59.8-day clean run to 13.5 days.

## Access dates

Data for this paper was accessed [ADD DATE — e.g. "between August and September 2026"].
KNMI, DOE and Zenodo archives may add data after this date; re-downloading will not change
the manuscript's results but may extend the calendar range of D1's platform beyond what is
described in Table I.

## Storage note

`gap_profile_rows.csv` (the full per-row error output behind Table VII and Fig. 2, roughly
60 MB) is not included in this git repository because it exceeds GitHub's soft size limit.
It is included in the archived Zenodo release of this code (see main README for the DOI) as
an additional file.
