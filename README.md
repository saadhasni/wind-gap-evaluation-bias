# Evaluation Bias From Data Gaps in Short-Term Wind Speed Forecasting

Code and pipeline for the paper "Evaluation Bias From Data Gaps in Short-Term Wind Speed Forecasting" (M. S. Hasni, A. Khalid, H. Ayoob, A. A. Arslan, M. Khalid).

This repository reproduces every table, figure, and reported number in the manuscript, plus the
released output it refers to, starting from the raw public data.

## Setup

Requires **Python 3.12** (the pinned xgboost 3.3 needs >= 3.12, TensorFlow 2.17 has no 3.13
wheels). The paper's numbers were produced on Windows; other platforms and library versions
can change individual gradient-boosting values by up to about 2.6 percentage points, while the
structural findings are unchanged (see Sect. 4.11 of the paper).

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Set your KNMI API key:

```bash
# Windows PowerShell
$env:KNMI_API_KEY = "your-key-here"

# permanent (restart terminal after)
[System.Environment]::SetEnvironmentVariable("KNMI_API_KEY", "your-key-here", "User")
```

## Getting the data

See `data/README.md` for exact files, dates, folders and licences.

1. D1 (ZephIR, Borssele Alpha, 21-26 Nov 2019): download the six 1-second-product files
   `ZephIR_windlidar_BSA_1s_*.CSV` manually from the KNMI Data Platform and place them in
   the repository root. `knmi_auto.py` does **not** fetch D1.
2. The six additional KNMI platform records (10-min product): run `knmi_auto.py` (needs
   `KNMI_API_KEY`; takes roughly 1 hour per platform including rate-limit pauses).
3. D2 (WFIP3 buoy lidar): download from the DOE Wind Data Hub, place in `buoy_data/`.
4. D3 (onshore turbine): download from Zenodo, place in `onshore/`.

## Running the pipeline

Run in this order. Steps 2 through 9 take an afternoon; step 10 (the injection experiment)
takes about 2 hours on the author's Windows machine, and step 16 (the window-depth sweep)
takes about three times as long as step 4. Both should be left running unattended.
Table and section numbers refer to the current manuscript.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 1 | `knmi_auto.py` | raw KNMI downloads into `knmi_*/` folders | ~1 hr/platform |
| 2 | `make_data_section.py` | Table 1, seven-day excerpt (`fig_data_overview.png`) | minutes |
| 3 | `make_knmi_table.py` | KNMI record statistics (`knmi_record_statistics.csv`) | minutes |
| 4 | `run_artifact_aligned.py` | **Table 3** (main result), block-bootstrap sensitivity | minutes |
| 5 | `run_artifact_decompose.py` | **Table 4** (channel decomposition), decomposition figure (`fig_s3_decomposition.png`) | minutes |
| 6 | `run_artifact_audit.py` | Sect. 3.4 (calendar-alignment audit) | minutes |
| 7 | `run_model_generalisation_aligned.py` | **Table 5**, LSTM results on D2/D3 (`model_generalisation_aligned.csv`) | ~20 min (LSTM is slow) |
| 8 | `run_rolling_origin_aligned.py` | **Table 6** | ~15 min |
| 9 | `run_gap_profile.py` | **Table 7**, data for Fig. 2 (`gap_profile_summary.csv`), pre-gap variability test (`gap_context_fig.png`) | ~30 min |
| - | `run_interpolation_sensitivity.py` | interpolation-policy sensitivity (Sect. 2.1) | minutes |
| 10 | `run_gap_injection.py` | injection results (`gap_injection_v3_results.csv`) | ~2 hours |
| 11 | `make_injection_summary.py` | **Table 8** (from the injection CSV) | seconds |
| 12 | `exposure_check.py` | checks Eq. (1) against the pipeline's valid-origin counts (`exposure_check.csv`), **Fig. 4** (`Figure_exposure.pdf`) | seconds |
| 13 | `mixture_check.py` | **Table 9** (`mixture_table.csv`, from steps 4 and 5), **Fig. 5** (`Figure_5.pdf`) | seconds |
| 14 | `injection_mixture_check.py` | exposure and error-inflation table for the injected runs (`injection_mixture_table.csv`, from step 10), **Fig. 7** (`Figure_7.pdf`) | seconds |
| 15 | `regen_fig1_v2.py`, `regen_fig2_v2.py`, `regen_fig3_v2.py` | **Figs. 1, 2 and 3** as vector PDFs (`Figure_1.pdf`, `Figure_2.pdf`, `Figure_3.pdf`); Fig. 2 from `gap_profile_summary.csv`, Fig. 3 from `gap_injection_v3_results.csv`, Fig. 1 needs no data | seconds |
| 16 | `run_window_depth.py` | **Table 10** (`window_depth_sensitivity.csv`), **Fig. 6** (`Figure_window_depth.pdf`) | hours |
| 17 | `run_clark_west.py` | Clark-West p-values for the A and C arms of Table 3 (`clark_west.csv`; refits the XGBoost models of step 4, Sect. 3.5) | minutes |
| 18 | `make_paper_numbers.py` | `paper_numbers.csv`: every number quoted in the text, collected from the CSVs (run last) | seconds |

`make_figs_2_3.py` is the earlier figure script (it produced the PNG versions of Fig. 1 and
the supplementary skill figure) and is kept for the audit trail.

The full feature specification is defined in `final_pipeline.py`. Some scripts also write
diagnostic figures that do not appear in the paper (for example `fig_s2_skill.png`,
`model_generalisation_aligned_fig.png`, `rolling_origin_aligned_fig.png`, `gap_profile_fig.png`
and `gap_injection_v3_fig_xgb.png`).

`final_pipeline.py`, `dataset_adapters.py`, and `knmi_adapter.py` are shared modules imported
by the scripts above; they are not run directly.

To run the window-depth sweep on both D2 and D3, set an environment variable first
(PowerShell: `$env:WD_DATASETS = "D2,D3"`); `WD_NBOOT` sets the number of bootstrap draws
(default 1000).

**Sanity check after step 4:** confirm D1 (the contiguous, gap-free record) returns
`bias = +0.00 [0.00, 0.00]` at every horizon. On a record with no gaps both protocols score
identical rows with identical features, so this is an identity check of the code path (not a
statistical calibration): if D1 is not exactly zero, something upstream is broken.

**Sanity check after step 12:** `exposure_check.py` must print
`Formula reproduces the pipeline counts exactly: True`. `run_window_depth.py` also prints a
check that its W = 60 run reproduces the Table 3 bias.

**A note on steps 13 and 14.** The "exact formula" column of `mixture_table.csv` and the
`bias_hat` column of `injection_mixture_table.csv` reproduce the measured bias by construction
(Eq. 4 is an accounting identity, see Sect. 3.7). They check the implementation; the
informative columns are the exposure `f` and the error ratios `rho_p` and `rho_m`.

## Repository layout

```
|-- final_pipeline.py              # feature engineering, segmentation, causal self-check
|-- dataset_adapters.py            # D1/D2 loaders
|-- knmi_adapter.py                # KNMI record loader
|-- knmi_auto.py                   # KNMI bulk downloader (needs KNMI_API_KEY)
|-- make_data_section.py           # Table 1, seven-day excerpt
|-- make_knmi_table.py             # KNMI record statistics
|-- run_artifact_aligned.py        # Table 3 (corrected block bootstrap lives here)
|-- run_artifact_decompose.py      # Table 4, decomposition figure
|-- run_artifact_audit.py          # Sect. 3.4
|-- run_model_generalisation_aligned.py   # Table 5, LSTM results
|-- run_rolling_origin_aligned.py  # Table 6
|-- run_gap_profile.py             # Table 7, data for Fig. 2, pre-gap test
|-- run_interpolation_sensitivity.py  # interpolation-policy sensitivity
|-- run_gap_injection.py           # injection runs (long-running)
|-- make_injection_summary.py      # Table 8
|-- exposure_check.py              # Eq. (1) check, Fig. 4
|-- mixture_check.py               # Table 9, Fig. 5
|-- injection_mixture_check.py     # injected-run exposure table, Fig. 7
|-- run_window_depth.py            # Table 10, Fig. 6 (long-running)
|-- run_clark_west.py              # Clark-West p-values (Sect. 3.5)
|-- regen_fig1_v2.py, regen_fig2_v2.py, regen_fig3_v2.py   # Figs. 1-3 (vector PDF)
|-- make_figs_2_3.py               # earlier figure script (audit trail)
|-- make_paper_numbers.py          # numbers quoted in the text
|-- *.csv                          # result files, one per script above
|-- paper_numbers.csv              # every number quoted in the text, from this run
|-- injection_summary_report.txt   # full report behind Table 8
|-- Figure_*.pdf                   # figures of the paper (Figs. 1-7, vector)
|-- *.png                          # earlier figure renderings
|-- data/README.md                 # data sources, files and licences
|-- LICENSE                        # MIT (code)
`-- archive/                       # superseded scripts and logs, not part of the reproduction path
```

## A note on the bootstrap

`run_artifact_aligned.py` contains the confidence-interval procedure discussed in Sect. 3.5
of the paper. Blocks are formed **within** contiguous segments (never spanning a gap) and both
protocols are resampled from the **same** block draw, since they score overlapping rows and are
strongly correlated. An earlier version of this code (kept in `archive/` for the audit trail)
resampled on row index and treated the two arms as independent; both defects were corrected
before the results reported here. If you are adapting this code for a different fragmented
time series, start from the corrected version and read that section first.

## License

Code: MIT (see `LICENSE`). Data: each source keeps its own licence; see `data/README.md`.

## Contact

M. S. Hasni: saadhasni14@gmail.com
