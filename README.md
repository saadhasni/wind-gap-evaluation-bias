# Evaluation Bias From Data Gaps in Short-Term Wind Forecasting for Grid Operations

Code and pipeline for the paper "Evaluation Bias From Data Gaps in Short-Term Wind Forecasting
for Grid Operations" (M. S. Hasni, A. Khalid, H. Ayoob, A. A. Arslan, M. Khalid).

This repository reproduces every table, figure, and reported number in the manuscript, plus the
released output it refers to, starting from the raw public data.

## Setup

Requires **Python 3.12** (the pinned xgboost 3.3 needs >= 3.12, TensorFlow 2.17 has no 3.13
wheels). The paper's numbers were produced on Windows; other platforms and library versions
can change individual gradient-boosting values by up to about 2.6 percentage points, while the
structural findings are unchanged (see Section IV-G of the paper).

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

1. D1 (ZephIR, Borssele Alpha, 21–26 Nov 2019): download the six 1-second-product files
   `ZephIR_windlidar_BSA_1s_*.CSV` manually from the KNMI Data Platform and place them in
   the repository root. `knmi_auto.py` does **not** fetch D1.
2. The six additional KNMI platform records (10-min product): run `knmi_auto.py` (needs
   `KNMI_API_KEY`; takes roughly 1 hour per platform including rate-limit pauses).
3. D2 (WFIP3 buoy lidar): download from the DOE Wind Data Hub, place in `buoy_data/`.
4. D3 (onshore turbine): download from Zenodo, place in `onshore/`.

## Running the pipeline

Run in this order. Steps 2 through 9 take an afternoon; step 10 (the injection experiment)
takes about 2 hours on the author's Windows machine and should be left running unattended.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 1 | `knmi_auto.py` | raw KNMI downloads into `knmi_*/` folders | ~1 hr/platform |
| 2 | `make_data_section.py` | Table I, seven-day excerpt (`fig_data_overview.png`) | minutes |
| 3 | `make_knmi_table.py` | KNMI record statistics (`knmi_record_statistics.csv`) | minutes |
| 4 | `run_artifact_aligned.py` | **Table III** (main result), block-bootstrap sensitivity | minutes |
| 5 | `run_artifact_decompose.py` | **Table IV** (channel decomposition), decomposition figure (`fig_s3_decomposition.png`) | minutes |
| 6 | `run_artifact_audit.py` | Section III-D (calendar-alignment audit) | minutes |
| 7 | `run_model_generalisation_aligned.py` | **Table V**, LSTM results on D2/D3 (`model_generalisation_aligned.csv`) | ~20 min (LSTM is slow) |
| 8 | `run_rolling_origin_aligned.py` | **Table VI** | ~15 min |
| 9 | `run_gap_profile.py` | **Table VII**, Fig. 2 (`gap_profile_fig_paper.png`), pre-gap variability test (`gap_context_fig.png`) | ~30 min |
| — | `run_interpolation_sensitivity.py` | interpolation-policy sensitivity (Section II-A) | minutes |
| 10 | `run_gap_injection.py` | injection results, Fig. 3 (`gap_injection_v3_fig.png`) | ~2 hours |
| 11 | `make_injection_summary.py` | **Table VIII** (from the injection CSV) | seconds |
| 12 | `make_figs_2_3.py` | Fig. 1 (`fig1_configurations.png`) | seconds |
| 13 | `make_paper_numbers.py` | `paper_numbers.csv`: every number quoted in the text, collected from the CSVs | seconds |

The full feature specification is defined in `final_pipeline.py`. Some scripts also write
diagnostic figures that do not appear in the paper (for example `fig_s2_skill.png`,
`model_generalisation_aligned_fig.png`, `rolling_origin_aligned_fig.png`, `gap_profile_fig.png`
and `gap_injection_v3_fig_xgb.png`).

`final_pipeline.py`, `dataset_adapters.py`, and `knmi_adapter.py` are shared modules imported
by the scripts above; they are not run directly.

**Sanity check after step 4:** confirm D1 (the contiguous, gap-free record) returns
`bias = +0.00 [0.00, 0.00]` at every horizon. On a record with no gaps both protocols score
identical rows with identical features, so this is an identity check of the code path (not a
statistical calibration): if D1 is not exactly zero, something upstream is broken.

## Repository layout

```
├── final_pipeline.py              # feature engineering, segmentation, causal self-check
├── dataset_adapters.py            # D1/D2 loaders
├── knmi_adapter.py                # KNMI record loader
├── knmi_auto.py                   # KNMI bulk downloader (needs KNMI_API_KEY)
├── make_data_section.py           # Table I, seven-day excerpt
├── make_knmi_table.py             # KNMI record statistics
├── run_artifact_aligned.py        # Table III — corrected block bootstrap lives here
├── run_artifact_decompose.py      # Table IV, decomposition figure
├── run_artifact_audit.py          # Section III-D
├── run_model_generalisation_aligned.py   # Table V, LSTM results
├── run_rolling_origin_aligned.py  # Table VI
├── run_gap_profile.py             # Table VII, Fig. 2, pre-gap test
├── run_interpolation_sensitivity.py  # interpolation-policy sensitivity
├── run_gap_injection.py           # injection runs, Fig. 3 (long-running)
├── make_injection_summary.py      # Table VIII
├── make_figs_2_3.py               # Fig. 1
├── make_paper_numbers.py          # numbers quoted in the text
├── *.csv                          # result files, one per script above
├── paper_numbers.csv              # every number quoted in the text, from this run
├── injection_summary_report.txt   # full report behind Table VIII
├── *.png                          # figures as submitted
├── data/README.md                 # data sources, files and licences
├── LICENSE                        # MIT (code)
└── archive/                       # superseded scripts and logs, not part of the reproduction path
```

## A note on the bootstrap

`run_artifact_aligned.py` contains the confidence-interval procedure discussed in Section III-E
of the paper. Blocks are formed **within** contiguous segments (never spanning a gap) and both
protocols are resampled from the **same** block draw, since they score overlapping rows and are
strongly correlated. An earlier version of this code (kept in `archive/` for the audit trail)
resampled on row index and treated the two arms as independent; both defects are described and
corrected in Section III-E. If you are adapting this code for a different fragmented time series,
start from the corrected version and read that section first.

## License

Code: MIT (see `LICENSE`). Data: each source keeps its own licence; see `data/README.md`.

## Contact

M. S. Hasni — saadhasni14@gmail.com
