# Evaluation Bias From Data Gaps in Short-Term Wind Forecasting for Grid Operations

Code and pipeline for the paper "Evaluation Bias From Data Gaps in Short-Term Wind Forecasting
for Grid Operations" (M. S. Hasni, A. Khalid, H. Ayoob, M. Khalid).

This repository reproduces every table, figure, and reported number in the manuscript and its
supplementary material, starting from the raw public data.


## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Set your KNMI API key (see `data/README.md` for how to obtain one):

```bash
# Windows PowerShell
$env:KNMI_API_KEY = "your-key-here"

# permanent (restart terminal after)
[System.Environment]::SetEnvironmentVariable("KNMI_API_KEY", "your-key-here", "User")
```

## Getting the data

See `data/README.md` for exact sources, DOIs, and where each file goes. In short:

1. D1 (ZephIR) and the six additional KNMI platform records: run `knmi_auto.py` (needs
   `KNMI_API_KEY`; takes roughly 1 hour per platform including rate-limit pauses).
2. D2 (WFIP3 buoy lidar): download from the DOE Wind Data Hub, place in `buoy_data/`.
3. D3 (onshore turbine): download from Zenodo, place in `onshore/`.

## Running the pipeline

Run in this order. Steps 2 through 8 take an afternoon; step 9 (the injection experiment)
takes 4 to 6 hours and should be left running unattended.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 1 | `knmi_auto.py` | raw KNMI downloads into `knmi_*/` folders | ~1 hr/platform |
| 2 | `make_data_section.py` | Table I, Fig. S1 | minutes |
| 3 | `run_artifact_aligned.py` | **Table III** (main result), block-bootstrap sensitivity | minutes |
| 4 | `run_artifact_decompose.py` | **Table IV** (channel decomposition), Fig. S3 | minutes |
| 5 | `run_artifact_audit.py` | Section III-D (calendar-alignment audit) | minutes |
| 6 | `run_model_generalisation_aligned.py` | **Table V**, Table S3 (LSTM on D2/D3), Fig. S4 | ~20 min (LSTM is slow) |
| 7 | `run_rolling_origin_aligned.py` | **Table VI**, Fig. S5 | ~15 min |
| 8 | `run_gap_profile.py` | **Table VII**, Fig. 2, Fig. S6 | ~30 min |
| 9 | `run_gap_injection.py` | **Table VIII**, Fig. 3 | 4–6 hours |
| — | `make_figs_2_3.py` | Fig. 1, Fig. S2 (re-render from CSVs) | seconds |

`final_pipeline.py`, `dataset_adapters.py`, and `knmi_adapter.py` are shared modules imported
by the scripts above; they are not run directly.

**Sanity check after step 3:** confirm D1 (the contiguous, gap-free record) returns
`bias = +0.00 [0.00, 0.00]` at every horizon. This is the paper's null calibration — if D1 is
not exactly zero, something upstream is broken and no other result should be trusted.

## Repository layout

```
├── final_pipeline.py              # feature engineering, segmentation, causal self-check
├── dataset_adapters.py            # D1/D2 loaders
├── knmi_adapter.py                # KNMI record loader
├── knmi_auto.py                   # KNMI bulk downloader (needs KNMI_API_KEY)
├── make_data_section.py           # Table I
├── run_artifact_aligned.py        # Table III — corrected block bootstrap lives here
├── run_artifact_decompose.py      # Table IV
├── run_artifact_audit.py          # Section III-D
├── run_model_generalisation_aligned.py   # Table V, Table S3
├── run_rolling_origin_aligned.py  # Table VI
├── run_gap_profile.py             # Table VII, Fig. 2
├── run_gap_injection.py           # Table VIII, Fig. 3 (long-running)
├── make_figs_2_3.py               # regenerates Fig. 1 and Fig. S2
├── *.csv                          # result files, one per script above
├── *_log.txt                      # console output of each run
├── *.png                          # figures as submitted
├── data/README.md                 # data sources and DOIs
├── requirements.txt
└── archive/                       # superseded scripts, not part of the reproduction path
```

## A note on the bootstrap

`run_artifact_aligned.py` contains the confidence-interval procedure discussed in Section III-E
of the paper. Blocks are formed **within** contiguous segments (never spanning a gap) and both
protocols are resampled from the **same** block draw, since they score overlapping rows and are
strongly correlated. An earlier version of this code (kept in `archive/` for the audit trail)
resampled on row index and treated the two arms as independent; both defects are described and
corrected in Section III-E. If you are adapting this code for a different fragmented time series,
start from the corrected version and read that section first.


## Contact

M. S. Hasni — saadhasni14@gmail.com
