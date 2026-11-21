"""Unit tests for the gap-aware core. They use synthetic data only, so they run
without any of the raw datasets:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import final_pipeline as FP                      # noqa: E402
from dataset_adapters import fill_short_gaps     # noqa: E402

STEP = pd.Timedelta('10min')


def make_series(n=600, gaps=(), seed=0):
    """AR(1)-like series on a 10-min grid with the given (start, length) gaps
    removed (rows dropped, as in the real records)."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 8.0
    for i in range(1, n):
        x[i] = 8.0 + 0.9 * (x[i - 1] - 8.0) + rng.normal(0, 0.5)
    s = pd.Series(x, index=pd.date_range('2024-01-01', periods=n, freq='10min'))
    keep = np.ones(n, bool)
    for a, ln in gaps:
        keep[a:a + ln] = False
    return s[keep]


def test_segment_ids_split_at_gaps():
    s = make_series(gaps=[(100, 5), (300, 20)])
    seg = FP.segment_ids(s.index, STEP)
    assert seg.max() == 2
    assert (np.diff(seg) >= 0).all()


def test_features_do_not_cross_gaps():
    s = make_series(gaps=[(200, 30)])
    f = FP.build_features_segmented(s, STEP)
    first_after = np.where(np.diff(f['_seg'].values) != 0)[0][0] + 1
    # lag_1 is undefined on the first row of a segment, never taken from before
    assert np.isnan(f['lag_1'].iloc[first_after])
    assert np.isnan(f['lag_60'].iloc[first_after + 10])


def test_causality_self_check_passes():
    s = make_series(gaps=[(150, 10), (350, 10)])
    assert FP.causality_self_check(s, STEP) > 0


def test_valid_rows_never_cross_a_gap():
    s = make_series(gaps=[(250, 15)])
    f = FP.build_features_segmented(s, STEP)
    H, L = 3, 12
    ok = np.where(FP.valid_rows_for_horizon(s, f, H, L))[0]
    seg = f['_seg'].values
    assert len(ok) > 0
    assert (seg[ok] == seg[ok + H]).all()
    assert (seg[ok] == seg[ok - (L - 1)]).all()


def test_contiguous_record_naive_equals_gap_aware():
    """The D1 identity: with no gaps both protocols score identical rows with
    identical features, so the bias must be exactly zero."""
    s = make_series(n=1500)
    f = FP.build_features_segmented(s, STEP)
    fn = FP.naive_features(s)
    cols = [c for c in f.columns if c != '_seg']
    both = f[cols].dropna().index.intersection(fn.dropna().index)
    assert np.allclose(f.loc[both, cols].values,
                       fn.loc[both, cols].values, equal_nan=True)


def test_naive_frame_target_time_jumps_over_gap():
    s = make_series(gaps=[(300, 40)])
    dn = FP.naive_frame(s, 1)
    lag = (dn['_target_ts'] - dn.index)
    assert lag.max() > STEP          # the naive target can be far in the future
    assert lag.min() == STEP


def test_fill_short_gaps_fills_only_short_interior_runs():
    idx = pd.date_range('2024-01-01', periods=20, freq='10min')
    v = np.arange(20, dtype=float)
    v[3:5] = np.nan        # short: filled
    v[10:16] = np.nan      # long: untouched
    out, n = fill_short_gaps(pd.Series(v, index=idx), limit=3)
    assert n == 2
    assert out.iloc[3:5].notna().all()
    assert out.iloc[10:16].isna().all()


def test_diebold_mariano_detects_a_better_model():
    rng = np.random.default_rng(1)
    e_bad = rng.normal(0, 1.0, 2000)
    e_good = rng.normal(0, 0.5, 2000)
    stat, p = FP.diebold_mariano(e_good, e_bad, h=1)
    assert stat < 0 and p < 1e-6


def test_diebold_mariano_identical_errors_not_significant():
    e = np.random.default_rng(2).normal(size=500)
    _, p = FP.diebold_mariano(e, e.copy(), h=1)
    assert p > 0.5


def test_block_bootstrap_ci_brackets_rmse():
    e = np.random.default_rng(3).normal(0, 1.0, 1000)
    lo, hi = FP.block_bootstrap_rmse_ci(e, n_boot=200)
    assert lo < np.sqrt(np.mean(e ** 2)) < hi


def test_bias_ci_zero_when_arms_identical():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_artifact_aligned import bias_ci
    s = make_series(n=800, gaps=[(400, 20)])
    seg = pd.Series(FP.segment_ids(s.index, STEP), index=s.index)
    rng = np.random.default_rng(4)
    em, ep = rng.normal(0, 0.8, len(s)), rng.normal(0, 1.0, len(s))
    lo, hi = bias_ci(em, ep, s.index, em, ep, s.index, seg, n_boot=300)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(0.0, abs=1e-9)


def test_config_is_single_source():
    import config
    assert config.xgb_params()['random_state'] == config.SEED
    assert config.xgb_params(random_state=7)['random_state'] == 7
    assert config.XGB_PARAMS['random_state'] == config.SEED   # not mutated
