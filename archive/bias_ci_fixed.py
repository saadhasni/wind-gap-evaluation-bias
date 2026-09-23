"""
=============================================================================
  bias_ci_fixed.py — corrected block bootstrap for the gap-handling bias
=============================================================================
  Drop-in replacement for bias_ci() in run_artifact_aligned.py
  (and for the identical function in add_statistics.py).

  WHAT WAS WRONG

  The original resampled each arm on its own circular block grid built from
  row positions:

      ih  = blocks drawn over positions 0..Th-1, wrapped
      idn = blocks drawn over positions 0..Tn-1, wrapped     <- independent

  Two separate defects.

  (1) BLOCKS SPAN GAPS. A block is `block` consecutive ROWS, and on a
      fragmented record consecutive rows are not consecutive in time. D3 has
      1,363 segments with a mean length of 28.8 samples against a block
      length of 50, so essentially every block straddles at least one
      outage. The bootstrap therefore assumes the series is contiguous —
      the assumption this paper exists to reject. It also wraps circularly
      from the end of the record to the beginning, joining December to
      January.

  (2) THE TWO ARMS ARE RESAMPLED INDEPENDENTLY. Both arms score the same
      calendar window and overlap heavily (on D3 the gap-aware rows are a
      subset of the naive rows), so their skills are strongly positively
      correlated. Drawing them independently destroys that correlation:

          Var(C - A) = VarC + VarA - 2Cov(C,A)      <- truth
          Var(C - A) = VarC + VarA                  <- what was computed

      With Cov > 0 the reported intervals are too WIDE. The published
      intervals are therefore conservative, not anti-conservative, and some
      results currently reported as "includes zero" may exclude it once
      this is corrected.

  WHAT THIS DOES INSTEAD

  One set of blocks, shared by both arms, and blocks that never cross a
  segment boundary:

    * every scored timestamp is assigned a segment id;
    * inside each segment, rows are cut into consecutive blocks of at most
      `block` rows, so no block spans an outage and none wraps the record;
    * each replicate draws block ids with replacement ONCE, then gathers
      whichever rows of each arm fall inside the drawn blocks, and
      recomputes both skills from those rows.

  Because the same draw feeds both arms, their covariance is preserved and
  the interval is on the difference rather than on two unrelated quantities.

  A useful side effect: on the contiguous control D1 the two arms score
  identical rows, so every replicate returns exactly 0.00 and the interval
  is [0.00, 0.00]. That is a strictly stronger control than the current
  [-6.61, +6.33] — the null is exact in the inference as well as in the
  point estimate.

  HOW TO USE IT

  In run_artifact_aligned.py, replace the whole bias_ci function with this
  one, and change the call inside run() from

      lo, hi = bias_ci(eh_m, eh_p, en_m, en_p)

  to

      seg_s = pd.Series(seg, index=series.index)
      lo, hi = bias_ci(eh_m, eh_p, series.index[te],
                       en_m, en_p, te2.index, seg_s)

  `seg` is already computed at the top of run(). Nothing else changes.

  Runtime is comparable to the original: a few seconds per dataset/horizon.
=============================================================================
"""
import numpy as np
import pandas as pd


def skill_from_err(err_model, err_pers):
    rm = np.sqrt(np.mean(np.asarray(err_model) ** 2))
    rp = np.sqrt(np.mean(np.asarray(err_pers) ** 2))
    return 100.0 * (1.0 - rm / rp)


def _segment_blocks(idx, seg_series, block):
    """Block id per row of `idx`. Blocks are consecutive runs of at most
    `block` rows and never cross a segment boundary."""
    seg = seg_series.reindex(idx).to_numpy()
    if np.isnan(seg.astype(float)).any():
        raise ValueError("some scored timestamps are absent from the "
                         "segment index; pass the full series index")
    b = np.empty(len(idx), dtype=np.int64)
    bid, count, prev = -1, block, None
    for i, s in enumerate(seg):
        if s != prev or count >= block:
            bid += 1
            count = 0
            prev = s
        b[i] = bid
        count += 1
    return b


def bias_ci(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg_series,
            n_boot=2000, block=50, seed=42, min_frac=0.5):
    """Block-bootstrap 95% CI for (naive skill - gap-aware skill).

    eh_m, eh_p, idx_h : gap-aware model errors, persistence errors, timestamps
    en_m, en_p, idx_n : naive model errors, persistence errors, timestamps
    seg_series        : pd.Series of segment id indexed by the full series index
    min_frac          : a replicate is discarded if either arm retains fewer
                        than this fraction of its rows, which guards the skill
                        denominator against a degenerate draw
    """
    eh_m, eh_p = np.asarray(eh_m, float), np.asarray(eh_p, float)
    en_m, en_p = np.asarray(en_m, float), np.asarray(en_p, float)
    idx_h, idx_n = pd.DatetimeIndex(idx_h), pd.DatetimeIndex(idx_n)

    # master grid: the union of both arms' scored timestamps, in time order
    master = idx_h.union(idx_n).sort_values()
    blk = _segment_blocks(master, seg_series, block)
    bmap = pd.Series(blk, index=master)

    bh = bmap.reindex(idx_h).to_numpy()
    bn = bmap.reindex(idx_n).to_numpy()
    n_blk = int(blk.max()) + 1

    # row positions belonging to each block, for each arm
    pos_h = [np.where(bh == k)[0] for k in range(n_blk)]
    pos_n = [np.where(bn == k)[0] for k in range(n_blk)]

    rng = np.random.default_rng(seed)
    out, tries, need_h, need_n = [], 0, min_frac * len(eh_m), min_frac * len(en_m)

    while len(out) < n_boot and tries < 5 * n_boot:
        tries += 1
        draw = rng.integers(0, n_blk, n_blk)
        ph = np.concatenate([pos_h[k] for k in draw if pos_h[k].size])
        pn = np.concatenate([pos_n[k] for k in draw if pos_n[k].size])
        if ph.size < need_h or pn.size < need_n:
            continue
        out.append(skill_from_err(en_m[pn], en_p[pn])
                   - skill_from_err(eh_m[ph], eh_p[ph]))

    if len(out) < n_boot // 2:
        raise RuntimeError(f"only {len(out)} usable replicates; "
                           f"lower `block` or `min_frac`")
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------------------
# Optional: block-length sensitivity, for Appendix B
# ---------------------------------------------------------------------------
def bias_ci_sensitivity(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg_series,
                        blocks=(25, 50, 100), **kw):
    """Return {block_length: (lo, hi)}. Report this rather than a single
    unmotivated block length."""
    return {b: bias_ci(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg_series,
                       block=b, **kw) for b in blocks}


# ---------------------------------------------------------------------------
# Self-test: synthetic fragmented record, no real data needed
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # a 10-min record broken into many short segments, like D3
    n = 6000
    t = pd.date_range('2020-01-01', periods=n, freq='10min')
    keep = np.ones(n, bool)
    for start in rng.choice(n - 10, 150, replace=False):
        keep[start:start + 4] = False
    t = t[keep]
    seg = pd.Series(np.cumsum(
        (t.to_series().diff() != pd.Timedelta('10min')).to_numpy()), index=t)
    seg.iloc[0] = 0

    # gap-aware arm scores a subset; naive arm scores everything
    idx_n = t[500:]
    idx_h = idx_n[rng.random(len(idx_n)) < 0.45]

    en_p = rng.normal(0, 1.0, len(idx_n))
    en_m = 0.97 * en_p + rng.normal(0, 0.25, len(idx_n))     # correlated arms
    hpos = idx_n.get_indexer(idx_h)
    eh_p, eh_m = en_p[hpos], en_m[hpos] * 0.99

    lo, hi = bias_ci(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg, n_boot=500)
    print(f"fragmented record   bias CI [{lo:+.2f}, {hi:+.2f}]")

    print("block sensitivity:", {b: (round(l, 2), round(h, 2)) for b, (l, h)
          in bias_ci_sensitivity(eh_m, eh_p, idx_h, en_m, en_p, idx_n, seg,
                                 n_boot=300).items()})

    # contiguous control: identical rows in both arms -> exactly zero
    t2 = pd.date_range('2020-01-01', periods=3000, freq='1min')
    seg2 = pd.Series(np.zeros(3000, dtype=int), index=t2)
    e_p = rng.normal(0, 1.0, 3000)
    e_m = 0.98 * e_p + rng.normal(0, 0.2, 3000)
    lo, hi = bias_ci(e_m, e_p, t2, e_m, e_p, t2, seg2, n_boot=500)
    print(f"contiguous control  bias CI [{lo:+.2f}, {hi:+.2f}]   "
          f"(must be exactly [0.00, 0.00])")
