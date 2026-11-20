import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
import run_artifact_aligned as RA
import run_artifact_decompose as RD
import run_window_depth as WD

STEP, SEQ, HORIZONS = pd.Timedelta('10min'), 36, (1, 3, 6)
NAME = 'D3 Onshore (many short gaps)'
OUT_LEAD = os.path.join(HERE, 'composition_by_lead_time.csv')
OUT_WD = os.path.join(HERE, 'window_depth_channels.csv')


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def lead_time_split(series):
    """Split the composition channel C - D into displaced rows whose target
    crosses a gap (lead time longer than H steps) and displaced rows where
    only the input window crosses a gap (lead time exactly H steps)."""
    fgap = FP.build_features_segmented(series, STEP)
    fnav = FP.naive_features(series)
    rows = []
    for H in HORIZONS:
        _, tr, te = RA.gap_aware_split(series, fgap, H, SEQ)
        tr2, teC, fc = RA.naive_aligned_split(series, H, tr, te, fnav)
        in_a = teC.index.isin(series.index[te])
        lead_ok = ((teC['_target_ts'] - teC.index.to_series()).to_numpy()
                   == H * STEP)
        tgt = ~in_a & ~lead_ok
        inp = ~in_a & lead_ok
        y, pers = teC['target'].to_numpy(), teC['lag_0'].to_numpy()
        for mname in RD.MODELS:
            pc, = RD.fit_predict(mname, tr2[fc].values, tr2['target'].values,
                                 teC[fc].values)
            sk_d = RD.skill(y[in_a], pc[in_a], pers[in_a])
            sk_e = RD.skill(y[in_a | inp], pc[in_a | inp], pers[in_a | inp])
            sk_c = RD.skill(y, pc, pers)
            ep, em = y - pers, y - pc
            ap, am = rmse(ep[in_a]), rmse(em[in_a])
            r = dict(dataset=NAME, horizon=H, model=mname,
                     n_valid=int(in_a.sum()), n_input_only=int(inp.sum()),
                     n_target_crossing=int(tgt.sum()),
                     rho_p_input_only=rmse(ep[inp]) / ap,
                     rho_m_input_only=rmse(em[inp]) / am,
                     rho_p_target_crossing=rmse(ep[tgt]) / ap,
                     rho_m_target_crossing=rmse(em[tgt]) / am,
                     skill_D=sk_d, skill_E=sk_e, skill_C=sk_c,
                     comp_total=sk_c - sk_d, comp_input_only=sk_e - sk_d,
                     comp_target_crossing=sk_c - sk_e)
            print(f"  H={H} {mname:<7} displaced: input-only {r['n_input_only']}, "
                  f"target-crossing {r['n_target_crossing']} | composition "
                  f"{r['comp_total']:+.2f} = {r['comp_input_only']:+.2f} "
                  f"(input-only) {r['comp_target_crossing']:+.2f} "
                  f"(target-crossing)")
            rows.append(r)
    return pd.DataFrame(rows).round(4)


def window_depth_channels(series):
    """Table 4 decomposition repeated at the window depths of Table 10."""
    rows = []
    for W in WD.WINDOWS:
        FP.build_features_segmented = (
            lambda s, step, W=W: WD.build_segmented(s, step, W))
        FP.naive_features = lambda s, W=W: WD.build_naive(s, W)
        RD.gap_aware_split = WD.make_split(W)
        out = []
        print(f"\n##### W = {W}")
        RD.run(NAME, series, STEP, SEQ, HORIZONS, out)
        for r in out:
            r['W'] = W
        rows += out
    return pd.DataFrame(rows)


if __name__ == '__main__':
    s = RA.load_dataset('D3')
    print('Composition channel by lead time (W = 60)')
    lead_time_split(s).to_csv(OUT_LEAD, index=False)
    print(f'Saved {OUT_LEAD}')
    wd = window_depth_channels(s)
    wd.to_csv(OUT_WD, index=False)
    print(f'\nSaved {OUT_WD}')
    print(wd[['W', 'horizon', 'model', 'bias_training', 'bias_composition',
              'bias_total', 'n_train_AB', 'n_train_CD']].to_string(index=False))
