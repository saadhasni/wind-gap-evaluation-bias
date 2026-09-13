import warnings, os, sys, glob, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# CONFIG 
SEED          = 42
TRAIN_RATIO   = 0.75
ARIMA_ORIGINS = 120          # strided ARIMA refit origins per horizon
RNN_EPOCHS    = 30
BOOT_N        = 500          # bootstrap resamples for RMSE CIs
BOOT_BLOCK    = 50           # circular block length (samples)
RUN_DEEP      = True         # set False to skip LSTM/GRU (much faster)
np.random.seed(SEED); tf.random.set_seed(SEED)

MODEL_NAMES = ['Persistence', 'ARIMA', 'XGBoost', 'LSTM', 'GRU']

#  SMALL UTILITIES
def rmse_mae(a, b):
    return (float(np.sqrt(mean_squared_error(a, b))),
            float(mean_absolute_error(a, b)))

def kalman_causal(z, Q=0.02, R=0.6):
    n, x, P = len(z), z[0], 1.0
    out = np.empty(n); out[0] = x
    for k in range(1, n):
        P += Q; K = P / (P + R); x += K * (z[k] - x); P *= (1 - K); out[k] = x
    return out

def diebold_mariano(e1, e2):
    """DM test on squared-error loss with Newey–West variance (lag 10)."""
    d = e1**2 - e2**2
    T = len(d); mu = d.mean(); nw = np.var(d, ddof=1)
    for L in range(1, 11):
        if L < T:
            nw += 2*(1 - L/11)*np.cov(d[:-L], d[L:])[0, 1]
    dm = mu / np.sqrt(nw / T)
    return float(dm), float(2*(1 - stats.norm.cdf(abs(dm))))

def block_bootstrap_rmse_ci(err, n_boot=BOOT_N, block=BOOT_BLOCK, seed=SEED):
    """95% CI for RMSE via circular block bootstrap of the error series."""
    rng = np.random.default_rng(seed)
    T = len(err)
    if T < block * 2:
        block = max(5, T // 4)
    n_blocks = int(np.ceil(T / block))
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        stats_[b] = np.sqrt(np.mean(err[idx[:T]]**2))
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))

# GAP-AWARE CORE (audited)
def segment_ids(index, step):
    d = index.to_series().diff()
    breaks = (d != step).to_numpy(copy=True)
    breaks[0] = False
    return np.cumsum(breaks)

def build_features_segmented(series, step):
    """Causal features computed WITHIN each contiguous segment (G2)."""
    seg = segment_ids(series.index, step)
    df = pd.DataFrame(index=series.index)
    df['_seg'] = seg
    g = series.groupby(seg)
    df['lag_0'] = series
    for lag in (1, 2, 3, 5, 10, 30, 60):
        df[f'lag_{lag}'] = g.shift(lag)
    for w in (5, 15, 60):
        df[f'roll_mean_{w}'] = g.transform(lambda s: s.rolling(w).mean())
        df[f'roll_std_{w}']  = g.transform(lambda s: s.rolling(w).std())
    df['wind_shear'] = g.diff()
    df['turbulence'] = (g.transform(lambda s: s.rolling(30).std())
                        / (g.transform(lambda s: s.rolling(30).mean()) + 1e-9))
    kv = np.empty(len(series))
    for _, idx in series.groupby(seg).groups.items():
        loc = series.index.get_indexer(idx)
        kv[loc] = kalman_causal(series.loc[idx].values)
    df['kalman'] = kv
    df['hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
    return df

def valid_rows_for_horizon(series, feat, H, seq_len):
    """Rows usable as a forecast origin: full features, and neither the
    seq_len input window nor the +H target crosses a segment gap (G3)."""
    seg = feat['_seg'].values
    n = len(series)
    fcols = [c for c in feat.columns if c != '_seg']
    no_nan = feat[fcols].notna().all(axis=1).to_numpy()
    p = np.arange(n)
    lo, tgt = p - (seq_len - 1), p + H
    ok = no_nan & (lo >= 0) & (tgt < n)
    lo_c, tgt_c = np.clip(lo, 0, n-1), np.clip(tgt, 0, n-1)
    return ok & (seg[lo_c] == seg) & (seg[tgt_c] == seg)

def build_rnn(cell, seq_len):
    m = Sequential([
        cell(32, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2), cell(16), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1)])
    m.compile(optimizer='adam', loss='mse')
    return m

# PER-DATASET EXPERIMENT 
def run_dataset(name, series, step, horizons, seq_len, all_rows_out):
    print(f"\n{'='*72}\n  DATASET: {name}"
          f"\n  seq_len={seq_len} steps ({seq_len*step}) | horizons={horizons}"
          f"\n{'='*72}")
    seg = segment_ids(series.index, step)
    print(f"  Samples {len(series)} | segments {seg.max()+1} | "
          f"{series.index[0]} -> {series.index[-1]}")

    feat = build_features_segmented(series, step)

    # G1: causality self-check
    chk = len(series) // 2
    fp = build_features_segmented(series.iloc[:chk+1], step)
    cols = [c for c in feat.columns if c != '_seg']
    diff = float((feat.iloc[chk][cols] - fp.iloc[chk][cols]).abs().max())
    assert diff < 1e-9, f"CAUSALITY VIOLATION ({diff})"
    print("  G1 causality check: PASSED")

    fcols = [c for c in feat.columns if c != '_seg']

    for H in horizons:
        mask = valid_rows_for_horizon(series, feat, H, seq_len)
        rows = np.where(mask)[0]
        if len(rows) < 800:
            print(f"  H={H}: only {len(rows)} valid rows — skipped"); continue

        split = int(len(rows) * TRAIN_RATIO)
        EMB = seq_len + H                       # G4 embargo
        tr_rows = rows[:split]
        te_rows = rows[split + EMB:]
        if len(te_rows) < 300:
            print(f"  H={H}: test too small after embargo — skipped"); continue

        Xtr = feat.iloc[tr_rows][fcols].values
        Xte = feat.iloc[te_rows][fcols].values
        ytr = series.values[tr_rows + H]
        yte = series.values[te_rows + H]
        ynow_te = feat.iloc[te_rows]['lag_0'].values
        res, errs = {}, {}

        # Persistence
        r, m = rmse_mae(yte, ynow_te)
        res['Persistence'] = (r, m); errs['Persistence'] = yte - ynow_te

        # ARIMA — strided true-H-step forecasts, in-segment causal history
        try:
            seg_all = feat['_seg'].values
            svals = series.values
            origins = np.linspace(0, len(te_rows)-1,
                                  min(ARIMA_ORIGINS, len(te_rows)), dtype=int)
            pr, tt, used_k = [], [], []
            for k in origins:
                pg = te_rows[k]; sid = seg_all[pg]
                # walk backwards inside the same segment (fast, causal)
                lo = pg
                while lo > 0 and seg_all[lo-1] == sid and pg - lo < 300:
                    lo -= 1
                hist = svals[lo:pg+1]
                if len(hist) < 30: continue
                fit = ARIMA(hist, order=(2, 0, 1)).fit(
                    method_kwargs={'maxiter': 25})
                pr.append(fit.forecast(H)[-1]); tt.append(yte[k])
                used_k.append(k)
            if len(tt) > 20:
                tt = np.array(tt); pr = np.array(pr)
                r, m = rmse_mae(tt, pr)
                res['ARIMA'] = (r, m)
                # fairness: persistence on the SAME origins, for honest skill
                pers_sub = np.array([ynow_te[k] for k in used_k])
                res['_pers_on_arima_origins'] = rmse_mae(tt, pers_sub)[0]
        except Exception as e:
            print(f"    ARIMA H={H} failed: {e}")

        # XGBoost
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, verbosity=0)
        xgb.fit(Xtr, ytr); xp = xgb.predict(Xte)
        r, m = rmse_mae(yte, xp)
        res['XGBoost'] = (r, m); errs['XGBoost'] = yte - xp

        # LSTM / GRU  (G6: scaler fit strictly pre-test)
        if RUN_DEEP:
            scaler = MinMaxScaler()
            scaler.fit(series.values[:te_rows[0]].reshape(-1, 1))
            ys_s = scaler.transform(series.values.reshape(-1, 1)).flatten()
            windows = np.lib.stride_tricks.sliding_window_view(ys_s, seq_len)
            Xs_tr = windows[tr_rows - seq_len + 1][..., None]
            Xs_te = windows[te_rows - seq_len + 1][..., None]
            ys_tr = ys_s[tr_rows + H]
            t_inv = yte
            es = EarlyStopping(patience=8, restore_best_weights=True, verbose=0)
            for nm, cell in (('LSTM', LSTM), ('GRU', GRU)):
                tf.random.set_seed(SEED)
                mdl = build_rnn(cell, seq_len)
                mdl.fit(Xs_tr, ys_tr, epochs=RNN_EPOCHS, batch_size=128,
                        validation_split=0.1, callbacks=[es], verbose=0)
                p = scaler.inverse_transform(
                    mdl.predict(Xs_te, verbose=0)).flatten()
                r, m = rmse_mae(t_inv, p)
                res[nm] = (r, m); errs[nm] = t_inv - p

        # report + stats
        rp = res['Persistence'][0]
        line = f"  H={H:>2} ({H*step}) valid={len(rows):6d} test={len(te_rows):5d} | "
        for nm in MODEL_NAMES:
            if nm in res:
                base = (res.get('_pers_on_arima_origins', rp)
                        if nm == 'ARIMA' else rp)
                sk = 0 if nm == 'Persistence' else 100*(1 - res[nm][0]/base)
                line += f"{nm[:4]} {res[nm][0]:.3f}({sk:+.0f}%) "
        print(line)

        dm_p = None
        if 'XGBoost' in errs:
            _, dm_p = diebold_mariano(errs['XGBoost'], errs['Persistence'])
            lo_p, hi_p = block_bootstrap_rmse_ci(errs['Persistence'])
            lo_x, hi_x = block_bootstrap_rmse_ci(errs['XGBoost'])
            print(f"        Persistence RMSE 95% CI [{lo_p:.3f},{hi_p:.3f}] | "
                  f"XGBoost [{lo_x:.3f},{hi_x:.3f}] | "
                  f"DM p={dm_p:.4f} [{'sig' if dm_p<0.05 else 'ns'}]")

        for nm in MODEL_NAMES:
            if nm in res:
                base = (res.get('_pers_on_arima_origins', rp)
                        if nm == 'ARIMA' else rp)
                sk = 0 if nm == 'Persistence' else 100*(1 - res[nm][0]/base)
                all_rows_out.append(dict(
                    dataset=name, horizon_steps=H,
                    horizon_time=str(H*step), model=nm,
                    rmse=round(res[nm][0], 4), mae=round(res[nm][1], 4),
                    skill_pct=round(sk, 1),
                    dm_p_vs_persistence=(round(dm_p, 4)
                                         if nm == 'XGBoost' and dm_p else None),
                    n_test=len(te_rows)))


#  MAIN 
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset_adapters import load_zephir, load_wfip3_buoy

    HERE = os.path.dirname(os.path.abspath(__file__))
    all_rows = []

    # 1) ZephIR — 1-min, seq_len 60 steps = 60 min
    try:
        s, meta = load_zephir(HERE, height_m=38, resample='1min')
        run_dataset('ZephIR300_offshore_1min', s, pd.Timedelta('1min'),
                    horizons=(1, 10, 30, 60), seq_len=60,
                    all_rows_out=all_rows)
    except FileNotFoundError:
        print("\n[skip] ZephIR CSVs not found next to this script")

    # 2) WFIP3 Buoy — 10-min, seq_len 36 steps = 6 h
    try:
        s, meta = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                                  height_m=38, resample='10min')
        run_dataset('WFIP3_Buoy130_offshore_10min', s, pd.Timedelta('10min'),
                    horizons=(1, 3, 6), seq_len=36, all_rows_out=all_rows)
    except FileNotFoundError:
        print("\n[skip] buoy_data/*.nc not found")

    # 3) Onshore turbine — 10-min, seq_len 36 steps = 6 h
    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run_dataset('Onshore_turbine_10min', s, pd.Timedelta('10min'),
                    horizons=(1, 3, 6), seq_len=36, all_rows_out=all_rows)
    except (IndexError, FileNotFoundError):
        print("\n[skip] onshore/*10min*.csv not found")

    #  results table + figures
    if all_rows:
        res_df = pd.DataFrame(all_rows)
        res_df.to_csv('final_results.csv', index=False)
        print(f"\nSaved final_results.csv ({len(res_df)} rows)")

        # Figure 1: skill vs horizon per dataset
        fig, axes = plt.subplots(1, res_df['dataset'].nunique(),
                                 figsize=(6*res_df['dataset'].nunique(), 4.5),
                                 squeeze=False)
        for ax, (ds, g) in zip(axes[0], res_df.groupby('dataset')):
            for nm, gg in g.groupby('model'):
                if nm == 'Persistence': continue
                ax.plot(gg['horizon_steps'], gg['skill_pct'],
                        marker='o', label=nm)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_title(ds, fontsize=9); ax.set_xlabel('Horizon (steps)')
            ax.set_ylabel('Skill vs persistence (%)'); ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig('final_fig_skill.png', dpi=140)
        plt.close(fig)

        # Figure 2: RMSE bars at shortest horizon per dataset
        fig, axes = plt.subplots(1, res_df['dataset'].nunique(),
                                 figsize=(6*res_df['dataset'].nunique(), 4.5),
                                 squeeze=False)
        for ax, (ds, g) in zip(axes[0], res_df.groupby('dataset')):
            h0 = g['horizon_steps'].min()
            gg = g[g['horizon_steps'] == h0]
            ax.bar(gg['model'], gg['rmse'], color='steelblue', alpha=0.85)
            ax.set_title(f'{ds}  (H={h0})', fontsize=9)
            ax.set_ylabel('RMSE (m/s)'); ax.tick_params(axis='x', rotation=25)
            ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig('final_fig_rmse.png', dpi=140)
        plt.close(fig)
        print("Saved final_fig_skill.png, final_fig_rmse.png")

    print("\nAll done.")