import warnings, os, sys, glob, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from dataset_adapters import load_zephir, load_wfip3_buoy

# ---------------- config ----------------
TRAIN_RATIO = 0.75
SEED        = 42
RUN_LSTM    = True     # set False for a fast run without the neural network
LSTM_EPOCHS = 20
MIN_TEST    = 200

MODELS_FAST = {
    'Ridge':        lambda: Ridge(alpha=1.0),
    'RandomForest': lambda: RandomForestRegressor(
                        n_estimators=200, max_depth=12, n_jobs=-1,
                        random_state=SEED),
    'XGBoost':      lambda: XGBRegressor(
                        n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, verbosity=0),
}


def naive_features(series):
    """Gap-ignoring feature construction (common practice)."""
    f = pd.DataFrame(index=series.index)
    f['lag_0'] = series
    for lag in (1, 2, 3, 5, 10, 30, 60):
        f[f'lag_{lag}'] = series.shift(lag)
    for w in (5, 15, 60):
        f[f'roll_mean_{w}'] = series.rolling(w).mean()
        f[f'roll_std_{w}'] = series.rolling(w).std()
    f['wind_shear'] = series.diff()
    f['turbulence'] = series.rolling(30).std() / (series.rolling(30).mean() + 1e-9)
    f['kalman'] = FP.kalman_causal(series.values)
    f['hour_sin'] = np.sin(2 * np.pi * f.index.hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * f.index.hour / 24)
    return f


def skill(y, pred, persist):
    rm = np.sqrt(mean_squared_error(y, pred))
    rp = np.sqrt(mean_squared_error(y, persist))
    return 100.0 * (1.0 - rm / rp)


def fit_predict_tabular(name, Xtr, ytr, Xte):
    """Ridge needs scaling; trees do not. Scaler is fit on TRAIN only."""
    if name == 'Ridge':
        sc = StandardScaler().fit(Xtr)
        m = MODELS_FAST[name]()
        m.fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    m = MODELS_FAST[name]()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def fit_predict_lstm(seq_len, series, tr_pos, te_pos, H, fit_end_pos):
    """LSTM on raw scaled sequences. tr_pos/te_pos are POSITIONS in `series`."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    sc = MinMaxScaler().fit(series.values[:fit_end_pos].reshape(-1, 1))
    ys = sc.transform(series.values.reshape(-1, 1)).flatten()
    win = np.lib.stride_tricks.sliding_window_view(ys, seq_len)
    Xtr = win[tr_pos - seq_len + 1][..., None]
    Xte = win[te_pos - seq_len + 1][..., None]
    ytr = ys[tr_pos + H]

    tf.random.set_seed(SEED)
    m = Sequential([
        LSTM(32, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2), LSTM(16), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1)])
    m.compile(optimizer='adam', loss='mse')
    m.fit(Xtr, ytr, epochs=LSTM_EPOCHS, batch_size=128, validation_split=0.1,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=0)
    return sc.inverse_transform(m.predict(Xte, verbose=0)).flatten()


def run_dataset(name, series, step, seq_len, horizons, out):
    print(f"\n{'='*74}\n  {name}\n{'='*74}")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = naive_features(series)

    for H in horizons:
        mask = FP.valid_rows_for_horizon(series, fgap, H, seq_len)
        rows = np.where(mask)[0]
        if len(rows) < 800:
            continue
        split = int(len(rows) * TRAIN_RATIO)
        EMB = seq_len + H
        tr, te = rows[:split], rows[split + EMB:]
        if len(te) < 300:
            continue

        # ---- honest (A): gap-aware features, gap-aware rows ----
        y_te_A = series.values[te + H]
        pers_A = fgap.iloc[te]['lag_0'].values
        Xtr_A, Xte_A = fgap.iloc[tr][gcols].values, fgap.iloc[te][gcols].values
        ytr_A = series.values[tr + H]

        # A's calendar test window
        a_start = series.index[te[0]]
        a_end = series.index[te[-1]]

        # ---- naive (C): naive features, all rows in A's window ----
        dn = fnav.copy()
        dn['target'] = series.shift(-H)
        dn = dn.dropna()
        fc = [c for c in dn.columns if c != 'target']
        s2 = int(len(dn) * TRAIN_RATIO)
        tr2 = dn.iloc[:s2]
        te2_all = dn.iloc[s2:]

        # ---- THE FIX: score C on A's calendar window ----
        te2 = te2_all[(te2_all.index >= a_start) & (te2_all.index <= a_end)]
        if len(te2) < MIN_TEST:
            print(f"  H={H}: aligned naive test only {len(te2)} rows — skipped")
            continue
        y_te_C = te2['target'].values
        pers_C = te2['lag_0'].values

        print(f"  H={H:>2} | valid={len(rows):6d} test A={len(te):5d} "
              f"C={len(te2):5d} (unaligned {len(te2_all)})  "
              f"expansion x{len(te2)/len(te):.2f}")

        names = list(MODELS_FAST) + (['LSTM'] if RUN_LSTM else [])
        for mname in names:
            t0 = time.time()
            if mname == 'LSTM':
                pa = fit_predict_lstm(seq_len, series, tr, te, H, te[0])
                sk_A = skill(y_te_A, pa, pers_A)

                # naive LSTM: positions of the naive rows within `series`.
                # The train/test boundary comes from the ORIGINAL split
                # point dn.index[s2], not from te2.index[0]; otherwise rows
                # between the two would migrate into training.
                pos_all = series.index.get_indexer(dn.index)
                ok = pos_all >= seq_len - 1
                pos_ok = pos_all[ok]
                idx_ok = dn.index[ok]
                boundary = dn.index[s2]
                n_tr2 = int((idx_ok < boundary).sum())
                tr_p = pos_ok[:n_tr2]
                te_p = pos_ok[n_tr2:]
                te_t = idx_ok[n_tr2:]
                keep = (te_t >= a_start) & (te_t <= a_end)
                te_p = te_p[keep]
                if len(te_p) < MIN_TEST or len(tr_p) < 300:
                    print(f"        {mname:<13} skipped "
                          f"(train {len(tr_p)}, test {len(te_p)})")
                    continue
                pc = fit_predict_lstm(seq_len, series, tr_p, te_p, H, te_p[0])
                sk_C = skill(series.values[te_p + H], pc,
                             series.values[te_p])
                n_c = len(te_p)
            else:
                pa = fit_predict_tabular(mname, Xtr_A, ytr_A, Xte_A)
                pc = fit_predict_tabular(mname, tr2[fc].values,
                                         tr2['target'].values, te2[fc].values)
                sk_A = skill(y_te_A, pa, pers_A)
                sk_C = skill(y_te_C, pc, pers_C)
                n_c = len(te2)

            bias = sk_C - sk_A
            print(f"        {mname:<13} honest {sk_A:+7.2f}%  "
                  f"naive {sk_C:+7.2f}%  bias {bias:+7.2f}   "
                  f"({time.time()-t0:.0f}s)")
            out.append(dict(dataset=name, horizon=H, model=mname,
                            skill_honest=round(sk_A, 2),
                            skill_naive=round(sk_C, 2),
                            bias=round(bias, 2),
                            n_test_honest=len(te), n_test_naive=n_c,
                            n_test_naive_unaligned=len(te2_all)))


if __name__ == '__main__':
    out = []
    try:
        s, _ = load_zephir(HERE, height_m=38, resample='1min')
        run_dataset('D1 ZephIR (contiguous)', s, pd.Timedelta('1min'), 60,
                    (1, 10, 30), out)
    except Exception as e:
        print('[skip D1]', e)

    try:
        s, _ = load_wfip3_buoy(os.path.join(HERE, 'buoy_data'),
                               height_m=38, resample='10min')
        run_dataset('D2 WFIP3 Buoy (few long gaps)', s, pd.Timedelta('10min'),
                    36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D2]', e)

    try:
        f = [x for x in glob.glob(os.path.join(HERE, 'onshore', '*.csv'))
             if '10min' in x][0]
        d = pd.read_csv(f); d['Time'] = pd.to_datetime(d['Time'])
        s = d.set_index('Time')['WindSpeed'].sort_index()
        s = s[~s.index.duplicated()].where(lambda x: x.between(0, 45)).dropna()
        run_dataset('D3 Onshore (many short gaps)', s, pd.Timedelta('10min'),
                    36, (1, 3, 6), out)
    except Exception as e:
        print('[skip D3]', e)

    if not out:
        print('No results produced.'); sys.exit(0)

    df = pd.DataFrame(out)
    df.to_csv('model_generalisation_aligned.csv', index=False)
    print(f"\nSaved model_generalisation_aligned.csv ({len(df)} rows)")

    print("\n" + "=" * 74)
    print("  MEAN BIAS BY DATASET AND MODEL (percentage points)")
    print("=" * 74)
    piv = df.pivot_table(index='dataset', columns='model', values='bias',
                         aggfunc='mean').round(2)
    print(piv.to_string())

    print("\n  Sign agreement per dataset/horizon (does every model family "
          "agree?)")
    agree = tot = 0
    for (ds, H), g in df.groupby(['dataset', 'horizon']):
        signs = np.sign(g['bias'].values)
        tot += 1
        ok = np.all(signs == signs[0])
        agree += int(ok)
        print(f"    {ds:<32} H={H:<3} "
              f"{'all agree' if ok else 'MIXED'}  "
              f"({', '.join(f'{m}:{b:+.1f}' for m, b in zip(g.model, g.bias))})")
    print(f"\n  Model families agree on sign in {agree}/{tot} cases")

    fig, axes = plt.subplots(1, df['dataset'].nunique(),
                             figsize=(5.6 * df['dataset'].nunique(), 4.3),
                             squeeze=False)
    for ax, (ds, g) in zip(axes[0], df.groupby('dataset', sort=False)):
        p = g.pivot_table(index='horizon', columns='model', values='bias')
        p.plot(kind='bar', ax=ax, width=0.8)
        ax.axhline(0, color='k', lw=1)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Bias: naive \u2212 honest (pts)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Gap-handling bias across model families '
                 '(calendar-aligned windows)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('model_generalisation_aligned_fig.png', dpi=150,
                bbox_inches='tight')
    print('Saved model_generalisation_aligned_fig.png')
