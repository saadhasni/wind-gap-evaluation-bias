import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_pipeline as FP
from run_artifact_aligned import (XGB, gap_aware_split, naive_aligned_split,
                                  run_all)

MODELS = ('XGBoost', 'Ridge')
OUT_CSV = os.path.join(HERE, 'artifact_decomposition.csv')
SHARES_CSV = os.path.join(HERE, 'artifact_channel_shares.csv')
FIG_S3 = os.path.join(HERE, 'fig_s3_decomposition.png')


def make_model(name):
    if name == 'XGBoost':
        return XGBRegressor(**XGB)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fit_predict(name, Xtr, ytr, *Xte):
    """Fit once, predict for each test matrix."""
    m = make_model(name)
    m.fit(Xtr, ytr)
    return [m.predict(X) for X in Xte]


def skill(y, pred, pers):
    return 100.0 * (1 - np.sqrt(mean_squared_error(y, pred))
                    / np.sqrt(mean_squared_error(y, pers)))


def feature_diff_report(fnav, fgap, rows, cols):
    """Where do naive and gap-aware features differ on gap-valid rows?"""
    a = fnav.iloc[rows][cols].to_numpy(float)
    b = fgap.iloc[rows][cols].to_numpy(float)
    d = np.abs(a - b)
    mx = d.max(axis=0)
    nz = [f"{c} {v:.3g}" for c, v in zip(cols, mx) if v > 0]
    # tolerance 1e-9 ignores float round-off in rolling sums
    n_diff = int((d > 1e-9).any(axis=1).sum())
    print(f"        B rows (train+test) {len(rows)}: {n_diff} rows with any "
          f"|naive - gap-aware| > 1e-9; max |diff| "
          f"{', '.join(nz) if nz else 'all columns 0'}")


def run(name, series, step, seq_len, horizons, out):
    print(f"\n{'=' * 82}\n  {name}\n{'=' * 82}")
    print(f"  G1 causality check: PASSED "
          f"({FP.causality_self_check(series, step)} points)")
    fgap = FP.build_features_segmented(series, step)
    gcols = [c for c in fgap.columns if c != '_seg']
    fnav = FP.naive_features(series)
    ncols = list(fnav.columns)
    assert ncols == gcols

    for H in horizons:
        sp = gap_aware_split(series, fgap, H, seq_len)
        if sp is None:
            print(f"  H={H}: too few gap-valid rows — skipped")
            continue
        _, tr, te = sp
        tr2, teC, fc = naive_aligned_split(series, H, tr, te, fnav)
        assert fc == ncols

        y_A = series.values[te + H]
        pers_A = fgap.iloc[te]['lag_0'].values
        # D: naive frame restricted to A's exact rows (all present, see split)
        teD = teC.loc[series.index[te]]
        assert np.allclose(teD['target'].values, y_A)

        print(f"  H={H:>2}  test rows: A/B/D {len(te)}   C {len(teC)}  "
              f"(expansion x{len(teC)/len(te):.2f})   "
              f"train rows: A/B {len(tr)}  C/D {len(tr2)} "
              f"(x{len(tr2)/len(tr):.2f})")
        feature_diff_report(fnav, fgap, np.concatenate([tr, te]), ncols)

        for mname in MODELS:
            # A
            pa, = fit_predict(mname, fgap.iloc[tr][gcols].values,
                              series.values[tr + H],
                              fgap.iloc[te][gcols].values)
            sk_a = skill(y_A, pa, pers_A)

            # B: naive features, gap-valid train, gap-valid test
            pb, = fit_predict(mname, fnav.iloc[tr][ncols].values,
                              series.values[tr + H],
                              fnav.iloc[te][ncols].values)
            sk_b = skill(y_A, pb, pers_A)

            # C and D share one model, trained on naive rows
            pd_, pc = fit_predict(mname, tr2[fc].values, tr2['target'].values,
                                  teD[fc].values, teC[fc].values)
            sk_d = skill(y_A, pd_, pers_A)
            sk_c = skill(teC['target'].values, pc, teC['lag_0'].values)

            feat, train = sk_b - sk_a, sk_d - sk_b
            comp, total = sk_c - sk_d, sk_c - sk_a

            print(f"        {mname:<8} A {sk_a:+7.2f}  B {sk_b:+7.2f}  "
                  f"D {sk_d:+7.2f}  C {sk_c:+7.2f}  ||  "
                  f"feature {feat:+6.2f}  training {train:+6.2f}  "
                  f"composition {comp:+6.2f}  total {total:+6.2f}")

            out.append(dict(
                dataset=name, horizon=H, model=mname,
                skill_A=round(sk_a, 2), skill_B=round(sk_b, 2),
                skill_D=round(sk_d, 2), skill_C=round(sk_c, 2),
                bias_feature=round(feat, 2),
                bias_training=round(train, 2),
                bias_composition=round(comp, 2),
                bias_total=round(total, 2),
                n_test_ABD=len(te), n_test_C=len(teC),
                n_train_AB=len(tr), n_train_CD=len(tr2),
                row_expansion=round(len(teC) / len(te), 2)))


CH = ('bias_feature', 'bias_training', 'bias_composition')


def channel_shares(df):
    """Two attributions per dataset:
    'mean_abs': share of mean |contribution| pooled over models and horizons
                (the method behind the paper's IV-B percentages);
    'signed_sum': per model, signed sum over horizons of each channel divided
                by the signed sum of the total bias."""
    rows = []
    for ds, g in df.groupby('dataset', sort=False):
        m = [g[c].abs().mean() for c in CH]
        tot = sum(m)
        rows.append(dict(dataset=ds, model='all', method='mean_abs',
                         feature=m[0], training=m[1], composition=m[2],
                         total=g.bias_total.abs().mean(),
                         **{f'share_{k}': (100 * v / tot if tot else np.nan)
                            for k, v in zip(('feature', 'training',
                                             'composition'), m)}))
        for mname, gm in g.groupby('model', sort=False):
            s_ = [gm[c].sum() for c in CH]
            tot = gm.bias_total.sum()
            rows.append(dict(dataset=ds, model=mname, method='signed_sum',
                             feature=s_[0], training=s_[1], composition=s_[2],
                             total=tot,
                             **{f'share_{k}': (100 * v / tot if abs(tot) > 1e-9
                                               else np.nan)
                                for k, v in zip(('feature', 'training',
                                                 'composition'), s_)}))
    return pd.DataFrame(rows).round(2)


def figure_s3(df, path=FIG_S3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'bias_feature': '#2a78d6', 'bias_training': '#eb6834',
              'bias_composition': '#1baf7a'}
    labels = {'bias_feature': 'feature (B−A)', 'bias_training':
              'training (D−B)', 'bias_composition': 'composition (C−D)'}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 3.8), sharey=True)
    for ax, mname in zip(np.atleast_1d(axes), MODELS):
        g = df[df.model == mname].reset_index(drop=True)
        x = np.arange(len(g))
        pos, neg = np.zeros(len(g)), np.zeros(len(g))
        for c in CH:
            v = g[c].to_numpy(float)
            base = np.where(v >= 0, pos, neg)
            ax.bar(x, v, 0.7, bottom=base, color=colors[c], label=labels[c],
                   edgecolor='white', linewidth=1)
            pos += np.clip(v, 0, None); neg += np.clip(v, None, 0)
        ax.scatter(x, g.bias_total, marker='D', s=18, color='black',
                   zorder=3, label='total (C−A)')
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d.split()[0]}\nH={h}" for d, h in
                            zip(g.dataset, g.horizon)], fontsize=8)
        ax.set_title(mname, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel('bias (percentage points of skill)')
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    out = []
    run_all(run, out)

    if not out:
        print('No results.'); sys.exit(0)
    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print('\n' + '=' * 82)
    print('  DECOMPOSITION (percentage points)')
    print('  feature = B-A   training = D-B   composition = C-D   '
          'total = C-A')
    print('=' * 82)
    print(df[['dataset', 'horizon', 'model', 'bias_feature', 'bias_training',
              'bias_composition', 'bias_total', 'row_expansion']]
          .to_string(index=False))

    print('\n' + '=' * 82)
    print('  WHICH CHANNEL DOMINATES, PER DATASET')
    print('  mean_abs  : share of mean |contribution| over models x horizons')
    print('  signed_sum: per model, signed channel sum / signed total sum')
    print('=' * 82)
    sh = channel_shares(df)
    sh.to_csv(SHARES_CSV, index=False)
    print(sh.to_string(index=False))
    print(f"Saved {SHARES_CSV}")

    figure_s3(df)
