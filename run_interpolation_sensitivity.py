import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_artifact_aligned as RA

OUT_CSV = os.path.join(HERE, 'interpolation_sensitivity.csv')
COLS = ['interpolate', 'dataset', 'segments', 'n_series', 'horizon',
        'skill_honest', 'skill_naive', 'bias_total', 'bias_ci_low',
        'bias_ci_high', 'ci_excludes_zero', 'dm_p_honest', 'dm_p_naive',
        'n_test_honest', 'n_test_naive', 'n_train_honest', 'n_train_naive']


if __name__ == '__main__':
    rows = []
    for name, step, seq_len, horizons in RA.DATASETS[:2]:      # D1, D2
        for interp in (True, False):
            try:
                s = RA.load_dataset(name[:2], interpolate=interp)
            except FileNotFoundError as e:
                print(f'[skip {name[:2]}]', e)
                continue
            out = []
            print(f"\n##### interpolate={interp}   {len(s)} samples")
            RA.run(name, s, step, seq_len, horizons, out)
            for r in out:
                r['interpolate'] = interp
                r['n_series'] = len(s)
            rows += out

    if not rows:
        print('No results.'); sys.exit(0)
    df = pd.DataFrame(rows)[COLS]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print('\n' + '=' * 78)
    print('  TABLE III, WITH AND WITHOUT SHORT-GAP INTERPOLATION')
    print('=' * 78)
    print(df[['dataset', 'horizon', 'interpolate', 'segments', 'skill_honest',
              'skill_naive', 'bias_total', 'bias_ci_low', 'bias_ci_high',
              'dm_p_honest', 'dm_p_naive']].to_string(index=False))
    w = df.pivot_table(index=['dataset', 'horizon'], columns='interpolate',
                       values='bias_total')
    w['change'] = w[False] - w[True]
    print('\n  bias without fill minus bias with fill (points):')
    print(w.to_string())
