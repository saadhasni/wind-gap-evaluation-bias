"""Run the whole pipeline in the README order.

    python run_all.py            # everything (many hours)
    python run_all.py --quick    # skip the long jobs (injection, window depth,
                                 # LSTM table, KNMI download); uses existing CSVs
    python run_all.py --from 9   # resume at step 9
    python run_all.py --list     # show the steps

Each step is a separate process, so a failure stops the run at that step and
names it. Step numbers match the README table.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (step, script, long_running)
STEPS = [
    (1, 'knmi_auto.py', True),
    (2, 'make_data_section.py', False),
    (3, 'make_knmi_table.py', False),
    (4, 'run_artifact_aligned.py', False),
    (5, 'run_artifact_decompose.py', False),
    (6, 'run_artifact_audit.py', False),
    (7, 'run_model_generalisation_aligned.py', True),
    (8, 'run_rolling_origin_aligned.py', False),
    (9, 'run_gap_profile.py', False),
    (9, 'run_interpolation_sensitivity.py', False),   # README row '-'
    (10, 'run_gap_injection.py', True),
    (11, 'make_injection_summary.py', False),
    (12, 'exposure_check.py', False),
    (13, 'mixture_check.py', False),
    (14, 'injection_mixture_check.py', False),
    (15, 'regen_fig1_v2.py', False),
    (15, 'regen_fig2_v2.py', False),
    (15, 'regen_fig3_v2.py', False),
    (16, 'run_window_depth.py', True),
    (17, 'run_clark_west.py', False),
    (18, 'run_channel_robustness.py', False),
    (19, 'run_robustness.py', True),
    (20, 'make_paper_numbers.py', False),   # always last
]


def main(argv):
    quick = '--quick' in argv
    start = int(argv[argv.index('--from') + 1]) if '--from' in argv else 0
    if '--list' in argv:
        for n, s, long_ in STEPS:
            print(f"{n:>3}  {s}{'   (long)' if long_ else ''}")
        return 0
    for n, script, long_ in STEPS:
        if n < start or (quick and long_):
            print(f"[skip] step {n}: {script}")
            continue
        print(f"\n[run ] step {n}: {script}", flush=True)
        t0 = time.time()
        if subprocess.run([sys.executable, os.path.join(HERE, script)],
                          cwd=HERE).returncode != 0:
            print(f"FAILED at step {n} ({script}); fix and resume with "
                  f"--from {n}")
            return 1
        print(f"[done] step {n} in {(time.time() - t0) / 60:.1f} min")
    print("\nAll steps finished. Run `python -m pytest tests -q` for the unit tests.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
