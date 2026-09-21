"""Run every script in order and regenerate every result, number and figure.

    python3 code/run_all.py          # writes results/, results/figs/, results/numbers.tex

Takes a few minutes. run_bench.py writes nothing unless bench measurements exist in
data/bench/.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["run_floor.py", "run_optimum.py", "run_stationarity.py", "run_value.py",
         "run_densities.py", "run_bench.py", "analyse.py", "build_numbers.py",
         "figures.py"]


def main():
    t0 = time.time()
    for step in STEPS:
        print(f"\n== {step}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=HERE, check=True)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
