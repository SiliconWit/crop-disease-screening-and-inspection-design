"""Summarise whatever bench measurements exist in data/bench/.

Reads the CSV files described in data/MANIFEST.md, skips any that are absent, and
writes one summary. Where the fitted densities and a measured recall both exist it
also computes the threshold the stationarity condition gives for them, and where goodput,
success rate and a latency log exist it compares measured latency with the renewal
model. It never fills a gap with an assumed value: a quantity that was not measured
is left out of the file.

    python3 run_bench.py             # writes ../results/bench.json if data exist
"""
import csv, json, os
import numpy as np
import bench as B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "bench")
OUT = os.path.join(ROOT, "results")


def read(name):
    """Rows of data/bench/<name> as dicts, or None if the file is absent."""
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    """Summarise each measurement that exists, and nothing else."""
    Z = {}
    idx = read("index_values.csv")
    if idx:
        Z["exposure_locked"] = all(r["exposure_locked"].strip().lower() in ("1", "true", "yes")
                                   for r in idx)
        for cls in ("healthy", "diseased"):
            v = [float(r["g"]) for r in idx if (r["label"] == "healthy") == (cls == "healthy")]
            if len(v) >= 2:
                a, b, clipped = B.fit_beta(v)
                Z[cls] = dict(n=len(v), a=a, b=b, clipped=clipped)
    tr = read("transfers.csv")
    if tr:
        Z["goodput"] = B.goodput_summary([r["bytes"] for r in tr], [r["seconds"] for r in tr])
        Z["p_success"] = B.first_try_success([r["attempts"] for r in tr])
    cur = read("burst_current.csv")
    if cur:
        Z["tx_current_mean_A"] = B.mean_current([r["t_s"] for r in cur],
                                                [r["current_A"] for r in cur])
    sl = read("sleep_current.csv")
    if sl:
        Z["sleep_current_A"] = B.mean_current([r["t_s"] for r in sl],
                                              [r["current_A"] for r in sl])
    ev = read("cloud_eval.csv")
    if ev:
        Z["cloud_recall"] = B.detection_recall([r["true"] for r in ev],
                                               [r["predicted"] for r in ev])
    lat = read("latency.csv")
    if lat and "goodput" in Z and "p_success" in Z:
        import uplink as U
        meas = float(np.median([float(r["seconds"]) for r in lat]))
        payload = float(np.median([float(r["bytes"]) for r in lat]))
        pred = U.offload_cost(payload, Z["goodput"]["median_bps"], Z["p_success"])["time_s"]
        Z["latency"] = dict(measured_s=meas, predicted_s=pred,
                            error_pct=100 * (pred - meas) / meas)
    if "healthy" in Z and "diseased" in Z:
        import screening as S
        from densities import swapped
        from stationarity import condition_roots
        h = dict(a=Z["healthy"]["a"], b=Z["healthy"]["b"])
        d = dict(a=Z["diseased"]["a"], b=Z["diseased"]["b"])
        with swapped(h, d):
            Z["lr_monotone_fitted"] = S.is_lr_monotone()
            if "cloud_recall" in Z:
                Z["condition_roots_measured"] = condition_roots(Z["cloud_recall"])
    if not Z:
        print("no bench measurements in data/bench/; nothing written")
        return
    os.makedirs(OUT, exist_ok=True)
    json.dump(Z, open(os.path.join(OUT, "bench.json"), "w"), indent=1)
    print(f"wrote results/bench.json: {', '.join(Z)}")


if __name__ == "__main__":
    main()
