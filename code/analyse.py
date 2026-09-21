"""Gather the per-sweep results files into the two files the numbers are built from.

    python3 analyse.py        # reads optimum, value, densities, stationarity;
                              # writes analysis.json and invariance.json

Nothing is recomputed here. If a number in either output is wrong, the fix belongs
in the script that produced it.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

ORDER_OPTIMUM = ["base", "target_miss", "conventional_interval", "lr_monotone",
                 "weights", "captures_per_visit", "search_bounds", "optimum",
                 "baseline_at_optimum", "at_target_miss", "at_target_miss_alt"]
ORDER_VALUE = ["regime", "regime_spread", "payload_sweep", "p_success_sweep",
               "weight_sensitivity"]


def merge(optimum, value, densities):
    """One analysis record, keys in a fixed order so the file diffs cleanly.

    Every key comes from exactly one input file. A missing key means that script
    was not run, and raises rather than leaving a gap.
    """
    out = {k: optimum[k] for k in ORDER_OPTIMUM}
    out["joint_value"] = value["joint_value"]
    out["density_sensitivity"] = densities["density_sensitivity"]
    out["condition_check"] = densities["condition_check"]
    for k in ORDER_VALUE:
        out[k] = value[k]
    return out


def main():
    """Write analysis.json from three results files, and invariance.json from one."""
    def load(n):
        with open(os.path.join(RES, n)) as fh:
            return json.load(fh)
    A = merge(load("optimum.json"), load("value.json"), load("densities.json"))
    with open(os.path.join(RES, "analysis.json"), "w") as fh:
        json.dump(A, fh, indent=1)
    with open(os.path.join(RES, "invariance.json"), "w") as fh:
        json.dump(load("stationarity.json"), fh, indent=1)
    print(f"wrote results/analysis.json ({len(A)} keys) and results/invariance.json")


if __name__ == "__main__":
    main()
