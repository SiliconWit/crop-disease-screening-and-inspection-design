"""Test the stationarity condition against the optimiser.

Three sweeps. Change the shape of the loss. Change the cloud recall. Add a cost
charged per visit rather than per offload. In each, the numerical optimum is
compared with the threshold the condition gives, and both are recorded, so that
agreement and disagreement are equally visible. The roots of the condition at the
base densities are recorded too, to show whether the root there is unique.

    python3 run_stationarity.py      # writes ../results/stationarity.json
"""
import json, os
from stationarity import analytic_theta, condition_roots, general_optimum, LOSSES, PRIOR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")


def main():
    """Loss forms, per-visit costs and recall, each against the condition.

    The signed departure (numerical minus analytic) is reported, not its absolute
    value, because a consistent sign is information. For the recall sweep the
    analytic threshold is recomputed at each recall.
    """
    Z = {"prior": PRIOR}
    t_star = analytic_theta(0.92)
    Z["analytic_theta"] = t_star
    Z["base_roots"] = condition_roots(0.92)
    print(f"threshold from the condition at cloud recall 0.92: {t_star:.6f}")
    print(f"roots of the condition on the scanned range: {Z['base_roots']}\n")

    print("loss shape")
    Z["loss_forms"] = []
    for name, fn in LOSSES.items():
        th, T, _ = general_optimum(fn)
        Z["loss_forms"].append(dict(name=name, theta=th, interval=T,
                                    error=th - t_star))
        print(f"  {name:<28} theta {th:.6f}  interval {T:>6.2f} d  "
              f"(departure {th - t_star:+.1e})")

    print("\ncost per visit")
    Z["per_visit"] = []
    for pv in (0.0, 5.0, 50.0, 500.0):
        th, T, _ = general_optimum(LOSSES["logistic (van der Plank)"], per_visit_J=pv)
        Z["per_visit"].append(dict(per_visit_J=pv, theta=th, interval=T,
                                   error=th - t_star))
        print(f"  {pv:>6.0f} J per visit   theta {th:.6f}  interval {T:>6.2f} d  "
              f"(departure {th - t_star:+.1e})")

    print("\ncloud recall")
    Z["recall"] = []
    for rho in (0.60, 0.70, 0.80, 0.92, 0.99):
        ta = analytic_theta(rho)
        th, T, _ = general_optimum(LOSSES["logistic (van der Plank)"],
                                   cloud_recall=rho)
        Z["recall"].append(dict(recall=rho, analytic=ta, numerical=th, interval=T))
        print(f"  recall {rho:.2f}   analytic {ta:.6f}   numerical {th:.6f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stationarity.json"), "w") as fh:
        json.dump(Z, fh, indent=1)
    print("\nwrote results/stationarity.json")


if __name__ == "__main__":
    main()
