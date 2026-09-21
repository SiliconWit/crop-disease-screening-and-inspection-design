"""Sweep the assumed densities, and check each optimum against the condition.

Fifteen pairs: three healthy densities against five diseased ones, from well
separated to heavily overlapping. For each pair, the joint optimum, the 2 percent
operating point, and every root of the stationarity condition for that pair. An
optimum on a bound of the search is a corner and is flagged; an interior optimum
is measured against the nearest root. For each root the file also records how
much of the diseased class the edge still sends there, and the delay and loss at
the shortest interval the search allows, which shows whether an interior optimum
could sit on that root at all.

    python3 run_densities.py         # writes ../results/densities.json
"""
import json, os
import risk as R
import screening as S
import epidemic as E
from densities import with_densities, swapped
from stationarity import condition_roots

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")

DISEASED_PAIRS = [(3.0, 12.0), (4.0, 10.0), (5.0, 9.0), (6.0, 8.0), (8.0, 7.0)]
HEALTHY_PAIRS = [(2.0, 24.0), (2.0, 18.0), (2.5, 14.0)]


def describe_root(t, rho=0.92):
    """The edge's send probability for a diseased leaf at threshold t, and the
    detection delay and loss even at the shortest interval allowed."""
    p = S.operating_point(t, R.BASE["prior"])["p_detect_given_diseased"]
    d = E.detection_delay(R.INTERVAL_BOUNDS[0], p * rho)
    return dict(theta=t, p_send_diseased=float(p), delay_at_shortest_interval=float(d),
                loss_at_shortest_interval=E.miss_cost(d, r=R.BASE["r"]))


def condition_check(h, d, theta, interval, rho=0.92):
    """The optimum's distance from the nearest root of the stationarity condition.

    An optimum on a bound of the search is a corner: the condition does not apply
    to it, and it is flagged rather than measured. For an interior optimum, every
    root is reported with the signed departure from the nearest one.
    """
    corner = R.on_bound(theta, interval)
    with swapped(h, d):
        roots = condition_roots(rho)
        described = [describe_root(t, rho) for t in roots]
    near = min(roots, key=lambda t: abs(t - theta)) if roots else None
    return dict(corner=corner, roots=roots, root_details=described, nearest_root=near,
                departure=None if (corner or near is None) else theta - near)


def main():
    """Run every pair and write the sweep and the condition check side by side."""
    grid, check = [], []
    for da, db in DISEASED_PAIRS:
        for ha, hb in HEALTHY_PAIRS:
            h, d = dict(a=ha, b=hb), dict(a=da, b=db)
            row = with_densities(h, d)
            grid.append(row)
            c = condition_check(h, d, row["theta"], row["interval"])
            c.update(healthy=h, diseased=d, theta=row["theta"], interval=row["interval"],
                     bhattacharyya=row["bhattacharyya"])
            check.append(c)
            dep = "corner" if c["corner"] else (
                "no root" if c["departure"] is None else f"{c['departure']:+.1e}")
            print(f"Be({ha},{hb}) Be({da},{db})  overlap {row['bhattacharyya']:.3f}  "
                  f"theta {row['theta']:.3f}  interval {row['interval']:>5.1f} d  "
                  f"roots {[round(t, 4) for t in c['roots']]}  departure {dep}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "densities.json"), "w") as fh:
        json.dump({"density_sensitivity": grid, "condition_check": check}, fh, indent=1)
    print("wrote results/densities.json")


if __name__ == "__main__":
    main()
