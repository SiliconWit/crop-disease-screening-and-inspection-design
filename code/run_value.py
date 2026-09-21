"""What optimising jointly is worth, and which deployment quantities move the design.

Four parts. The joint optimum against the two one-variable procedures a
practitioner would actually use. The optimum over a grid of apparent infection
rate and link goodput. The optimum as the payload and the per-attempt success
probability of the link are varied. And the optimum as each cost weight is scaled
up and down tenfold. Every optimum records its distance from the threshold given
by the stationarity condition and whether it lies on a bound of the search.

    python3 run_value.py             # writes ../results/value.json
"""
import json, os
import screening as S
import risk as R
from stationarity import analytic_theta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")

RS = [0.08, 0.12, 0.18, 0.25, 0.35, 0.45]
GOODPUTS = [3000.0, 6000.0, 12000.0, 20000.0, 40000.0, 70000.0]
PAYLOADS = [10000, 15000, 25000, 35000, 50000]
P_SUCCESS = [0.50, 0.70, 0.85, 0.95, 0.99]
WEIGHT_CASES = [("energy x10", (1, 10, 1)), ("energy /10", (1, 0.1, 1)),
                ("data x10", (1, 1, 10)), ("data /10", (1, 1, 0.1)),
                ("loss x10", (10, 1, 1)), ("loss /10", (0.1, 1, 1))]


def _row(oo, t_star, **extra):
    """One optimum as a results row."""
    return dict(extra, theta=oo["theta"], interval=oo["interval_days"],
                offload=oo["offload_rate"], risk=oo["risk"], miss=oo["miss_rate"],
                delay=oo["delay_days"], on_bound=oo["on_bound"],
                departure=oo["theta"] - t_star)


def joint_value():
    """The joint optimum against two sequential procedures.

    Procedure one fixes the interval at a conventional weekly schedule and tunes
    the threshold. Procedure two fixes the threshold at the 2 percent miss rate and
    tunes the interval. Both ratios are reported: the gain of the joint optimum,
    1 - joint / procedure, and the excess cost of the procedure, procedure / joint
    - 1. They are different numbers and must be quoted as what they are.
    """
    BASE = R.BASE
    o = R.optimise(**BASE)
    r_joint = o["risk"]
    th2 = S.threshold_for_miss_rate(R.TARGET_MISS, BASE["prior"])
    th_fix, r_fix_iv = R.opt_theta_given(R.CONVENTIONAL_INTERVAL, **BASE)
    iv_fix, r_fix_th = R.opt_interval_given(th2, **BASE)
    out = dict(
        joint=r_joint,
        theta_only_at_conventional_interval=dict(theta=th_fix, risk=r_fix_iv,
                                                 interval=R.CONVENTIONAL_INTERVAL),
        interval_only_at_target_theta=dict(interval=iv_fix, risk=r_fix_th, theta=th2),
        gain_over_theta_only=1 - r_joint / r_fix_iv,
        gain_over_interval_only=1 - r_joint / r_fix_th,
        excess_theta_only=r_fix_iv / r_joint - 1,
        excess_interval_only=r_fix_th / r_joint - 1)
    print(f"joint risk {r_joint:.5f}; threshold alone at {R.CONVENTIONAL_INTERVAL:.0f} d "
          f"{r_fix_iv:.5f}; interval alone at the 2% threshold {r_fix_th:.5f}")
    return out


def regime(t_star):
    """The joint optimum over the (infection rate, goodput) grid.

    Returns the rows and a summary of how far the interval moves along each axis,
    one axis at a time with the other held at a middle value. A factor between the
    smallest and largest interval is easier to compare across axes than a
    difference in days.
    """
    rows = []
    for gp in GOODPUTS:
        for rr in RS:
            kw = dict(R.BASE); kw.update(r=rr, goodput=gp)
            rows.append(_row(R.optimise(**kw), t_star, r=rr, goodput=gp))
    iv = {(d["r"], d["goodput"]): d["interval"] for d in rows}
    spread = dict(
        over_r_at_median_goodput=[iv[(rr, 12000.0)] for rr in RS],
        over_goodput_at_median_r=[iv[(0.18, gp)] for gp in GOODPUTS],
        median_goodput=12000.0, median_r=0.18,
        r_range=[min(RS), max(RS)], goodput_range=[min(GOODPUTS), max(GOODPUTS)])
    a_, b_ = spread["over_r_at_median_goodput"], spread["over_goodput_at_median_r"]
    print(f"interval across r: {min(a_):.1f} to {max(a_):.1f} d; "
          f"across goodput: {min(b_):.1f} to {max(b_):.1f} d")
    return rows, spread


def link(t_star):
    """The joint optimum as the payload and the per-attempt success probability vary."""
    pay, ps = [], []
    for B in PAYLOADS:
        pay.append(_row(R.optimise(**dict(R.BASE, payload=B)), t_star, payload=B))
    for p in P_SUCCESS:
        ps.append(_row(R.optimise(**dict(R.BASE, p_success=p)), t_star, p_success=p))
    for name, rows in (("payload", pay), ("p_success", ps)):
        print(f"  {name}: interval {min(d['interval'] for d in rows):.1f} to "
              f"{max(d['interval'] for d in rows):.1f} d, largest threshold departure "
              f"{max(abs(d['departure']) for d in rows):.1e}")
    return pay, ps


def weights(t_star):
    """The joint optimum with each weight scaled, one at a time.

    The weights are module globals in risk.py. They are restored afterwards even
    if an optimisation fails, or every later number would be computed with the
    wrong weights and nothing would say so.
    """
    w0 = (R.C_LOSS, R.C_ENERGY, R.C_DATA)
    out = []
    try:
        for name, mult in WEIGHT_CASES:
            R.C_LOSS, R.C_ENERGY, R.C_DATA = (w * m for w, m in zip(w0, mult))
            oo = R.optimise(**R.BASE)
            out.append(_row(oo, t_star, name=name, multipliers=list(mult)))
            print(f"  {name:<12} theta {oo['theta']:.4f}  interval "
                  f"{oo['interval_days']:>5.1f} d")
    finally:
        R.C_LOSS, R.C_ENERGY, R.C_DATA = w0
    return out


def main():
    """Run the four parts and write them to one file."""
    t_star = analytic_theta(R.BASE["cloud_recall"], R.BASE["prior"])
    Z = {"joint_value": joint_value()}
    Z["regime"], Z["regime_spread"] = regime(t_star)
    Z["payload_sweep"], Z["p_success_sweep"] = link(t_star)
    Z["weight_sensitivity"] = weights(t_star)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "value.json"), "w") as fh:
        json.dump(Z, fh, indent=1)
    print("wrote results/value.json")


if __name__ == "__main__":
    main()
