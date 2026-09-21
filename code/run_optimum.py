"""The joint optimum, what it is made of, and the operating point at a fixed edge
miss rate.

Three designs are reported, each compared with offloading everything at the same
interval:

  * the unconstrained joint optimum at the base parameters;
  * the threshold set for a 2 percent edge miss rate, with the interval re-tuned
    for that threshold at the base infection rate (the reported operating point);
  * the same threshold at r = 0.30, with the interval held at the joint optimum
    for r = 0.30 rather than re-tuned, reported for comparison.

    python3 run_optimum.py           # writes ../results/optimum.json
"""
import json, os
import screening as S
import risk as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")

ALT_R = 0.30


def main():
    """Optimum, its term balance, the baseline, and the 2 percent operating points.

    The share of each term at the optimum matters: an objective whose minimum is
    dominated by one term is really an optimisation of that term, and whatever
    does not enter it cannot move the answer.
    """
    BASE = R.BASE
    Z = {"base": BASE, "target_miss": R.TARGET_MISS,
         "conventional_interval": R.CONVENTIONAL_INTERVAL,
         "lr_monotone": S.is_lr_monotone(),
         "weights": dict(loss=R.C_LOSS, energy=R.C_ENERGY, data=R.C_DATA),
         "captures_per_visit": R.CAPTURES_PER_VISIT,
         "search_bounds": dict(theta=list(R.THETA_BOUNDS),
                               interval=list(R.INTERVAL_BOUNDS))}

    o = R.optimise(**BASE)
    b = R.unconditional_baseline(o["interval_days"], **BASE)
    o["volume_cut"] = 1 - o["bytes_per_day"] / b["bytes_per_day"]
    o["energy_cut"] = 1 - o["energy_J_per_day"] / b["energy_J_per_day"]
    o["extra_delay"] = o["delay_days"] - b["delay_days"]
    o["cost_loss"] = R.C_LOSS * o["loss"]
    o["cost_energy"] = R.C_ENERGY * o["energy_J_per_day"]
    o["cost_data"] = R.C_DATA * o["bytes_per_day"]
    # controllable energy spent per inspection visit, the unit a per-visit cost
    # must be compared in
    o["energy_J_per_visit"] = o["energy_J_per_day"] * o["interval_days"]
    tot = o["cost_loss"] + o["cost_energy"] + o["cost_data"]
    o["share_loss"] = o["cost_loss"] / tot
    o["share_energy"] = o["cost_energy"] / tot
    o["share_data"] = o["cost_data"] / tot
    Z["optimum"] = o
    Z["baseline_at_optimum"] = b
    print(f"joint optimum: theta {o['theta']:.4f}, interval {o['interval_days']:.2f} d, "
          f"miss {o['miss_rate']:.4f}, offload {o['offload_rate']:.4f}")
    print(f"  term shares: loss {o['share_loss']:.3f}, energy {o['share_energy']:.4f}, "
          f"airtime {o['share_data']:.3f}; {o['energy_J_per_visit']:.1f} J per visit")

    th2 = S.threshold_for_miss_rate(R.TARGET_MISS, BASE["prior"])
    i2, _ = R.opt_interval_given(th2, **BASE)
    op2 = R.risk(th2, i2, **BASE, verbose=True)
    b2 = R.unconditional_baseline(i2, **BASE)
    op2.update(theta=th2, interval_days=i2, on_bound=R.on_bound(th2, i2),
               volume_cut=1 - op2["bytes_per_day"] / b2["bytes_per_day"],
               extra_delay=op2["delay_days"] - b2["delay_days"])
    Z["at_target_miss"] = op2
    print(f"at {R.TARGET_MISS:.0%} miss: theta {th2:.4f}, interval {i2:.2f} d, "
          f"offload {op2['offload_rate']:.4f}, volume cut {op2['volume_cut']:.4f}, "
          f"extra delay {op2['extra_delay']:+.2f} d")

    kw = dict(BASE, r=ALT_R)
    o3 = R.optimise(**kw)
    i3 = o3["interval_days"]
    op3 = R.risk(th2, i3, **kw, verbose=True)
    b3 = R.unconditional_baseline(i3, **kw)
    op3.update(r=ALT_R, theta=th2, interval_days=i3,
               volume_cut=1 - op3["bytes_per_day"] / b3["bytes_per_day"],
               extra_delay=op3["delay_days"] - b3["delay_days"])
    Z["at_target_miss_alt"] = op3
    print(f"at {R.TARGET_MISS:.0%} miss, r = {ALT_R}, interval of the joint optimum "
          f"{i3:.2f} d: extra delay {op3['extra_delay']:+.2f} d, "
          f"volume cut {op3['volume_cut']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "optimum.json"), "w") as fh:
        json.dump(Z, fh, indent=1)
    print("wrote results/optimum.json")


if __name__ == "__main__":
    main()
