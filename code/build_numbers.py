"""Turn the results files into LaTeX macros, so no number is typed by hand.

Every reported value is a macro written here from a results file, and
rounding happens here and nowhere else. Where a quantity is a range over a sweep,
the sweep it comes from is part of the macro name, and optima that lie on a bound
of the search are separated from interior ones before a range is taken. The
statements made about the results (no corner in the parameter sweeps,
corners exactly where the condition has no root, a single root at the base
densities) are checked here, and the build stops if one of them fails.

    python3 build_numbers.py         # writes ../results/numbers.tex and verify_table.tex
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
L = []


def cmd(n, v):
    L.append(r"\newcommand{\%s}{%s}" % (n, v))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sci(x, digits=0):
    """A magnitude in scientific notation, typeset as maths: 1.3e-06 -> 1.3 x 10^-6."""
    if x == 0:
        return r"\ensuremath{0}"
    e = int(np.floor(np.log10(abs(x))))
    m = abs(x) / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return r"\ensuremath{%.*f\times 10^{%d}}" % (digits, m, e)


def esci(x):
    """A signed error in scientific notation, the way a table should show one."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if abs(round(m)) >= 10:
        m, e = m / 10, e + 1
    return r"$%+.0f\times 10^{%d}$" % (m, e)


def rng(rows, key="interval", d=1):
    v = [r[key] for r in rows]
    return f"{min(v):.{d}f}", f"{max(v):.{d}f}"


def check(ok, what):
    if not ok:
        raise SystemExit(f"build_numbers: expected {what}, but the results do not show it")


def main():
    with open(os.path.join(RES, "analysis.json")) as fh:
        A = json.load(fh)
    with open(os.path.join(RES, "invariance.json")) as fh:
        I = json.load(fh)

    # --- base case ---------------------------------------------------------
    b = A["base"]
    cmd("Prior", pct(b["prior"], 0)); cmd("Rbase", f"{b['r']:.2f}")
    cmd("Goodput", f"{b['goodput'] / 1000:.0f}"); cmd("Payload", f"{b['payload'] / 1000:.0f}")
    cmd("Psuccess", f"{b['p_success']:.2f}"); cmd("Recall", f"{b['cloud_recall']:.2f}")
    cmd("Captures", A["captures_per_visit"])
    cmd("TargetMiss", pct(A["target_miss"], 0))
    cmd("ConvInterval", f"{A['conventional_interval']:.0f}")
    cmd("LrMonotone", "monotone" if A["lr_monotone"] else "NOT MONOTONE")
    w = A["weights"]
    cmd("Wloss", f"{w['loss']:g}"); cmd("Wenergy", f"{w['energy']:g}")
    cmd("Wdata", f"{w['data']:g}")
    iv_lo, iv_hi = A["search_bounds"]["interval"]
    cmd("IvBoundLo", f"{iv_lo:g}"); cmd("IvBoundHi", f"{iv_hi:g}")

    # --- the joint optimum and the operating points --------------------------
    o = A["optimum"]
    check(not o["on_bound"], "the base optimum is interior")
    cmd("OptTheta", f"{o['theta']:.3f}"); cmd("OptInterval", f"{o['interval_days']:.1f}")
    cmd("OptOffload", pct(o["offload_rate"])); cmd("OptMiss", pct(o["miss_rate"]))
    cmd("OptDelay", f"{o['delay_days']:.1f}")
    cmd("OptVolumeCut", pct(o["volume_cut"], 0))
    cmd("OptExtraDelay", f"{o['extra_delay']:.1f}")
    cmd("ShareLoss", pct(o["share_loss"], 0))
    cmd("ShareData", pct(o["share_data"], 0))
    cmd("ShareEnergy", pct(o["share_energy"], 1))
    cmd("EnergyDay", f"{o['energy_J_per_day']:.1f}")
    cmd("EnergyVisit", f"{o['energy_J_per_visit']:.0f}")
    cmd("IdleDay", f"{o['idle_J_per_day']:.1f}")

    t = A["at_target_miss"]
    check(not t["on_bound"], "the interval of the operating point is interior")
    cmd("TgtTheta", f"{t['theta']:.3f}"); cmd("TgtInterval", f"{t['interval_days']:.1f}")
    cmd("TgtOffload", pct(t["offload_rate"], 0))
    cmd("TgtVolumeCut", pct(t["volume_cut"], 0))
    cmd("TgtExtraDelay", f"{t['extra_delay']:.1f}")
    t3 = A["at_target_miss_alt"]
    cmd("AltR", f"{t3['r']:.2f}"); cmd("AltInterval", f"{t3['interval_days']:.1f}")
    cmd("AltExtraDelay", f"{t3['extra_delay']:.1f}")
    cmd("AltVolumeCut", pct(t3["volume_cut"], 0))

    j = A["joint_value"]
    cmd("GainOverThetaOnly", pct(j["gain_over_theta_only"], 1))
    cmd("GainOverIntervalOnly", pct(j["gain_over_interval_only"], 1))
    cmd("CostOverThetaOnly", pct(j["excess_theta_only"], 1))
    cmd("CostOverIntervalOnly", pct(j["excess_interval_only"], 1))

    # --- the parameter sweeps, each range attributed to its own sweep --------
    t_star = I["analytic_theta"]
    s = A["regime_spread"]
    a_, b_ = s["over_r_at_median_goodput"], s["over_goodput_at_median_r"]
    cmd("IvRlo", f"{min(a_):.1f}"); cmd("IvRhi", f"{max(a_):.1f}")
    cmd("IvRfactor", f"{max(a_) / min(a_):.1f}")
    cmd("IvRgoodput", f"{s['median_goodput'] / 1000:.0f}")
    cmd("IvGlo", f"{min(b_):.1f}"); cmd("IvGhi", f"{max(b_):.1f}")
    cmd("IvGfactor", f"{max(b_) / min(b_):.2f}")
    cmd("IvGr", f"{s['median_r']:.2f}")
    cmd("GoodputFactor", f"{s['goodput_range'][1] / s['goodput_range'][0]:.0f}")
    cmd("GoodputLo", f"{s['goodput_range'][0] / 1000:.0f}")
    cmd("GoodputHi", f"{s['goodput_range'][1] / 1000:.0f}")
    cmd("Rlo", f"{s['r_range'][0]:.2f}"); cmd("Rhi", f"{s['r_range'][1]:.2f}")

    reg, pay, ps, wts = (A["regime"], A["payload_sweep"], A["p_success_sweep"],
                         A["weight_sensitivity"])
    sweeps = reg + pay + ps + wts
    check(not any(d["on_bound"] for d in sweeps),
          "no optimum in the regime, payload, success-probability or weight sweeps "
          "lies on a search bound")
    lo, hi = rng(reg); cmd("RegimeIvLo", lo); cmd("RegimeIvHi", hi)
    cmd("NRegime", len(reg))
    lo, hi = rng(pay); cmd("PayloadIvLo", lo); cmd("PayloadIvHi", hi)
    pv_ = [d["payload"] for d in pay]
    cmd("PayloadLo", f"{min(pv_) / 1000:.0f}"); cmd("PayloadHi", f"{max(pv_) / 1000:.0f}")
    cmd("PayloadFactor", f"{max(pv_) / min(pv_):.0f}")
    lo, hi = rng(ps); cmd("PsIvLo", lo); cmd("PsIvHi", hi)
    pp = [d["p_success"] for d in ps]
    cmd("PsLo", f"{min(pp):.2f}"); cmd("PsHi", f"{max(pp):.2f}")
    lo, hi = rng(wts); cmd("WeightIvLo", lo); cmd("WeightIvHi", hi)
    cmd("WeightFactor", f"{max(max(d['multipliers']) for d in wts):.0f}")
    lo, hi = rng(sweeps); cmd("SweepIvLo", lo); cmd("SweepIvHi", hi)
    cmd("NSweep", len(sweeps))
    cmd("SweepThetaErr", sci(max(abs(d["departure"]) for d in sweeps), 1))

    # --- the tight optimiser: loss forms, per-visit cost, recall -------------
    cmd("AnalyticTheta", f"{t_star:.4f}")
    roots = I["base_roots"]
    check(len(roots) == 1 and abs(roots[0] - t_star) < 1e-9,
          "the condition has a single root at the base densities")
    cmd("LossFormErr", sci(max(abs(d["error"]) for d in I["loss_forms"])))
    lo, hi = rng(I["loss_forms"]); cmd("LossFormIvLo", lo); cmd("LossFormIvHi", hi)
    cmd("NLossForms", len(I["loss_forms"]))
    pv = {d["per_visit_J"]: d for d in I["per_visit"]}
    small, large = sorted(k for k in pv if k > 0)[0], max(pv)
    cmd("PvSmallJ", f"{small:.0f}"); cmd("PvLargeJ", f"{large:.0f}")
    cmd("PvRatio", f"{large / small:.0f}")
    cmd("PvSmall", f"{pv[small]['theta']:.4f}")
    cmd("PvSmallErr", sci(abs(pv[small]["error"])))
    cmd("PvLarge", f"{pv[large]['theta']:.3f}")
    cmd("PvLargeErr", f"{abs(pv[large]['error']):.3f}")
    cmd("PvSmallShare", pct(small / o["energy_J_per_visit"], 0))
    rec = I["recall"]
    cmd("RecallLo", f"{min(d['recall'] for d in rec):.2f}")
    cmd("RecallHi", f"{max(d['recall'] for d in rec):.2f}")
    cmd("RecallThetaHi", f"{max(d['numerical'] for d in rec):.3f}")
    cmd("RecallThetaLo", f"{min(d['numerical'] for d in rec):.3f}")
    cmd("RecallErr", sci(max(abs(d["numerical"] - d["analytic"]) for d in rec)))

    # --- the density sweep: corners apart from interior optima ---------------
    g = A["density_sensitivity"]
    inner = [d for d in g if not d["on_bound"]]
    corner = [d for d in g if d["on_bound"]]
    cmd("DensN", len(g)); cmd("DensNInterior", len(inner)); cmd("DensNCorner", len(corner))
    cmd("OverlapLo", f"{min(d['bhattacharyya'] for d in g):.2f}")
    cmd("OverlapHi", f"{max(d['bhattacharyya'] for d in g):.2f}")
    ov_in = max(d["bhattacharyya"] for d in inner)
    ov_co = min(d["bhattacharyya"] for d in corner)
    check(ov_co > ov_in, "the corners are the most overlapping pairs")
    cmd("OverlapInteriorHi", f"{ov_in:.2f}"); cmd("OverlapCornerLo", f"{ov_co:.2f}")
    check(all(abs(d["interval"] - iv_lo) < 1e-6 for d in corner),
          "every corner has its interval on the lower bound of the search")
    lo, hi = rng(inner, "theta", 2); cmd("DensThetaLo", lo); cmd("DensThetaHi", hi)
    lo, hi = rng(inner, "interval", 1); cmd("DensIvLo", lo); cmd("DensIvHi", hi)
    lo, hi = rng(corner, "theta", 2); cmd("DensCornerThetaLo", lo); cmd("DensCornerThetaHi", hi)
    cmd("DensCutLo", pct(min(d['at_target_volume_cut'] for d in g), 0))
    cmd("DensCutHi", pct(max(d['at_target_volume_cut'] for d in g), 0))

    cc = A["condition_check"]
    check(all((not c["roots"]) == c["corner"] for c in cc),
          "the condition has no root for exactly the pairs whose optimum is a corner")
    cmd("DensRootErr", sci(max(abs(c["departure"]) for c in cc if not c["corner"]), 1))
    multi = [c for c in cc if len(c["roots"]) > 1]
    cmd("DensNMultiRoot", len(multi))
    extra = [r for c in multi for r in c["root_details"] if r["theta"] != c["nearest_root"]]
    check(all(abs(c["nearest_root"] - c["roots"][0]) < 1e-12 for c in multi),
          "at every pair with several roots the optimum sits on the lowest")
    cmd("ExtraRootLo", f"{min(r['theta'] for r in extra):.2f}")
    cmd("ExtraRootHi", f"{max(r['theta'] for r in extra):.2f}")
    cmd("ExtraRootPsend", sci(max(r["p_send_diseased"] for r in extra), 1))
    cmd("ExtraRootDelay", f"{min(r['delay_at_shortest_interval'] for r in extra):.0f}")
    cmd("ExtraRootLoss", f"{min(r['loss_at_shortest_interval'] for r in extra):.2f}")

    # --- the verification table --------------------------------------------
    # Generated whole rather than assembled from macros, because a table body
    # built out of \input inside a tabular breaks \multicolumn.
    rows = [r"\multicolumn{4}{@{}l}{\emph{loss function, with everything else fixed}}\\"]
    for d in I["loss_forms"]:
        rows.append(r"\quad %s & %.6f & %.2f & %s \\" %
                    (d["name"], d["theta"], d["interval"], esci(d["error"])))
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{4}{@{}l}{\emph{cloud recall, the one quantity that should move it}}\\")
    for d in rec:
        rows.append(r"\quad $\rho = %.2f$ & %.6f & %.2f & %s \\" %
                    (d["recall"], d["numerical"], d["interval"],
                     esci(d["numerical"] - d["analytic"])))
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{4}{@{}l}{\emph{a cost per visit, which the proof says should break it}}\\")
    for d in I["per_visit"]:
        rows.append(r"\quad $A = %.0f$ J per visit & %.6f & %.2f & %s \\" %
                    (d["per_visit_J"], d["theta"], d["interval"], esci(d["error"])))

    table = (r"""\begin{table}[htbp]
\centering\footnotesize
\caption{Numerical joint optima from the multi-start optimiser. The final column is the
distance from the threshold given by condition~\eqref{eq:cond}, which for the recall block
is recomputed at that recall. The interval moves by an order of magnitude; the threshold
moves only where the recall changes or the cost structure violates the proposition's
hypothesis.}
\label{tab:verify}
\begin{tabular}{@{}lrrr@{}}
\toprule
 & optimal $\theta$ & optimal $T$ (days) & departure from \eqref{eq:cond} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "verify_table.tex"), "w") as fh:
        fh.write(table)
    print("wrote results/verify_table.tex")

    with open(os.path.join(RES, "numbers.tex"), "w") as fh:
        fh.write("% generated by code/build_numbers.py; do not edit\n")
        fh.write("\n".join(L) + "\n")
    print(f"wrote results/numbers.tex, {len(L)} macros")


if __name__ == "__main__":
    main()
