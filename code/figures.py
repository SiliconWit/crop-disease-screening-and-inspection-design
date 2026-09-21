"""Every figure, drawn from the results files. No number is retyped.

Each figure is drawn at the width it is printed.

    python3 figures.py               # writes ../results/figs/*.pdf
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figs")
NAVY, GOLD, SLATE, RED, MOSS = "#0F284D", "#B0892C", "#5A6473", "#8C2F1F", "#3E6B4F"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 200,
    "savefig.bbox": "tight"})


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name)); plt.close(fig); print(name)


def fig_optima(A, I):
    """Optimal interval and optimal threshold against r, one curve per goodput.

    The threshold from the stationarity condition is drawn on the threshold panel,
    and that panel's vertical range is fixed deliberately: an automatic range would
    make a change of one part in a million look like a trend.
    """
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    gps = sorted({d["goodput"] for d in A["regime"]})
    rs = sorted({d["r"] for d in A["regime"]})
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(gps)))
    for c, gp in zip(cmap, gps):
        row = [d for d in A["regime"] if d["goodput"] == gp]
        row.sort(key=lambda d: d["r"])
        ax[0].plot([d["r"] for d in row], [d["interval"] for d in row],
                   color=c, marker="o", ms=2.6, label=f"{gp / 1000:.0f} kbps")
        ax[1].plot([d["r"] for d in row], [d["theta"] for d in row],
                   color=c, marker="o", ms=2.6)
    ax[1].axhline(I["analytic_theta"], color=RED, ls="--", lw=1.0)
    ax[1].text(rs[-1], I["analytic_theta"] + 0.012,
               "stationarity condition, which contains\nneither $r$ nor the link",
               color=RED, fontsize=6.4, ha="right")
    ax[0].set_xlabel("apparent infection rate $r$ (per day)")
    ax[0].set_ylabel("optimal interval (days)")
    ax[0].set_title("the interval responds")
    ax[0].legend(frameon=False, fontsize=6.2, ncol=2, handlelength=1.1,
                 title="goodput", title_fontsize=6.2)
    ax[1].set_xlabel("apparent infection rate $r$ (per day)")
    ax[1].set_ylabel(r"optimal threshold $\theta$")
    ax[1].set_ylim(I["analytic_theta"] - 0.1, I["analytic_theta"] + 0.1)
    ax[1].set_title("the threshold does not")
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig1_invariance.pdf")


def fig_densities(A):
    """Optimal threshold, interval and volume saving against density overlap.

    Pairs whose joint optimum lies on a bound of the search are corners, drawn as
    open markers in the first two panels: their threshold and interval are set by
    the search region, not by the densities. The volume saving at a fixed miss rate
    depends only on the threshold that gives that miss rate, so it is valid for
    every pair.
    """
    g = sorted(A["density_sensitivity"], key=lambda d: d["bhattacharyya"])
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.3))
    inner = [d for d in g if not d["on_bound"]]
    corner = [d for d in g if d["on_bound"]]
    miss = f"{100 * A['target_miss']:.0f}"
    for a, key, lab in ((ax[0], "theta", r"optimal threshold $\theta$"),
                        (ax[1], "interval", "optimal interval (days)")):
        a.plot([d["bhattacharyya"] for d in inner], [d[key] for d in inner],
               marker="o", ms=3, ls="none", color=NAVY, label="interior optimum")
        a.plot([d["bhattacharyya"] for d in corner], [d[key] for d in corner],
               marker="o", ms=3.4, ls="none", mfc="white", mec=RED, mew=0.9,
               label="interval on search bound")
        a.set_xlabel("density overlap (Bhattacharyya)")
        a.set_ylabel(lab)
    ax[0].legend(frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.3)
    ax[2].plot([d["bhattacharyya"] for d in g], [d["at_target_volume_cut"] for d in g],
               marker="o", ms=3, ls="none", color=NAVY)
    ax[2].set_xlabel("density overlap (Bhattacharyya)")
    ax[2].set_ylabel(f"volume cut at {miss}% miss")
    ax[2].set_ylim(0, 1.05)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig2_densities.pdf")


def fig_surface(A):
    """The objective over the design plane, and the share of each term at the optimum.

    The logarithm is contoured, since the cost spans orders of magnitude across the
    plane, and a share too small to hold its own text is labelled outside its bar.
    """
    import screening as S, risk as R
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    base = A["base"]
    ths = np.linspace(0.06, 0.60, 90)
    ivs = np.linspace(1.5, 40.0, 90)
    Z = np.array([[R.risk(t, i, **base) for i in ivs] for t in ths])
    cs = ax[0].contourf(ivs, ths, np.log10(Z), levels=22, cmap="viridis")
    ax[0].plot(A["optimum"]["interval_days"], A["optimum"]["theta"], marker="*",
               ms=11, color="white", mec=RED, mew=0.9)
    ax[0].set_xlabel("inspection interval (days)")
    ax[0].set_ylabel(r"screening threshold $\theta$")
    ax[0].set_title("expected daily cost, $\\log_{10}$")
    fig.colorbar(cs, ax=ax[0], pad=0.02, fraction=0.05)

    o = A["optimum"]
    parts = [("late detection", o["share_loss"], NAVY),
             ("airtime", o["share_data"], GOLD),
             ("energy", o["share_energy"], MOSS)]
    left = 0.0
    for lab, v, col in parts:
        ax[1].barh([0], [v], left=[left], color=col, edgecolor="none", height=0.5)
        if v > 0.05:
            ax[1].text(left + v / 2, 0, f"{lab}\n{100 * v:.0f}%", ha="center",
                       va="center", color="white", fontsize=7)
        left += v
    ax[1].text(left, 0.42, f"energy {100 * o['share_energy']:.1f}%", ha="right",
               fontsize=6.6, color=MOSS)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(-0.6, 0.8)
    ax[1].set_yticks([]); ax[1].set_xlabel("share of the objective at the optimum")
    ax[1].set_title("why link speed barely matters")
    for s in ("left", "right", "top"):
        ax[1].spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig3_surface.pdf")


def main():
    """Draw the three figures from analysis.json and invariance.json."""
    with open(os.path.join(RES, "analysis.json")) as fh:
        A = json.load(fh)
    with open(os.path.join(RES, "invariance.json")) as fh:
        I = json.load(fh)
    fig_optima(A, I)
    fig_densities(A)
    fig_surface(A)


if __name__ == "__main__":
    main()
