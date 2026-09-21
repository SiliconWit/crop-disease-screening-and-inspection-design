"""The three building blocks on their own, before anything is composed.

Everything here is closed form or nearly so, which makes it the cheapest place to
catch a units slip. Nothing is optimised.

    python3 run_floor.py             # writes ../results/floor.json
"""
import json, os
import screening as S
import uplink as U
import epidemic as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")

PRIOR = 0.15
PROBE_THETAS = [0.10, 0.20, 0.30, 0.40]
BASE_LINK = dict(payload_bytes=25000, goodput_bps=20000.0, p_success=0.85)


def main():
    """Screening, uplink and epidemic, each reported at fixed probe values.

    The probe values are fixed in advance and are not optima, so the file reports
    the same quantities whatever the optimisers later find.
    """
    Z = {"config": dict(prior=PRIOR, healthy=dict(S.HEALTHY), diseased=dict(S.DISEASED),
                        link=BASE_LINK, v_supply=U.V_SUPPLY, i_idle=U.I_IDLE,
                        i_active=U.I_ACTIVE, i_tx_mean=U.I_TX_MEAN, t_attach=U.T_ATTACH)}

    # screening
    Z["lr_monotone"] = S.is_lr_monotone()
    Z["probes"] = [dict(theta=t, **S.operating_point(t, PRIOR)) for t in PROBE_THETAS]
    th = S.threshold_for_miss_rate(0.02, PRIOR)
    Z["at_2pct_miss"] = dict(theta=th, **S.operating_point(th, PRIOR))
    print(f"likelihood ratio monotone: {Z['lr_monotone']}")
    print(f"threshold for a 2% edge miss rate: {th:.4f}, "
          f"offload {Z['at_2pct_miss']['offload_rate']:.4f}")

    # uplink
    c = U.offload_cost(**BASE_LINK)
    d = U.duty_cycle_cost(10.0, 0.2, BASE_LINK["payload_bytes"], BASE_LINK["goodput_bps"],
                          BASE_LINK["p_success"], 20)
    Z["offload"] = c
    Z["duty_probe"] = dict(interval_days=10.0, offload_rate=0.2, captures=20, **d)
    print(f"one offload: {c['time_s']:.2f} s, {c['energy_J']:.2f} J; "
          f"probe schedule: {d['controllable_J_per_day']:.2f} J/day controllable, "
          f"{d['idle_J_per_day']:.2f} J/day asleep")

    # epidemic
    Z["epidemic"] = dict(
        days_to_5pct={str(r): float(E.time_to_severity(0.05, r=r)) for r in (0.15, 0.30, 0.45)},
        severity_far_out=float(E.severity(1000.0, r=0.45)),
        delay_T10_p08=float(E.detection_delay(10.0, 0.8)),
        delay_T10_p1=float(E.detection_delay(10.0, 1.0)))
    print(f"days to 5% severity at r = 0.15: {Z['epidemic']['days_to_5pct']['0.15']:.2f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "floor.json"), "w") as fh:
        json.dump(Z, fh, indent=1)
    print("wrote results/floor.json")


if __name__ == "__main__":
    main()
