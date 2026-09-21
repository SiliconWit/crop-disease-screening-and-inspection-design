"""A renewal model of the 2G uplink, and what one offload costs.

The SIM800L is GPRS multislot class 12, so its nominal ceiling is 85.6 kbps. A
loaded rural cell delivers a fraction of that, so goodput is an input here rather
than a constant, and it is swept in run_value.py.
"""
import numpy as np

NOMINAL_BPS = 85_600.0      # GPRS class 12 ceiling, SIM800L datasheet

# Representative SIM800L figures, assumed from the datasheet rather than measured. The
# datasheet gives the peak transmit current; the model uses the mean over a burst.
V_SUPPLY = 4.0              # V
I_IDLE = 5.0e-5             # A, deep sleep with the modem powered down between
                            #    visits. 2 mA is the figure for a registered modem
                            #    and is what a node looks like if power to the
                            #    modem is never cut.
I_ACTIVE = 0.100            # A, modem awake, not transmitting
I_TX_MEAN = 0.350           # A, mean over a transmit burst, not the 2 A peak
T_ATTACH = 3.0              # s, PDP context activation when starting from idle


def offload_cost(payload_bytes, goodput_bps, p_success=0.85, attach=True):
    """Expected time and energy for one image offload.

    Retransmissions are geometric with success probability p_success per attempt,
    so the expected number of attempts is 1/p_success, and each failed attempt
    costs a full transmission before the next begins. Goodput is in bits per
    second and the payload in bytes. The attach time is spent awake but not
    transmitting.
    """
    bits = 8.0 * payload_bytes
    t_one = bits / goodput_bps
    attempts = 1.0 / p_success
    t_tx = attempts * t_one
    t_total = t_tx + (T_ATTACH if attach else 0.0)
    e = V_SUPPLY * (I_TX_MEAN * t_tx + I_ACTIVE * (T_ATTACH if attach else 0.0))
    return dict(time_s=t_total, energy_J=e, attempts=attempts,
                bytes_on_air=payload_bytes * attempts)


def duty_cycle_cost(interval_days, offload_rate, payload_bytes, goodput_bps,
                    p_success=0.85, captures_per_inspection=1):
    """Energy per day and bytes per day for a given capture schedule.

    The node wakes every `interval_days`, takes `captures_per_inspection` images,
    screens them locally, and offloads a fraction `offload_rate` of them. The sleep
    budget is reported separately from the part the two design variables control:
    sleep depends on neither, and an optimiser handed the total would be minimising
    a constant with a small correction on top.
    """
    per_visit = captures_per_inspection
    sent = per_visit * offload_rate
    c = offload_cost(payload_bytes, goodput_bps, p_success)
    e_visit = sent * c["energy_J"]
    b_visit = sent * c["bytes_on_air"]
    idle_J = V_SUPPLY * I_IDLE * interval_days * 86400.0
    return dict(energy_J_per_day=(e_visit + idle_J) / interval_days,
                controllable_J_per_day=e_visit / interval_days,
                idle_J_per_day=idle_J / interval_days,
                bytes_per_day=b_visit / interval_days,
                latency_s=c["time_s"],
                sent_per_visit=sent)
