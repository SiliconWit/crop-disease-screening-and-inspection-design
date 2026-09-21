# Data

The model has two kinds of input.

**Assumed.** Every constant at the top of `code/screening.py`, `code/uplink.py` and
`code/risk.py`: the two Beta densities of the leaf index, the modem currents, the attach
time, the goodput, the retransmission probability, the cloud recall and the cost weights.
They are stated assumptions, not measurements.

**Measured.** Bench measurements, read by `code/run_bench.py` from `data/bench/` in the
formats below. A file that does not exist is skipped, never replaced with an assumed
value. No bench files are included in this repository.

## Bench file formats

| File | Columns | Contents |
|---|---|---|
| `index_values.csv` | `image_id, label, g, exposure_locked, captured_at` | the leaf index for each labelled image; `label` is `healthy` or a disease name; `exposure_locked` is `true` or `false` |
| `transfers.csv` | `started_at, bytes, seconds, attempts, cell` | one row per image upload; `seconds` is wall clock including retries |
| `burst_current.csv` | `t_s, current_A` | one current capture covering a whole transmission |
| `sleep_current.csv` | `t_s, current_A` | a current capture of the node asleep, with the modem powered down |
| `cloud_eval.csv` | `image_id, true, predicted` | the cloud classifier on a labelled set it was not trained on |
| `latency.csv` | `started_at, bytes, seconds` | end-to-end offload time, capture to cloud response |
