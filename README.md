# Scout: screening threshold and inspection interval for a crop disease camera node

How a crop disease camera node on a 2G link should set its screening threshold and its
inspection interval, and what each depends on.

## Requirements

Python 3.8 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

## Running

```bash
python3 code/run_all.py
```

`run_all.py` runs, in order, `run_floor.py`, `run_optimum.py`, `run_stationarity.py`,
`run_value.py`, `run_densities.py`, `run_bench.py`, `analyse.py`, `build_numbers.py` and
`figures.py`; each can also be run on its own. The whole sequence takes a few minutes.

## Outputs

Everything is written to `results/`: the JSON results files, `numbers.tex` (every
reported value as a LaTeX macro), `verify_table.tex` and the figures in `results/figs/`.

## Data

The model inputs are stated assumptions, listed in `data/MANIFEST.md` together with the
file formats `run_bench.py` reads for bench measurements.

## Citation

See `CITATION.cff`.

## License

MIT
