# SAQ-Decoder

This repository includes a BANR-SAQ baseline implementation with:

- Shared multiscale renormalization cell (`banr_saq/models/renorm_cell.py`)
- Boundary auto-classification from parity checks (`banr_saq/data/boundary_classifier.py`)
- RG pooling and multiscale encoder (`banr_saq/models/pool_coarsen.py`, `banr_saq/models/banr_saq_encoder.py`)
- Logical, entropy, RG consistency, and syndrome consistency losses (`banr_saq/losses/`)
- CPND post-processing helpers (`banr_saq/postprocess/cpnd.py`)
- Minimal train/eval scripts (`train.py`, `evaluate.py`)

## Installation

Install dependencies before running train/eval:

```bash
pip install -r requirements.txt
```

If dependencies are missing, scripts now raise a clear startup error with install hints.

## Where to choose toric vs rotated

Use the `--family` flag on CLI:

```bash
python train.py --family rotated --distances 5,7,9,11 --p-init 0.05 --p-final 0.2 --n-p 4 --steps 100
python train.py --family toric --distances 7,9,11 --p-init 0.05 --p-final 0.2 --n-p 4 --steps 100

python evaluate.py --family rotated --distance 9 --samples 8
python evaluate.py --family toric --distance 9 --samples 8
```

Internally this is selected in `precompute_distance(..., family=...)`, which calls
`build_css_code(..., family=...)` in `banr_saq/data/code_builder.py`.


## Multi-distance training

Training is distance-mixed by design in `train.py`:

- pass `--distances` as a comma-separated list
- each optimizer step uniformly samples one distance from that list
- each sampled distance uses its own precomputed cache
- error-probability grid is configured via `--p-init`, `--p-final`, `--n-p`

For example, `--p-init 0.05 --p-final 0.2 --n-p 4` gives
`p = [0.05, 0.10, 0.15, 0.20]`, and each step samples uniformly from that grid.

Example:

```bash
python train.py --family rotated --distances 5,7,9,11 --p-init 0.05 --p-final 0.2 --n-p 4 --steps 200
```

## Where data is generated

Data is generated on-the-fly in:

- `banr_saq/data/batch_builder.py::make_single_sample`
- error sampled via `banr_saq/data/noise_sampler.py`
- syndrome computed as `H @ e mod 2`
- logical class computed as `L @ e mod 2`

## Quickstart

```bash
python train.py --family rotated --distances 5,7,9,11 --p-init 0.05 --p-final 0.2 --n-p 4 --steps 10
python evaluate.py --family rotated --distance 7
```

These scripts run single-sample (or tiny-sample) sanity flows on generated CSS-like codes.
