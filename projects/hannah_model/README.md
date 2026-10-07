# Hannah model project

This directory adds a standalone Hannah project while leaving the repository's existing files unchanged.

## Components

- The historical reconstruction contains the recovered Transformer, trust-aware attention, Hannah embedding, two-epoch training entry point, historical targets, provenance log, and independent geodesic arithmetic checks.
- `experimental_numpy/` contains a separate small model and reverse-mode differentiation implementation, full-window training regimen, tests, and an independent geodesic audit. It does not require PyTorch.

## Run the historical reconstruction

Install dependencies from `requirements.txt`, then run:

```bash
PYTHONPATH=src python scripts/train_historical.py --data-dir /path/to/gid-clean/intake_full
```

The original theorem corpus is required. The runner writes batch and epoch metrics and resumable checkpoints. See `HISTORICAL_TARGETS.md` and `RECONSTRUCTION_LOG.md` for configuration and evidence limits.

## Run the NumPy experiment

From `experimental_numpy/`:

```bash
python test_numpy_lab.py
python independent_geodesic_audit.py
python train_regimen.py --help
```

The regimen defaults to two full-window epochs. It requires the corpus path and options shown by `--help`. The tiny tests verify implementation behavior; they do not reproduce historical corpus metrics or establish broader geometric claims.

## Evidence status

The historical dataset, complete training logs/checkpoint, and dependencies needed to replay the historical run are not bundled. No full-corpus retraining result is claimed in this project snapshot. Quinn-related attached code is not silently merged into either implementation.
