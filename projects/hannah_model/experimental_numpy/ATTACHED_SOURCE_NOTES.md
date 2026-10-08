# Notes on attached training files

- `geocai_numpy_demo.py` runs a NumPy forward-pass demonstration. It does not compute gradients or update weights. It ran in this workspace; its demo accuracy came from random inputs and random labels, so it is only a wiring/forward-pass result.
- `geocai_training_framework.py` is a PyTorch model and trainer, not a NumPy-only training framework.
- `train_with_quinn.py` imports PyTorch, `QUINNEngine`, and `LeechQuantizer`. Its `compute_geodesic_gradient` currently ends with `grad * adapted_lr` and comments that full geodesic computation still needs manifold projection. The QUINN training claims therefore remain a port/reconstruction target, not code executed by this NumPy lab.
- `Trust Manifold Transformer Training1(1).txt` contains the recovered historical demo trainer. It matches the 20-file, 128-token, batch-16, two-epoch setup, but depends on PyTorch and a hard-coded `/home/father/gid-clean/intake_full` path.
- `train_hannah_quinn.py` instead configures three epochs, batch size 4, sequence length 64, and subsamples windows by eight. It computes scheduler multipliers but keeps the DataLoader batch size and optimizer learning rate fixed, so the advertised rapid-feeding changes are not applied by this script. Its model/data-store imports are project-local dependencies absent from this isolated lab.
- `track_ingestion_velocity.py` is a JSON convergence-file monitor, not a trainer or model validator. It labels a character count as bytes and tracks a single max iteration across two arms, which can hide arm-specific progress changes.

These source files were left unchanged. The NumPy mini-model is a new, isolated experimental branch.
