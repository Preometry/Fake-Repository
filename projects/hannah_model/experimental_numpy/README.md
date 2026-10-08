# Hannah NumPy training lab

This is a separate, PyTorch-free prototype. It does not edit or replace the Fake-Repository, the recovered Trust-Manifold source, or the attached files.

It includes a small NumPy reverse-mode autodiff engine, Adam, and a configurable mini Hannah character model. The model retains the recovered 22-channel remainder mapping with `r=23` excluded, learned channel/position/winding tables, remainder projection, trust-distance channel mixing, normalized spherical projections, `acos(clamp(q·k))` geodesic attention, causal masking, residual feed-forward blocks, and tied input/output token weights.

Adam is the baseline update rule here. The QUINN integration remains a separate next port: the supplied `train_with_quinn.py` imports PyTorch plus external `QUINNEngine` and Leech modules, and its gradient method itself labels the full geodesic projection as requiring additional implementation. This prototype does not present that integration as already reproduced.

The miniature smoke runner defaults to 16D, one block, two heads, 8D sphere, and sequence length 16. The historical regimen runner uses the recovered 256D, four-layer, four-head, 128-token shape and matches the recovered **3,062,016** trainable parameter count. It is an independently implemented training path, not yet a historical reproduction: this version's GELU uses the tanh approximation, the source corpus is missing here, and a full-scale NumPy run is expensive.

## Test the engine

```bash
python3 test_numpy_lab.py
```

The checks compare autodiff gradients (including a model geodesic-attention parameter) to finite differences, verify repeated embedding-index gradients, check the historical parameter count, validate causal geodesic attention, run model forward/backward with finite gradients, and train a tiny periodic character task until its loss falls.

A one-batch run at the full historical shape also completed without PyTorch: 3,062,016 parameters, batch 16 × 128, finite gradients and one Adam update. It took 3.022 seconds and peaked at about 1.5 GiB RSS on this runtime. The rough two-epoch projection is 42.8 hours; see `BENCHMARK.md` for conditions and limits.

## Run the complete two-epoch baseline

```bash
python3 train_regimen.py --data-dir /path/to/intake_full --output-dir run_hannah
```

Defaults match the reported shape and regimen: first 20 sorted `THRM-*.yaml` files, 128-token windows, batch size 16, 256/1024 dimensions, four layers and heads, 20D sphere, Adam at 1e-3, and two full passes over all next-character windows. It records SHA-256 for every source file, corpus counts, every batch loss, epoch summaries, and checkpoints. A resumable checkpoint is written every 1,000 batches by default; continue with `--resume run_hannah/checkpoint_latest.npz`.

The historical 20-file corpus is not present in this workspace, so the full historical run has not been performed here. `historical_comparison.json` extracts the reported checkpoint metrics and marks corpus-count matches; equal counts alone do not establish that the source text is identical. Compare historical metrics only after checking file hashes and tokenization/update details.

## Independent geodesic check

```bash
python3 independent_geodesic_audit.py
```

This recomputes spherical angles from the model's normalized query/key vectors using an independent `atan2` formula, checks causal attention, and compares a geodesic projection gradient against central finite differences. `test_numpy_lab.py` also checks reverse-mode derivatives, the historical parameter count, and a tiny learning task.

## Attached-code observation

`geocai_numpy_demo.py` executes a NumPy forward-pass demo but does not calculate gradients or update model weights. `geocai_training_framework.py` and `train_with_quinn.py` still rely on PyTorch and other external Quinn/Leech modules. The files are retained as source context, not silently treated as an existing PyTorch-free Hannah trainer.
