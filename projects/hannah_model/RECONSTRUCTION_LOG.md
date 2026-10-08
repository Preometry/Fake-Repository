# Reconstruction log

## 2026-10-04 — Trust-Manifold training reconstruction

- Recovered `trust_aware_attention.py`, `hannah_embedding.py`, `transformer_lm.py`, `transformer_lm_trust.py`, `test_trust_hannah.py`, `train_trust_manifold.py`, and `demo_train_fast.py` from embedded file writes in the `Hannah.py` JSONL archive. Exact snapshots are retained separately and hashed in `archive/SHA256SUMS`.
- The canonical model modules are copied from the recovered snapshots without changes. The original full trainer has a syntax error in its tqdm postfix expression, fixed only in the new reconstructed entry point; the original remains archived.
- The reconstructed runner keeps the logged demo settings, makes the corpus directory configurable, validates the source file count and resulting vocabulary, saves model/optimizer/vocabulary state after each epoch, supports periodic checkpoints, and writes batch and epoch loss records.
- The transcript records 408,359 characters, vocabulary 133, 3,062,016 parameters, and 25,515 batches/epoch for the demo configuration. It also contains separate corpus-size statements (118 files / 1.66M characters) that cannot be reconciled without the historical corpus manifest.
- Verification in this runtime: Python syntax compilation and independent NumPy geodesic identity checks. PyTorch is not installed, and the historical theorem corpus is absent, so the recovered neural test suite and two-epoch training have not run here.
- Independent geometry checks confirm symmetry, guarded endpoint behavior, content sensitivity, and the `softmax(-d^2/T)` weight transform. They also confirm cosine and geodesic scores preserve nearest-neighbor ranking on normalized vectors while producing different weight distributions. The recovered PyTorch implementation clamps to `[-1+1e-7, 1-1e-7]`, so its self-distance is slightly above zero and antipodal distance slightly below π.
- Next evidence needed for a faithful training reproduction: the exact 20 theorem YAML files (or a hash manifest and matching files), any synthesis files actually included, historical random seed / library versions, and full epoch logs. A checkpoint was not found in the archive.

## Evidence labels

- Historical source snapshots: `exact` (embedded code content recovered from transcript).
- `src/hannah_model/`: `exact` copies of the recovered module snapshots.
- `scripts/train_historical.py`: `reconstructed`; it preserves the recorded model/data/training settings while fixing paths and adding checkpointing/metrics.
- NumPy geodesic checks: `extended` diagnostics; they validate the arccos formula independently of the neural network.
- Two-epoch contemporary training and historical loss reproduction: `blocked` pending PyTorch and the source corpus.
