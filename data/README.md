# Training data

Place the historical theorem corpus here as:

```text
data/intake_full/THRM-*.yaml
```

The historical reconstruction uses the first 20 sorted `THRM-*.yaml` files, character-level tokenization, sequence length 128, batch size 16, Adam at 1e-3, and 2 epochs.

The exact historical 20-file manifest has not yet been recovered, so the trainer records the selected file list and SHA-256 hashes for every run.
