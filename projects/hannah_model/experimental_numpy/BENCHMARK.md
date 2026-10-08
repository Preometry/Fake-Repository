# NumPy Hannah prototype benchmark

Date: 2026-10-07

## Full historical shape: one synthetic batch

- Architecture dimensions: vocabulary 133, width 256, FFN 1024, 4 layers, 4 heads, sphere dimension 20, sequence length 128.
- Trainable parameter count: **3,062,016**, exactly matching the recovered historical count.
- Batch: 16 sequences of 128 randomly generated token IDs; dropout disabled for this deterministic wiring/gradient check.
- Result: one forward/backward/Adam update completed; loss **4.934627**, all updated parameters finite.
- Elapsed time: **3.022 s** for this batch; peak process RSS: **1,601,320 kB**.

Linear extrapolation at this single-batch rate is about **21.4 hours per 25,515-batch epoch**, or **42.8 hours for two epochs**. This is a rough planning estimate from one synthetic batch on this runtime, not an observed corpus run; data loading, dropout, hardware, and run-to-run timing will change it. The historical report was about 14h43m for epoch 1, so this NumPy version is not yet runtime-matched.

## Tiny corpus smoke run

A separate two-epoch run on a generated 38-character repeating `abab...` file completed three updates per epoch and wrote JSONL metrics plus both checkpoints. Its loss means were 1.0920 and 1.0622. This validates the CLI/checkpoint path only; it is not Hannah corpus evidence.

The full-window regimen runner was also smoke-tested on a generated short corpus: 7 batches per epoch, two epochs, both checkpoints, historical comparison report, and successful checkpoint reload without duplicate metrics. The tiny task's mean loss declined from 1.0692 to 1.0309. The independent geodesic audit passed, including an `atan2` distance recomputation and a nontrivial central-difference gradient check. These remain implementation checks, not a rerun on the historical corpus.

The training corpus was not available in this workspace. Consequently, neither the reported historical batch losses nor the full 25,515-batch-per-epoch regimen has been reproduced here. The two-epoch wall-clock estimate above is based on one synthetic full-size update.
