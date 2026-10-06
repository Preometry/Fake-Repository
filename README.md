# Fake Hannah Bobanna

A runnable repository home for the reconstructed **Hannah Trust-Manifold Transformer**.

This repo is intentionally recovery-first: the recovered model mechanisms are kept intact, exact historical snapshots are archived separately, and compatibility wiring is kept in the test layer instead of silently rewriting the recovered model.

## One-command local home

```bash
git clone https://github.com/Preometry/Fake-Repository.git ~/HANNAH
cd ~/HANNAH
./setup.sh
```

Then:

```bash
source .venv/bin/activate
./scripts/run_tests.sh
```

To run the reconstructed historical 2-epoch trainer after placing theorem files in `data/intake_full/`:

```bash
python scripts/train_historical.py --data-dir data/intake_full
```

## What is here

- `src/hannah_model/transformer_lm.py` — recovered baseline GPT-style transformer.
- `src/hannah_model/transformer_lm_trust.py` — Hannah trust-manifold transformer integration.
- `src/hannah_model/trust_aware_attention.py` — spectral projection and geodesic trust-aware attention.
- `src/hannah_model/hannah_embedding.py` — Hannah remainder/topological embedding machinery.
- `tests/test_trust_hannah.py` — supplied persistence-compatible test wiring (`attn(x,x,x)` layer by layer).
- `scripts/train_historical.py` — path-configurable reconstruction of the historical two-epoch trainer.
- `scripts/independent_geodesic_checks.py` — independent numerical checks of the recovered geodesic arithmetic.
- `archive/exact_snapshots/` — recovered source snapshots preserved without model rewrites.
- `HISTORICAL_TARGETS.md` and `RECONSTRUCTION_LOG.md` — recovered targets, provenance, and unresolved evidence gaps.

## Historical run target

The concrete recovered demo configuration uses:

- first 20 sorted `THRM-*.yaml` files
- character-level next-character prediction
- sequence length 128
- batch size 16
- Adam learning rate `1e-3`
- 2 epochs
- 4 transformer layers / 4 heads / 256 model dimension / 1024 feed-forward dimension
- historical vocabulary size 133 and reported 3,062,016 trainable parameters

The exact original 20-file corpus manifest and random seed are not yet recovered. Every reconstructed run records corpus hashes and runtime metadata rather than pretending those missing pieces are known.

## Current wiring note

The recovered attention module is a normal Q/K/V interface. The persistence test therefore performs self-attention explicitly:

```python
out = attn1(x, x, x)[0]
out = norm1(out)
out = attn2(out, out, out)[0]
```

This is test/integration wiring; the recovered model modules remain intact.
