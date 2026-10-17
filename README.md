# ATLAS

**Adaptive Topology & Local Activation Stepping**: an experimental approach to neural network
training that spends compute where the network currently needs it, instead of updating every
weight on every step.

> **Status: active research.** This repository describes the idea, the design and what we have
> measured so far. The implementation, tests and full benchmark records will be published when
> the method is complete. Every result below is preliminary, from small models, and labeled
> with the task it came from.

## The idea

A training step normally treats a network as one block: every weight receives a gradient,
and every weight is updated. But at any moment some regions of a weight matrix are changing
quickly while others have mostly settled. ATLAS asks whether a trainer can notice that and
skip work on the quiet regions, first in the optimizer, and then in the forward and backward
passes themselves.

ATLAS splits each weight matrix into **tiles**, gives every tile a **score** for how much it
currently wants to change, and lets a **gate** decide which tiles are worked on this step.
Tiles that are not selected **sleep**. The design questions are what the score should be, how
the gate should choose, what a sleeping tile should do, and how to make skipping work actually
save time.

## Design

### Tiles and coarse-to-fine refinement

Each 2D weight matrix is divided into an R × R grid of tiles. Training starts coarse
(R = 1, the whole matrix) and refines on a schedule (1 → 2 → 4). Refined tiles inherit their
parent's state, so refinement never discards optimizer history.

### Scores

A tile's score is its gradient norm relative to its weight norm, normalized against that
tile's own slowly decaying peak, and scaled by global training progress:

```
ratio    = ‖gradient tile‖ / ‖weight tile‖
score    = ratio / max(ratio, 0.994 · previous peak) · progress^0.45     (clipped to [0, 1])
```

The score is a relative gating signal: "how active is this tile compared with its own
recent history". It is **not** an estimate of how far a weight is from a correct value.

### Gates

- **Budget gate (current default).** The top *k*% of tiles by score update each step, chosen
  across all matrices together. This fixes the amount of skipped work by construction.
- **Threshold gate (original).** A tile updates when its score reaches a threshold that adapts
  to the mean score. On transformers this stops gating late in training (see below).

A tile that has slept too long is woken, so no tile can starve.

### What a sleeping tile does

The original design let sleeping tiles **accumulate** their gradients and apply them on
waking with a catch-up multiplier. The current design **drops** them instead: a waking tile
trains on its current gradient. Fresh updates trained better than accumulated ones in every
paired comparison, and they are what make skipping the backward pass possible.

### Skipping the passes

An optimizer-only scheme has a hard ceiling: the forward and backward passes still run. So
ATLAS **plans one step ahead**. After each update the gate chooses which tiles the *next*
step computes, and the network's linear layers skip the rest:

- **Backward-sparse.** The forward pass is exact; only planned tiles' weight-gradient matmuls
  run. This has cost no measurable quality in any run so far.
- **Fully sparse.** Sleeping tiles are also left out of the training forward pass and the
  input gradient. This is the "fuzzy" mode: the network trained on a given step is a sparser
  version of the real one. ATLAS can switch back to full passes when training stalls.

### Seeing sleeping tiles without computing them

A skipped tile has no gradient, so its score goes stale. But its gradient norm has a cheap
upper bound from tensors the backward pass already has:

```
‖dW tile‖ = ‖dYᵀ[rows] · X[cols]‖ ≤ ‖dY[rows]‖ · ‖X[cols]‖
```

Scaling that bound per matrix by the true/bound ratio of the tiles that *were* computed gives
every tile a fresh estimate each step. In practice the calibrated estimate came within about
3–5% of the true gradient-norm ratios, at a small fraction of the cost.

## Engineering lessons

- **Tiles must not mean tiny kernels.** A naive per-tile implementation launched about 16,700
  GPU kernels per optimizer step, against about 1,100 for the native optimizer. Treating each
  matrix's tiles as one strided batch, then batching same-shaped tiles *across* matrices,
  brought ATLAS's optimizer step from 10× slower than native to faster than native.
- **Threshold gates can quietly stop gating.** The adaptive threshold settled at about 60% of
  the mean score while scores rose and bunched together, so nearly every tile cleared it late
  in training. Smoothing the loss signal did not help; a fixed budget does.
- **Regular schedules hurt.** Rotating which tiles update on a fixed cycle trained worse than
  top-*k* even while updating more parameters, and freezing whole matrices for several steps
  at a time was far worse. Irregular selection, whether by score or at random, does not have
  this problem.
- **Scores seem to matter across matrices, not within them.** At equal update volume, which
  tiles of a matrix update barely matters (top-*k* beat random by 0.004 in one run). But
  letting scores decide how much of the budget each matrix receives beat splitting it
  evenly. Our working hypothesis: scores are for *allocating* work between matrices, and
  placement inside a matrix can be chosen for hardware efficiency.
- **Tune for sparsity.** Updating tiles less often wants a larger step: doubling the learning
  rate improved ATLAS three times as much as it improved the dense baseline.
- **Measure, then guess.** Several plausible speedups (a background planner thread, batching
  only the device-to-host transfers) turned out to be minor or negative once profiled.

## Preliminary results

**Task:** causal language models trained from scratch on TinyStories, six transformer blocks,
width 288 to 640 (6.6M to 31M parameters), 2M training tokens per run, on one AMD Radeon
RX 7900 XT. Every configuration shares initialization and minibatches within a seed. ATLAS
uses the budget gate at 25%, fresh updates and backward-sparse passes; both ATLAS and the
native Muon baseline use twice the default Muon learning rate.

*Equal-time deficit* compares ATLAS's final validation loss with native Muon's loss after the
same training time. It is the number that decides whether skipping work pays: positive means
the baseline is ahead.

![ATLAS training time and equal-time deficit by model size](scaling.png)

| Model size | Seeds | Time vs native Muon | Loss gap | Equal-time deficit |
| :--- | ---: | ---: | ---: | ---: |
| 6.6M | 1 | 0.99× | +0.25 | +0.25 |
| 13.4M | 1 | 0.90× | +0.23 | +0.18 |
| 25.2M | 1 | 0.86× | +0.28 | +0.19 |
| 31.0M | 3 | **0.82×** | +0.25 | **+0.13** |

What this does and does not show:

- **ATLAS gets relatively faster as models grow.** The work it skips grows with matrix size
  while its bookkeeping stays roughly fixed. At 31M it trains 18% faster than native Muon.
- **Its equal-time deficit shrinks with size,** from about 0.25 to 0.13, but native Muon is
  still ahead at every size tested. The improvement comes from speed: the raw quality gap at
  a fixed number of steps stays around 0.2–0.25.
- **These are short runs on small models.** Every run uses the same 2M tokens, so larger
  models are increasingly undertrained. Most points are single seeds. Whether the trend
  continues, and whether it crosses over, is exactly what the next experiments test.

## Open questions

- **The best budget at scale.** At small sizes, a larger budget closed much of the quality gap
  at some cost in speed. Now that ATLAS is faster than the baseline at 31M, a 35–65% budget
  may come out ahead at equal time.
- **Larger models and longer training,** with more seeds, to see whether the scaling trend
  holds and crosses over.
- **Matrix-level allocation:** if scores mainly matter for dividing work between matrices,
  a per-matrix score could replace per-tile scoring, with tiles inside each matrix placed for
  hardware efficiency.
- **Making sparse forward passes work.** Fully sparse training is the fastest mode, but the
  sparse network drifts from the full one, and switching back to full passes has not yet
  recovered enough quality.
- **Fixed-shape steps** (a constant budget per matrix shape) would allow GPU graph capture of
  the whole update, removing the remaining launch overhead.
- Adaptive tile sizes, other optimizers and other tasks.

## Credits and license

ATLAS was conceived by Jacob ([hearthlink](https://github.com/hearthlink)), who proposed the
tile metadata, threshold gating, resolution, sparse-pass and scaling ideas. It is developed by
Hearthlink: Jacob working with AI collaborators (Claude, Codex and Gemini). Muon follows its
published Newton–Schulz convention, with PyTorch's implementation as the parity reference.

Licensed under the [GNU General Public License v3.0 or later](LICENSE).
