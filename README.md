# ATLAS

**Adaptive Topology & Local Activation Stepping**: an experimental approach to neural network
training that spends compute where the network currently needs it, instead of updating every
weight on every step.

> **Status: active research.** This branch describes the idea, the design and what we have
> measured so far. The implementation, tests and full benchmark records will be published when
> the method is complete. Every result below is preliminary, from small models, and labeled
> with the task it came from.

## The idea

A training step normally treats a network as one block: every weight receives a gradient,
and every weight is updated. But at any moment some regions of a weight matrix are changing
quickly while others have mostly settled. ATLAS asks whether a trainer can notice that and
skip work on the quiet regions, first in the optimizer, and eventually in the forward and
backward passes themselves.

ATLAS splits each weight matrix into **tiles**, gives every tile a **score** for how much it
currently wants to change, and lets a **gate** decide which tiles are worked on this step.
Tiles that are not selected **sleep**. The design questions are what the score should be, how
the gate should choose, what a sleeping tile should do, and how to make skipping work actually
save time.

## Design

### Tiles and coarse-to-fine refinement

Each 2D weight matrix is divided into an R × R grid of tiles. Training starts coarse
(R = 1, the whole matrix) and refines on a schedule (1 → 2 → 4). Refined tiles inherit their
parent's state, so refinement never discards optimizer history. Coarse tiles are cheap and
stable early on; fine tiles let the gate target smaller regions later.

### Scores

A tile's score is its gradient norm relative to its weight norm, normalized against that
tile's own slowly decaying peak, and scaled by global training progress (the current loss
relative to the first loss):

```
ratio    = ‖gradient tile‖ / ‖weight tile‖
score    = ratio / max(ratio, 0.994 · previous peak) · progress^0.45     (clipped to [0, 1])
```

The score is a relative gating signal: "how active is this tile compared with its own
recent history". It is **not** an estimate of how far a weight is from a correct value. An
optional *local* mode blends in errors from small detached probes attached to each block,
used only as gating signals.

### Gates

- **Threshold gate.** A tile updates when its score reaches a threshold τ that adapts to the
  group's mean score.
- **Budget gate.** The top *k*% of tiles by score update each step. This guarantees a fixed
  amount of skipped work by construction.

Either way, a tile that has slept for `max_sleep_steps` is woken, so no tile can starve.

### Sleeping and catching up

Two policies for a sleeping tile:

1. **Accumulate.** Keep summing its gradients. When it wakes, apply one update from the
   mean accumulated gradient, scaled by a bounded catch-up multiplier
   `min(1.85, steps_slept^0.58)`. Momentum and other optimizer state advance only when the
   tile updates.
2. **Drop.** A sleeping tile receives no gradient at all, so nothing accumulates. When it
   wakes it trains on its current gradient, with the same catch-up multiplier. This is what
   makes skipping the passes possible (below).

ATLAS supports AdamW and Muon updates per tile. Orthogonalizing each Muon tile separately
gives a different update direction from orthogonalizing the whole matrix, so tiled Muon is a
variant of Muon, not just a cheaper way to compute it.

### Skipping the passes

An optimizer-only scheme has a hard ceiling. The forward pass and the full backward pass
still run, so the most it can save is the optimizer's share of a step. To go further, ATLAS
can **plan one step ahead**. After each update the gate chooses which tiles will be computed
on the *next* step, from their latest scores, boosted the longer a tile has waited. The
network's linear layers then skip the sleeping tiles:

- **Backward-sparse.** The forward pass is exact. Only the weight-gradient matmuls of
  planned tiles are computed.
- **Fully sparse.** Sleeping tiles are also left out of the training forward pass and the
  input-gradient matmul, so every matmul of those layers shrinks with the awake share.

Fully sparse training is deliberately "fuzzy". The network being trained on a given step is a
sparser version of the real one, and different tiles are adjusted on different steps.
Evaluation always uses every tile.

### Switching back to full passes

The expectation is that fuzzy training makes fast early progress and then stalls, because
independently adjusted tiles stop agreeing with each other. So ATLAS can **switch to full
passes** when the smoothed training loss stops improving (a plateau detector), or at a set
step. Optimizer state carries across the switch.

## Engineering lessons

- **Tiles must not mean tiny kernels.** A naive per-tile implementation launched about 16,700
  GPU kernels per optimizer step, against about 1,100 for the native optimizer, and the GPU
  sat idle waiting for them. Treating each matrix's tiles as one strided batch (batched norms
  and a single batched Newton–Schulz iteration per matrix) cut the launches tenfold and made
  an optimizer step about 6× faster, at the same quality.
- **Threshold gates can quietly stop gating.** On a 20M-parameter transformer the adaptive
  threshold settled at about 60% of the mean score, while scores rose and bunched together
  late in training, so nearly every tile cleared it. Smoothing the loss signal did not help.
  The cause is the threshold rule, not noise. Budget gates avoid this by construction.
- **Measure, then guess.** Several plausible speedups (for example, batching only the
  device-to-host transfers) turned out to be minor once profiled.

## Preliminary results

**Task:** a causal language model trained from scratch on TinyStories (about 11.5M
parameters, 6 transformer blocks, width 384, 2M training tokens per run), 5 paired seeds,
one AMD Radeon RX 7900 XT. All configurations share the same initialization and minibatches
within each seed. Gated configurations use a 25% budget.

| Configuration | Validation loss | Time vs native Muon | Matmul work in gated layers |
| :--- | ---: | ---: | ---: |
| Native Muon (reference) | 2.78 | 1.00× | 100% |
| ATLAS, threshold gate | 2.83 | 1.36× | 100% |
| ATLAS, 25% budget gate | 3.10 | 1.17× | 100% |
| ATLAS, backward-sparse | 3.05 | 1.01× | 78% |
| ATLAS, fully sparse | 3.67 | 0.85× | 30% |
| ATLAS, fully sparse → full passes on plateau | 3.05 | 1.01× | 75% |

What this does and does not show:

- Native Muon still trains better at equal time. ATLAS does not yet beat it.
- Fully sparse training is the fastest configuration measured, and switching to full passes
  recovers most of its quality gap. The plateau detector fired at nearly the same point in
  every seed.
- Most of the remaining quality cost comes from updating only 25% of tiles, not from
  skipping passes. Backward-sparse training was slightly *better* than the same budget with
  full passes.
- These are short, fixed-learning-rate runs on small models. They are not convergence results
  or claims about large-scale training.

## Open questions

- Budget schedules: start dense and anneal the active share, or adapt it to the score
  distribution.
- A dense warm-up before sparse passes. A poor sparse start can persist after switching back.
- Whether fresh-gradient (drop) updates beat accumulated updates even with full passes.
- Does a tile's score predict which updates matter? An oracle comparison of top-*k* against
  random-*k* tiles would show whether targeting beats a global schedule.
- Adaptive tile sizes: split a tile when its halves disagree, merge when they agree, with a
  minimum size set by measured kernel cost.
- Moving cheap decision logic off the critical path, for example planning tiles on a CPU
  thread while the GPU computes.
- Larger models, longer training and other tasks.

## Credits and license

ATLAS was conceived by Jacob ([hearthlink](https://github.com/hearthlink)), who proposed the
tile metadata, threshold gating, resolution and sparse-pass ideas. It is developed by
Hearthlink: Jacob working with AI collaborators (Claude, Codex and Gemini). Muon follows its
published Newton–Schulz convention, with PyTorch's implementation as the parity reference.

Licensed under the [GNU General Public License v3.0 or later](LICENSE).
