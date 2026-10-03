# Changelog

## [0.1.0] — 2026-10-04

First release.

### Added

- `ftdoctor <target>` for a training output directory, a single checkpoint, a
  `trainer_state.json`, or a model on the HuggingFace Hub (`owner/model`).
- Noise-aware checkpoint selection: a band of 2x the eval's own step-to-step
  noise (or 1%), restricted to checkpoints that actually exist on disk or on
  the Hub.
- Selection metric recovered from the Trainer's `best_metric`, so runs selected
  on accuracy, WER or BLEU are ranked on that, not on eval loss.
- Checks: overfitting, stopped improving, loss vs selection metric
  disagreement, no eval, NaN / zero loss, loss flat or rising, loss spikes
  (including periodic and epoch-boundary jumps), gradient spikes, zero
  gradients and fp16 overflow, zero-learning-rate tails, eval below train,
  unfinished runs. RL runs are recognised and handed to rldoctor.
- `ftdoctor prune`: strip resume-only state or delete whole checkpoints, dry
  run by default, every path re-checked before deletion.
- Terminal, Markdown and JSON reports; `--fail-on` for CI.
- `validation/`: ten public runs with pinned verdicts, chosen from a sweep of 89.
