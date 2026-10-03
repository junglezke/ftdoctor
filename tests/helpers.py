"""Build trainer_state.json payloads and output directories with known shapes."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np


def make_state(
    n_steps: int = 1000,
    log_every: int = 10,
    eval_every: int = 100,
    overfit_after: Optional[int] = None,
    eval_noise: float = 0.004,
    metric: Optional[str] = None,
    seed: int = 0,
    lr_peak: float = 2e-4,
    loss_fn=None,
    grad_fn=None,
    extra_train: Optional[Dict[int, Dict[str, float]]] = None,
    best_metric_key: Optional[str] = None,
    epochs: float = 3.0,
) -> dict:
    """A plausible HF Trainer history.

    Training loss decays smoothly. Eval loss follows it down and, if
    ``overfit_after`` is set, turns upward from that step while training loss
    keeps falling.
    """
    rng = np.random.default_rng(seed)
    log: List[dict] = []
    for step in range(log_every, n_steps + 1, log_every):
        frac = step / n_steps
        warm = min(1.0, step / max(n_steps * 0.03, 1))
        lr = lr_peak * warm * (1 - frac) if frac < 1 else 0.0
        loss = loss_fn(step) if loss_fn else 1.8 * math.exp(-3 * frac) + 0.25 + rng.normal(0, 0.02)
        grad = grad_fn(step) if grad_fn else 0.8 * math.exp(-frac) + abs(rng.normal(0, 0.03))
        entry = {
            "epoch": epochs * frac,
            "grad_norm": grad,
            "learning_rate": lr,
            "loss": loss,
            "step": step,
        }
        if extra_train and step in extra_train:
            entry.update(extra_train[step])
        log.append(entry)
        if eval_every and step % eval_every == 0:
            base = 1.6 * math.exp(-3 * frac) + 0.45
            if overfit_after is not None and step > overfit_after:
                base = 1.6 * math.exp(-3 * overfit_after / n_steps) + 0.45
                base += 0.35 * (step - overfit_after) / n_steps
            ev = {"epoch": epochs * frac, "eval_loss": base + rng.normal(0, eval_noise), "step": step,
                  "eval_runtime": 12.3, "eval_samples_per_second": 40.1}
            if metric == "accuracy":
                # accuracy keeps improving even while loss rises
                ev["eval_accuracy"] = 0.70 + 0.2 * frac + rng.normal(0, 0.003)
            log.append(ev)
    log.append({"epoch": epochs, "step": n_steps, "train_runtime": 3600.0, "train_loss": 0.5,
                "total_flos": 1e15, "train_samples_per_second": 3.0})
    state = {
        "global_step": n_steps,
        "max_steps": n_steps,
        "num_train_epochs": epochs,
        "epoch": epochs,
        "save_steps": eval_every,
        "eval_steps": eval_every,
        "logging_steps": log_every,
        "best_metric": None,
        "best_model_checkpoint": None,
        "best_global_step": None,
        "log_history": log,
        "stateful_callbacks": {"TrainerControl": {"args": {"should_training_stop": True}}},
    }
    if best_metric_key:
        evals = [(e["step"], e[best_metric_key]) for e in log if best_metric_key in e]
        sign = 1 if "accuracy" in best_metric_key else -1
        step, value = max(evals, key=lambda sv: sign * sv[1])
        state.update(best_metric=value, best_global_step=step,
                     best_model_checkpoint=f"out/checkpoint-{step}")
    return state


def truncate(state: dict, step: int) -> dict:
    """The copy of the state a checkpoint at ``step`` would carry."""
    out = dict(state)
    out["global_step"] = step
    out["log_history"] = [e for e in state["log_history"] if e["step"] <= step and "train_runtime" not in e]
    return out


def make_output_dir(root: str, state: dict, checkpoint_steps: List[int],
                    weights_bytes: int = 4000, optimizer_bytes: int = 8000) -> str:
    """A training output directory with checkpoint-N/ folders and fake files."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "trainer_state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    for step in checkpoint_steps:
        ckpt = os.path.join(root, f"checkpoint-{step}")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "trainer_state.json"), "w", encoding="utf-8") as fh:
            json.dump(truncate(state, step), fh)
        with open(os.path.join(ckpt, "model.safetensors"), "wb") as fh:
            fh.write(b"\0" * weights_bytes)
        with open(os.path.join(ckpt, "optimizer.pt"), "wb") as fh:
            fh.write(b"\0" * optimizer_bytes)
        with open(os.path.join(ckpt, "scheduler.pt"), "wb") as fh:
            fh.write(b"\0" * 100)
        with open(os.path.join(ckpt, "rng_state.pth"), "wb") as fh:
            fh.write(b"\0" * 50)
    return root
