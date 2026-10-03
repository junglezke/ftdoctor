"""Read a fine-tuning run into one structure, whatever shape it arrived in.

A HuggingFace ``Trainer`` run leaves its whole history in ``trainer_state.json``
-- every logged loss, learning rate, gradient norm and eval metric -- and every
``checkpoint-N/`` directory carries a copy of it up to that step. That file is
the input. It can arrive as:

* the file itself,
* a training output directory (``./outputs/``) containing ``checkpoint-*/``,
* a single ``checkpoint-N/`` directory,
* a model repository on the HuggingFace Hub (see :mod:`ftdoctor.hub`).

Which checkpoints actually *exist* matters more than it looks. A recommendation
to "keep checkpoint-480" is useless when ``save_total_limit`` deleted it, or
when the run saved per epoch and step 480 was never written. So checkpoints are
only ever taken from the directories really present -- on disk or on the Hub --
and never inferred from ``save_steps``, which is meaningless under
``save_strategy="epoch"`` and is still recorded as 500.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")

#: Files that are training state rather than the model. Removing them from a
#: checkpoint you will never resume from frees most of its size and keeps the
#: weights loadable.
RESUME_ONLY_FILES = re.compile(
    r"^(optimizer\.(pt|bin)|scheduler\.pt|rng_state(_\d+)?\.pth|scaler\.pt|"
    r"global_step\d+|zero_to_fp32\.py|latest)$"
)


@dataclass
class Series:
    """One metric over training steps."""

    steps: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return int(self.steps.size)

    def at(self, step: float) -> Optional[float]:
        hit = np.where(self.steps == step)[0]
        return float(self.values[hit[0]]) if hit.size else None


@dataclass
class Checkpoint:
    """A checkpoint that really exists, with what it costs to keep."""

    step: int
    path: Optional[str]
    size_bytes: Optional[int] = None
    resume_only_bytes: Optional[int] = None
    where: str = "disk"  # "disk" | "hub"

    @property
    def name(self) -> str:
        return f"checkpoint-{self.step}"


@dataclass
class History:
    """A fine-tuning run, reduced to what the checks need."""

    name: str
    source: str
    train: Dict[str, Series]  # "loss", "learning_rate", "grad_norm", ...
    evals: Dict[str, Series]  # "eval_loss", "eval_accuracy", ...
    meta: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    #: Steps where a training metric was logged as NaN or inf. Kept apart from
    #: the series, where a missing value only ever means "not logged here".
    nonfinite: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def final_step(self) -> int:
        candidates = [int(self.meta.get("global_step") or 0)]
        candidates += [int(s.steps[-1]) for s in self.train.values() if len(s)]
        candidates += [int(s.steps[-1]) for s in self.evals.values() if len(s)]
        return max(candidates)

    @property
    def max_steps(self) -> Optional[int]:
        value = self.meta.get("max_steps")
        return int(value) if isinstance(value, (int, float)) and value > 0 else None

    def has_train(self, key: str, min_points: int = 3) -> bool:
        return key in self.train and len(self.train[key]) >= min_points

    def checkpoint_steps(self) -> List[int]:
        return sorted(c.step for c in self.checkpoints)

    @property
    def is_rl(self) -> bool:
        """True for an RL run (GRPO, PPO, RLOO) rather than supervised fine-tuning.

        In those, the logged "loss" is a policy-gradient surrogate that sits near
        zero by construction, and exactly-zero gradients mean degenerate reward
        groups. Every loss-based check here would misread them -- one public GRPO
        run was diagnosed as "trained on nothing, labels masked". DPO is not
        matched: its loss is a real log-likelihood ratio and its keys are
        ``rewards/chosen`` and friends, not a scalar ``reward``.
        """
        keys = set(self.train)
        markers = {"reward", "frac_reward_zero_std", "objective/rlhf_reward", "ppo/loss/policy"}
        return bool(keys & markers) or any(k.startswith(("actor/", "critic/")) for k in keys)


# -- parsing ----------------------------------------------------------------


def from_trainer_state(state: Dict[str, Any], name: str = "run", source: str = "") -> History:
    """Build a :class:`History` from a parsed ``trainer_state.json``."""
    log = state.get("log_history")
    if not isinstance(log, list):
        raise ValueError("not a trainer_state.json: there is no `log_history` list")

    train: Dict[str, List[Tuple[float, float]]] = {}
    evals: Dict[str, List[Tuple[float, float]]] = {}
    nonfinite: Dict[str, List[int]] = {}
    summary: Dict[str, Any] = {}

    for entry in log:
        if not isinstance(entry, dict) or "step" not in entry:
            continue
        step = entry["step"]
        if not isinstance(step, (int, float)):
            continue
        # The closing summary row (train_runtime, train_loss, ...) describes the
        # whole run; it is not a point on any curve.
        if "train_runtime" in entry:
            summary.update({k: v for k, v in entry.items() if k != "step"})
            continue
        for key, value in entry.items():
            if key == "step" or isinstance(value, bool):
                continue
            if key == "epoch" and any(k.startswith("eval_") for k in entry):
                continue  # an eval row's epoch duplicates the training row's
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                nonfinite.setdefault(key, []).append(int(step))
                continue
            target = evals if key.startswith("eval_") else train
            target.setdefault(key, []).append((float(step), float(value)))

    def to_series(points: List[Tuple[float, float]]) -> Series:
        points.sort(key=lambda p: p[0])
        steps = np.array([p[0] for p in points], dtype=float)
        values = np.array([p[1] for p in points], dtype=float)
        return Series(steps, values)

    meta = {
        k: state.get(k)
        for k in (
            "global_step",
            "max_steps",
            "num_train_epochs",
            "epoch",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "train_batch_size",
            "best_metric",
            "best_model_checkpoint",
            "best_global_step",
            "total_flos",
        )
    }
    meta.update(summary)
    control = (state.get("stateful_callbacks") or {}).get("TrainerControl", {}).get("args", {})
    if isinstance(control, dict) and "should_training_stop" in control:
        meta["should_training_stop"] = bool(control["should_training_stop"])

    return History(
        name=name,
        source=source,
        train={k: to_series(v) for k, v in train.items() if not _is_bookkeeping(k)},
        evals={k: to_series(v) for k, v in evals.items() if not _is_bookkeeping(k)},
        meta=meta,
        nonfinite=nonfinite,
    )


def _is_bookkeeping(key: str) -> bool:
    """Throughput and timing fields: logged next to metrics, never a metric."""
    return bool(
        re.search(r"(runtime|per_second|samples|num_tokens|total_flos|model_preparation)", key)
    )


# -- loading from disk ------------------------------------------------------


def load(target: str) -> History:
    """Load a run from a file, an output directory, a checkpoint, or the Hub."""
    if os.path.isfile(target):
        return _load_file(target)
    if os.path.isdir(target):
        return _load_dir(target)

    from . import hub

    if hub.looks_like_repo_id(target):
        return hub.load(target)
    raise FileNotFoundError(
        f"{target!r} is not a file, a directory, or a HuggingFace repo id (owner/name)"
    )


def _load_file(path: str) -> History:
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    parent = os.path.dirname(os.path.abspath(path))
    run_dir = os.path.dirname(parent) if _CHECKPOINT_DIR.match(os.path.basename(parent)) else parent
    if os.path.basename(path) == "trainer_state.json":
        name = os.path.basename(run_dir) or "run"
    else:
        name = os.path.splitext(os.path.basename(path))[0]
    history = from_trainer_state(state, name=name, source=path)
    history.checkpoints = scan_checkpoints(run_dir)
    return history


def _load_dir(path: str) -> History:
    path = os.path.abspath(path)
    if _CHECKPOINT_DIR.match(os.path.basename(path)):
        # Pointed at one checkpoint: diagnose the run it belongs to.
        return _load_file(os.path.join(path, "trainer_state.json"))

    candidates = []
    root_state = os.path.join(path, "trainer_state.json")
    if os.path.isfile(root_state):
        candidates.append(root_state)
    for checkpoint in scan_checkpoints(path, measure=False):
        state_path = os.path.join(checkpoint.path or "", "trainer_state.json")
        if os.path.isfile(state_path):
            candidates.append(state_path)
    if not candidates:
        raise FileNotFoundError(
            f"no trainer_state.json in {path} or in any checkpoint-*/ beneath it"
        )

    # The most complete history is the one that reached the furthest step.
    best_path, best_step = candidates[0], -1
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as handle:
                step = int(json.load(handle).get("global_step") or 0)
        except (OSError, ValueError):
            continue
        if step > best_step:
            best_path, best_step = candidate, step

    with open(best_path, encoding="utf-8") as handle:
        state = json.load(handle)
    history = from_trainer_state(state, name=os.path.basename(path), source=best_path)
    history.checkpoints = scan_checkpoints(path)
    return history


def scan_checkpoints(run_dir: str, measure: bool = True) -> List[Checkpoint]:
    """Every ``checkpoint-N/`` directly inside ``run_dir``, with sizes."""
    found: List[Checkpoint] = []
    try:
        entries = os.listdir(run_dir)
    except OSError:
        return found
    for entry in entries:
        match = _CHECKPOINT_DIR.match(entry)
        full = os.path.join(run_dir, entry)
        if not match or not os.path.isdir(full):
            continue
        checkpoint = Checkpoint(step=int(match.group(1)), path=full)
        if measure:
            checkpoint.size_bytes, checkpoint.resume_only_bytes = _measure(full)
        found.append(checkpoint)
    return sorted(found, key=lambda c: c.step)


def _measure(directory: str) -> Tuple[int, int]:
    """(total bytes, bytes that only matter for resuming training)."""
    total = resume_only = 0
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        size = _tree_size(full)
        total += size
        if RESUME_ONLY_FILES.match(name):
            resume_only += size
    return total, resume_only


def _tree_size(path: str) -> int:
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total
