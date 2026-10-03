"""Reclaim the disk space a finished fine-tune is sitting on.

A 7B full fine-tune that saved every 500 steps leaves tens of checkpoints of
tens of gigabytes each, most of which nobody will ever load. Two modes:

``resume-state`` (the default, and the only one offered without an eval)
    From every checkpoint except the latest, delete only what is needed to
    *resume training* -- optimizer moments, scheduler, RNG state, DeepSpeed
    shards. The model weights stay, so every checkpoint can still be loaded and
    evaluated. The optimizer state is usually the bulk of a checkpoint (Adam
    keeps two fp32 moments per parameter), so this recovers most of the space
    and destroys nothing you would evaluate.

``full``
    Delete whole checkpoints, keeping the recommended one and the latest.
    Only available when the run has an eval to base the recommendation on.

Nothing is deleted without ``apply()``, which re-checks every target: it must
sit directly inside the run directory, be named ``checkpoint-N``, contain a
``trainer_state.json``, and not be one of the checkpoints being kept.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

from .diagnosis import Diagnosis
from .history import RESUME_ONLY_FILES, Checkpoint

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")


@dataclass
class Action:
    checkpoint: Checkpoint
    #: Paths that will be removed: the checkpoint itself, or files inside it.
    targets: List[str]
    freed_bytes: int


@dataclass
class Plan:
    run_dir: str
    mode: str
    keep: List[int]
    actions: List[Action] = field(default_factory=list)
    refused: Optional[str] = None

    @property
    def freed_bytes(self) -> int:
        return sum(a.freed_bytes for a in self.actions)


def plan(
    diagnosis: Diagnosis, mode: str = "resume-state", keep_extra: Iterable[int] = ()
) -> Plan:
    local = [c for c in diagnosis.history.checkpoints if c.where == "disk" and c.path]
    run_dir = os.path.dirname(local[0].path) if local else ""
    keep: Set[int] = set(int(s) for s in keep_extra)
    if local:
        keep.add(max(c.step for c in local))
    p = diagnosis.pick
    if p is not None and p.keep_step is not None:
        keep.add(p.keep_step)

    result = Plan(run_dir=run_dir, mode=mode, keep=sorted(keep))
    if not local:
        result.refused = (
            "no checkpoint directories on local disk -- point prune at the training output "
            "directory that contains checkpoint-*/"
        )
        return result
    if mode == "full" and (p is None or p.keep_step is None):
        result.refused = (
            "deleting whole checkpoints needs an eval-based recommendation, and this run has "
            "none. Use the default --mode resume-state, which keeps every model's weights."
        )
        return result

    for c in local:
        if c.step in keep:
            continue
        if mode == "full":
            result.actions.append(Action(c, [c.path], c.size_bytes or 0))
        else:
            targets = [
                os.path.join(c.path, name)
                for name in sorted(os.listdir(c.path))
                if RESUME_ONLY_FILES.match(name)
            ]
            if targets:
                result.actions.append(Action(c, targets, c.resume_only_bytes or 0))
    return result


def apply(p: Plan) -> Tuple[int, List[str]]:
    """Carry out a plan. Returns (bytes freed, log lines)."""
    if p.refused:
        raise RuntimeError(p.refused)
    run_dir = os.path.realpath(p.run_dir)
    freed = 0
    log: List[str] = []
    for action in p.actions:
        ckpt_dir = os.path.realpath(action.checkpoint.path or "")
        _verify_checkpoint_dir(ckpt_dir, run_dir, p.keep)
        for target in action.targets:
            real = os.path.realpath(target)
            if real != ckpt_dir and os.path.dirname(real) != ckpt_dir:
                raise RuntimeError(f"refusing to delete {target}: not inside {ckpt_dir}")
            if os.path.isdir(real):
                shutil.rmtree(real)
            elif os.path.exists(real):
                os.remove(real)
            log.append(f"deleted {target}")
        freed += action.freed_bytes
    return freed, log


def _verify_checkpoint_dir(path: str, run_dir: str, keep: List[int]) -> None:
    name = os.path.basename(path)
    match = _CHECKPOINT_DIR.match(name)
    if not match:
        raise RuntimeError(f"refusing to touch {path}: not a checkpoint-N directory")
    if os.path.dirname(path) != run_dir:
        raise RuntimeError(f"refusing to touch {path}: not directly inside {run_dir}")
    if not os.path.isfile(os.path.join(path, "trainer_state.json")):
        raise RuntimeError(f"refusing to touch {path}: no trainer_state.json, so not a Trainer checkpoint")
    if int(match.group(1)) in keep:
        raise RuntimeError(f"refusing to touch {path}: it is a checkpoint being kept")


def human(size: Optional[int]) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
