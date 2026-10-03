"""Which checkpoint to keep -- and how sure we are about it.

``argmin(eval_loss)`` is the obvious answer and a bad one. Across 24 public
fine-tunes with enough evaluations, the final eval was not the minimum in 17
of them -- but in most of those 17 the difference was 1-2%, well inside the
step-to-step noise of the eval itself. Telling someone "your last checkpoint
is not the best" in those cases is technically true and practically wrong.

So a pick comes with a noise band. The band is the larger of:

* twice the eval metric's own step-to-step noise, estimated from successive
  differences (robust to the downward trend every training curve has), and
* a 1% relative floor, because a curve evaluated on a large eval set can be so
  smooth that its statistical noise is far below anything worth acting on.

Any eval inside the band is reported as equivalent to the best. Only an eval
outside it counts as genuinely worse.

The pick is also restricted to checkpoints that exist. The best eval often
lands on a step that was never saved, or was saved and then deleted by
``save_total_limit``; recommending it is useless. When the checkpoint list is
unknown (a bare ``trainer_state.json``), the report says so instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import stats
from .history import History
from .metrics import Selection

#: Relative difference below which two evals are treated as the same.
REL_FLOOR = 0.01
#: Width of the noise band in units of the eval's own noise.
NOISE_MULT = 2.0


@dataclass
class Pick:
    selection: Selection
    steps: np.ndarray
    values: np.ndarray
    best_step: int
    best_value: float
    band: float  # in the metric's own units
    noise: float
    final_step: int
    final_value: float
    equivalent_steps: List[int] = field(default_factory=list)
    #: The checkpoint to keep, when the existing checkpoints are known.
    keep_step: Optional[int] = None
    keep_value: Optional[float] = None
    checkpoints_known: bool = False
    #: Eval steps that have no checkpoint behind them.
    unsaved_best: bool = False

    @property
    def final_is_equivalent(self) -> bool:
        return self.final_step in self.equivalent_steps

    @property
    def final_gap(self) -> float:
        """How much worse the final eval is than the best, in metric units (>= 0)."""
        return float(abs(self.final_value - self.best_value))

    @property
    def final_gap_rel(self) -> float:
        return self.final_gap / max(abs(self.best_value), 1e-12)

    @property
    def final_gap_noise(self) -> float:
        return self.final_gap / self.noise if self.noise > 0 else float("inf")

    @property
    def steps_after_best(self) -> int:
        return max(self.final_step - self.best_step, 0)


def pick(history: History, selection: Selection) -> Optional[Pick]:
    series = history.evals.get(selection.key)
    if series is None or len(series) < 2:
        return None

    steps = series.steps.astype(int)
    values = series.values
    score = selection.score(values)  # smaller is better

    best_i = int(np.argmin(score))
    best_score = float(score[best_i])

    noise = stats.diff_sigma(score) if score.size >= 4 else 0.0
    band = max(NOISE_MULT * noise, REL_FLOOR * abs(best_score))
    equivalent = [int(s) for s, v in zip(steps, score, strict=True) if v - best_score <= band]

    result = Pick(
        selection=selection,
        steps=steps,
        values=values,
        best_step=int(steps[best_i]),
        best_value=float(values[best_i]),
        band=band,
        noise=noise,
        final_step=int(steps[-1]),
        final_value=float(values[-1]),
        equivalent_steps=equivalent,
    )

    saved = set(history.checkpoint_steps())
    if saved:
        result.checkpoints_known = True
        evaluated_and_saved = [i for i, s in enumerate(steps) if int(s) in saved]
        if evaluated_and_saved:
            keep_i = min(evaluated_and_saved, key=lambda i: score[i])
            result.keep_step = int(steps[keep_i])
            result.keep_value = float(values[keep_i])
        result.unsaved_best = result.best_step not in saved
    return result


def describe_band(p: Pick) -> str:
    if p.noise > 0 and NOISE_MULT * p.noise >= REL_FLOOR * abs(p.best_value):
        return f"{NOISE_MULT:.0f}x the eval's own step-to-step noise ({p.band:.4g})"
    return f"a {REL_FLOOR:.0%} relative floor ({p.band:.4g}) -- the eval is smoother than that"
