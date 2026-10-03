"""Put it together: one call from a run to a verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import checks, metrics, picker
from .findings import Finding, Severity
from .history import History

__version__ = "0.1.0"


@dataclass
class Row:
    """One checkpoint in the keep/delete table."""

    step: int
    value: Optional[float]
    size_bytes: Optional[int]
    status: str  # "keep" | "equivalent" | "worse" | "not evaluated"
    is_final: bool


@dataclass
class Diagnosis:
    history: History
    selection: Optional[metrics.Selection]
    pick: Optional[picker.Pick]
    findings: List[Finding]
    version: str = __version__

    @property
    def worst(self) -> Severity:
        return max((f.severity for f in self.findings), default=Severity.OK)

    @property
    def problems(self) -> List[Finding]:
        return sorted(self.findings, key=lambda f: -int(f.severity))

    def table(self) -> List[Row]:
        """Every checkpoint that exists, and what to do with it."""
        rows: List[Row] = []
        if not self.history.checkpoints:
            return rows
        last = max(c.step for c in self.history.checkpoints)
        evaluated = {}
        if self.pick is not None:
            evaluated = {int(s): float(v) for s, v in zip(self.pick.steps, self.pick.values, strict=True)}
        for c in self.history.checkpoints:
            value = evaluated.get(c.step)
            if self.pick is None or value is None:
                status = "not evaluated"
            elif c.step == self.pick.keep_step:
                status = "keep"
            elif c.step in self.pick.equivalent_steps:
                status = "equivalent"
            else:
                status = "worse"
            rows.append(Row(c.step, value, c.size_bytes, status, c.step == last))
        return rows

    @property
    def headline(self) -> str:
        p = self.pick
        parts = []
        if p is not None and p.keep_step is not None:
            parts.append(f"keep checkpoint-{p.keep_step}")
        elif p is not None:
            parts.append(f"best eval at step {p.best_step}")
        top = [f for f in self.problems if f.severity >= Severity.WARNING]
        if top:
            parts.append(f"{top[0].title.lower()}")
        if not parts:
            return f"{self.history.name}: nothing to report."
        return f"{self.history.name}: " + "; ".join(parts) + "."

    def exit_code(self, fail_on: Severity = Severity.CRITICAL) -> int:
        return 1 if self.worst >= fail_on else 0

    def to_dict(self) -> Dict[str, Any]:
        p = self.pick
        return {
            "version": self.version,
            "run": {
                "name": self.history.name,
                "source": self.history.source,
                "final_step": self.history.final_step,
                "max_steps": self.history.max_steps,
                "eval_metrics": sorted(self.history.evals),
                "checkpoints": [
                    {"step": c.step, "size_bytes": c.size_bytes, "where": c.where}
                    for c in self.history.checkpoints
                ],
            },
            "selection": None
            if self.selection is None
            else {
                "metric": self.selection.key,
                "better": self.selection.better,
                "reason": self.selection.reason,
            },
            "pick": None
            if p is None
            else {
                "best_step": p.best_step,
                "best_value": p.best_value,
                "final_step": p.final_step,
                "final_value": p.final_value,
                "noise_band": p.band,
                "equivalent_steps": p.equivalent_steps,
                "final_is_equivalent": p.final_is_equivalent,
                "keep_step": p.keep_step,
                "checkpoints_known": p.checkpoints_known,
            },
            "worst": self.worst.name,
            "headline": self.headline,
            "findings": [f.to_dict() for f in self.findings],
        }

    def print(self) -> None:
        from .report import emit, render_terminal

        emit(render_terminal(self))


def diagnose(
    history: History,
    metric: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
) -> Diagnosis:
    selection = metrics.choose(history, override=metric, higher_is_better=higher_is_better)
    # An RL run's evals (if any) are rewards, not a held-out fit -- do not rank on them
    # unless asked to explicitly.
    if history.is_rl and metric is None:
        selection = None
    p = picker.pick(history, selection) if selection is not None else None
    findings = checks.run_all(history, selection, p)
    return Diagnosis(history=history, selection=selection, pick=p, findings=findings)
