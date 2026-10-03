"""The shape of a verdict."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List


class Severity(enum.IntEnum):
    """Ordered, so ``max()`` over findings is the run's worst state."""

    OK = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return {0: "OK", 1: "INFO", 2: "WARN", 3: "CRIT"}[int(self)]


@dataclass
class Finding:
    check: str
    title: str
    severity: Severity
    summary: str
    evidence: List[str] = field(default_factory=list)
    fix: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "title": self.title,
            "severity": self.severity.name,
            "summary": self.summary,
            "evidence": self.evidence,
            "fix": self.fix,
            "metrics": self.metrics,
        }
