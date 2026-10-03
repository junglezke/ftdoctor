"""Which eval metric decides "best", and which way is better.

Getting this wrong is the easiest way for a tool like this to be confidently
wrong. The motivating case is a public audio-classification fine-tune whose
``eval_loss`` rose 37% from its minimum while ``eval_accuracy`` kept improving
to the very last step. Judged on loss, the run overfit badly; judged on
accuracy -- the metric its author actually selected on -- the final checkpoint
was the best one. Both readings are true, and only one of them is the question
the author asked.

So the selection metric is, in order of preference:

1. the one you pass explicitly;
2. the one the Trainer itself used: ``trainer_state.json`` records
   ``best_metric`` but not its name, so we find the eval series that contains
   that exact value at ``best_global_step``;
3. ``eval_loss``;
4. the only eval metric logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .history import History

#: Name fragments for metrics where smaller is better.
_LOWER = ("loss", "wer", "cer", "perplexity", "ppl", "error", "mae", "mse", "rmse", "ter")
#: Name fragments for metrics where larger is better.
_HIGHER = (
    "accuracy",
    "acc",
    "bleu",
    "chrf",
    "comet",
    "rouge",
    "f1",
    "exact_match",
    "precision",
    "recall",
    "auc",
    "map",
    "ndcg",
    "spearman",
    "pearson",
    "matthews",
    "score",
    "reward",
)


@dataclass
class Selection:
    """The metric checkpoints are ranked on."""

    key: str
    #: -1 when lower is better, +1 when higher is better.
    sign: int
    reason: str

    @property
    def label(self) -> str:
        # "loss" alone reads as the *training* loss, which is the one thing this
        # tool must never confuse with the eval.
        if self.key == "eval_loss":
            return "eval loss"
        return self.key[len("eval_") :] if self.key.startswith("eval_") else self.key

    @property
    def better(self) -> str:
        return "lower" if self.sign < 0 else "higher"

    def score(self, values: np.ndarray) -> np.ndarray:
        """Map values so that smaller is always better."""
        return np.asarray(values, dtype=float) * (-self.sign)


def direction(name: str) -> Optional[int]:
    """-1 when smaller is better, +1 when larger is better, None if unknown."""
    tokens = re.split(r"[^a-z0-9+]+", name.lower())
    joined = "_".join(tokens)
    for fragment in _LOWER:
        if fragment in tokens or joined.endswith(fragment):
            return -1
    for fragment in _HIGHER:
        if fragment in tokens or any(t.startswith(fragment) for t in tokens):
            return 1
    return None


def choose(
    history: History, override: Optional[str] = None, higher_is_better: Optional[bool] = None
) -> Optional[Selection]:
    """Pick the selection metric, or None when the run logged no eval metric."""
    usable = {k: s for k, s in history.evals.items() if len(s) >= 2}
    if not usable:
        return None

    if override:
        key = override if override.startswith("eval_") else f"eval_{override}"
        if key not in usable:
            raise KeyError(
                f"{override!r} is not an eval metric in this run; "
                f"available: {', '.join(sorted(usable))}"
            )
        sign = _sign_for(key, usable[key].values, higher_is_better)
        return Selection(key, sign, "you chose it")

    recorded = _trainer_choice(history, usable)
    if recorded is not None:
        key, sign = recorded
        return Selection(
            key, sign, "the metric your Trainer selected on (it matches `best_metric`)"
        )

    if "eval_loss" in usable:
        return Selection("eval_loss", -1, "eval_loss, the default")

    key = sorted(usable)[0]
    sign = _sign_for(key, usable[key].values, higher_is_better)
    return Selection(key, sign, "the only eval metric logged")


def _sign_for(key: str, values: np.ndarray, higher_is_better: Optional[bool]) -> int:
    if higher_is_better is not None:
        return 1 if higher_is_better else -1
    sign = direction(key)
    if sign is None:
        raise ValueError(
            f"cannot tell whether {key} is better high or low; pass "
            "--higher-is-better or --lower-is-better"
        )
    return sign


def _trainer_choice(history: History, usable) -> Optional[tuple]:
    """Recover ``metric_for_best_model`` from the value the Trainer recorded."""
    best = history.meta.get("best_metric")
    if not isinstance(best, (int, float)):
        return None
    best_step = history.meta.get("best_global_step")
    matches = []
    for key, series in usable.items():
        if isinstance(best_step, (int, float)):
            value = series.at(float(best_step))
            hit = value is not None and abs(value - best) <= 1e-9 * max(1.0, abs(best))
        else:
            hit = bool(np.any(np.abs(series.values - best) <= 1e-9 * max(1.0, abs(best))))
        if hit:
            matches.append(key)
    if len(matches) != 1:
        return None
    key = matches[0]
    sign = direction(key)
    if sign is None:
        # Name says nothing: the Trainer kept the best, so best is the extreme.
        values = usable[key].values
        sign = 1 if best >= float(np.max(values)) - 1e-12 else -1
    return key, sign
