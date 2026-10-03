"""Terminal, Markdown and JSON reports.

The terminal report answers the question people open it with -- *which
checkpoint do I keep?* -- on the first screen, before any finding.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from typing import List, Optional

from .diagnosis import Diagnosis, Row
from .findings import Finding, Severity
from .picker import describe_band
from .prune import human

_RESET = "\033[0m"
_STYLE = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[91m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "cyan": "\033[36m",
}
_SEV_STYLE = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "blue",
    Severity.OK: "green",
}


def emit(text: str) -> None:
    """Print text a legacy console code page may not be able to encode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text)


def _color_ok() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _unicode_ok() -> bool:
    if os.environ.get("FTDOCTOR_ASCII"):
        return False
    try:
        "─·▍≈".encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class _Paint:
    def __init__(self, on: bool) -> None:
        self.on = on

    def __call__(self, text: str, *styles: str) -> str:
        if not self.on or not styles:
            return text
        return "".join(_STYLE[s] for s in styles) + text + _RESET


def render_terminal(
    d: Diagnosis, width: Optional[int] = None, color: Optional[bool] = None, unicode: Optional[bool] = None
) -> str:
    width = width or min(shutil.get_terminal_size((96, 24)).columns, 96)
    paint = _Paint(_color_ok() if color is None else color)
    u = _unicode_ok() if unicode is None else unicode
    rule, dot, bar, approx = ("─", "·", "▍", "≈") if u else ("-", "*", ">", "~")
    h = d.history
    out: List[str] = [""]

    out.append("  " + paint("ftdoctor", "bold", "cyan") + paint(f" {d.version}", "dim") + "   " + paint(h.name, "bold"))
    out.append("  " + paint(rule * (width - 4), "dim"))
    facts = [f"{h.final_step} steps"]
    epochs = h.meta.get("epoch")
    if isinstance(epochs, (int, float)) and epochs:
        facts.append(f"{epochs:.3g} epochs")
    n_evals = max((len(s) for s in h.evals.values()), default=0)
    facts.append(f"{n_evals} evals" if n_evals else "no evals")
    if h.checkpoints:
        sizes = [c.size_bytes for c in h.checkpoints if c.size_bytes]
        where = "on the Hub" if h.checkpoints[0].where == "hub" else "on disk"
        total = f" ({human(sum(sizes))})" if sizes else ""
        noun = "checkpoint" if len(h.checkpoints) == 1 else "checkpoints"
        facts.append(f"{len(h.checkpoints)} {noun} {where}{total}")
    out.append("  " + paint(f" {dot} ".join(facts), "dim"))
    if d.selection is not None:
        out.append(
            "  "
            + paint(
                f"ranked on {d.selection.label} ({d.selection.better} is better) -- {d.selection.reason}",
                "dim",
            )
        )
    out.append("")

    out += _keep_block(d, width, paint, bar, approx)

    problems = [f for f in d.problems if f.severity > Severity.OK]
    out.append(_rule("FINDINGS", width, paint, rule))
    out.append("")
    if not problems:
        out.append(paint("  Nothing wrong with this run.", "green"))
        out.append("")
    for f in problems:
        out += _finding_block(f, width, paint, dot)

    out += _disk_block(d, width, paint, rule)
    return "\n".join(out)


def _keep_block(d: Diagnosis, width: int, paint: _Paint, bar: str, approx: str) -> List[str]:
    p = d.pick
    out: List[str] = []
    if p is None:
        out.append("  " + paint(bar + " ", "yellow") + paint("No eval metric -- there is nothing to rank checkpoints on.", "bold"))
        out.append("")
        return out

    label = p.selection.label
    h = d.history
    if p.keep_step is not None:
        head = f"KEEP checkpoint-{p.keep_step}   {label} {p.keep_value:.4g}"
    else:
        head = f"BEST eval at step {p.best_step}   {label} {p.best_value:.4g}"
    out.append("  " + paint(bar + " ", "green") + paint(head, "bold"))

    last_ckpt = max(h.checkpoint_steps(), default=p.final_step)
    # "final checkpoint" only when the last eval really is the last checkpoint;
    # otherwise say what was measured -- the last eval.
    subject = (
        f"the final checkpoint (step {p.final_step})"
        if last_ckpt <= p.final_step
        else f"the last eval (step {p.final_step})"
    )
    if p.final_is_equivalent:
        verdict = (
            f"{subject} is within noise of the best -- keeping it is fine"
            if p.final_step != p.best_step
            else f"{subject} is the best one"
        )
    else:
        verdict = (
            f"{subject} is {p.final_gap_rel:.0%} worse -- outside the noise band, so the "
            "difference is real"
        )
    for chunk in textwrap.wrap(verdict, width - 8):
        out.append("    " + paint(chunk, "dim"))
    if p.keep_step is None and not p.checkpoints_known:
        for chunk in textwrap.wrap(
            "which checkpoints still exist is not recorded in trainer_state.json -- point ftdoctor at "
            "the output directory to restrict the pick to checkpoints you actually have",
            width - 8,
        ):
            out.append("    " + paint(chunk, "dim"))
    elif p.unsaved_best:
        for chunk in textwrap.wrap(
            f"the best eval, at step {p.best_step}, was never saved as a checkpoint", width - 8
        ):
            out.append("    " + paint(chunk, "dim"))
    out.append("")

    rows = d.table()
    if rows:
        out += _table(rows, label, paint, approx)
    return out


def _table(rows: List[Row], label: str, paint: _Paint, approx: str) -> List[str]:
    # Show what matters: kept, final, equivalents, and a few worse ones; fold the rest.
    important = [r for r in rows if r.status in ("keep", "equivalent") or r.is_final]
    others = [r for r in rows if r not in important]
    shown = sorted(important + others[: max(0, 10 - len(important))], key=lambda r: r.step)
    hidden = len(rows) - len(shown)
    out = ["    " + paint(f"{'checkpoint':<18}{label:>14}{'size':>12}   ", "dim")]
    for r in shown:
        value = f"{r.value:.4g}" if r.value is not None else "-"
        status = {
            "keep": paint("keep", "green", "bold"),
            "equivalent": paint(f"{approx} same, within noise", "dim"),
            "worse": paint("worse", "yellow"),
            "not evaluated": paint("not evaluated", "dim"),
        }[r.status]
        final = paint("  (final)", "dim") if r.is_final else ""
        out.append(f"    {'checkpoint-' + str(r.step):<18}{value:>14}{human(r.size_bytes):>12}   {status}{final}")
    if hidden:
        out.append("    " + paint(f"... {hidden} more", "dim"))
    out.append("")
    return out


def _finding_block(f: Finding, width: int, paint: _Paint, dot: str) -> List[str]:
    style = _SEV_STYLE[f.severity]
    out = [f"  {paint(' ' + f.severity.label + ' ', 'bold', style)} {paint(f.title, 'bold')}"]
    body = width - 8
    for chunk in textwrap.wrap(f.summary, body):
        out.append("        " + chunk)
    out.append("")
    if f.evidence:
        for item in f.evidence:
            chunks = textwrap.wrap(item, body - 4) or [""]
            out.append("          " + paint(dot, "dim") + " " + paint(chunks[0], "dim"))
            out += ["            " + paint(c, "dim") for c in chunks[1:]]
        out.append("")
    if f.fix:
        out.append("        " + paint("fix", "dim", "bold"))
        for i, item in enumerate(f.fix, 1):
            chunks = textwrap.wrap(item, body - 5) or [""]
            out.append(f"          {paint(str(i) + '.', 'dim')} {chunks[0]}")
            out += ["             " + c for c in chunks[1:]]
        out.append("")
    return out


def _disk_block(d: Diagnosis, width: int, paint: _Paint, rule: str) -> List[str]:
    local = [c for c in d.history.checkpoints if c.where == "disk" and c.size_bytes]
    if len(local) < 2:
        return []
    from .prune import plan

    resume = plan(d, mode="resume-state")
    full = plan(d, mode="full")
    out = [_rule("DISK", width, paint, rule), ""]
    keep = ", ".join(f"checkpoint-{s}" for s in resume.keep)
    text = (
        f"{len(local)} checkpoints use {human(sum(c.size_bytes for c in local))}. "
        f"`ftdoctor prune` would keep {keep} untouched and strip resume-only state "
        f"(optimizer, scheduler, RNG) from the rest: {human(resume.freed_bytes)} back, every "
        "model still loadable."
    )
    if not full.refused and full.freed_bytes:
        text += f" `--mode full` deletes the other checkpoints outright: {human(full.freed_bytes)}."
    for chunk in textwrap.wrap(text, width - 6):
        out.append("  " + chunk)
    out.append("  " + paint("Nothing is deleted without --apply.", "dim"))
    out.append("")
    return out


def _rule(label: str, width: int, paint: _Paint, rule: str) -> str:
    text = f"{rule * 2} {label} "
    return "  " + paint(text + rule * max(width - len(text) - 4, 0), "dim")


def render_markdown(d: Diagnosis) -> str:
    h = d.history
    out = [f"## ftdoctor — `{h.name}`", ""]
    p = d.pick
    if p is None:
        out += ["**No eval metric was logged**, so there is nothing to rank checkpoints on.", ""]
    else:
        label = p.selection.label
        if p.keep_step is not None:
            out.append(f"**Keep `checkpoint-{p.keep_step}`** — {label} {p.keep_value:.4g}.")
        else:
            out.append(f"**Best eval at step {p.best_step}** — {label} {p.best_value:.4g}.")
        if p.final_is_equivalent:
            out.append(f"The final checkpoint (step {p.final_step}) is within noise of the best.")
        else:
            out.append(
                f"The final checkpoint (step {p.final_step}) is **{p.final_gap_rel:.0%} worse** "
                f"(outside the noise band: {describe_band(p)})."
            )
        out.append("")
        rows = d.table()
        if rows:
            out += [f"| checkpoint | {label} | size | |", "|---|---|---|---|"]
            for r in rows:
                value = f"{r.value:.4g}" if r.value is not None else "–"
                note = r.status + (" (final)" if r.is_final else "")
                out.append(f"| `checkpoint-{r.step}` | {value} | {human(r.size_bytes)} | {note} |")
            out.append("")
    for f in d.problems:
        if f.severity == Severity.OK:
            continue
        out += [f"### {f.severity.label} — {f.title}", "", f.summary, ""]
        out += [f"- {e}" for e in f.evidence]
        if f.evidence:
            out.append("")
        if f.fix:
            out += ["**Fix**", ""] + [f"{i}. {x}" for i, x in enumerate(f.fix, 1)] + [""]
    out.append(f"<sub>generated by [ftdoctor](https://github.com/junglezke/ftdoctor) v{d.version}</sub>")
    return "\n".join(out)


def render_json(d: Diagnosis) -> str:
    return json.dumps(d.to_dict(), indent=2, default=str)


def render(d: Diagnosis, fmt: str = "terminal") -> str:
    renderers = {"terminal": render_terminal, "markdown": render_markdown, "md": render_markdown, "json": render_json}
    if fmt not in renderers:
        raise KeyError(f"unknown format {fmt!r}; choose from {', '.join(sorted(renderers))}")
    return renderers[fmt](d)
