"""Command line.

    ftdoctor ./outputs                       # a training output directory
    ftdoctor ./outputs/checkpoint-1200       # or one checkpoint
    ftdoctor trainer_state.json              # or the file
    ftdoctor someone/their-finetuned-model   # or a model on the HuggingFace Hub
    ftdoctor prune ./outputs                 # show what can be deleted
    ftdoctor prune ./outputs --apply         # ...and delete it
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .diagnosis import __version__, diagnose
from .findings import Severity
from .history import load
from .report import emit, render, render_terminal

_FAIL_ON = {"warning": Severity.WARNING, "critical": Severity.CRITICAL}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metric", help="rank checkpoints on this eval metric (e.g. accuracy, wer)")
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--higher-is-better", dest="higher", action="store_true", default=None)
    direction.add_argument("--lower-is-better", dest="higher", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftdoctor",
        description="Did your fine-tune overfit? Which checkpoint should you keep? What can you delete?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"ftdoctor {__version__}")
    sub = parser.add_subparsers(dest="command")

    diag = sub.add_parser("diagnose", help="diagnose a run (the default command)")
    diag.add_argument("target", help="output dir, checkpoint dir, trainer_state.json, or owner/model")
    _common(diag)
    diag.add_argument("-f", "--format", default="terminal", choices=["terminal", "markdown", "json"])
    diag.add_argument("-o", "--output", help="write the report to a file")
    diag.add_argument("--fail-on", choices=["never", "warning", "critical"], default="never")
    diag.add_argument("--no-color", action="store_true")

    prune = sub.add_parser("prune", help="free disk space held by checkpoints you will not use")
    prune.add_argument("target", help="training output directory containing checkpoint-*/")
    _common(prune)
    prune.add_argument(
        "--mode",
        choices=["resume-state", "full"],
        default="resume-state",
        help="resume-state (default): strip optimizer/scheduler/RNG state, keep weights. "
        "full: delete whole checkpoints except the kept ones.",
    )
    prune.add_argument("--keep", type=int, nargs="+", default=[], metavar="STEP", help="also keep these checkpoints")
    prune.add_argument("--apply", action="store_true", help="actually delete (otherwise just show the plan)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `ftdoctor TARGET` is shorthand for `ftdoctor diagnose TARGET`.
    if argv and argv[0] not in ("diagnose", "prune", "-h", "--help", "--version"):
        argv.insert(0, "diagnose")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        history = load(args.target)
        result = diagnose(history, metric=args.metric, higher_is_better=args.higher)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ftdoctor: {exc}", file=sys.stderr)
        return 2

    if args.command == "prune":
        return _prune(result, args)

    if args.format == "terminal" and args.no_color:
        text = render_terminal(result, color=False)
    else:
        text = render(result, args.format)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        emit(text)
    if args.fail_on == "never":
        return 0
    return result.exit_code(_FAIL_ON[args.fail_on])


def _prune(result, args) -> int:
    from .prune import apply, human, plan

    p = plan(result, mode=args.mode, keep_extra=args.keep)
    if p.refused:
        print(f"ftdoctor: {p.refused}", file=sys.stderr)
        return 2
    lines = [
        "",
        f"  run:  {p.run_dir}",
        f"  mode: {p.mode}",
        f"  keep: {', '.join(f'checkpoint-{s}' for s in p.keep)}",
        "",
    ]
    if not p.actions:
        lines.append("  nothing to remove.")
        emit("\n".join(lines))
        return 0
    for action in p.actions:
        what = "whole checkpoint" if p.mode == "full" else ", ".join(
            t.replace("\\", "/").rsplit("/", 1)[-1] for t in action.targets
        )
        lines.append(f"  {action.checkpoint.name:<18}{human(action.freed_bytes):>10}   {what}")
    lines += ["", f"  total: {human(p.freed_bytes)} from {len(p.actions)} checkpoint(s)", ""]
    if not args.apply:
        lines.append("  dry run -- nothing was deleted. Re-run with --apply to delete the above.")
        emit("\n".join(lines))
        return 0
    emit("\n".join(lines))
    freed, log = apply(p)
    for line in log:
        print(f"  {line}")
    print(f"\n  freed {human(freed)}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
