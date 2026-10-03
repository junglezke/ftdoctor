"""Diagnose every cached public run and compare against the pinned verdicts.

    python validation/fetch.py
    python validation/run.py            # print the table
    python validation/run.py --check    # exit 1 if any verdict changed

Run this before changing a threshold. Unit tests show a check *can* fire; these
runs show whether it fires on fine-tunes that exist -- and, as importantly,
that it stays quiet on the ones that are fine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ftdoctor import diagnose, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 on any verdict drift")
    args = parser.parse_args()

    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))["runs"]
    drift = 0
    print(f"{'run':<30}{'steps':>7}  findings")
    print("-" * 100)
    for run in runs:
        path = HERE / ".cache" / f"{run['id']}.json"
        if not path.exists():
            print(f"{run['id']:<30}  not cached -- run validation/fetch.py first")
            drift += 1
            continue
        result = diagnose(load(str(path)))
        got = {f.check: f.severity.name for f in result.findings}
        ok = got == run["expect"]
        drift += not ok
        shown = ", ".join(f"{k}:{v}" for k, v in sorted(got.items())) or "nothing to report"
        print(f"{run['id']:<30}{result.history.final_step:>7}  [{'ok' if ok else 'DRIFT'}] {shown}")
        if not ok:
            print(f"{'':<39}expected: {run['expect'] or 'nothing'}")
    print()
    if drift:
        print(
            f"{drift} run(s) changed verdict. If the new verdict is right, update runs.json and "
            "say why in the commit; if not, the change broke something real."
        )
    return 1 if (args.check and drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
