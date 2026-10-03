"""Regenerate docs/assets/report.svg, the banner in the README.

    python tools/make_banner.py

The banner is ftdoctor's real output on a public model, read live from the Hub,
so the image cannot drift from what the tool prints. Needs network access.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from ansi2svg import render  # noqa: E402

from ftdoctor import diagnose, load  # noqa: E402
from ftdoctor.report import render_terminal  # noqa: E402

REPO = "anshpunia8597/mistral-7b-medical-qa-finetuned"


def main() -> None:
    lines = render_terminal(diagnose(load(REPO)), width=90, color=True, unicode=True).split("\n")
    # Header, the keep block and its table: the answer people open the report for.
    stop = next(i for i, line in enumerate(lines) if "FINDINGS" in line)
    svg = render("\n".join(lines[1:stop]).rstrip(), f"ftdoctor {REPO}")
    out = HERE.parent / "docs" / "assets" / "report.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
