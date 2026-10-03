"""Render ANSI terminal output to a self-contained SVG.

Used to generate the banner images in the project READMEs. GitHub renders SVG
through <img>, which blocks scripts and external fonts, so everything here is
plain <text> with inline fills and a generic monospace stack.
"""

from __future__ import annotations

import re
import sys
from html import escape

ANSI = re.compile(r"\033\[([0-9;]*)m")

# Tokyo-night-ish, chosen for contrast on both GitHub themes.
BG = "#16161e"
FG = "#c0caf5"
COLORS = {
    "2": "#565f89",  # dim
    "1": None,  # bold: handled as weight
    "31": "#f7768e",
    "91": "#ff7a93",
    "33": "#e0af68",
    "34": "#7aa2f7",
    "32": "#9ece6a",
    "36": "#7dcfff",
    "35": "#bb9af7",
}

CHAR_W = 8.4
LINE_H = 19.0
PAD_X = 22.0
PAD_TOP = 44.0
PAD_BOTTOM = 20.0


def parse_line(line: str):
    """Split an ANSI line into (text, color, bold) runs."""
    runs, pos = [], 0
    color, bold = None, False
    for match in ANSI.finditer(line):
        chunk = line[pos : match.start()]
        if chunk:
            runs.append((chunk, color, bold))
        for code in (match.group(1) or "0").split(";"):
            if code in ("", "0"):
                color, bold = None, False
            elif code == "1":
                bold = True
            elif code == "2":
                color = COLORS["2"]
            elif code in COLORS and COLORS[code]:
                color = COLORS[code]
        pos = match.end()
    tail = line[pos:]
    if tail:
        runs.append((tail, color, bold))
    return runs


def render(text: str, title: str) -> str:
    lines = text.rstrip("\n").split("\n")
    cols = max((len(ANSI.sub("", line)) for line in lines), default=80)
    width = cols * CHAR_W + PAD_X * 2
    height = len(lines) * LINE_H + PAD_TOP + PAD_BOTTOM

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,&quot;Liberation Mono&quot;,monospace" font-size="13">',
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="10" fill="{BG}"/>',
        # window chrome
        '<circle cx="24" cy="22" r="6" fill="#f7768e"/>',
        '<circle cx="44" cy="22" r="6" fill="#e0af68"/>',
        '<circle cx="64" cy="22" r="6" fill="#9ece6a"/>',
        f'<text x="{width / 2:.0f}" y="27" fill="#565f89" font-size="12" '
        f'text-anchor="middle">{escape(title)}</text>',
    ]

    for row, line in enumerate(lines):
        y = PAD_TOP + row * LINE_H + 12
        col = 0
        for chunk, color, bold in parse_line(line):
            if chunk.strip():
                x = PAD_X + col * CHAR_W
                attrs = f' fill="{color or FG}"'
                if bold:
                    attrs += ' font-weight="600"'
                # textLength pins each run to an exact width so the layout
                # holds whatever monospace font the viewer actually resolves.
                # Without it the SVG overflows its own viewBox on any font
                # whose advance differs from our estimate.
                out.append(
                    f'<text x="{x:.1f}" y="{y:.1f}"{attrs} '
                    f'textLength="{len(chunk) * CHAR_W:.1f}" '
                    f'lengthAdjust="spacingAndGlyphs" '
                    f'xml:space="preserve">{escape(chunk)}</text>'
                )
            col += len(chunk)

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "terminal"
    data = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    sys.stdout.buffer.write(render(data, title).encode("utf-8"))
