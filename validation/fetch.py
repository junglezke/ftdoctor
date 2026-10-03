"""Download the public training logs listed in runs.json into validation/.cache/.

    python validation/fetch.py

The logs are other people's work, so they are cached locally and never
committed. Re-running is a no-op for files already present.
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / ".cache"


def main() -> int:
    CACHE.mkdir(exist_ok=True)
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))["runs"]
    for run in runs:
        target = CACHE / f"{run['id']}.json"
        if target.exists() and target.stat().st_size > 0:
            print(f"cached   {run['id']}")
            continue
        url = f"{run['source'].rstrip('/')}/resolve/main/{run['file']}"
        print(f"fetching {run['id']}  <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        except OSError as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
