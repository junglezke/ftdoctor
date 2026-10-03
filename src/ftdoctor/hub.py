"""Read a run straight from a model repository on the HuggingFace Hub.

    ftdoctor diagnose someone/their-finetuned-model

Thousands of public fine-tunes upload ``trainer_state.json`` next to their
weights, usually without the author ever looking at it again. This module
finds it, downloads it, and lists which ``checkpoint-N/`` folders the repo
really holds -- so a recommendation is always a checkpoint you can download.

Standard library only. A token is read from ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``) for private repos and sent to the Hub endpoint and
nowhere else; public repos need none.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .history import Checkpoint, History, from_trainer_state

_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
_CHECKPOINT_FILE = re.compile(r"^(?:.*/)?checkpoint-(\d+)/")


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def normalise(target: str) -> str:
    """Accept ``owner/name``, ``hf://owner/name`` or a full Hub URL."""
    target = target.strip()
    if target.startswith("hf://"):
        target = target[len("hf://") :]
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if target.startswith(prefix):
            target = target[len(prefix) :]
    parts = [p for p in target.split("/") if p]
    return "/".join(parts[:2]) if len(parts) >= 2 else target


def looks_like_repo_id(target: str) -> bool:
    return bool(_REPO_ID.match(normalise(target)))


def _request(url: str, timeout: int = 60) -> bytes:
    headers = {"User-Agent": "ftdoctor"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and url.startswith(_endpoint()):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_files(repo_id: str) -> List[Dict[str, Any]]:
    """Repository files with sizes, from the public model API."""
    url = f"{_endpoint()}/api/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=true"
    try:
        payload = json.loads(_request(url))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise FileNotFoundError(
                f"{repo_id!r} is not a local path, and no public Hub repo of that name was "
                "found (set HF_TOKEN to read a private one)"
            ) from None
        raise
    return [s for s in payload.get("siblings", []) if isinstance(s, dict) and "rfilename" in s]


def choose_state_file(files: List[Dict[str, Any]]) -> Optional[str]:
    """The ``trainer_state.json`` most likely to hold the full history.

    The root copy is written at the end of training and covers every step. A
    ``checkpoint-N`` copy covers steps up to N, so the highest N is the next
    best. Anything else (an adapter subfolder, ``last-checkpoint/``) comes after.
    """
    names = [f["rfilename"] for f in files]
    states = [n for n in names if n == "trainer_state.json" or n.endswith("/trainer_state.json")]
    if not states:
        return None
    if "trainer_state.json" in states:
        return "trainer_state.json"

    def rank(name: str):
        match = _CHECKPOINT_FILE.match(name)
        return (1, int(match.group(1))) if match else (0, -name.count("/"))

    return max(states, key=rank)


def hub_checkpoints(files: List[Dict[str, Any]]) -> List[Checkpoint]:
    """``checkpoint-N/`` folders present in the repo, with their total size."""
    sizes: Dict[int, int] = {}
    resume_only: Dict[int, int] = {}
    from .history import RESUME_ONLY_FILES

    for item in files:
        match = _CHECKPOINT_FILE.match(item["rfilename"])
        if not match:
            continue
        step = int(match.group(1))
        size = int(item.get("size") or (item.get("lfs") or {}).get("size") or 0)
        sizes[step] = sizes.get(step, 0) + size
        leaf = item["rfilename"].split(f"checkpoint-{step}/", 1)[-1].split("/", 1)[0]
        if RESUME_ONLY_FILES.match(leaf):
            resume_only[step] = resume_only.get(step, 0) + size
    return [
        Checkpoint(
            step=step,
            path=None,
            size_bytes=sizes[step] or None,
            resume_only_bytes=resume_only.get(step),
            where="hub",
        )
        for step in sorted(sizes)
    ]


def load(target: str) -> History:
    repo_id = normalise(target)
    files = list_files(repo_id)
    state_file = choose_state_file(files)
    if state_file is None:
        raise FileNotFoundError(
            f"{repo_id} has no trainer_state.json, so there is no training history to "
            "read. It is written by the HuggingFace Trainer; the author would need to "
            "upload it (or a checkpoint-N/ folder) alongside the weights."
        )
    url = f"{_endpoint()}/{repo_id}/resolve/main/{urllib.parse.quote(state_file)}"
    state = json.loads(_request(url))
    history = from_trainer_state(state, name=repo_id, source=f"hf://{repo_id}/{state_file}")
    history.checkpoints = hub_checkpoints(files)
    return history
