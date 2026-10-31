#!/usr/bin/env python
"""Execute notebook/code_track.ipynb in place and save it with its outputs.

Usage (from the repository root, with the project venv):
    scripts/.venv/Scripts/python scripts/run_notebook.py          # Windows
    scripts/.venv/bin/python scripts/run_notebook.py              # macOS / Linux

The notebook runs with `notebook/` as its working directory and the venv's own
Python (the default `python3` kernel of the ipykernel installed in that venv).
Each cell may take up to 15 minutes. Pass another notebook path as the first
argument to run a different file.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebook"
DEFAULT_NOTEBOOK = NOTEBOOK_DIR / "code_track.ipynb"
CELL_TIMEOUT_S = 15 * 60
KERNEL_NAME = "python3"


def main() -> int:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    if sys.platform == "win32":
        # The default Proactor loop does not work with pyzmq on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT_S,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    print(f"Executing {path} with the {KERNEL_NAME!r} kernel ({sys.executable})")
    t0 = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - t0
    nbformat.write(nb, path)
    print(f"Done in {elapsed:.1f} s. Saved {path} ({path.stat().st_size / 1e6:.2f} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
