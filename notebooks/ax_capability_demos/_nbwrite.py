from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(filename: str, cells: list) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    path = ROOT / filename
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
