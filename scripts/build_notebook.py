"""Convert the cell source to a Colab notebook without an extra dependency."""
import ast
import json
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "strata_l4_gsq_rco_iq3_xxs.py"
cells = []
kind = None
lines = []


def append_cell():
    if kind is None:
        return
    text = "".join(lines).strip() + "\n"
    if kind == "code" and not text.startswith("!"):
        ast.parse(text)
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


for line in source.read_text().splitlines(keepends=True):
    if line.startswith("# %%"):
        append_cell()
        kind = "markdown" if "[markdown]" in line else "code"
        lines = []
    elif kind == "markdown":
        lines.append(line[2:] if line.startswith("# ") else line.removeprefix("#"))
    else:
        lines.append(line)
append_cell()
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "Strata L4 / A100 GSQ-RCO IQ3_XXS", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
for i, cell in enumerate(cells):
    cell["id"] = f"strata-colab-{i:02d}"
target = source.with_suffix(".ipynb")
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(target)
