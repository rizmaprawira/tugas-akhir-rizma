from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = Path("/Users/rizzie/ClimateData")
DATA_ROOT = Path("/Users/rizzie/TugasAkhir/data_processing/data")
RESULTS_ROOT = Path("/Users/rizzie/TugasAkhir/data_processing/results")


SOURCE_REPLACEMENTS = {
    "../../../external/ClimateData": str(EXTERNAL_ROOT),
    "../../../data/intermediate/divided_correlation": str(DATA_ROOT / "intermediate" / "divided_correlation"),
    "../../../results": str(RESULTS_ROOT),
}


def rewrite_notebook(path: Path) -> None:
    nb = nbformat.read(path.open(), as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type not in {"code", "markdown"}:
            continue
        source = cell.source
        new_source = source
        for old, new in SOURCE_REPLACEMENTS.items():
            new_source = new_source.replace(old, new)
        if new_source != source:
            cell.source = new_source
            changed = True
    if changed:
        nbformat.write(nb, path.open("w"))


def main() -> None:
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        rewrite_notebook(path)
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
