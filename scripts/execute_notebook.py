"""Exécute `notebooks/01_exploration.ipynb` et enregistre ses résultats dans le fichier.

À lancer après `scripts/generate_notebook.py` (qui régénère le notebook sans
résultats) pour que les graphiques soient visibles sans avoir à relancer le
notebook, par exemple lors d'un aperçu sur GitHub :

    python scripts/generate_notebook.py
    python scripts/execute_notebook.py

Les sorties HTML volumineuses (l'animation `to_jshtml`, plusieurs Mo) sont
retirées pour garder le fichier léger ; la cellule les régénère à chaque exécution.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

_NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "01_exploration.ipynb"
_MAX_HTML_OUTPUT_BYTES = 200_000
_TIMEOUT_SECONDS_PER_CELL = 900


def _strip_heavy_html_outputs(notebook: nbformat.NotebookNode) -> int:
    """Remplace les sorties HTML trop lourdes par une courte note. Retourne leur nombre."""
    stripped = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for index, output in enumerate(cell.outputs):
            html = output.get("data", {}).get("text/html", "")
            if len("".join(html)) > _MAX_HTML_OUTPUT_BYTES:
                cell.outputs[index] = nbformat.v4.new_output(
                    "stream",
                    name="stdout",
                    text="[Animation non conservée dans le fichier (trop volumineuse) : "
                    "exécuter cette cellule pour la voir.]\n",
                )
                stripped += 1
    return stripped


def main() -> None:
    notebook = nbformat.read(_NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=_TIMEOUT_SECONDS_PER_CELL, resources={"metadata": {"path": str(_NOTEBOOK.parent)}}).execute()
    stripped = _strip_heavy_html_outputs(notebook)
    nbformat.write(notebook, _NOTEBOOK)
    size_kb = _NOTEBOOK.stat().st_size / 1024
    print(f"Notebook exécuté : {_NOTEBOOK} ({size_kb:.0f} Ko, {stripped} sortie(s) volumineuse(s) retirée(s))")


if __name__ == "__main__":
    main()
