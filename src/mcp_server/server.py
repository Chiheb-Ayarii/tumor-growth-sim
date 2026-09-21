"""Serveur MCP exposant les simulations tumorales comme outils appelables par un agent.

Enregistre les fonctions de `mcp_server.tools` comme outils MCP. Lancement
(transport stdio, celui qu'utilisent Claude Desktop et la plupart des
clients MCP locaux) :

    python -m mcp_server
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import tools


@contextmanager
def _report_invalid_input() -> Iterator[None]:
    """Transforme les erreurs de saisie en `ToolError`, dont le message est transmis à l'agent.

    Sans ça, le SDK traite une `ValueError`/`KeyError` comme un plantage du
    serveur et ne renvoie à l'agent qu'un message générique : il ne saurait
    pas quel paramètre corriger.
    """
    try:
        yield
    except KeyError as exc:
        raise ToolError(f"Paramètre manquant : {exc.args[0]!r} (voir get_model_info).") from exc
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Paramètres invalides : {exc}") from exc


mcp = MCPServer(
    name="tumor-growth-sim",
    instructions=(
        "Simulations de croissance tumorale (Gompertz 0D, Fisher-KPP 2D) à visée "
        "pédagogique et exploratoire, pas clinique. Commencer par get_model_info "
        "pour connaître les modèles et leurs paramètres."
    ),
)


@mcp.tool()
def get_model_info() -> dict[str, Any]:
    """Décrit les modèles disponibles (Gompertz, Fisher-KPP), leurs paramètres et leurs références."""
    return tools.get_model_info()


@mcp.tool()
def simulate_tumor_growth(
    model: str,
    params: dict[str, Any],
    include_final_grid: bool = False,
    max_grid_preview_size: int = 40,
) -> dict[str, Any]:
    """Simule la croissance tumorale avec le modèle `gompertz` ou `fisher_kpp`.

    `params` dépend du modèle (voir get_model_info). Pour fisher_kpp, la réponse
    contient le rayon effectif au cours du temps, pas la grille complète ;
    `include_final_grid=True` ajoute un aperçu réduit de la grille finale.
    """
    with _report_invalid_input():
        return tools.simulate_tumor_growth(model, params, include_final_grid, max_grid_preview_size)


@mcp.tool()
def compare_surgical_scenarios(
    grid_size: int,
    D: float,
    r: float,
    K: float,
    initial_tumor_position: list[int],
    initial_radius: float,
    n_steps: int,
    resection_step: int,
    resection_center: list[int],
    resection_radius: float,
    domain_size_mm: float | None = None,
) -> dict[str, Any]:
    """Compare la croissance Fisher-KPP avec et sans résection chirurgicale simulée.

    À `resection_step`, la densité tumorale est mise à zéro dans un disque
    (`resection_center`, `resection_radius` en cellules de grille), puis la
    simulation continue. Renvoie le rayon effectif au cours du temps pour les
    deux scénarios. Idéalisation : résection instantanée, sans effet sur le tissu voisin.
    Attention : un rayon effectif tombé à 0 ne prouve pas l'élimination. Si le disque
    ne couvre pas la faible queue de densité au-delà du bord visible, la tumeur repousse.
    Vérifier le rayon en fin de simulation, pas seulement juste après la résection.
    """
    with _report_invalid_input():
        return tools.compare_surgical_scenarios(
            grid_size=grid_size,
            D=D,
            r=r,
            K=K,
            initial_tumor_position=initial_tumor_position,
            initial_radius=initial_radius,
            n_steps=n_steps,
            resection_step=resection_step,
            resection_center=resection_center,
            resection_radius=resection_radius,
            domain_size_mm=domain_size_mm,
        )


def main() -> None:
    """Lance le serveur en transport stdio."""
    mcp.run(transport="stdio")
