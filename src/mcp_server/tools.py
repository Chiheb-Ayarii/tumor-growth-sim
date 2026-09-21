"""Logique des outils MCP, en fonctions Python pures et testables.

Séparé de `server.py` (qui se contente d'enregistrer ces fonctions comme
outils MCP) pour pouvoir tester cette logique avec pytest sans dépendre du
protocole MCP lui-même — le protocole n'est qu'une fine couche de transport
au-dessus de ces fonctions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.diffusion_reaction import _laplacian_neumann, simulate_2d
from simulation.gompertz import simulate

_MODEL_INFO: dict[str, dict[str, Any]] = {
    "gompertz": {
        "description": (
            "Modèle 0D de croissance tumorale à volume unique : "
            "dV/dt = r . V . ln(K / V). La croissance ralentit progressivement "
            "à mesure que le volume V approche de la capacité limite K."
        ),
        "parameters": {
            "V0": "Volume initial (mm³), strictement positif.",
            "r": "Taux de croissance intrinsèque (/jour), strictement positif.",
            "K": "Capacité limite / volume maximal asymptotique (mm³), strictement positif.",
            "t_span": "Intervalle de temps [t0, tf] en jours, t0 < tf.",
            "n_points": "Nombre de points de temps également espacés (>= 2).",
        },
        "reference": (
            "Vaghi et al. 2020, PLOS Computational Biology 16(2):e1007178 "
            "(xénogreffe murine, cancer du sein) : V0=1 mm³, K=2600 mm³, "
            "r≈0.0738 /jour."
        ),
    },
    "fisher_kpp": {
        "description": (
            "Modèle 2D d'invasion tumorale par diffusion-réaction : "
            "du/dt = D . laplacien(u) + r . u . (1 - u/K), sur une grille carrée. "
            "Historiquement utilisé pour l'invasion des gliomes."
        ),
        "parameters": {
            "grid_size": "Nombre de cellules par côté de la grille (entier positif).",
            "D": "Coefficient de diffusion (mm²/jour), strictement positif.",
            "r": "Taux de prolifération intrinsèque (/jour), strictement positif.",
            "K": "Densité maximale locale / capacité du tissu, strictement positive.",
            "initial_tumor_position": "Indices [i, j] du centre du foyer initial.",
            "initial_radius": "Rayon initial du foyer, en nombre de cellules de grille.",
            "n_steps": "Nombre de pas de temps à simuler (entier positif).",
            "domain_size_mm": "Taille physique du domaine carré, en mm (optionnel, 30 mm par défaut).",
        },
        "reference": (
            "Swanson et al. 2002, J Neurol Sci 216(1):1-10 (gliome) : "
            "ρ=0.012 /jour, D_gris=0.13 mm²/jour, D_blanc=0.65 mm²/jour."
        ),
    },
}

_SUPPORTED_MODELS = tuple(_MODEL_INFO)


def get_model_info() -> dict[str, Any]:
    """Décrit les modèles de simulation disponibles et leurs paramètres.

    Returns:
        Un dictionnaire `{nom_du_modèle: {description, parameters, reference}}`.
    """
    return _MODEL_INFO


def _effective_radius_mm(grid: np.ndarray, K: float, dx: float) -> float:
    """Rayon effectif d'un foyer tumoral (seuil u > K/2), en mm."""
    return float(np.sqrt(np.sum(grid > 0.5 * K) / np.pi) * dx)


def _downsample_grid(grid: np.ndarray, max_size: int) -> list[list[float]]:
    """Réduit une grille carrée à au plus `max_size` x `max_size` cellules.

    Utilise une moyenne par blocs quand la taille d'origine est un multiple
    exact de la taille cible (pas d'aliasing) ; sinon, un sous-échantillonnage
    régulier par indices. Sert uniquement à garder la réponse MCP compacte —
    ce n'est pas la grille utilisée pour les calculs.
    """
    n = grid.shape[0]
    if n <= max_size:
        return grid.tolist()
    if n % max_size == 0:
        factor = n // max_size
        reduced = grid.reshape(max_size, factor, max_size, factor).mean(axis=(1, 3))
    else:
        indices = np.linspace(0, n - 1, max_size).round().astype(int)
        reduced = grid[np.ix_(indices, indices)]
    return reduced.tolist()


def simulate_tumor_growth(
    model: str,
    params: dict[str, Any],
    include_final_grid: bool = False,
    max_grid_preview_size: int = 40,
) -> dict[str, Any]:
    """Simule la croissance tumorale avec le modèle et les paramètres donnés.

    Args:
        model: `"gompertz"` ou `"fisher_kpp"`.
        params: Paramètres du modèle choisi (voir `get_model_info()` pour le
            détail de chacun). Pour `fisher_kpp`, `initial_tumor_position`
            peut être une liste `[i, j]` (convertie en tuple).
        include_final_grid: Pour `fisher_kpp` uniquement — si True, inclut
            une version réduite de la grille finale dans la réponse (voir
            `max_grid_preview_size`). Ignoré pour `gompertz`.
        max_grid_preview_size: Taille maximale (par côté) de la grille
            renvoyée si `include_final_grid=True` — la grille complète
            n'est jamais renvoyée telle quelle, pour garder une réponse
            compacte.

    Returns:
        Pour `gompertz` : `{"model", "t", "V"}`.
        Pour `fisher_kpp` : `{"model", "t", "effective_radius_mm", "dx", "dt",
        [+ "final_grid_preview" si demandé]}`.

    Raises:
        ValueError: Si `model` n'est pas reconnu, ou si les paramètres sont
            invalides (validation déléguée à `simulate`/`simulate_2d`).
    """
    if model not in _SUPPORTED_MODELS:
        raise ValueError(f"model doit être l'un de {_SUPPORTED_MODELS!r}, reçu model={model!r}")

    if model == "gompertz":
        t, V = simulate(
            V0=params["V0"],
            r=params["r"],
            K=params["K"],
            t_span=tuple(params["t_span"]),
            n_points=params["n_points"],
        )
        return {"model": model, "t": t.tolist(), "V": V.tolist()}

    position = params["initial_tumor_position"]
    kwargs: dict[str, Any] = dict(
        grid_size=params["grid_size"],
        D=params["D"],
        r=params["r"],
        K=params["K"],
        initial_tumor_position=tuple(position),
        initial_radius=params["initial_radius"],
        n_steps=params["n_steps"],
    )
    if "domain_size_mm" in params:
        kwargs["domain_size_mm"] = params["domain_size_mm"]

    result = simulate_2d(**kwargs)
    K = params["K"]
    effective_radius_mm = [_effective_radius_mm(grid, K, result.dx) for grid in result.grids]

    response: dict[str, Any] = {
        "model": model,
        "t": result.t.tolist(),
        "effective_radius_mm": effective_radius_mm,
        "dx": result.dx,
        "dt": result.dt,
    }
    if include_final_grid:
        response["final_grid_preview"] = _downsample_grid(result.grids[-1], max_grid_preview_size)
    return response


def compare_surgical_scenarios(
    grid_size: int,
    D: float,
    r: float,
    K: float,
    initial_tumor_position: list[int] | tuple[int, int],
    initial_radius: float,
    n_steps: int,
    resection_step: int,
    resection_center: list[int] | tuple[int, int],
    resection_radius: float,
    domain_size_mm: float | None = None,
) -> dict[str, Any]:
    """Compare la croissance tumorale (Fisher-KPP) avec et sans résection chirurgicale simulée.

    Simule la croissance normalement jusqu'à `resection_step`, puis simule
    deux scénarios en parallèle à partir de cet instant, sur le nombre de
    pas restant :

    - **Sans résection** : la simulation continue sans modification.
    - **Avec résection** : la densité tumorale est mise à zéro dans un
        disque de centre `resection_center` et de rayon `resection_radius`
        (cellules de grille) — une résection chirurgicale idéalisée,
        instantanée et parfaitement localisée — puis la simulation continue.

    Attention : « tumeur invisible » n'est pas « tumeur éliminée ». Le rayon
    effectif ne compte que les cellules de densité > K/2 ; la densité déborde
    du bord visible sous forme d'une queue très faible mais non nulle, qui se
    remultiplie. Si le disque réséqué ne couvre pas toute cette queue, le rayon
    effectif tombe à 0 juste après la résection puis la tumeur repousse. Dans
    le schéma numérique la densité ne se propage que d'une cellule par pas :
    au-delà d'une distance finie elle vaut exactement 0, et une résection qui
    enlève toutes les cellules de densité non nulle élimine la tumeur pour de
    bon. Dans l'équation continue cette queue est infinie, donc cette
    élimination exacte est une propriété du schéma et non un résultat biologique.

    Args:
        grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps,
            domain_size_mm : voir `simulate_2d`.
        resection_step: Pas de temps auquel la résection est simulée
            (doit être strictement compris entre 0 et `n_steps`).
        resection_center: Indices [i, j] du centre de la zone réséquée.
        resection_radius: Rayon de la zone réséquée, en cellules de grille.

    Returns:
        Un dictionnaire `{t, effective_radius_mm_no_resection,
        effective_radius_mm_with_resection, resection_time}` — les deux
        trajectoires de rayon effectif partagent le même axe de temps
        (identique jusqu'à `resection_step`, puis divergent).

    Raises:
        ValueError: Si `resection_step` n'est pas strictement compris entre
            0 et `n_steps`, ou si les autres paramètres sont invalides
            (validation déléguée à `simulate_2d`).
    """
    if not (0 < resection_step < n_steps):
        raise ValueError(
            f"resection_step doit être strictement compris entre 0 et n_steps={n_steps!r}, "
            f"reçu resection_step={resection_step!r}"
        )

    common_kwargs: dict[str, Any] = dict(
        grid_size=grid_size,
        D=D,
        r=r,
        K=K,
        initial_tumor_position=tuple(initial_tumor_position),
        initial_radius=initial_radius,
    )
    if domain_size_mm is not None:
        common_kwargs["domain_size_mm"] = domain_size_mm

    # Phase commune : jusqu'à resection_step, les deux scénarios sont identiques.
    phase1 = simulate_2d(n_steps=resection_step, **common_kwargs)
    n_remaining = n_steps - resection_step

    # Scénario 1 : pas de résection, on continue tel quel à partir de l'état courant.
    t_rel, radii_no_resection_rel = _continue_simulation(
        initial_grid=phase1.grids[-1], dx=phase1.dx, dt=phase1.dt, D=D, r=r, K=K, n_steps=n_remaining
    )

    # Scénario 2 : on réséque (densité mise à zéro dans un disque), puis on continue.
    resected_grid = phase1.grids[-1].copy()
    i0, j0 = resection_center
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    resected_grid[(ii - i0) ** 2 + (jj - j0) ** 2 <= resection_radius**2] = 0.0
    _, radii_with_resection_rel = _continue_simulation(
        initial_grid=resected_grid, dx=phase1.dx, dt=phase1.dt, D=D, r=r, K=K, n_steps=n_remaining
    )

    radii_phase1 = [_effective_radius_mm(g, K, phase1.dx) for g in phase1.grids]
    t_phase1 = phase1.t.tolist()
    t_phase2 = [phase1.t[-1] + dt_rel for dt_rel in t_rel[1:]]

    return {
        "t": t_phase1 + t_phase2,
        "effective_radius_mm_no_resection": radii_phase1 + radii_no_resection_rel[1:],
        "effective_radius_mm_with_resection": radii_phase1 + radii_with_resection_rel[1:],
        "resection_time": float(phase1.t[-1]),
    }


def _continue_simulation(
    initial_grid: np.ndarray, dx: float, dt: float, D: float, r: float, K: float, n_steps: int
) -> tuple[list[float], list[float]]:
    """Poursuit une simulation Fisher-KPP de `n_steps` pas à partir d'une grille donnée.

    Réutilise le pas de temps `dt` déjà validé (CFL) par la simulation
    initiale, pour rester cohérent avec elle plutôt que d'en recalculer un
    nouveau. Retourne `(t_relatifs, rayons_effectifs_mm)`, y compris l'état
    initial (t=0), avec la même formule d'évolution que `simulate_2d`.
    """
    u = initial_grid.copy()
    radii = [_effective_radius_mm(u, K, dx)]
    t = [0.0]
    for step in range(1, n_steps + 1):
        laplacian = _laplacian_neumann(u, dx)
        u = u + dt * (D * laplacian + r * u * (1 - u / K))
        u = np.clip(u, 0.0, K)
        radii.append(_effective_radius_mm(u, K, dx))
        t.append(step * dt)
    return t, radii
