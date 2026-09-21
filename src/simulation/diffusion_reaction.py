"""Modèle de diffusion-réaction (Fisher-KPP) pour l'invasion tumorale.

Équation aux dérivées partielles :
    du/dt = D * laplacien(u) + r * u * (1 - u/K)

où u(x, y, t) représente une densité de cellules tumorales, D le
coefficient de diffusion (migration cellulaire), r le taux de
prolifération et K la densité maximale locale (capacité du tissu).

La résolution se fait par différences finies explicites sur une grille 2D
régulière (schéma d'Euler en temps, laplacien à 5 points en espace), sans
dépendance à scipy.integrate.

Le modèle de diffusion-réaction est implémenté selon l'équation de Fisher-KPP
(Fisher-Kolmogorov-Petrovsky-Piskunov), qui combine une diffusion linéaire
non bornée et un terme de croissance logistique locale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from simulation.base import TumorGrowthModel

# Taille physique du domaine carré simulé, en millimètres. Fixée en
# constante de module plutôt qu'en paramètre pour ne pas alourdir la
# signature publique de `simulate_2d`.
#
# Doit rester grande par rapport à l'échelle de longueur de diffusion
# sqrt(D/r) pour les valeurs de D et r typiquement utilisées dans ce
# projet : un domaine trop petit force un pas spatial dx = _DOMAIN_SIZE_MM
# / grid_size trop fin, ce qui rend le patch tumoral initial (quelques
# cellules de rayon) physiquement minuscule et sous-critique — la
# diffusion dilue alors la population plus vite que la réaction logistique
# ne peut la faire croître, et la tumeur s'éteint au lieu de se propager
# (phénomène de patch sous-critique, cf. Kierstead-Slobodkin-Skellam),
# au lieu de suivre la vitesse de front théorique de Fisher-KPP.
_DOMAIN_SIZE_MM = 30.0


@dataclass(frozen=True)
class FisherKPPResult:
    """Résultat d'une simulation de diffusion-réaction Fisher-KPP.

    Attributes:
        t: Instants correspondant à chaque grille enregistrée, de forme
            (n_steps + 1,).
        grids: Densité tumorale à chaque instant, de forme
            (n_steps + 1, grid_size, grid_size).
        dx: Pas spatial utilisé (mm), dérivé de `_DOMAIN_SIZE_MM` et de
            `grid_size`.
        dt: Pas de temps utilisé, calculé automatiquement pour respecter
            la condition de stabilité CFL.
    """

    t: NDArray[np.float64]
    grids: NDArray[np.float64]
    dx: float
    dt: float

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Permet le déballage `t, grids = simulate_2d(...)`."""
        return iter((self.t, self.grids))


def _validate_params(
    grid_size: int,
    D: float,
    r: float,
    K: float,
    initial_tumor_position: tuple[int, int],
    initial_radius: float,
    n_steps: int,
) -> None:
    if grid_size <= 0:
        raise ValueError(f"grid_size doit être strictement positif, reçu grid_size={grid_size!r}")
    if not (D > 0):
        raise ValueError(f"D doit être strictement positif, reçu D={D!r}")
    if not (r > 0):
        raise ValueError(f"r doit être strictement positif, reçu r={r!r}")
    if not (K > 0):
        raise ValueError(f"K doit être strictement positif, reçu K={K!r}")
    if n_steps <= 0:
        raise ValueError(f"n_steps doit être strictement positif, reçu n_steps={n_steps!r}")

    if not (0 < initial_radius < grid_size):
        raise ValueError(
            "initial_radius doit être strictement compris entre 0 et grid_size, "
            f"reçu initial_radius={initial_radius!r} pour grid_size={grid_size!r}"
        )

    i0, j0 = initial_tumor_position
    if not (0 <= i0 < grid_size and 0 <= j0 < grid_size):
        raise ValueError(
            "initial_tumor_position doit désigner un point de la grille "
            f"[0, grid_size), reçu {initial_tumor_position!r} pour grid_size={grid_size!r}"
        )


def _initial_grid(
    grid_size: int, K: float, initial_tumor_position: tuple[int, int], initial_radius: float
) -> NDArray[np.float64]:
    i0, j0 = initial_tumor_position
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    inside_tumor = (ii - i0) ** 2 + (jj - j0) ** 2 <= initial_radius**2
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    u[inside_tumor] = K
    return u


def _laplacian_neumann(u: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Laplacien à 5 points avec conditions aux limites de Neumann (flux nul).

    Le bord est répliqué (`mode="edge"`) plutôt que périodique ou nul, ce
    qui modélise un tissu borné sans perte ni apport de matière tumorale
    aux frontières du domaine simulé.
    """
    padded = np.pad(u, pad_width=1, mode="edge")
    return (
        padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * padded[1:-1, 1:-1]
    ) / dx**2


def _max_dt_cfl(dx: float, D: float) -> float:
    """Limite théorique de dt pour la stabilité CFL en diffusion 2D explicite."""
    return dx ** 2 / (4 * D)


def simulate_2d(
    grid_size: int,
    D: float,
    r: float,
    K: float,
    initial_tumor_position: tuple[int, int],
    initial_radius: float,
    n_steps: int,
    dt: float | None = None,
    domain_size_mm: float = _DOMAIN_SIZE_MM,
) -> FisherKPPResult:
    """Simule l'invasion tumorale par un modèle de diffusion-réaction Fisher-KPP.

    Args:
        grid_size: Nombre de cellules par côté de la grille carrée
            (doit être strictement positif).
        D: Coefficient de diffusion, en mm²/unité de temps (doit être
            strictement positif).
        r: Taux de prolifération intrinsèque (doit être strictement positif).
        K: Densité maximale locale / capacité du tissu (doit être
            strictement positive).
        initial_tumor_position: Indices de grille (i, j) du centre de la
            tumeur initiale (et non des coordonnées physiques en mm).
        initial_radius: Rayon initial de la tumeur, en nombre de cellules
            de grille (doit être strictement compris entre 0 et grid_size).
        n_steps: Nombre de pas de temps à simuler après l'état initial
            (doit être strictement positif).
        dt: Pas de temps optionnel (en jours ou unité cohérente).
            Si non fourni, il est calculé automatiquement pour respecter la
            condition CFL `dt <= dx**2/(4*D)` avec une marge de sécurité.
            Si fourni, il doit vérifier `dt <= dx**2/(4*D)`, sinon une
            `ValueError` est levée.
        domain_size_mm: Taille physique du domaine carré simulé, en mm.
            Vaut `_DOMAIN_SIZE_MM` par défaut ; à augmenter explicitement
            pour des régimes de paramètres où `D` et `r` demandent un
            domaine plus grand pour éviter à la fois l'extinction d'un
            foyer sous-critique et les effets de bord (voir le commentaire
            sur `_DOMAIN_SIZE_MM` plus haut dans ce fichier).

    Returns:
        Un FisherKPPResult contenant les instants `t`, les grilles `grids`
        de forme (n_steps + 1, grid_size, grid_size), ainsi que les pas
        `dx` et `dt` effectivement utilisés. Déballable directement via
        `t, grids = simulate_2d(...)`.

    Raises:
        ValueError: Si un des paramètres est invalide ou si le `dt` fourni
            viole la condition CFL.
    """
    _validate_params(grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps)

    dx = domain_size_mm / grid_size
    max_dt = _max_dt_cfl(dx, D)  # dx**2 / (4*D)

    if dt is not None:
        if dt <= 0:
            raise ValueError(f"dt doit être strictement positif, reçu dt={dt!r}")
        if dt > max_dt:
            raise ValueError(
                f"Condition CFL violée : dt={dt:.6f} > max_dt={max_dt:.6f} "
                f"(dx**2/(4*D) avec dx={dx:.6f}, D={D:.6f})"
            )
        # dt fourni par l'utilisateur : on l'utilise tel quel
        user_dt = dt
    else:
        # Calcul automatique avec marge de sécurité 10 %
        user_dt = 0.9 * min(max_dt, 2.0 / r)

    u = _initial_grid(grid_size, K, initial_tumor_position, initial_radius)
    grids = np.empty((n_steps + 1, grid_size, grid_size), dtype=np.float64)
    grids[0] = u

    for step in range(1, n_steps + 1):
        laplacian = _laplacian_neumann(u, dx)
        u = u + user_dt * (D * laplacian + r * u * (1 - u / K))
        u = np.clip(u, 0.0, K)
        grids[step] = u

    t = np.arange(n_steps + 1) * user_dt
    return FisherKPPResult(t=t, grids=grids, dx=dx, dt=user_dt)


class FisherKPPModel(TumorGrowthModel):
    """Modèle de diffusion-réaction Fisher-KPP conforme à TumorGrowthModel.

    Le modèle de diffusion-réaction s'appuie sur l'équation de Fisher-KPP
    (également appelée équation Fisher-Kolmogorov-Petrovsky-Piskunov) :

    du/dt = D * laplacien(u) + r * u * (1 - u/K)

    où `D` est le coefficient de diffusion (migration cellulaire), `r` le taux
    de prolifération et `K` la densité maximale locale (capacité du tissu).
    La croissance logistique locale reproduit le comportement saturé vu dans
    le modèle de Gompertz, mais à échelle spatiale.

    Le modèle est conçu pour être instancié via le constructeur
    `FisherKPPModel(...)` ou invoqué directement via la fonction
    `simulate_2d(...)`.
    """

    name = "fisher_kpp"

    def __init__(
        self,
        grid_size: int,
        D: float,
        r: float,
        K: float,
        initial_tumor_position: tuple[int, int],
        initial_radius: float,
        n_steps: int,
        dt: float | None = None,
        domain_size_mm: float = _DOMAIN_SIZE_MM,
    ) -> None:
        _validate_params(grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps)
        self.grid_size = grid_size
        self.D = D
        self.r = r
        self.K = K
        self.initial_tumor_position = initial_tumor_position
        self.initial_radius = initial_radius
        self.n_steps = n_steps
        self.domain_size_mm = domain_size_mm

        if dt is not None:
            if dt <= 0:
                raise ValueError(f"dt doit être strictement positif, reçu dt={dt!r}")
            dx = domain_size_mm / grid_size
            max_dt = dx**2 / (4 * D)
            if dt > max_dt:
                raise ValueError(
                    f"Condition CFL violée : dt={dt:.6f} > max_dt={max_dt:.6f} "
                    f"(dx**2/(4*D) avec dx={dx:.6f}, D={D:.6f})"
                )
        self._dt = dt  # None = automatique, sinon validé CFL

    def simulate(self) -> FisherKPPResult:
        return simulate_2d(
            self.grid_size,
            self.D,
            self.r,
            self.K,
            self.initial_tumor_position,
            self.initial_radius,
            self.n_steps,
            dt=self._dt,
            domain_size_mm=self.domain_size_mm,
        )


# Alias de compatibilité : le modèle de diffusion-réaction s'appuie sur FisherKPPModel
DiffusionReactionModel = FisherKPPModel
