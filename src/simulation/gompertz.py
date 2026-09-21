"""Modèle de croissance de Gompertz.

Équation différentielle :
    dV/dt = r * V * ln(K / V)

où V est le volume tumoral, r le taux de croissance et K la capacité
limite (volume maximal asymptotique). Ce modèle est classiquement utilisé
pour décrire une croissance tumorale qui ralentit progressivement à mesure
que le volume approche de K, par exemple sous l'effet de contraintes
d'apport en oxygène et en nutriments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from simulation.base import TumorGrowthModel

# Plancher numérique appliqué à V dans le second membre uniquement, pour
# éviter un log(0) en cas de sous-dépassement de l'intégrateur. V ne peut
# mathématiquement pas atteindre 0 lorsque V0 > 0 (Gompertz reste positif).
_V_FLOOR = 1e-12


@dataclass(frozen=True)
class GompertzResult:
    """Résultat d'une simulation de Gompertz.

    Attributes:
        t: Instants d'évaluation, de forme (n_points,).
        V: Volume tumoral à chaque instant, de forme (n_points,).
    """

    t: NDArray[np.float64]
    V: NDArray[np.float64]

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Permet le déballage `t, V = simulate(...)`."""
        return iter((self.t, self.V))


def _validate_params(
    V0: float, r: float, K: float, t_span: tuple[float, float], n_points: int
) -> None:
    if not (V0 > 0):
        raise ValueError(f"V0 doit être strictement positif, reçu V0={V0!r}")
    if not (K > 0):
        raise ValueError(f"K doit être strictement positif, reçu K={K!r}")
    if not (r > 0):
        raise ValueError(f"r doit être strictement positif, reçu r={r!r}")

    t0, tf = t_span
    if not (np.isfinite(t0) and np.isfinite(tf)):
        raise ValueError(f"t_span doit contenir des valeurs finies, reçu t_span={t_span!r}")
    if not (t0 < tf):
        raise ValueError(f"t_span doit vérifier t0 < tf, reçu t_span={t_span!r}")

    if n_points < 2:
        raise ValueError(f"n_points doit être >= 2, reçu n_points={n_points!r}")


def _rhs(t: float, y: NDArray[np.float64], r: float, K: float) -> list[float]:
    V = max(float(y[0]), _V_FLOOR)
    return [r * V * np.log(K / V)]


def simulate(
    V0: float, r: float, K: float, t_span: tuple[float, float], n_points: int
) -> GompertzResult:
    """Simule la croissance de Gompertz par intégration numérique.

    Args:
        V0: Volume tumoral initial (doit être strictement positif).
        r: Taux de croissance intrinsèque (doit être strictement positif).
        K: Capacité limite / volume maximal asymptotique (doit être
            strictement positif). Si V0 > K, le volume décroît vers K
            (régression tumorale) : ce cas est valide.
        t_span: Intervalle de temps (t0, tf) sur lequel intégrer, avec
            t0 < tf.
        n_points: Nombre de points de temps également espacés sur
            lesquels évaluer la solution (doit être >= 2).

    Returns:
        Un GompertzResult contenant les tableaux temps `t` et volume `V`,
        déballable directement via `t, V = simulate(...)`.

    Raises:
        ValueError: Si un des paramètres est invalide.
        RuntimeError: Si l'intégration numérique échoue.
    """
    _validate_params(V0, r, K, t_span, n_points)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = solve_ivp(
        _rhs,
        t_span=t_span,
        y0=[V0],
        method="RK45",
        t_eval=t_eval,
        args=(r, K),
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"L'intégration de Gompertz a échoué : {solution.message}")

    return GompertzResult(t=solution.t, V=solution.y[0])


class GompertzModel(TumorGrowthModel):
    """Modèle de croissance de Gompertz conforme à l'interface TumorGrowthModel."""

    name = "gompertz"

    def __init__(
        self, V0: float, r: float, K: float, t_span: tuple[float, float], n_points: int
    ) -> None:
        _validate_params(V0, r, K, t_span, n_points)
        self.V0 = V0
        self.r = r
        self.K = K
        self.t_span = t_span
        self.n_points = n_points

    def simulate(self) -> GompertzResult:
        return simulate(self.V0, self.r, self.K, self.t_span, self.n_points)
