"""Calibration de paramètres à partir de données (synthétiques) bruitées.

Ce module illustre le schéma « jumeau numérique » : un modèle mécaniste
(Gompertz ou Fisher-KPP) dont les paramètres ne sont pas connus a priori
sont estimés à partir d'observations bruitées, comme ce serait le cas avec
de vraies mesures de bioproduction (volume tumoral, densité cellulaire,
titre produit, ...). La calibration est faite par moindres carrés non
linéaires (`scipy.optimize.least_squares`), avec estimation de l'incertitude
sur les paramètres à partir de la matrice de covariance approchée au point
optimal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

import simulation.diffusion_reaction as dr
from simulation.gompertz import simulate

# Bornes par défaut pour (r, K), assez larges pour ne pas biaiser la
# calibration mais empêcher l'optimiseur d'explorer des valeurs non
# physiques (r <= 0 ou K <= 0, qui font diverger le modèle de Gompertz).
_DEFAULT_R_BOUNDS = (1e-6, 10.0)
_DEFAULT_K_BOUNDS = (1e-3, np.inf)

# Bornes par défaut pour (D, r) du modèle Fisher-KPP. `_DEFAULT_D_BOUNDS`
# sert aussi à fixer un pas de temps CFL sûr pour toute la calibration (voir
# `_safe_fixed_dt`) : l'augmenter élargit la plage de D explorable mais
# rend le pas de temps utilisé plus petit, donc la calibration plus lente.
_DEFAULT_D_BOUNDS = (1e-4, 0.05)
_DEFAULT_FISHER_R_BOUNDS = (1e-4, 1.0)


def _param_std_errors(result: OptimizeResult, n_obs: int, n_params: int) -> tuple[float, float]:
    """Écarts-types approchés des 2 premiers paramètres d'un ajustement `least_squares`.

    Approximation classique des moindres carrés non linéaires : la
    covariance des paramètres est estimée à partir du jacobien au point
    optimal, mise à l'échelle par la variance résiduelle. Valable localement
    (bruit faible à modéré), pas une garantie globale.
    """
    try:
        residual_variance = 2 * result.cost / max(n_obs - n_params, 1)
        covariance = residual_variance * np.linalg.inv(result.jac.T @ result.jac)
        return (
            float(np.sqrt(max(covariance[0, 0], 0.0))),
            float(np.sqrt(max(covariance[1, 1], 0.0))),
        )
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


@dataclass(frozen=True)
class CalibrationResult:
    """Résultat de la calibration des paramètres (r, K) du modèle de Gompertz.

    Attributes:
        r: Taux de croissance estimé.
        K: Capacité limite estimée.
        r_std_err: Écart-type approché de l'estimation de `r`, dérivé de la
            matrice de covariance au point optimal (approximation locale,
            valable pour un bruit d'observation faible à modéré).
        K_std_err: Écart-type approché de l'estimation de `K`.
        cost: Valeur finale de la fonction de coût (demi-somme des carrés
            des résidus) atteinte par l'optimiseur.
        success: Indique si l'optimiseur a convergé.
        n_observations: Nombre de points de données utilisés pour la
            calibration.
    """

    r: float
    K: float
    r_std_err: float
    K_std_err: float
    cost: float
    success: bool
    n_observations: int


def generate_noisy_gompertz_observations(
    V0: float,
    r: float,
    K: float,
    t_span: tuple[float, float],
    n_points: int,
    noise_std_rel: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Génère des observations synthétiques bruitées d'une croissance de Gompertz.

    Simule la trajectoire « vraie » avec les paramètres fournis, puis ajoute
    un bruit gaussien multiplicatif (proportionnel au volume local, ce qui
    est plus réaliste que du bruit additif pour des mesures de volume/densité
    biologique qui ne peuvent être négatives à l'échelle du signal observé).

    Args:
        V0: Volume tumoral initial « vrai ».
        r: Taux de croissance « vrai ».
        K: Capacité limite « vraie ».
        t_span: Intervalle de temps (t0, tf) sur lequel simuler.
        n_points: Nombre de points de mesure également espacés.
        noise_std_rel: Écart-type du bruit multiplicatif, en fraction du
            signal (par exemple 0.05 pour un bruit de mesure de ±5 %).
            Doit être positif ou nul.
        seed: Graine du générateur aléatoire, pour des observations
            reproductibles.

    Returns:
        Un tuple `(t_obs, V_obs)` des instants et volumes observés (bruités,
        et tronqués à 0 pour rester physiquement valides).

    Raises:
        ValueError: Si `noise_std_rel` est strictement négatif.
    """
    if noise_std_rel < 0:
        raise ValueError(f"noise_std_rel doit être positif ou nul, reçu noise_std_rel={noise_std_rel!r}")

    t_true, V_true = simulate(V0=V0, r=r, K=K, t_span=t_span, n_points=n_points)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=1.0, scale=noise_std_rel, size=V_true.shape)
    V_obs = np.clip(V_true * noise, a_min=0.0, a_max=None)

    return t_true, V_obs


def calibrate_gompertz(
    t_obs: NDArray[np.float64],
    V_obs: NDArray[np.float64],
    V0: float,
    r_init: float = 0.1,
    K_init: float | None = None,
) -> CalibrationResult:
    """Calibre (r, K) du modèle de Gompertz à partir d'observations bruitées.

    Le volume initial `V0` est supposé connu (mesuré directement), ce qui
    correspond au cas usuel où l'état initial d'un procédé est observable
    mais où les paramètres cinétiques (taux de croissance, capacité limite)
    ne le sont pas et doivent être estimés a posteriori.

    Args:
        t_obs: Instants des observations, de forme (n,).
        V_obs: Volumes observés (potentiellement bruités), de forme (n,).
        V0: Volume initial connu, utilisé comme condition initiale fixe de
            la simulation (n'est pas ré-estimé).
        r_init: Valeur initiale de `r` pour l'optimiseur.
        K_init: Valeur initiale de `K` pour l'optimiseur. Si non fournie,
            `max(V_obs)` est utilisé comme estimation de départ raisonnable
            (K est nécessairement >= au volume maximal observé).

    Returns:
        Un CalibrationResult contenant les paramètres estimés et leur
        incertitude approchée.

    Raises:
        ValueError: Si `t_obs` et `V_obs` n'ont pas la même forme, ou si
            moins de 2 observations sont fournies (il faut au moins autant
            de points que de paramètres à estimer, ici 2 : r et K).
    """
    t_obs = np.asarray(t_obs, dtype=np.float64)
    V_obs = np.asarray(V_obs, dtype=np.float64)

    if t_obs.shape != V_obs.shape:
        raise ValueError(
            f"t_obs et V_obs doivent avoir la même forme, reçu t_obs.shape={t_obs.shape!r} "
            f"et V_obs.shape={V_obs.shape!r}"
        )
    if t_obs.shape[0] < 2:
        raise ValueError(f"Au moins 2 observations sont nécessaires, reçu {t_obs.shape[0]}")

    if K_init is None:
        K_init = max(float(np.max(V_obs)), V0) * 1.5

    def _residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        r, K = params
        t_span = (float(t_obs[0]), float(t_obs[-1]))
        if t_span[0] == t_span[1]:
            t_span = (t_span[0], t_span[1] + 1.0)
        _, V_pred_full = simulate(V0=V0, r=r, K=K, t_span=t_span, n_points=max(len(t_obs), 2))
        V_pred = np.interp(t_obs, np.linspace(t_span[0], t_span[1], len(V_pred_full)), V_pred_full)
        return V_pred - V_obs

    result = least_squares(
        _residuals,
        x0=[r_init, K_init],
        bounds=([_DEFAULT_R_BOUNDS[0], _DEFAULT_K_BOUNDS[0]], [_DEFAULT_R_BOUNDS[1], _DEFAULT_K_BOUNDS[1]]),
        method="trf",
    )

    r_fit, K_fit = result.x
    n_obs = t_obs.shape[0]
    n_params = 2
    r_std_err, K_std_err = _param_std_errors(result, n_obs, n_params)

    return CalibrationResult(
        r=float(r_fit),
        K=float(K_fit),
        r_std_err=r_std_err,
        K_std_err=K_std_err,
        cost=float(result.cost),
        success=bool(result.success),
        n_observations=n_obs,
    )


@dataclass(frozen=True)
class FisherKPPCalibrationResult:
    """Résultat de la calibration des paramètres (D, r) du modèle Fisher-KPP.

    Attributes:
        D: Coefficient de diffusion estimé.
        r: Taux de prolifération estimé.
        D_std_err: Écart-type approché de l'estimation de `D`.
        r_std_err: Écart-type approché de l'estimation de `r`.
        cost: Valeur finale de la fonction de coût atteinte par l'optimiseur.
        success: Indique si l'optimiseur a convergé.
        n_observations: Nombre de points de données utilisés pour la
            calibration.
    """

    D: float
    r: float
    D_std_err: float
    r_std_err: float
    cost: float
    success: bool
    n_observations: int


def _total_mass_mm2(result: dr.FisherKPPResult) -> NDArray[np.float64]:
    """Masse tumorale totale (intégrale de la densité sur la grille) à chaque instant, en mm².

    Utilisée comme observable de calibration plutôt que le rayon effectif du
    front (seuil u > K/2, utilisé ailleurs pour la mesure de vitesse) : ce
    seuil est une fonction en escalier des paramètres (le nombre de cellules
    de grille au-dessus du seuil ne varie que par sauts discrets), ce qui
    annule le gradient numérique vu par l'optimiseur et bloque la
    convergence de `least_squares` dès la première itération. La masse
    totale, elle, varie continûment avec D et r et se calibre normalement.
    """
    return np.array([np.sum(grid) for grid in result.grids]) * result.dx**2


def _safe_fixed_dt(grid_size: int, D_max: float) -> float:
    """Pas de temps fixe, valide (condition CFL) pour tout D <= D_max sur ce maillage.

    Le pas de temps `dt` de `simulate_2d` dépend normalement de D (condition
    CFL). Pendant une calibration, D change à chaque essai de l'optimiseur :
    si dt changeait aussi, l'axe des temps simulé ne correspondrait plus à
    t_obs d'un essai à l'autre. Fixer dt une fois pour toutes (sûr pour
    n'importe quel D testé dans `_DEFAULT_D_BOUNDS`) évite ce problème et
    permet de comparer directement les courbes simulées aux observations.
    """
    dx = dr._DOMAIN_SIZE_MM / grid_size
    return 0.9 * dx**2 / (4 * D_max)


def generate_noisy_fisher_kpp_observations(
    grid_size: int,
    D: float,
    r: float,
    K: float,
    initial_tumor_position: tuple[int, int],
    initial_radius: float,
    n_steps: int,
    noise_std_rel: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Génère une courbe bruitée de masse tumorale totale (modèle Fisher-KPP).

    Analogue à `generate_noisy_gompertz_observations`, mais pour le modèle
    spatial : simule la « vraie » tumeur avec (D, r) connus, calcule la
    masse totale (intégrale de la densité, voir `_total_mass_mm2`) à chaque
    instant, puis ajoute un bruit gaussien multiplicatif pour imiter une
    mesure imparfaite (par exemple un volume tumoral estimé par imagerie).

    Args:
        grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps:
            Voir `simulate_2d`.
        noise_std_rel: Écart-type du bruit multiplicatif, en fraction du
            signal. Doit être positif ou nul.
        seed: Graine du générateur aléatoire, pour des observations
            reproductibles.

    Returns:
        Un tuple `(t_obs, mass_obs_mm2)`.

    Raises:
        ValueError: Si `noise_std_rel` est strictement négatif, ou si D
            dépasse `_DEFAULT_D_BOUNDS[1]` (la borne utilisée pour fixer un
            pas de temps sûr, partagée avec `calibrate_fisher_kpp`).
    """
    if noise_std_rel < 0:
        raise ValueError(f"noise_std_rel doit être positif ou nul, reçu noise_std_rel={noise_std_rel!r}")
    if D > _DEFAULT_D_BOUNDS[1]:
        raise ValueError(
            f"D={D!r} dépasse la borne supérieure {_DEFAULT_D_BOUNDS[1]} utilisée pour fixer un pas de "
            "temps sûr pendant la calibration (_DEFAULT_D_BOUNDS)."
        )

    dt = _safe_fixed_dt(grid_size, _DEFAULT_D_BOUNDS[1])
    result = dr.simulate_2d(grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps, dt=dt)
    mass_true = _total_mass_mm2(result)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=1.0, scale=noise_std_rel, size=mass_true.shape)
    mass_obs = np.clip(mass_true * noise, a_min=0.0, a_max=None)

    return result.t, mass_obs


def calibrate_fisher_kpp(
    t_obs: NDArray[np.float64],
    mass_obs_mm2: NDArray[np.float64],
    grid_size: int,
    K: float,
    initial_tumor_position: tuple[int, int],
    initial_radius: float,
    n_steps: int,
    D_init: float = 0.01,
    r_init: float = 0.05,
) -> FisherKPPCalibrationResult:
    """Calibre (D, r) du modèle Fisher-KPP à partir de la masse tumorale observée.

    K, la géométrie de la grille et la condition initiale sont supposés
    connus ; seuls les paramètres cinétiques D (migration cellulaire) et r
    (prolifération) sont estimés, comme pour `calibrate_gompertz`.

    Le pas de temps est fixé (voir `_safe_fixed_dt`), indépendamment de D,
    pour rester valide (condition CFL) pour toute valeur de D testée par
    l'optimiseur ; t_obs doit donc provenir d'observations générées avec ce
    même pas de temps fixe (voir `generate_noisy_fisher_kpp_observations`).

    Args:
        t_obs: Instants des observations, de forme (n_steps + 1,).
        mass_obs_mm2: Masse tumorale totale observée (mm²), de forme
            (n_steps + 1,).
        grid_size, K, initial_tumor_position, initial_radius, n_steps: Voir
            `simulate_2d` — supposés connus et identiques à ceux utilisés
            lors de la génération des observations.
        D_init: Valeur initiale de D pour l'optimiseur.
        r_init: Valeur initiale de r pour l'optimiseur.

    Returns:
        Un FisherKPPCalibrationResult contenant les paramètres estimés et
        leur incertitude approchée.

    Raises:
        ValueError: Si t_obs et mass_obs_mm2 n'ont pas la même forme, ou si
            moins de 2 observations sont fournies.
    """
    t_obs = np.asarray(t_obs, dtype=np.float64)
    mass_obs_mm2 = np.asarray(mass_obs_mm2, dtype=np.float64)

    if t_obs.shape != mass_obs_mm2.shape:
        raise ValueError(
            "t_obs et mass_obs_mm2 doivent avoir la même forme, reçu "
            f"t_obs.shape={t_obs.shape!r} et mass_obs_mm2.shape={mass_obs_mm2.shape!r}"
        )
    if t_obs.shape[0] < 2:
        raise ValueError(f"Au moins 2 observations sont nécessaires, reçu {t_obs.shape[0]}")

    dt = _safe_fixed_dt(grid_size, _DEFAULT_D_BOUNDS[1])

    def _residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        D, r = params
        result = dr.simulate_2d(grid_size, D, r, K, initial_tumor_position, initial_radius, n_steps, dt=dt)
        return _total_mass_mm2(result) - mass_obs_mm2

    result = least_squares(
        _residuals,
        x0=[D_init, r_init],
        bounds=(
            [_DEFAULT_D_BOUNDS[0], _DEFAULT_FISHER_R_BOUNDS[0]],
            [_DEFAULT_D_BOUNDS[1], _DEFAULT_FISHER_R_BOUNDS[1]],
        ),
        method="trf",
    )

    D_fit, r_fit = result.x
    n_obs = t_obs.shape[0]
    n_params = 2
    D_std_err, r_std_err = _param_std_errors(result, n_obs, n_params)

    return FisherKPPCalibrationResult(
        D=float(D_fit),
        r=float(r_fit),
        D_std_err=D_std_err,
        r_std_err=r_std_err,
        cost=float(result.cost),
        success=bool(result.success),
        n_observations=n_obs,
    )
