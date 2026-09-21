"""Surrogate models (PyTorch) estimant directement les paramètres depuis une courbe observée.

Alternative à la calibration classique (`calibration.py`, moindres carrés
itératifs qui re-simulent le modèle à chaque essai) : un petit réseau de
neurones (MLP) apprend directement la relation « courbe observée bruitée ->
paramètres », à partir d'un grand nombre d'exemples synthétiques générés par
les modèles mécanistes eux-mêmes. Une fois entraîné, il répond en une
fraction de seconde par lot, sans aucune nouvelle simulation.

L'intérêt pratique de cette approche dépend du coût de la simulation
sous-jacente : pour Gompertz (une ODE 0D, déjà très rapide à simuler), le
gain est marginal voire nul par rapport aux moindres carrés. Pour Fisher-KPP
(une PDE 2D nettement plus coûteuse), remplacer des dizaines de simulations
par un unique passage avant dans le réseau devient un vrai gain de temps.
Le notebook compare les deux approches sur les deux modèles, chiffres à
l'appui, plutôt que de le supposer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from simulation.calibration import (
    generate_noisy_fisher_kpp_observations,
    generate_noisy_gompertz_observations,
)


@dataclass(frozen=True)
class SurrogateDataset:
    """Jeu de données synthétique pour l'entraînement d'un surrogate model.

    Attributes:
        X: Courbes observées bruitées, de forme (n_samples, n_features).
        y: Paramètres vrais correspondants, de forme (n_samples, 2).
        param_names: Noms des deux paramètres estimés (ex: `("r", "K")`).
        param_bounds: Bornes `(min, max)` utilisées pour tirer chaque
            paramètre à l'entraînement ; réutilisées pour dénormaliser les
            sorties du réseau lors de la prédiction.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    param_names: tuple[str, str]
    param_bounds: tuple[tuple[float, float], tuple[float, float]]


def generate_gompertz_surrogate_dataset(
    n_samples: int,
    V0: float,
    r_bounds: tuple[float, float],
    K_bounds: tuple[float, float],
    t_span: tuple[float, float],
    n_points: int,
    noise_std_rel: float,
    seed: int,
) -> SurrogateDataset:
    """Génère `n_samples` exemples (courbe bruitée -> (r, K)) pour un surrogate Gompertz.

    Chaque exemple tire `(r, K)` uniformément dans les bornes fournies,
    simule la courbe bruitée correspondante (`generate_noisy_gompertz_observations`),
    et garde la paire (courbe, paramètres vrais) comme exemple d'entraînement.

    Args:
        n_samples: Nombre d'exemples à générer.
        V0: Volume initial, fixé et identique pour tous les exemples (connu).
        r_bounds: Bornes `(min, max)` du taux de croissance à échantillonner.
        K_bounds: Bornes `(min, max)` de la capacité limite à échantillonner.
        t_span, n_points: Voir `simulate` — identiques pour tous les exemples.
        noise_std_rel: Écart-type du bruit multiplicatif appliqué à chaque courbe.
        seed: Graine du générateur aléatoire, pour un jeu de données reproductible.

    Returns:
        Un `SurrogateDataset` prêt à être passé à `train_surrogate`.
    """
    rng = np.random.default_rng(seed)
    r_samples = rng.uniform(*r_bounds, size=n_samples)
    K_samples = rng.uniform(*K_bounds, size=n_samples)
    sub_seeds = rng.integers(0, 2**31 - 1, size=n_samples)

    X = np.empty((n_samples, n_points), dtype=np.float64)
    for i in range(n_samples):
        _, V_obs = generate_noisy_gompertz_observations(
            V0=V0,
            r=float(r_samples[i]),
            K=float(K_samples[i]),
            t_span=t_span,
            n_points=n_points,
            noise_std_rel=noise_std_rel,
            seed=int(sub_seeds[i]),
        )
        X[i] = V_obs

    y = np.stack([r_samples, K_samples], axis=1)
    return SurrogateDataset(X=X, y=y, param_names=("r", "K"), param_bounds=(r_bounds, K_bounds))


def generate_fisher_kpp_surrogate_dataset(
    n_samples: int,
    grid_size: int,
    K: float,
    initial_tumor_position: tuple[int, int],
    initial_radius: float,
    n_steps: int,
    D_bounds: tuple[float, float],
    r_bounds: tuple[float, float],
    noise_std_rel: float,
    seed: int,
) -> SurrogateDataset:
    """Génère `n_samples` exemples (masse bruitée -> (D, r)) pour un surrogate Fisher-KPP.

    Analogue à `generate_gompertz_surrogate_dataset`, mais pour le modèle
    spatial : `(D, r)` sont tirés uniformément dans leurs bornes, la grille,
    `K` et la condition initiale sont fixés et identiques pour tous les
    exemples (connus).

    Args:
        n_samples: Nombre d'exemples à générer.
        grid_size, K, initial_tumor_position, initial_radius, n_steps: Voir
            `simulate_2d` — identiques pour tous les exemples.
        D_bounds: Bornes `(min, max)` du coefficient de diffusion à
            échantillonner (doit respecter `_DEFAULT_D_BOUNDS` du module
            `calibration`, qui fixe le pas de temps sûr).
        r_bounds: Bornes `(min, max)` du taux de prolifération à échantillonner.
        noise_std_rel: Écart-type du bruit multiplicatif appliqué à chaque courbe.
        seed: Graine du générateur aléatoire, pour un jeu de données reproductible.

    Returns:
        Un `SurrogateDataset` prêt à être passé à `train_surrogate`.
    """
    rng = np.random.default_rng(seed)
    D_samples = rng.uniform(*D_bounds, size=n_samples)
    r_samples = rng.uniform(*r_bounds, size=n_samples)
    sub_seeds = rng.integers(0, 2**31 - 1, size=n_samples)

    X = np.empty((n_samples, n_steps + 1), dtype=np.float64)
    for i in range(n_samples):
        _, mass_obs = generate_noisy_fisher_kpp_observations(
            grid_size=grid_size,
            D=float(D_samples[i]),
            r=float(r_samples[i]),
            K=K,
            initial_tumor_position=initial_tumor_position,
            initial_radius=initial_radius,
            n_steps=n_steps,
            noise_std_rel=noise_std_rel,
            seed=int(sub_seeds[i]),
        )
        X[i] = mass_obs

    y = np.stack([D_samples, r_samples], axis=1)
    return SurrogateDataset(X=X, y=y, param_names=("D", "r"), param_bounds=(D_bounds, r_bounds))


class _MLP(nn.Module):
    """Petit perceptron multicouche : courbe normalisée -> 2 paramètres normalisés dans [0, 1]."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedSurrogate:
    """Surrogate entraîné, prêt à prédire des paramètres à partir d'une courbe brute.

    Attributes:
        model: Réseau PyTorch entraîné.
        x_scale: Facteur d'échelle (division) appliqué aux courbes en
            entrée, calculé une fois sur le jeu d'entraînement.
        param_names: Noms des deux paramètres estimés.
        param_bounds: Bornes utilisées pour dénormaliser les sorties.
        train_losses: Perte MSE (sur sorties normalisées) par époque, sur
            le jeu d'entraînement.
        val_losses: Même chose sur le jeu de validation.
    """

    model: nn.Module
    x_scale: float
    param_names: tuple[str, str]
    param_bounds: tuple[tuple[float, float], tuple[float, float]]
    train_losses: list[float]
    val_losses: list[float]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prédit les 2 paramètres pour un lot de courbes brutes (non normalisées).

        Args:
            X: Courbes observées, de forme (n_features,) ou (n_samples, n_features).

        Returns:
            Un tableau de forme (n_samples, 2) des paramètres prédits, dans
            leur échelle physique d'origine (dénormalisés).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.model.eval()
        with torch.no_grad():
            X_norm = torch.tensor(X / self.x_scale, dtype=torch.float32)
            y_norm = self.model(X_norm).numpy()

        (lo1, hi1), (lo2, hi2) = self.param_bounds
        p1 = y_norm[:, 0] * (hi1 - lo1) + lo1
        p2 = y_norm[:, 1] * (hi2 - lo2) + lo2
        return np.stack([p1, p2], axis=1)


def train_surrogate(
    dataset: SurrogateDataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[TrainedSurrogate, dict[str, NDArray[np.float64]]]:
    """Entraîne un surrogate MLP sur un `SurrogateDataset`, avec split train/val/test.

    Les courbes en entrée sont normalisées par un facteur d'échelle global
    (le maximum observé sur le jeu d'entraînement) plutôt que par exemple
    individuellement : une normalisation par exemple effacerait l'échelle
    absolue de chaque courbe, qui est justement le signal principal utilisé
    pour distinguer un `K` (ou une masse totale) petit d'un grand. Les
    paramètres cibles sont normalisés dans [0, 1] via leurs bornes de tirage
    (connues, pas ré-estimées depuis les données).

    Args:
        dataset: Jeu de données généré par `generate_gompertz_surrogate_dataset`
            ou `generate_fisher_kpp_surrogate_dataset`.
        val_fraction: Fraction des exemples réservée à la validation (suivi
            de la perte pendant l'entraînement).
        test_fraction: Fraction des exemples tenue à l'écart de tout
            entraînement, retournée dans `splits` pour permettre une
            comparaison a posteriori avec la calibration classique sur
            exactement les mêmes exemples.
        hidden_dim: Largeur des couches cachées du MLP.
        epochs: Nombre d'époques d'entraînement.
        lr: Taux d'apprentissage (Adam).
        batch_size: Taille de mini-lot.
        seed: Graine (numpy pour le split des données, PyTorch pour
            l'initialisation des poids et le mélange des lots).

    Returns:
        Un tuple `(surrogate, splits)` où `splits` contient `X_test` et
        `y_test`, les exemples non utilisés pendant l'entraînement.

    Raises:
        ValueError: Si `val_fraction + test_fraction >= 1`.
    """
    if val_fraction + test_fraction >= 1:
        raise ValueError(
            f"val_fraction + test_fraction doit être < 1, reçu {val_fraction!r} + {test_fraction!r}"
        )

    rng = np.random.default_rng(seed)
    n = dataset.X.shape[0]
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    x_scale = float(np.max(dataset.X[train_idx]))
    (lo1, hi1), (lo2, hi2) = dataset.param_bounds

    def _to_tensors(idx: NDArray[np.int64]) -> tuple[torch.Tensor, torch.Tensor]:
        X_norm = dataset.X[idx] / x_scale
        y_norm = np.stack(
            [
                (dataset.y[idx, 0] - lo1) / (hi1 - lo1),
                (dataset.y[idx, 1] - lo2) / (hi2 - lo2),
            ],
            axis=1,
        )
        return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)

    X_train, y_train = _to_tensors(train_idx)
    X_val, y_val = _to_tensors(val_idx)

    torch.manual_seed(seed)
    model = _MLP(input_dim=dataset.X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    n_train = X_train.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        train_losses.append(epoch_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val), y_val).item())

    surrogate = TrainedSurrogate(
        model=model,
        x_scale=x_scale,
        param_names=dataset.param_names,
        param_bounds=dataset.param_bounds,
        train_losses=train_losses,
        val_losses=val_losses,
    )
    splits = {"X_test": dataset.X[test_idx], "y_test": dataset.y[test_idx]}
    return surrogate, splits
