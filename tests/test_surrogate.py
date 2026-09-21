import numpy as np
import pytest

from simulation.surrogate import (
    generate_fisher_kpp_surrogate_dataset,
    generate_gompertz_surrogate_dataset,
    train_surrogate,
)

# Paramètres volontairement réduits (peu d'exemples, petites grilles) pour
# garder les tests rapides : ils vérifient que le pipeline fonctionne
# correctement (formes, reproductibilité, baisse de la perte), pas la
# précision finale du modèle — celle-ci est démontrée dans le notebook avec
# un jeu de données bien plus grand.


def test_gompertz_dataset_shapes_and_reproducibility():
    kwargs = dict(
        n_samples=40, V0=50.0, r_bounds=(0.02, 0.15), K_bounds=(500.0, 5000.0),
        t_span=(0, 300), n_points=20, noise_std_rel=0.05,
    )
    ds1 = generate_gompertz_surrogate_dataset(seed=1, **kwargs)
    ds2 = generate_gompertz_surrogate_dataset(seed=1, **kwargs)

    assert ds1.X.shape == (40, 20)
    assert ds1.y.shape == (40, 2)
    assert ds1.param_names == ("r", "K")
    assert np.array_equal(ds1.X, ds2.X)
    assert np.array_equal(ds1.y, ds2.y)


def test_fisher_kpp_dataset_shapes_and_reproducibility():
    kwargs = dict(
        n_samples=15, grid_size=20, K=1.0, initial_tumor_position=(10, 10),
        initial_radius=2, n_steps=30, D_bounds=(0.001, 0.02), r_bounds=(0.02, 0.1),
        noise_std_rel=0.05,
    )
    ds1 = generate_fisher_kpp_surrogate_dataset(seed=2, **kwargs)
    ds2 = generate_fisher_kpp_surrogate_dataset(seed=2, **kwargs)

    assert ds1.X.shape == (15, 31)
    assert ds1.y.shape == (15, 2)
    assert ds1.param_names == ("D", "r")
    assert np.array_equal(ds1.X, ds2.X)
    assert np.array_equal(ds1.y, ds2.y)


def test_train_surrogate_reduces_loss_and_predicts_right_shape():
    ds = generate_gompertz_surrogate_dataset(
        n_samples=120, V0=50.0, r_bounds=(0.02, 0.15), K_bounds=(500.0, 5000.0),
        t_span=(0, 300), n_points=20, noise_std_rel=0.05, seed=3,
    )
    surrogate, splits = train_surrogate(ds, val_fraction=0.15, test_fraction=0.15, epochs=30, seed=0)

    assert surrogate.train_losses[-1] < surrogate.train_losses[0]
    assert splits["X_test"].shape[0] == splits["y_test"].shape[0]

    y_pred = surrogate.predict(splits["X_test"])
    assert y_pred.shape == splits["y_test"].shape


def test_train_surrogate_predictions_correlate_with_true_params():
    # Sanity check peu exigeant : le surrogate doit faire mieux que prédire
    # la même valeur pour tout le monde, pas être d'une précision parfaite.
    ds = generate_gompertz_surrogate_dataset(
        n_samples=200, V0=50.0, r_bounds=(0.02, 0.15), K_bounds=(500.0, 5000.0),
        t_span=(0, 300), n_points=30, noise_std_rel=0.03, seed=4,
    )
    surrogate, splits = train_surrogate(ds, epochs=60, seed=0)
    y_pred = surrogate.predict(splits["X_test"])
    y_true = splits["y_test"]

    corr_r = np.corrcoef(y_pred[:, 0], y_true[:, 0])[0, 1]
    corr_K = np.corrcoef(y_pred[:, 1], y_true[:, 1])[0, 1]
    assert corr_r > 0.5
    assert corr_K > 0.5


def test_predict_accepts_single_curve():
    ds = generate_gompertz_surrogate_dataset(
        n_samples=60, V0=50.0, r_bounds=(0.02, 0.15), K_bounds=(500.0, 5000.0),
        t_span=(0, 300), n_points=20, noise_std_rel=0.05, seed=5,
    )
    surrogate, _ = train_surrogate(ds, epochs=20, seed=0)
    y_pred = surrogate.predict(ds.X[0])
    assert y_pred.shape == (1, 2)


def test_invalid_fractions_raise_value_error():
    ds = generate_gompertz_surrogate_dataset(
        n_samples=20, V0=50.0, r_bounds=(0.02, 0.15), K_bounds=(500.0, 5000.0),
        t_span=(0, 300), n_points=10, noise_std_rel=0.05, seed=6,
    )
    with pytest.raises(ValueError):
        train_surrogate(ds, val_fraction=0.6, test_fraction=0.6, epochs=5)
