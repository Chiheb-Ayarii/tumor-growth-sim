import numpy as np
import pytest

from simulation.calibration import calibrate_fisher_kpp, generate_noisy_fisher_kpp_observations

# Paramètres partagés par les tests : un maillage volontairement petit et un
# nombre de pas modeste pour garder la calibration (qui doit re-simuler à
# chaque itération de l'optimiseur) rapide en tests, tout en restant dans un
# régime où la tumeur se propage réellement (voir l'audit initial sur
# l'extinction de patch sous-critique).
_GRID_SIZE = 50
_K = 1.0
_POSITION = (25, 25)
_RADIUS = 3
_N_STEPS = 150


def test_calibration_recovers_true_params_from_noise_free_data():
    D_true, r_true = 0.005, 0.05
    t_obs, mass_obs = generate_noisy_fisher_kpp_observations(
        grid_size=_GRID_SIZE, D=D_true, r=r_true, K=_K,
        initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
        noise_std_rel=0.0, seed=1,
    )

    result = calibrate_fisher_kpp(
        t_obs, mass_obs, grid_size=_GRID_SIZE, K=_K,
        initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
    )

    assert result.success
    assert result.D == pytest.approx(D_true, rel=1e-2)
    assert result.r == pytest.approx(r_true, rel=1e-2)


def test_calibration_recovers_true_params_within_tolerance_from_noisy_data():
    D_true, r_true = 0.005, 0.05
    t_obs, mass_obs = generate_noisy_fisher_kpp_observations(
        grid_size=_GRID_SIZE, D=D_true, r=r_true, K=_K,
        initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
        noise_std_rel=0.05, seed=42,
    )

    result = calibrate_fisher_kpp(
        t_obs, mass_obs, grid_size=_GRID_SIZE, K=_K,
        initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
    )

    assert result.success
    assert result.D == pytest.approx(D_true, rel=0.3)
    assert result.r == pytest.approx(r_true, rel=0.2)
    assert np.isfinite(result.D_std_err)
    assert np.isfinite(result.r_std_err)


def test_noisy_observations_reproducible_with_same_seed():
    kwargs = dict(
        grid_size=_GRID_SIZE, D=0.005, r=0.05, K=_K,
        initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
        noise_std_rel=0.05,
    )
    t1, m1 = generate_noisy_fisher_kpp_observations(seed=7, **kwargs)
    t2, m2 = generate_noisy_fisher_kpp_observations(seed=7, **kwargs)

    assert np.array_equal(t1, t2)
    assert np.array_equal(m1, m2)


def test_d_above_bound_raises_value_error():
    with pytest.raises(ValueError):
        generate_noisy_fisher_kpp_observations(
            grid_size=_GRID_SIZE, D=1.0, r=0.05, K=_K,
            initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
            noise_std_rel=0.05, seed=1,
        )


def test_negative_noise_std_raises_value_error():
    with pytest.raises(ValueError):
        generate_noisy_fisher_kpp_observations(
            grid_size=_GRID_SIZE, D=0.005, r=0.05, K=_K,
            initial_tumor_position=_POSITION, initial_radius=_RADIUS, n_steps=_N_STEPS,
            noise_std_rel=-0.1, seed=1,
        )


def test_mismatched_shapes_raise_value_error():
    with pytest.raises(ValueError):
        calibrate_fisher_kpp(
            t_obs=np.array([0.0, 1.0, 2.0]), mass_obs_mm2=np.array([1.0, 2.0]),
            grid_size=_GRID_SIZE, K=_K, initial_tumor_position=_POSITION,
            initial_radius=_RADIUS, n_steps=_N_STEPS,
        )
