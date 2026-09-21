import numpy as np
import pytest

from simulation.calibration import calibrate_gompertz, generate_noisy_gompertz_observations
from simulation.gompertz import simulate


def test_calibration_recovers_true_params_from_noise_free_data():
    V0, r_true, K_true = 50.0, 0.08, 2000.0
    t_obs, V_obs = simulate(V0=V0, r=r_true, K=K_true, t_span=(0, 300), n_points=60)

    result = calibrate_gompertz(t_obs, V_obs, V0=V0)

    assert result.success
    assert result.r == pytest.approx(r_true, rel=1e-3)
    assert result.K == pytest.approx(K_true, rel=1e-3)


def test_calibration_recovers_true_params_within_tolerance_from_noisy_data():
    V0, r_true, K_true = 50.0, 0.08, 2000.0
    t_obs, V_obs = generate_noisy_gompertz_observations(
        V0=V0, r=r_true, K=K_true, t_span=(0, 300), n_points=60, noise_std_rel=0.03, seed=42
    )

    result = calibrate_gompertz(t_obs, V_obs, V0=V0)

    assert result.success
    assert result.r == pytest.approx(r_true, rel=0.15)
    assert result.K == pytest.approx(K_true, rel=0.1)
    # L'incertitude estimée doit être finie et strictement positive avec du bruit.
    assert result.r_std_err > 0
    assert result.K_std_err > 0
    assert np.isfinite(result.r_std_err)
    assert np.isfinite(result.K_std_err)


def test_noisy_observations_reproducible_with_same_seed():
    kwargs = dict(V0=50.0, r=0.08, K=2000.0, t_span=(0, 300), n_points=60, noise_std_rel=0.05)
    t1, V1 = generate_noisy_gompertz_observations(seed=7, **kwargs)
    t2, V2 = generate_noisy_gompertz_observations(seed=7, **kwargs)

    assert np.array_equal(t1, t2)
    assert np.array_equal(V1, V2)


def test_noisy_observations_differ_with_different_seed():
    kwargs = dict(V0=50.0, r=0.08, K=2000.0, t_span=(0, 300), n_points=60, noise_std_rel=0.05)
    _, V1 = generate_noisy_gompertz_observations(seed=1, **kwargs)
    _, V2 = generate_noisy_gompertz_observations(seed=2, **kwargs)

    assert not np.array_equal(V1, V2)


def test_noisy_observations_stay_non_negative():
    # Bruit relatif volontairement fort pour vérifier le clipping à 0.
    t_obs, V_obs = generate_noisy_gompertz_observations(
        V0=10.0, r=0.05, K=1000.0, t_span=(0, 50), n_points=100, noise_std_rel=0.8, seed=123
    )
    assert np.all(V_obs >= 0.0)


def test_negative_noise_std_raises_value_error():
    with pytest.raises(ValueError):
        generate_noisy_gompertz_observations(
            V0=10.0, r=0.05, K=1000.0, t_span=(0, 50), n_points=10, noise_std_rel=-0.1, seed=1
        )


def test_mismatched_shapes_raise_value_error():
    with pytest.raises(ValueError):
        calibrate_gompertz(t_obs=np.array([0.0, 1.0, 2.0]), V_obs=np.array([1.0, 2.0]), V0=1.0)


def test_too_few_observations_raise_value_error():
    with pytest.raises(ValueError):
        calibrate_gompertz(t_obs=np.array([0.0]), V_obs=np.array([1.0]), V0=1.0)
