import numpy as np
import pytest

from simulation.gompertz import GompertzModel, simulate


def test_converges_to_carrying_capacity_from_below():
    K = 1000.0
    t, V = simulate(V0=10.0, r=0.1, K=K, t_span=(0, 500), n_points=200)
    assert V[-1] == pytest.approx(K, rel=1e-2)


def test_converges_to_carrying_capacity_from_above():
    K = 1000.0
    t, V = simulate(V0=5000.0, r=0.1, K=K, t_span=(0, 500), n_points=200)
    assert V[-1] == pytest.approx(K, rel=1e-2)


def test_no_nan_or_negative_values():
    t, V = simulate(V0=10.0, r=0.2, K=1000.0, t_span=(0, 200), n_points=100)
    assert np.all(np.isfinite(V))
    assert np.all(V > 0)


def test_monotonic_growth_when_below_carrying_capacity():
    # Tolérance liée à la précision de l'intégrateur près du plateau
    # d'équilibre : on ne peut pas exiger une monotonie stricte au bit près.
    K = 1000.0
    t, V = simulate(V0=10.0, r=0.1, K=K, t_span=(0, 500), n_points=200)
    assert np.all(np.diff(V) >= -1e-6 * K)


def test_monotonic_decay_when_above_carrying_capacity():
    K = 1000.0
    t, V = simulate(V0=5000.0, r=0.1, K=K, t_span=(0, 500), n_points=200)
    assert np.all(np.diff(V) <= 1e-6 * K)


def test_unpacking_returns_time_and_volume_arrays():
    t, V = simulate(V0=10.0, r=0.1, K=1000.0, t_span=(0, 100), n_points=50)
    assert t.shape == (50,)
    assert V.shape == (50,)


def test_model_class_matches_function():
    params = dict(V0=10.0, r=0.15, K=800.0, t_span=(0, 300), n_points=80)
    t_func, V_func = simulate(**params)
    model = GompertzModel(**params)
    t_model, V_model = model.simulate()
    assert np.allclose(t_func, t_model)
    assert np.allclose(V_func, V_model)


@pytest.mark.parametrize(
    "invalid_params",
    [
        dict(V0=0.0, r=0.1, K=1000.0, t_span=(0, 100), n_points=50),
        dict(V0=-10.0, r=0.1, K=1000.0, t_span=(0, 100), n_points=50),
        dict(V0=10.0, r=0.1, K=0.0, t_span=(0, 100), n_points=50),
        dict(V0=10.0, r=0.1, K=-5.0, t_span=(0, 100), n_points=50),
        dict(V0=10.0, r=0.0, K=1000.0, t_span=(0, 100), n_points=50),
        dict(V0=10.0, r=-0.1, K=1000.0, t_span=(0, 100), n_points=50),
        dict(V0=10.0, r=0.1, K=1000.0, t_span=(100, 100), n_points=50),
        dict(V0=10.0, r=0.1, K=1000.0, t_span=(100, 0), n_points=50),
        dict(V0=10.0, r=0.1, K=1000.0, t_span=(0, 100), n_points=1),
    ],
)
def test_invalid_parameters_raise_value_error(invalid_params):
    with pytest.raises(ValueError):
        simulate(**invalid_params)
