import numpy as np
import pytest

from simulation.diffusion_reaction import (
    DiffusionReactionModel,
    FisherKPPModel,
    simulate_2d,
)


def test_no_divergence():
    t, grids = simulate_2d(
        grid_size=50, D=0.01, r=0.1, K=1.0,
        initial_tumor_position=(25, 25), initial_radius=3, n_steps=100,
    )
    assert np.all(np.isfinite(grids))


def test_values_bounded_between_zero_and_k():
    K = 1.0
    t, grids = simulate_2d(
        grid_size=50, D=0.01, r=0.1, K=K,
        initial_tumor_position=(25, 25), initial_radius=3, n_steps=100,
    )
    assert np.all(grids >= 0.0)
    assert np.all(grids <= K)


def test_initial_condition_is_a_disk_at_k():
    K = 1.0
    grid_size = 30
    position = (15, 15)
    radius = 4
    t, grids = simulate_2d(
        grid_size=grid_size, D=0.01, r=0.1, K=K,
        initial_tumor_position=position, initial_radius=radius, n_steps=1,
    )
    initial_grid = grids[0]

    i0, j0 = position
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    inside = (ii - i0) ** 2 + (jj - j0) ** 2 <= radius**2

    assert np.all(initial_grid[inside] == K)
    assert np.all(initial_grid[~inside] == 0.0)


def test_tumor_front_advances_over_time():
    K = 1.0
    t, grids = simulate_2d(
        grid_size=60, D=0.02, r=0.2, K=K,
        initial_tumor_position=(30, 30), initial_radius=3, n_steps=200,
    )
    area_initial = np.sum(grids[0] > K / 2)
    area_final = np.sum(grids[-1] > K / 2)
    assert area_final > area_initial


def test_front_speed_matches_kpp_theory():
    """Le rayon effectif du front (seuil u > K/2) doit progresser, une fois le
    régime transitoire de courbure passé, à une vitesse proche de la vitesse
    théorique de Fisher-KPP c = 2*sqrt(r*D).

    Ce test protège contre un patch initial sous-critique (le rayon initial
    trop petit par rapport à l'échelle de diffusion sqrt(D/r)) qui provoque
    l'extinction de la tumeur au lieu de sa propagation — un tel cas ferait
    échouer cette assertion plutôt que de passer inaperçu.
    """
    D, r, K = 0.005, 0.05, 1.0
    grid_size = 200
    result = simulate_2d(
        grid_size=grid_size, D=D, r=r, K=K,
        initial_tumor_position=(grid_size // 2, grid_size // 2),
        initial_radius=3, n_steps=450,
    )
    radii_mm = np.array(
        [np.sqrt(np.sum(grid > 0.5 * K) / np.pi) for grid in result.grids]
    ) * result.dx

    start = int(0.6 * len(result.t))
    slope, _ = np.polyfit(result.t[start:-1], radii_mm[start:-1], 1)

    c_theory = 2 * np.sqrt(r * D)
    assert slope == pytest.approx(c_theory, rel=0.15)


def test_front_speed_converges_toward_theory_as_r_times_t_grows():
    """La vitesse du front dépend du produit r*t : elle est loin de c = 2*sqrt(r*D)
    au début, puis s'en approche progressivement.

    En prenant sqrt(D/r) comme unité de longueur, l'équation ne dépend plus que de
    r*t. C'est ce qui explique l'écart d'environ 57 % observé dans le notebook avec
    la prolifération lente des gliomes (r*t ~ 2 seulement après ~170 jours) : ce
    n'est pas un défaut du code mais un régime transitoire, ici vérifié en
    prolongeant la simulation jusqu'à r*t ~ 10 avec des paramètres rapides.
    """
    D, r, K = 0.005, 0.05, 1.0
    length = np.sqrt(D / r)
    grid_size, domain_mm = 240, 60 * length
    dx = domain_mm / grid_size
    n_steps = int(np.ceil(12 / (r * 0.9 * dx**2 / (4 * D)))) + 1

    result = simulate_2d(
        grid_size=grid_size, D=D, r=r, K=K,
        initial_tumor_position=(grid_size // 2, grid_size // 2),
        initial_radius=round(3.4 * length / dx), n_steps=n_steps, domain_size_mm=domain_mm,
    )
    radii_mm = np.array([np.sqrt(np.sum(g > 0.5 * K) / np.pi) for g in result.grids]) * result.dx
    c_theory = 2 * np.sqrt(r * D)

    def speed_ratio(r_times_t):
        window = (result.t >= (r_times_t - 1) / r) & (result.t <= (r_times_t + 1) / r)
        return np.polyfit(result.t[window], radii_mm[window], 1)[0] / c_theory

    early, middle, late = speed_ratio(2), speed_ratio(6), speed_ratio(10)
    assert early < 0.6
    assert early < middle < late
    assert late > 0.8


def test_dt_respects_cfl_bound():
    D = 0.05
    result = simulate_2d(
        grid_size=40, D=D, r=0.1, K=1.0,
        initial_tumor_position=(20, 20), initial_radius=3, n_steps=10,
    )
    assert result.dt <= result.dx**2 / (4 * D)


def test_custom_dt_violating_cfl_raises_value_error():
    with pytest.raises(ValueError, match="Condition CFL violée"):
        simulate_2d(
            grid_size=40,
            D=0.05,
            r=0.1,
            K=1.0,
            initial_tumor_position=(20, 20),
            initial_radius=3,
            n_steps=10,
            dt=100.0,
        )


def test_model_class_matches_function():
    params = dict(
        grid_size=40, D=0.01, r=0.1, K=1.0,
        initial_tumor_position=(20, 20), initial_radius=3, n_steps=50,
    )
    t_func, grids_func = simulate_2d(**params)
    model = FisherKPPModel(**params)
    t_model, grids_model = model.simulate()
    assert np.allclose(grids_func, grids_model)


def test_diffusion_reaction_model_alias_points_to_fisher_kpp():
    assert DiffusionReactionModel is FisherKPPModel
    model = DiffusionReactionModel(
        grid_size=40,
        D=0.01,
        r=0.1,
        K=1.0,
        initial_tumor_position=(20, 20),
        initial_radius=3,
        n_steps=5,
    )
    assert model.simulate()


@pytest.mark.parametrize(
    "invalid_params",
    [
        dict(grid_size=0, D=0.01, r=0.1, K=1.0, initial_tumor_position=(0, 0), initial_radius=1, n_steps=10),
        dict(grid_size=40, D=0.0, r=0.1, K=1.0, initial_tumor_position=(20, 20), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=-0.01, r=0.1, K=1.0, initial_tumor_position=(20, 20), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.0, K=1.0, initial_tumor_position=(20, 20), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=0.0, initial_tumor_position=(20, 20), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=1.0, initial_tumor_position=(20, 20), initial_radius=0, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=1.0, initial_tumor_position=(20, 20), initial_radius=40, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=1.0, initial_tumor_position=(-1, 20), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=1.0, initial_tumor_position=(20, 40), initial_radius=3, n_steps=10),
        dict(grid_size=40, D=0.01, r=0.1, K=1.0, initial_tumor_position=(20, 20), initial_radius=3, n_steps=0),
    ],
)
def test_invalid_parameters_raise_value_error(invalid_params):
    with pytest.raises(ValueError):
        simulate_2d(**invalid_params)