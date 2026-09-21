import asyncio

import numpy as np
import pytest

pytest.importorskip("mcp", reason="extra optionnel : pip install -e .[mcp]")

from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import tools
from mcp_server.server import mcp

_FISHER_PARAMS = dict(
    grid_size=30, D=0.01, r=0.1, K=1.0,
    initial_tumor_position=[15, 15], initial_radius=3, n_steps=60,
)
_COMPARE_KWARGS = dict(
    grid_size=30, D=0.01, r=0.1, K=1.0, initial_tumor_position=[15, 15],
    initial_radius=3, n_steps=80, resection_step=30, resection_center=[15, 15],
)


def test_get_model_info_describes_both_models():
    info = tools.get_model_info()
    assert set(info) == {"gompertz", "fisher_kpp"}
    for model_info in info.values():
        assert {"description", "parameters", "reference"} <= set(model_info)


def test_simulate_gompertz_returns_time_and_volume():
    result = tools.simulate_tumor_growth(
        "gompertz", {"V0": 1.0, "r": 0.0738, "K": 2600.0, "t_span": [0, 90], "n_points": 10}
    )
    assert len(result["t"]) == len(result["V"]) == 10
    assert result["V"][0] == pytest.approx(1.0)
    assert result["V"][-1] < 2600.0


def test_simulate_fisher_kpp_returns_radius_curve_without_full_grid_by_default():
    result = tools.simulate_tumor_growth("fisher_kpp", _FISHER_PARAMS)
    assert len(result["t"]) == len(result["effective_radius_mm"]) == 61
    assert "final_grid_preview" not in result
    assert result["effective_radius_mm"][-1] > result["effective_radius_mm"][0]


def test_simulate_fisher_kpp_final_grid_preview_is_downsampled():
    result = tools.simulate_tumor_growth(
        "fisher_kpp", _FISHER_PARAMS, include_final_grid=True, max_grid_preview_size=10
    )
    preview = np.array(result["final_grid_preview"])
    assert preview.shape == (10, 10)


def test_downsample_grid_non_divisible_size_still_returns_requested_size():
    preview = tools._downsample_grid(np.ones((30, 30)), 7)
    assert np.array(preview).shape == (7, 7)


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="model doit être"):
        tools.simulate_tumor_growth("logistique", {})


@pytest.mark.parametrize("bad_step", [0, 80, 100])
def test_resection_step_outside_range_raises_value_error(bad_step):
    kwargs = {**_COMPARE_KWARGS, "resection_step": bad_step, "resection_radius": 4}
    with pytest.raises(ValueError, match="resection_step"):
        tools.compare_surgical_scenarios(**kwargs)


def test_partial_resection_lowers_radius_then_scenarios_share_history_before_it():
    result = tools.compare_surgical_scenarios(**_COMPARE_KWARGS, resection_radius=3)
    no_res = result["effective_radius_mm_no_resection"]
    with_res = result["effective_radius_mm_with_resection"]

    assert no_res[:31] == with_res[:31]  # identiques jusqu'à la résection incluse
    assert with_res[31] < no_res[31]  # effet immédiat de la résection
    assert len(result["t"]) == len(no_res) == len(with_res) == 81


def test_complete_resection_eliminates_tumor_permanently():
    result = tools.compare_surgical_scenarios(**_COMPARE_KWARGS, resection_radius=40)
    assert result["effective_radius_mm_with_resection"][-1] == 0.0
    assert result["effective_radius_mm_no_resection"][-1] > 0.0


def test_resection_hiding_the_visible_tumor_still_lets_the_invisible_tail_regrow():
    # Un disque qui couvre le bord visible (rayon effectif 0 juste après) mais pas la queue
    # de densité très faible qui le dépasse : la tumeur repousse. Un disque qui couvre toute
    # la queue (densité non nulle jusqu'à ~33,5 cellules ici) l'élimine réellement.
    kwargs = dict(
        grid_size=60, D=0.01, r=0.1, K=1.0, initial_tumor_position=[30, 30], initial_radius=4,
        n_steps=200, resection_step=40, resection_center=[30, 30],
    )
    hidden = tools.compare_surgical_scenarios(resection_radius=24, **kwargs)
    eliminated = tools.compare_surgical_scenarios(resection_radius=36, **kwargs)

    assert hidden["effective_radius_mm_with_resection"][41] == 0.0
    assert hidden["effective_radius_mm_with_resection"][-1] > 10.0
    assert eliminated["effective_radius_mm_with_resection"][-1] == 0.0


def test_server_registers_the_three_tools():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"get_model_info", "simulate_tumor_growth", "compare_surgical_scenarios"}


def test_server_call_tool_end_to_end():
    result = asyncio.run(
        mcp.call_tool(
            "simulate_tumor_growth",
            {"model": "gompertz", "params": {"V0": 1.0, "r": 0.07, "K": 2600.0, "t_span": [0, 30], "n_points": 4}},
        )
    )
    assert not result.is_error
    assert '"model": "gompertz"' in result.content[0].text


def test_server_invalid_model_error_message_reaches_the_agent():
    with pytest.raises(ToolError, match="model doit être"):
        asyncio.run(mcp.call_tool("simulate_tumor_growth", {"model": "inconnu", "params": {}}))


def test_server_missing_parameter_error_names_the_parameter():
    with pytest.raises(ToolError, match="'V0'"):
        asyncio.run(mcp.call_tool("simulate_tumor_growth", {"model": "gompertz", "params": {}}))


def test_server_invalid_value_error_message_reaches_the_agent():
    bad = {"V0": -1.0, "r": 0.07, "K": 2600.0, "t_span": [0, 30], "n_points": 4}
    with pytest.raises(ToolError, match="V0 doit être strictement positif"):
        asyncio.run(mcp.call_tool("simulate_tumor_growth", {"model": "gompertz", "params": bad}))
