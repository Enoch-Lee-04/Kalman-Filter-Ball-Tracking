"""The no-render batch harness is deterministic and its bookkeeping adds up."""
import pytest

from config import apply_overrides, load_config
from tools import batch


def test_batch_is_deterministic_and_consistent():
    cfg = load_config()
    a = batch.run_batch(cfg, seeds=[0, 1, 2], frames=1200)
    b = batch.run_batch(cfg, seeds=[0, 1, 2], frames=1200)
    assert a == b

    for run in a["per_run"]:
        assert run["frames"] == 1200
        finished = run["hits"] + run["misses"]
        assert finished == len(run["strike_records"])
        assert run["strikes"] - finished in (0, 1)  # at most one strike in flight
    agg = a["aggregate"]
    assert agg["strikes"] == agg["hits"] + agg["misses"] + agg["unfinished"]


def test_standing_still_never_strikes():
    result = batch.run_batch(load_config(), seeds=[0], frames=1500, allow_strikes=False)
    assert result["aggregate"]["strikes"] == 0


def test_cli_writes_json(tmp_path):
    out = tmp_path / "r.json"
    md = tmp_path / "s.md"
    assert batch.main(["--seeds", "3", "--frames", "300", "--quiet",
                       "--json", str(out), "--md", str(md)]) == 0
    assert out.exists() and md.read_text(encoding="utf-8").startswith("### ")


def test_parse_seeds():
    assert batch.parse_seeds("0-3") == [0, 1, 2, 3]
    assert batch.parse_seeds("1,4, 9") == [1, 4, 9]
    assert batch.parse_seeds("3-5,10") == [3, 4, 5, 10]
    with pytest.raises(ValueError):
        batch.parse_seeds("5-3")


def test_overrides_are_typed_and_strict():
    cfg = load_config()
    out = apply_overrides(cfg, ["forward_model.residual_threshold_px=6",
                                "robot.start_pos=[1.0, -2.0]"])
    assert out["forward_model"]["residual_threshold_px"] == 6
    assert out["robot"]["start_pos"] == [1.0, -2.0]
    assert cfg["forward_model"]["residual_threshold_px"] == 3  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["forward_model.residual_thresh=6"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["forward_model.residual_threshold_px"])
