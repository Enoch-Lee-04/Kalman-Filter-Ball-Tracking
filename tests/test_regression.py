"""The seed-7 headless run must keep producing the recorded numbers."""
from tools import regression


def test_seed7_matches_golden():
    expected = regression.load_golden()
    actual = regression.measure(expected["seed"], expected["frames"])
    problems = regression.compare(expected, actual)
    assert not problems, (
        "seed-7 results changed:\n  " + "\n  ".join(problems)
        + "\nIf intended, run `python -m tools.regression --record` and commit "
          "tests/golden/seed7.json in the same commit."
    )


def test_compare_flags_count_and_residual_changes():
    base = {"strikes": 2, "hits": 2, "misses": 0,
            "residual_px": {"count": 10, "median": 5.0}}
    assert regression.compare(base, base) == []
    moved = {**base, "hits": 1, "residual_px": {"count": 10, "median": 5.001}}
    problems = regression.compare(base, moved)
    assert any(p.startswith("hits:") for p in problems)
    assert any("residual_px.median" in p for p in problems)
    # last-bit float noise is tolerated
    noise = {**base, "residual_px": {"count": 10, "median": 5.0 + 1e-9}}
    assert regression.compare(base, noise) == []
