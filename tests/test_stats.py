import random

from otf.evaluation.stats import block_bootstrap_ci, dm_test, latex_table


def test_dm_test_detects_a_clear_winner():
    rng = random.Random(1)
    deltas = [0.02 + rng.gauss(0.0, 0.005) for _ in range(60)]
    stat, p = dm_test(deltas)
    assert stat > 0 and p < 0.01


def test_dm_test_neutral_on_noise():
    rng = random.Random(4)
    deltas = [rng.gauss(0.0, 0.01) for _ in range(60)]
    _, p = dm_test(deltas)
    assert p > 0.05


def test_dm_test_degenerate_inputs():
    assert dm_test([]) == (0.0, 1.0)
    assert dm_test([0.1]) == (0.0, 1.0)


def test_block_bootstrap_ci_brackets_the_mean():
    rng = random.Random(3)
    xs = [0.01 + rng.gauss(0.0, 0.02) for _ in range(80)]
    lo, hi = block_bootstrap_ci(xs)
    m = sum(xs) / len(xs)
    assert lo < m < hi


def test_latex_table_escapes_and_structures():
    out = latex_table("t1", "50% of a_b", ["Name", "Val"], [["x_y", 1.0]])
    assert r"\%" in out and r"a\_b" in out and r"\toprule" in out
    assert "tab:t1" in out
