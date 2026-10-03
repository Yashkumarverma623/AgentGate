"""Unit tests for statistical metrics against hand-computed ground truth fixtures."""

from agentgate.stats.metrics import (
    bootstrap_ci_pass,
    compute_pass_k_unbiased,
    compute_pass_metrics,
    paired_bootstrap_comparison,
)


def test_compute_pass_k_hand_computed() -> None:
    # Test case 1: n=4, k=2, c=2 => comb(2,2)/comb(4,2) = 1/6
    assert abs(compute_pass_k_unbiased(c=2, n=4, k=2) - (1.0 / 6.0)) < 1e-6

    # Test case 2: n=4, k=4, c=4 => 1.0
    assert abs(compute_pass_k_unbiased(c=4, n=4, k=4) - 1.0) < 1e-6

    # Test case 3: n=2, k=2, c=1 => 0.0
    assert compute_pass_k_unbiased(c=1, n=2, k=2) == 0.0

    # Test case 4: n=2, k=2, c=2 => 1.0
    assert compute_pass_k_unbiased(c=2, n=2, k=2) == 1.0

    # Test case 5: c < k => 0.0
    assert compute_pass_k_unbiased(c=0, n=2, k=2) == 0.0


def test_compute_pass_metrics_two_tasks() -> None:
    # Task 10: 2 trials, both success [1, 1] => p1=1.0, p2=1.0
    # Task 20: 2 trials, 1 success [1, 0] => p1=0.5, p2=0.0
    # Expected overall: p1 = (1.0 + 0.5)/2 = 0.75, p2 = (1.0 + 0.0)/2 = 0.50
    outcomes = {10: [1, 1], 20: [1, 0]}
    p1, p2 = compute_pass_metrics(outcomes, k=2)
    assert abs(p1 - 0.75) < 1e-6
    assert abs(p2 - 0.50) < 1e-6


def test_bootstrap_ci_deterministic() -> None:
    outcomes = {i: [1, 1] if i < 10 else [0, 0] for i in range(20)}
    ci_p1, ci_pk = bootstrap_ci_pass(outcomes, k=2, num_bootstrap=1000, seed=42)
    assert 0.30 <= ci_p1[0] <= ci_p1[1] <= 0.70
    assert 0.30 <= ci_pk[0] <= ci_pk[1] <= 0.70


def test_paired_comparison_identical() -> None:
    outcomes = {i: [1, 0] for i in range(20)}
    res = paired_bootstrap_comparison(outcomes, outcomes, num_bootstrap=500, seed=42)
    assert abs(res.delta_pass_1) < 1e-6
    assert res.p_value == 1.0
    assert res.verdict == "NO_SIGNIFICANT_CHANGE"


def test_paired_comparison_regression() -> None:
    base = {i: [1, 1] for i in range(30)}
    cand = {i: [0, 0] for i in range(30)}
    res = paired_bootstrap_comparison(base, cand, num_bootstrap=500, seed=42)
    assert res.delta_pass_1 == -1.0
    assert res.delta_ci[1] < 0.0
    assert res.p_value < 0.001
    assert res.verdict == "REGRESSION"


def test_paired_comparison_improvement() -> None:
    base = {i: [0, 0] for i in range(30)}
    cand = {i: [1, 1] for i in range(30)}
    res = paired_bootstrap_comparison(base, cand, num_bootstrap=500, seed=42)
    assert res.delta_pass_1 == 1.0
    assert res.delta_ci[0] > 0.0
    assert res.p_value < 0.001
    assert res.verdict == "IMPROVEMENT"
