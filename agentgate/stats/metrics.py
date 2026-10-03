"""Statistical evaluation engine: pass^1, pass^k, bootstrap CIs, and McNemar paired comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy import stats


@dataclass
class RunSummaryMetrics:
    total_tasks: int
    num_trials: int
    pass_1: float
    pass_1_ci: Tuple[float, float]
    pass_k: float
    pass_k_ci: Tuple[float, float]
    k: int
    mean_steps: float
    median_steps: float
    mean_duration_seconds: float
    p50_duration_seconds: float
    p95_duration_seconds: float
    total_tokens: int
    mean_tokens_per_task: float
    total_would_have_cost: float
    would_have_cost_per_task: float
    would_have_cost_per_success: float


@dataclass
class PairedComparisonResult:
    baseline_run_id: str
    candidate_run_id: str
    baseline_pass_1: float
    candidate_pass_1: float
    delta_pass_1: float
    delta_ci: Tuple[float, float]
    p_value: float
    contingency_table: Dict[str, int]
    verdict: Literal["REGRESSION", "IMPROVEMENT", "NO_SIGNIFICANT_CHANGE"]
    details: str


def compute_pass_k_unbiased(c: int, n: int, k: int) -> float:
    """Computes comb(c, k) / comb(n, k) for a single task."""
    if n < k or c < k:
        return 0.0
    return float(math.comb(c, k) / math.comb(n, k))


def compute_pass_metrics(
    task_outcomes: Dict[int, List[int]],
    k: int = 2,
) -> Tuple[float, float]:
    """Computes pass^1 and pass^k over a dictionary of task_id -> list of trial binary rewards."""
    if not task_outcomes:
        return 0.0, 0.0

    task_pass_1_list = []
    task_pass_k_list = []

    for task_id, rewards in task_outcomes.items():
        n = len(rewards)
        c = sum(rewards)
        p1 = c / n if n > 0 else 0.0
        pk = compute_pass_k_unbiased(c, n, k)
        task_pass_1_list.append(p1)
        task_pass_k_list.append(pk)

    pass_1 = float(np.mean(task_pass_1_list))
    pass_k = float(np.mean(task_pass_k_list))
    return pass_1, pass_k


def bootstrap_ci_pass(
    task_outcomes: Dict[int, List[int]],
    k: int = 2,
    num_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculates 95% bootstrap confidence intervals for pass^1 and pass^k over tasks."""
    if not task_outcomes:
        return (0.0, 0.0), (0.0, 0.0)

    task_ids = sorted(task_outcomes.keys())
    N = len(task_ids)
    rng = np.random.RandomState(seed)

    # Precompute per-task metrics
    p1_array = np.array([np.mean(task_outcomes[tid]) for tid in task_ids])
    pk_array = np.array(
        [
            compute_pass_k_unbiased(sum(task_outcomes[tid]), len(task_outcomes[tid]), k)
            for tid in task_ids
        ]
    )

    bootstrap_indices = rng.choice(N, size=(num_bootstrap, N), replace=True)
    boot_p1 = np.mean(p1_array[bootstrap_indices], axis=1)
    boot_pk = np.mean(pk_array[bootstrap_indices], axis=1)

    low_q = (alpha / 2.0) * 100.0
    high_q = (1.0 - alpha / 2.0) * 100.0

    ci_p1 = (float(np.percentile(boot_p1, low_q)), float(np.percentile(boot_p1, high_q)))
    ci_pk = (float(np.percentile(boot_pk, low_q)), float(np.percentile(boot_pk, high_q)))
    return ci_p1, ci_pk


def paired_bootstrap_comparison(
    baseline_outcomes: Dict[int, List[int]],
    candidate_outcomes: Dict[int, List[int]],
    num_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    tolerance: float = 0.0,
) -> PairedComparisonResult:
    """Performs paired bootstrap test on delta pass^1 and McNemar's test for regression gating."""
    shared_task_ids = sorted(set(baseline_outcomes.keys()) & set(candidate_outcomes.keys()))
    if not shared_task_ids:
        raise ValueError("No common tasks found between baseline and candidate runs.")

    N = len(shared_task_ids)
    base_p1_per_task = np.array([np.mean(baseline_outcomes[tid]) for tid in shared_task_ids])
    cand_p1_per_task = np.array([np.mean(candidate_outcomes[tid]) for tid in shared_task_ids])

    base_p1 = float(np.mean(base_p1_per_task))
    cand_p1 = float(np.mean(cand_p1_per_task))
    delta_p1 = cand_p1 - base_p1

    rng = np.random.RandomState(seed)
    boot_indices = rng.choice(N, size=(num_bootstrap, N), replace=True)

    boot_base = np.mean(base_p1_per_task[boot_indices], axis=1)
    boot_cand = np.mean(cand_p1_per_task[boot_indices], axis=1)
    boot_deltas = boot_cand - boot_base

    low_q = (alpha / 2.0) * 100.0
    high_q = (1.0 - alpha / 2.0) * 100.0
    delta_ci = (float(np.percentile(boot_deltas, low_q)), float(np.percentile(boot_deltas, high_q)))

    # McNemar's test on binary task success (all trials succeed)
    n00 = n01 = n10 = n11 = 0
    for tid in shared_task_ids:
        b_succ = int(all(r == 1 for r in baseline_outcomes[tid]))
        c_succ = int(all(r == 1 for r in candidate_outcomes[tid]))
        if b_succ == 0 and c_succ == 0:
            n00 += 1
        elif b_succ == 0 and c_succ == 1:
            n01 += 1
        elif b_succ == 1 and c_succ == 0:
            n10 += 1
        else:
            n11 += 1

    contingency = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    discordant = n01 + n10

    if discordant == 0:
        p_value = 1.0
    elif discordant < 25:
        # Exact binomial test
        # p-value = 2 * min(P(X <= min(n01, n10)), 0.5)
        k_min = min(n01, n10)
        p_value = float(2.0 * stats.binom.cdf(k_min, discordant, 0.5))
        p_value = min(1.0, p_value)
    else:
        # Chi-square with Edwards continuity correction
        chi2 = ((abs(n01 - n10) - 1.0) ** 2) / discordant
        p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))

    verdict: Literal["REGRESSION", "IMPROVEMENT", "NO_SIGNIFICANT_CHANGE"] = "NO_SIGNIFICANT_CHANGE"
    details = ""

    # Determine Verdict
    # REGRESSION if delta is negative and statistically significant OR drop exceeds tolerance
    if delta_ci[1] < -tolerance and p_value < alpha:
        verdict = "REGRESSION"
        details = f"Statistically significant regression detected: delta={delta_p1:+.4f} (95% CI {delta_ci}), p={p_value:.4f}"
    elif delta_p1 < -tolerance and p_value < 0.10:
        verdict = "REGRESSION"
        details = f"Regression exceeded tolerance ({tolerance}): delta={delta_p1:+.4f} (95% CI {delta_ci}), p={p_value:.4f}"
    elif delta_ci[0] > tolerance and p_value < alpha:
        verdict = "IMPROVEMENT"
        details = f"Statistically significant improvement: delta={delta_p1:+.4f} (95% CI {delta_ci}), p={p_value:.4f}"
    else:
        verdict = "NO_SIGNIFICANT_CHANGE"
        details = f"No significant change detected: delta={delta_p1:+.4f} (95% CI {delta_ci}), p={p_value:.4f}"

    return PairedComparisonResult(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_pass_1=base_p1,
        candidate_pass_1=cand_p1,
        delta_pass_1=delta_p1,
        delta_ci=delta_ci,
        p_value=p_value,
        contingency_table=contingency,
        verdict=verdict,
        details=details,
    )
