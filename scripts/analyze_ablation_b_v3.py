"""Statistical analysis and paired comparison for ablB_v3 vs baseline_v1_steps30 (first 10 tasks)."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from agentgate.stats.metrics import (
    bootstrap_ci_pass,
    compute_pass_metrics,
    paired_bootstrap_comparison,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    first_10_tasks = [16, 0, 10, 17, 3, 40, 22, 2, 31, 1]

    with sqlite3.connect("results/agentgate.db") as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # 1. Fetch Candidate (ablB_v3)
        cur.execute(
            """
            SELECT task_id, trial_id, success, reward, steps, would_have_cost, 
                   total_prompt_tokens, total_completion_tokens, total_tokens, 
                   duration_seconds, trajectory_json, info_json, error, status
            FROM trial_results
            WHERE run_id = 'ablB_v3'
            ORDER BY task_id, trial_id
            """
        )
        cand_rows = cur.fetchall()

        # 2. Fetch Baseline slice (baseline_v1_steps30 on the same 10 tasks)
        placeholders = ",".join("?" for _ in first_10_tasks)
        cur.execute(
            f"""
            SELECT task_id, trial_id, success, reward, steps, would_have_cost,
                   total_prompt_tokens, total_completion_tokens, total_tokens,
                   duration_seconds, trajectory_json, info_json, error, status
            FROM trial_results
            WHERE run_id = 'baseline_v1_steps30' AND task_id IN ({placeholders})
            ORDER BY task_id, trial_id
            """,
            first_10_tasks,
        )
        base_rows = cur.fetchall()

    if not cand_rows:
        print("[ERROR] No trials found for ablB_v3 in results/agentgate.db")
        return

    print("================================================================================")
    print("      ABLATION B (Prompt v3: degraded multi-factor) vs BASELINE SLICE          ")
    print("================================================================================")
    print(f"Candidate Run ID:   ablB_v3 (Prompt v3: degraded multi-factor)")
    print(f"Reference Baseline: baseline_v1_steps30 (Prompt v1, max_steps=30)")
    print(f"Task Subset (10):   {first_10_tasks}")
    print(f"Candidate Trials:   {len(cand_rows)} completed / {len(first_10_tasks)*2} planned")
    print(f"Baseline Trials:    {len(base_rows)} / 20")

    # Group outcomes
    cand_outcomes: dict[int, list[int]] = defaultdict(list)
    cand_trial_map: dict[tuple[int, int], int] = {}
    cand_steps: list[int] = []
    cand_costs: list[float] = []
    cand_prompt_tokens: int = 0
    cand_comp_tokens: int = 0
    cand_total_tokens: int = 0

    for r in cand_rows:
        cand_outcomes[r["task_id"]].append(int(r["success"]))
        cand_trial_map[(r["task_id"], r["trial_id"])] = int(r["success"])
        cand_steps.append(r["steps"])
        cand_costs.append(r["would_have_cost"])
        cand_prompt_tokens += r["total_prompt_tokens"]
        cand_comp_tokens += r["total_completion_tokens"]
        cand_total_tokens += r["total_tokens"]

    base_outcomes: dict[int, list[int]] = defaultdict(list)
    base_trial_map: dict[tuple[int, int], int] = {}
    base_steps: list[int] = []
    base_costs: list[float] = []

    for r in base_rows:
        base_outcomes[r["task_id"]].append(int(r["success"]))
        base_trial_map[(r["task_id"], r["trial_id"])] = int(r["success"])
        base_steps.append(r["steps"])
        base_costs.append(r["would_have_cost"])

    # Metrics
    cand_p1, cand_pk = compute_pass_metrics(cand_outcomes, k=2)
    (cand_ci_p1, cand_ci_pk) = bootstrap_ci_pass(cand_outcomes, k=2, num_bootstrap=10000, seed=42)
    cand_trial_pass1 = sum(sum(v) for v in cand_outcomes.values()) / max(1, sum(len(v) for v in cand_outcomes.values()))

    base_p1, base_pk = compute_pass_metrics(base_outcomes, k=2)
    (base_ci_p1, base_ci_pk) = bootstrap_ci_pass(base_outcomes, k=2, num_bootstrap=10000, seed=42)
    base_trial_pass1 = sum(sum(v) for v in base_outcomes.values()) / max(1, sum(len(v) for v in base_outcomes.values()))

    print("\n--- 1. PERFORMANCE SUMMARY ---")
    print(f"Baseline (10 tasks, steps30, v1):")
    print(f"  Trial-level pass^1: {base_trial_pass1:.2%} (20/20)")
    print(f"  Task-level  pass^1: {base_p1:.2%} (95% CI: [{base_ci_p1[0]:.2%}, {base_ci_p1[1]:.2%}])")
    print(f"  Task-level  pass^2: {base_pk:.2%} (95% CI: [{base_ci_pk[0]:.2%}, {base_ci_pk[1]:.2%}])")
    print(f"  Mean steps/trial:   {np.mean(base_steps):.1f}")
    print(f"  Total cost:         ${sum(base_costs):.5f}")

    print(f"\nCandidate ablB_v3 (Prompt v3: degraded):")
    print(f"  Trial-level pass^1: {cand_trial_pass1:.2%} ({sum(sum(v) for v in cand_outcomes.values())}/{sum(len(v) for v in cand_outcomes.values())})")
    print(f"  Task-level  pass^1: {cand_p1:.2%} (95% CI: [{cand_ci_p1[0]:.2%}, {cand_ci_p1[1]:.2%}])")
    print(f"  Task-level  pass^2: {cand_pk:.2%} (95% CI: [{cand_ci_pk[0]:.2%}, {cand_ci_pk[1]:.2%}])")
    print(f"  Mean steps/trial:   {np.mean(cand_steps):.1f}")
    print(f"  Total tokens:       {cand_total_tokens:,} ({cand_prompt_tokens:,} prompt + {cand_comp_tokens:,} comp)")
    print(f"  Would-have-cost:    ${sum(cand_costs):.5f}")

    # Paired Bootstrap Comparison
    paired_comp = paired_bootstrap_comparison(
        baseline_outcomes=base_outcomes,
        candidate_outcomes=cand_outcomes,
        num_bootstrap=10000,
        seed=42,
    )

    print("\n--- 2. PAIRED STATISTICAL COMPARISON (Task-Level Paired Bootstrap) ---")
    print(f"  Delta pass^1 (Task-Level): {paired_comp.delta_pass_1:+.2%}")
    print(f"  95% Paired Bootstrap CI:   [{paired_comp.delta_ci[0]:+.2%}, {paired_comp.delta_ci[1]:+.2%}]")
    print(f"  Bootstrap p-value:         {paired_comp.p_value:.4f}")
    print(f"  Statistical Verdict:       {paired_comp.verdict}")

    # Paired Trials McNemar
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0
    paired_keys = sorted(set(base_trial_map.keys()) & set(cand_trial_map.keys()))
    for k in paired_keys:
        b = base_trial_map[k]
        c = cand_trial_map[k]
        if b == 0 and c == 0:
            n00 += 1
        elif b == 0 and c == 1:
            n01 += 1
        elif b == 1 and c == 0:
            n10 += 1
        elif b == 1 and c == 1:
            n11 += 1

    b_disc = n01
    c_disc = n10
    total_disc = b_disc + c_disc
    if total_disc > 0:
        p_val_mcnemar = float(stats.binom.pmf(b_disc, total_disc, 0.5) * 2)
        p_val_mcnemar = min(1.0, p_val_mcnemar)
    else:
        p_val_mcnemar = 1.0

    print("\n--- 3. MCNEMAR'S TEST ON PAIRED TRIALS ---")
    print(f"  Both Passed (n11):          {n11}")
    print(f"  Both Failed (n00):          {n00}")
    print(f"  Gained Successes (n01):     {n01} (Baseline 0 -> Candidate 1)")
    print(f"  Lost Successes / Regr (n10):{n10} (Baseline 1 -> Candidate 0)")
    print(f"  McNemar Exact p-value:      {p_val_mcnemar:.4f}")

    print("\n--- 4. FAILED TRIALS AUDIT (Every failed trial in ablB_v3) ---")
    failed_trials = [r for r in cand_rows if not r["success"]]
    if not failed_trials:
        print("  None! All trials passed.")
    else:
        for r in failed_trials:
            info = json.loads(r["info_json"]) if r["info_json"] else {}
            end_reason = "unknown"
            if r["trajectory_json"]:
                traj = json.loads(r["trajectory_json"])
                if traj and traj[-1].get("content") == "###STOP###":
                    end_reason = "customer_simulator_stopped (###STOP###)"
                elif r["steps"] >= 30:
                    end_reason = "max_steps_reached (30)"
                elif r["error"]:
                    end_reason = f"error: {r['error']}"
                else:
                    end_reason = "agent_completed_goal_not_satisfied"

            print(f"  Task {r['task_id']}, Trial {r['trial_id']}:")
            print(f"    Success: {bool(r['success'])}, Reward: {r['reward']}")
            print(f"    Steps Used: {r['steps']}")
            print(f"    Duration: {r['duration_seconds']:.1f}s")
            print(f"    End Reason: {end_reason}")
            if r["error"]:
                print(f"    Error: {r['error']}")
            print()

    # Per-task comparison table
    print("\n--- 5. PER-TASK BREAKDOWN ---")
    print(f"{'Task ID':<10} | {'Baseline (v1)':<15} | {'Candidate (v3)':<15} | {'Delta':<10}")
    print("-" * 58)
    for tid in first_10_tasks:
        b_p = np.mean(base_outcomes[tid]) if tid in base_outcomes else 0.0
        c_p = np.mean(cand_outcomes[tid]) if tid in cand_outcomes else 0.0
        print(f"{tid:<10} | {b_p:<15.1%} | {c_p:<15.1%} | {c_p - b_p:+.1%}")


if __name__ == "__main__":
    main()
