"""Phase 2 Run: baseline_v1_steps30 with max_steps=30 and paired comparison against baseline_v1."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

import numpy as np
from scipy import stats

from agentgate.adapter.tau_bench import TauBenchRetailAdapter
from agentgate.agent.models import AgentConfig
from agentgate.runner.runner import EvaluationRunner
from agentgate.stats.metrics import (
    bootstrap_ci_pass,
    compute_pass_metrics,
    paired_bootstrap_comparison,
)


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set.")
        sys.exit(1)

    agent_model = "gemini/gemini-3.1-flash-lite"
    user_model = "gemini/gemma-4-26b-a4b-it"
    run_id = "baseline_v1_steps30"
    baseline_run_id = "baseline_v1"

    adapter = TauBenchRetailAdapter(task_split="test")
    runner = EvaluationRunner()

    task_ids = adapter.get_stratified_subset_ids(count=15)
    num_trials = 2
    concurrency = 3

    # Exact same configuration as baseline_v1 except max_steps=30
    config = AgentConfig(
        model=agent_model,
        user_model=user_model,
        max_steps=30,  # Only changed field
        system_prompt_version="v1",
        tool_description_variant="standard",
        temperature=0.0,
    )

    print("==================================================")
    print("      RUN: baseline_v1_steps30 (max_steps=30)     ")
    print("==================================================")
    print(f"Run ID:            {run_id}")
    print(f"Agent Model:       {agent_model}")
    print(f"User Simulator:    {user_model}")
    print(f"Tasks ({len(task_ids)}):        {task_ids}")
    print(f"Trials per Task:   {num_trials} (Total trials: {len(task_ids) * num_trials})")
    print(f"Max Steps Cap:     {config.max_steps} (canonical tau-bench default)")
    print(f"Concurrency:       {concurrency} workers (pacer lock: 4.5s/call, 300 cap/day)")

    remaining_quota = runner.pacer.get_remaining_quota(agent_model)
    used_quota = runner.pacer.get_used_quota(agent_model)
    print(f"Daily Quota:       {used_quota} used / {remaining_quota} remaining today")

    # Run the evaluation
    manifest = await runner.run(
        run_id=run_id,
        config=config,
        task_ids=task_ids,
        num_trials=num_trials,
        adapter=adapter,
        concurrency=concurrency,
    )

    print("\n==================================================")
    print("      RUN COMPLETE: ANALYZING & COMPARING        ")
    print("==================================================")

    # 1. Fetch results from SQLite for both runs
    with sqlite3.connect("results/agentgate.db") as conn:
        cursor = conn.cursor()

        # Candidate run results
        cursor.execute(
            """SELECT task_id, trial_id, success, reward, steps, done, error, trajectory_json, status 
               FROM trial_results WHERE run_id = ? ORDER BY task_id, trial_id""",
            (run_id,),
        )
        cand_rows = cursor.fetchall()

        # Baseline run results
        cursor.execute(
            """SELECT task_id, trial_id, success, reward, steps, done, error, trajectory_json, status 
               FROM trial_results WHERE run_id = ? ORDER BY task_id, trial_id""",
            (baseline_run_id,),
        )
        base_rows = cursor.fetchall()

    cand_outcomes: dict[int, list[int]] = {}
    cand_completed = 0
    cand_infra = 0
    cand_failed_trials = []

    for tid, trid, succ, rew, steps, done, err, traj_json, status in cand_rows:
        if status == "infra_error":
            cand_infra += 1
            continue
        cand_completed += 1
        cand_outcomes.setdefault(tid, []).append(int(succ))
        if succ == 0:
            # Determine end reason
            end_reason = "unknown"
            if steps >= config.max_steps and not done:
                end_reason = f"max_steps ({config.max_steps})"
            elif err:
                end_reason = f"error: {err[:60]}"
            else:
                try:
                    traj = json.loads(traj_json) if traj_json else []
                    if traj and "###STOP###" in str(traj[-1].get("content", "")):
                        end_reason = "simulator ###STOP###"
                    else:
                        end_reason = "agent final reply without completing goal"
                except Exception:
                    end_reason = "completed without reward"
            cand_failed_trials.append(
                {"task_id": tid, "trial_id": trid, "steps": steps, "reason": end_reason}
            )

    base_outcomes: dict[int, list[int]] = {}
    for tid, trid, succ, rew, steps, done, err, traj_json, status in base_rows:
        if status != "infra_error":
            base_outcomes.setdefault(tid, []).append(int(succ))

    # 1. Statistical pass^1 (trial-level AND task-level) and pass^2
    all_k = (len(cand_outcomes) == len(task_ids)) and all(
        len(rewards) >= num_trials for rewards in cand_outcomes.values()
    )

    print("\n--- 1. STATISTICAL PERFORMANCE METRICS (baseline_v1_steps30, max_steps=30) ---")
    if cand_outcomes:
        # Task-level metrics
        task_ids_sorted = sorted(cand_outcomes.keys())
        task_p1_list = [float(np.mean(cand_outcomes[tid])) for tid in task_ids_sorted]
        task_pass_1 = float(np.mean(task_p1_list))

        rng = np.random.RandomState(42)
        # Bootstrap CI for task-level pass^1 (resampling tasks)
        boot_task_p1 = [float(np.mean(rng.choice(task_p1_list, size=len(task_p1_list), replace=True))) for _ in range(10000)]
        ci_task_p1 = (float(np.percentile(boot_task_p1, 2.5)), float(np.percentile(boot_task_p1, 97.5)))

        # Trial-level metrics
        all_trial_rewards = [succ for tid in task_ids_sorted for succ in cand_outcomes[tid]]
        trial_pass_1 = float(np.mean(all_trial_rewards))
        boot_trial_p1 = [float(np.mean(rng.choice(all_trial_rewards, size=len(all_trial_rewards), replace=True))) for _ in range(10000)]
        ci_trial_p1 = (float(np.percentile(boot_trial_p1, 2.5)), float(np.percentile(boot_trial_p1, 97.5)))

        print(f"Tasks Evaluated:              {len(cand_outcomes)}/{len(task_ids)} tasks ({cand_completed} trials)")
        print(f"pass^1 (Trial-Level):         {trial_pass_1 * 100:.2f}%  (95% bootstrap CI: [{ci_trial_p1[0] * 100:.2f}%, {ci_trial_p1[1] * 100:.2f}%])")
        print(f"pass^1 (Task-Level):          {task_pass_1 * 100:.2f}%  (95% bootstrap CI: [{ci_task_p1[0] * 100:.2f}%, {ci_task_p1[1] * 100:.2f}%])")

        if all_k:
            p1_calc, pk_calc = compute_pass_metrics(cand_outcomes, k=num_trials)
            _, cik = bootstrap_ci_pass(cand_outcomes, k=num_trials, num_bootstrap=10000, seed=42)
            print(f"pass^2 (Task-Level Unbiased): {pk_calc * 100:.2f}%  (95% bootstrap CI: [{cik[0] * 100:.2f}%, {cik[1] * 100:.2f}%])")
        else:
            print("pass^2 (Task-Level Unbiased): [WITHHELD] Incomplete run — pass^2 requires all tasks to finish both trials.")

    # 2. Paired Comparison Against baseline_v1
    print("\n--- 2. PAIRED COMPARISON: baseline_v1 (max_steps=15) vs baseline_v1_steps30 (max_steps=30) ---")
    shared_tasks = sorted(set(base_outcomes.keys()) & set(cand_outcomes.keys()))
    print(f"Shared Tasks Analyzed: {len(shared_tasks)}")

    # Primary: Task-level paired bootstrap
    paired_res = paired_bootstrap_comparison(
        baseline_outcomes=base_outcomes,
        candidate_outcomes=cand_outcomes,
        num_bootstrap=10000,
        seed=42,
        alpha=0.05,
        tolerance=0.0,
    )

    print("\n[PRIMARY TEST: Task-Level Paired Bootstrap]")
    print(f"  Baseline pass^1 (Task-Level):  {paired_res.baseline_pass_1 * 100:.2f}%")
    print(f"  Candidate pass^1 (Task-Level): {paired_res.candidate_pass_1 * 100:.2f}%")
    print(f"  Delta pass^1 (Task-Level):     {paired_res.delta_pass_1 * 100:+.2f}% (95% bootstrap CI: [{paired_res.delta_ci[0] * 100:+.2f}%, {paired_res.delta_ci[1] * 100:+.2f}%])")
    print(f"  Task-Level Statistical Verdict: {paired_res.verdict}")
    print(f"  Details: {paired_res.details}")

    # Secondary: McNemar's test on 30 paired trials
    base_pair_dict = {(r[0], r[1]): r[2] for r in base_rows if r[8] != "infra_error"}
    cand_pair_dict = {(r[0], r[1]): r[2] for r in cand_rows if r[8] != "infra_error"}
    shared_pairs = sorted(set(base_pair_dict.keys()) & set(cand_pair_dict.keys()))

    n00 = n01 = n10 = n11 = 0
    gains_list = []
    losses_list = []

    for pair in shared_pairs:
        b_s = base_pair_dict[pair]
        c_s = cand_pair_dict[pair]
        if b_s == 0 and c_s == 0:
            n00 += 1
        elif b_s == 0 and c_s == 1:
            n01 += 1
            gains_list.append(pair)
        elif b_s == 1 and c_s == 0:
            n10 += 1
            losses_list.append(pair)
        else:
            n11 += 1

    discordant = n01 + n10
    if discordant == 0:
        p_val_trials = 1.0
    else:
        k_min = min(n01, n10)
        p_val_trials = float(2.0 * stats.binom.cdf(k_min, discordant, 0.5))
        p_val_trials = min(1.0, p_val_trials)

    print(f"\n[SECONDARY TEST: McNemar's Test on {len(shared_pairs)} Paired Trials]")
    print("  (Note: Trials are clustered by task (k=2); McNemar treats trials independently and serves as a secondary check.)")
    print(f"  Contingency Table:")
    print(f"    Both Failed (0,0):          {n00}")
    print(f"    Baseline 0 -> Candidate 1:  {n01} (gained successes)")
    print(f"    Baseline 1 -> Candidate 0:  {n10} (lost successes / regressions)")
    print(f"    Both Succeeded (1,1):       {n11}")
    print(f"  Discordant Pairs: {discordant} (n01={n01}, n10={n10})")
    print(f"  McNemar Exact Two-Sided p-value: {p_val_trials:.4f}")

    print(f"\n[GAINS LIST: Trials that Failed in baseline_v1 and Passed in baseline_v1_steps30]")
    print(f"  Total Gains Count: {len(gains_list)} (n01 = {n01})")
    for tid, trid in gains_list:
        print(f"  • Task {tid:2d}, Trial {trid}")
    assert len(gains_list) == n01, f"Mismatch: len(gains_list)={len(gains_list)} != n01={n01}"
    print("  ✅ Verification: len(gains_list) strictly matches n01.")

    # 3. Failed trials breakdown
    print("\n--- 3. FAILED TRIALS BREAKDOWN (baseline_v1_steps30) ---")
    print(f"Total Failures: {len(cand_failed_trials)}")
    for f in cand_failed_trials:
        print(f"  Task {f['task_id']:2d}, Trial {f['trial_id']}: {f['steps']} steps | Ended by: {f['reason']}")

    # 4. Calls, Retries, Tokens, Cost
    retries = runner.pacer.get_retry_count()
    used_today = runner.pacer.get_used_quota(agent_model)
    rem_today = runner.pacer.get_remaining_quota(agent_model)

    print("\n--- 4. USAGE & RESOURCE ACCOUNTING ---")
    print(f"Agent API Calls Used Today: {used_today} / 300 (remaining: {rem_today})")
    print(f"Transient Retries Absorbed: {retries}")
    print(f"Total Tokens:               {manifest.total_tokens:,}")
    print(f"Commercial 'Would-Have-Cost': ${manifest.total_would_have_cost:.4f}")

    # 5. Exact config differences
    print("\n--- 5. CONFIGURATION DIFFERENCES ---")
    with open("results/manifests/baseline_v1.json") as f:
        b_manifest = json.load(f)
    with open(f"results/manifests/{run_id}.json") as f:
        c_manifest = json.load(f)

    b_cfg = b_manifest.get("agent_config", {})
    c_cfg = c_manifest.get("agent_config", {})
    diffs = {}
    all_keys = set(b_cfg.keys()) | set(c_cfg.keys())
    for k in sorted(all_keys):
        if b_cfg.get(k) != c_cfg.get(k):
            diffs[k] = {"baseline_v1": b_cfg.get(k), "baseline_v1_steps30": c_cfg.get(k)}

    print(f"Differences in AgentConfig between baseline_v1 and {run_id}:")
    for k, v in diffs.items():
        print(f"  • {k}: baseline_v1={v['baseline_v1']} -> {run_id}={v['baseline_v1_steps30']}")
    if len(diffs) == 1 and "max_steps" in diffs:
        print("  ✅ Confirmed: The ONLY difference between the two runs is max_steps (15 -> 30).")


if __name__ == "__main__":
    asyncio.run(main())
