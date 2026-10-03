"""Phase 2 Baseline Evaluation: 15 stratified tasks x k=2 with concurrency=3 and kill/resume support."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

from agentgate.adapter.tau_bench import TauBenchRetailAdapter
from agentgate.agent.models import AgentConfig
from agentgate.runner.runner import EvaluationRunner


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    agent_model = "gemini/gemini-3.1-flash-lite"
    user_model = "gemini/gemma-4-26b-a4b-it"

    adapter = TauBenchRetailAdapter(task_split="test")
    runner = EvaluationRunner()

    # First 15 stratified tasks across all retail domain action types
    task_ids = adapter.get_stratified_subset_ids(count=15)
    num_trials = 2
    concurrency = 3
    run_id = "baseline_v1"

    config = AgentConfig(
        model=agent_model,
        user_model=user_model,
        max_steps=15,
        system_prompt_version="v1",
        tool_description_variant="standard",
    )

    print("==================================================")
    print("         PHASE 2: BASELINE EVALUATION             ")
    print("==================================================")
    print(f"Run ID:            {run_id}")
    print(f"Agent Model:       {agent_model}")
    print(f"User Simulator:    {user_model}")
    print(f"Tasks ({len(task_ids)}):        {task_ids}")
    print(f"Trials per Task:   {num_trials} (Total trials: {len(task_ids) * num_trials})")
    print(f"Concurrency:       {concurrency} workers (pacer lock enforced: 4.5s/call, 400 cap/day)")

    remaining_quota = runner.pacer.get_remaining_quota(agent_model)
    used_quota = runner.pacer.get_used_quota(agent_model)
    print(f"Daily Quota:       {used_quota} used / {remaining_quota} remaining today")

    manifest = await runner.run(
        run_id=run_id,
        config=config,
        task_ids=task_ids,
        num_trials=num_trials,
        adapter=adapter,
        concurrency=concurrency,
    )

    print("\n==================================================")
    print("             BASELINE RESULTS SUMMARY             ")
    print("==================================================")
    print(f"Status:            {manifest.status}")
    print(f"Completed Trials:  {manifest.completed_trials}/{manifest.total_trials}")
    print(f"Successful Trials: {manifest.successful_trials}")
    print(f"Infra-Error Trials:{manifest.infra_error_trials} (excluded from pass metrics)")
    print(f"Total Tokens:      {manifest.total_tokens:,}")
    print(f"Would-Have-Cost:   ${manifest.total_would_have_cost:.4f}")

    used_quota = runner.pacer.get_used_quota(agent_model)
    remaining_quota = runner.pacer.get_remaining_quota(agent_model)
    retries = runner.pacer.get_retry_count()
    print(f"Calls Used Today:  {used_quota} / 400 (remaining: {remaining_quota})")
    print(f"Transient Retries: {retries} (503 / 429 backoffs caught)")

    # Query DB to compute exact statistical pass^1 and pass^k with 95% bootstrap CIs
    import sqlite3
    from agentgate.stats.metrics import compute_pass_metrics, bootstrap_ci_pass

    with sqlite3.connect("results/agentgate.db") as conn:
        cursor = conn.execute(
            "SELECT task_id, success, status FROM trial_results WHERE run_id = ? ORDER BY task_id, trial_id",
            (run_id,),
        )
        rows = cursor.fetchall()

    task_outcomes: dict[int, list[int]] = {}
    completed_trials = 0
    infra_error_trials = 0
    for tid, succ, st in rows:
        if st == "infra_error":
            infra_error_trials += 1
            continue
        completed_trials += 1
        task_outcomes.setdefault(tid, []).append(int(succ))

    if task_outcomes:
        import numpy as np

        all_k_completed = (len(task_outcomes) == len(task_ids)) and all(
            len(rewards) >= num_trials for rewards in task_outcomes.values()
        )

        if all_k_completed:
            pass_1, pass_2 = compute_pass_metrics(task_outcomes, k=num_trials)
            ci_1, ci_2 = bootstrap_ci_pass(task_outcomes, k=num_trials, num_bootstrap=10000, seed=42)
            print("\n---------------- STATISTICAL METRICS ----------------")
            print(f"Tasks Evaluated:   {len(task_outcomes)}/{len(task_ids)} tasks ({completed_trials} trials)")
            print(f"pass^1:            {pass_1 * 100:.2f}%  (95% bootstrap CI: [{ci_1[0] * 100:.2f}%, {ci_1[1] * 100:.2f}%])")
            print(f"pass^2:            {pass_2 * 100:.2f}%  (95% bootstrap CI: [{ci_2[0] * 100:.2f}%, {ci_2[1] * 100:.2f}%])")
            print(f"Infra Errors:      {infra_error_trials} (excluded from pass metrics, re-run on resume)")
            print("-----------------------------------------------------")
        else:
            # Incomplete run: only report pass^1 across completed tasks, withhold pass^2
            task_ids_sorted = sorted(task_outcomes.keys())
            p1_list = [float(np.mean(task_outcomes[tid])) for tid in task_ids_sorted]
            pass_1 = float(np.mean(p1_list))
            rng = np.random.RandomState(42)
            boot_p1 = [float(np.mean(rng.choice(p1_list, size=len(p1_list), replace=True))) for _ in range(10000)]
            ci_1 = (float(np.percentile(boot_p1, 2.5)), float(np.percentile(boot_p1, 97.5)))

            print("\n---------------- STATISTICAL METRICS (INCOMPLETE RUN) ----------------")
            print(f"Tasks Evaluated:   {len(task_outcomes)}/{len(task_ids)} tasks ({completed_trials} completed trials)")
            print(f"pass^1:            {pass_1 * 100:.2f}%  (95% bootstrap CI: [{ci_1[0] * 100:.2f}%, {ci_1[1] * 100:.2f}%])")
            print(f"pass^2:            [WITHHELD] Incomplete run — pass^2 requires all {len(task_ids)} tasks to complete both trials.")
            print(f"Infra Errors:      {infra_error_trials} (excluded from pass metrics, queued to re-run on resume)")
            print("----------------------------------------------------------------------")

    if manifest.status == "paused_quota":
        resets_in = runner.pacer.get_next_reset_str()
        print(f"\n⚠️  RUNNER PAUSED AT DAILY CAP: Reached daily limit today.")
        print(f"    Runner automatically pauses at cap and will resume cleanly after Pacific midnight (in {resets_in}).")


if __name__ == "__main__":
    asyncio.run(main())
