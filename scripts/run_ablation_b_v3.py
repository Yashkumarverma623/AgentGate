"""Run Ablation B candidate: ablB_v3 (Prompt v3: degraded multi-factor).

Tasks: First 10 tasks in stratified order [16, 0, 10, 17, 3, 40, 22, 2, 31, 1], k=2.
Baseline comparison: Same 10 tasks in baseline_v1_steps30.
"""

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
    run_id = "ablB_v3"
    baseline_run_id = "baseline_v1_steps30"

    adapter = TauBenchRetailAdapter(task_split="test")
    runner = EvaluationRunner()

    # First 10 tasks from stratified subset
    full_stratified = adapter.get_stratified_subset_ids(count=15)
    task_ids = full_stratified[:10]
    num_trials = 2
    concurrency = 3

    config = AgentConfig(
        model=agent_model,
        user_model=user_model,
        max_steps=30,
        system_prompt_version="v3",
        tool_description_variant="standard",
        temperature=0.0,
    )

    print("==================================================")
    print("      RUN: ablB_v3 (Prompt v3: degraded)          ")
    print("==================================================")
    print(f"Run ID:            {run_id}")
    print(f"Agent Model:       {agent_model}")
    print(f"User Simulator:    {user_model}")
    print(f"System Prompt:     v3 (degraded multi-factor)")
    print(f"Tasks ({len(task_ids)}):        {task_ids}")
    print(f"Trials per Task:   {num_trials} (Total trials: {len(task_ids) * num_trials})")
    print(f"Max Steps Cap:     {config.max_steps}")
    print(f"Concurrency:       {concurrency} workers")

    remaining_quota = runner.pacer.get_remaining_quota(agent_model)
    used_quota = runner.pacer.get_used_quota(agent_model)
    print(f"Daily Quota:       {used_quota} used / {remaining_quota} remaining today")

    # Run the evaluation
    start_time = time.time()
    manifest = await runner.run(
        run_id=run_id,
        config=config,
        task_ids=task_ids,
        num_trials=num_trials,
        adapter=adapter,
        concurrency=concurrency,
    )
    duration = time.time() - start_time

    print(f"\nCompleted in {duration:.1f}s.")


if __name__ == "__main__":
    asyncio.run(main())
