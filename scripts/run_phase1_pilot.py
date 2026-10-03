"""Phase 1 Pilot: Verify Gemini tool-calling, Gemma simulator, and execute 1 + (5 x 2) trials."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from agentgate.adapter.tau_bench import TauBenchRetailAdapter
from agentgate.agent.models import AgentConfig
from agentgate.runner.runner import EvaluationRunner
from agentgate.tracing.tracer import LocalTracer
from agentgate.stats.metrics import compute_pass_metrics, bootstrap_ci_pass


async def verify_models(agent_model: str, user_model: str) -> bool:
    import litellm

    print(f"\n[1/3] Verifying LiteLLM tool-calling on {agent_model}...")
    tools_probe = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]
    messages_probe = [
        {"role": "user", "content": "What is 42 + 58? Use the calculate tool to compute it."}
    ]
    try:
        res = await litellm.acompletion(
            model=agent_model,
            messages=messages_probe,
            tools=tools_probe,
            temperature=0.0,
        )
        msg = res.choices[0].message
        tc = getattr(msg, "tool_calls", None)
        print(f"[OK] Agent tool calling response: content='{msg.content}', tool_calls={tc}")
    except Exception as e:
        print(f"[ERROR] Error verifying agent model {agent_model}: {e}")
        return False

    print(f"\n[2/3] Verifying simulated user on {user_model}...")
    if "gemma" in user_model.lower():
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        clean_model = user_model.replace("gemini/", "").replace("models/", "")
        if clean_model in ("gemma-4-26b", "gemma-26b"):
            clean_model = "gemma-4-26b-a4b-it"
        elif clean_model in ("gemma-4-31b", "gemma-31b"):
            clean_model = "gemma-4-31b-it"

        for attempt in range(3):
            try:
                res_sim = client.models.generate_content(
                    model=clean_model,
                    contents="You are a customer who wants to cancel order #W12345. Answer concisely.\nAgent: Hi! How can I help you today?\nCustomer:",
                )
                sim_msg = res_sim.text.strip() if res_sim.text else ""
                print(f"[OK] User simulator ({clean_model}) response: '{sim_msg}'")
                return True
            except Exception as e:
                print(f"User simulator attempt {attempt + 1} notice on {clean_model}: {e}")
                if attempt == 2:
                    print(f"[ERROR] User simulator failed after 3 attempts on {clean_model}: {e}")
                    return False
                time.sleep(2.0)
    else:
        sim_probe_messages = [
            {"role": "system", "content": "You are a customer who wants to cancel order #W12345. Answer concisely."},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        for attempt in range(3):
            try:
                res_sim = await litellm.acompletion(
                    model=user_model,
                    messages=sim_probe_messages,
                    temperature=0.0,
                )
                sim_msg = res_sim.choices[0].message.content
                print(f"[OK] User simulator response: '{sim_msg}'")
                return True
            except Exception as e:
                print(f"User simulator attempt {attempt + 1} notice on {user_model}: {e}")
                if attempt == 2:
                    print(f"[ERROR] Notice on user simulator model {user_model}: {e}")
                    return False
                time.sleep(2.0)
    return True


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("Please export GEMINI_API_KEY or add it to .env before running the live pilot.")
        sys.exit(1)

    agent_model = "gemini/gemini-3.1-flash-lite"
    user_model = "gemini/gemma-4-26b-a4b-it"

    print("==================================================")
    print("           AGENTGATE PHASE 1 PILOT                ")
    print("==================================================")
    print(f"Agent Model:    {agent_model}")
    print(f"User Simulator: {user_model}")

    models_ok = await verify_models(agent_model, user_model)
    if not models_ok:
        print("Aborting pilot due to model verification failure.")
        sys.exit(1)

    adapter = TauBenchRetailAdapter(task_split="test")
    runner = EvaluationRunner()

    config = AgentConfig(
        model=agent_model,
        user_model=user_model,
        max_steps=15,
        system_prompt_version="v1",
        tool_description_variant="standard",
    )

    # 1. Clean up errored trials from previous run
    import sqlite3
    with sqlite3.connect(runner.db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM trial_results WHERE run_id = 'pilot_5x2' AND (status = 'infra_error' OR error IS NOT NULL)"
        )
        deleted = cursor.rowcount
        conn.commit()
    if deleted > 0:
        print(f"\n[Cleanup] Cleared {deleted} errored trials from 'pilot_5x2' for clean re-run.")

    # 2. Re-run 5 tasks x 2 trials (re-uses task 0 trials, executes tasks 1-4)
    pilot_task_ids = [0, 1, 2, 3, 4]
    print(f"\nRunning 5 tasks x 2 trials on pilot task IDs: {pilot_task_ids}...")
    manifest_5x2 = await runner.run(
        run_id="pilot_5x2",
        config=config,
        task_ids=pilot_task_ids,
        num_trials=2,
        adapter=adapter,
        concurrency=1,
    )

    print("\n==================================================")
    print("              PILOT METRICS REPORT                ")
    print("==================================================")
    print(f"Run ID: {manifest_5x2.run_id}")
    print(f"Status: {manifest_5x2.status}")
    print(f"Completed trials: {manifest_5x2.completed_trials}/{manifest_5x2.total_trials}")
    print(f"Infra-error trials (excluded): {manifest_5x2.infra_error_trials}")
    print(f"Successful trials: {manifest_5x2.successful_trials}")
    print(f"Total Tokens: {manifest_5x2.total_tokens:,}")
    print(f"Total 'Would-Have-Cost': ${manifest_5x2.total_would_have_cost:.4f}")

    # Calculate empirical calls per trial on COMPLETED trials only
    with sqlite3.connect(runner.db_path) as conn:
        cursor = conn.execute(
            "SELECT steps, tool_calls_count, total_prompt_tokens, total_completion_tokens, would_have_cost, success FROM trial_results WHERE run_id = ? AND (status != 'infra_error' OR status IS NULL)",
            (manifest_5x2.run_id,),
        )
        rows = cursor.fetchall()
        cursor_infra = conn.execute(
            "SELECT COUNT(*) FROM trial_results WHERE run_id = ? AND status = 'infra_error'",
            (manifest_5x2.run_id,),
        )
        infra_count = cursor_infra.fetchone()[0]

        if rows:
            avg_steps = sum(r[0] for r in rows) / len(rows)
            avg_tools = sum(r[1] for r in rows) / len(rows)
            avg_tokens = sum(r[2] + r[3] for r in rows) / len(rows)
            avg_cost = sum(r[4] for r in rows) / len(rows)
            succ_count = sum(r[5] for r in rows)
            pass_rate = (succ_count / len(rows)) * 100.0

            print(f"\nEmpirical Metrics (from {len(rows)} completed trials, {infra_count} infra errors excluded):")
            print(f"  • Pass Rate:                 {pass_rate:.1f}% ({succ_count}/{len(rows)})")
            print(f"  • Avg Agent LLM Calls/Trial: {avg_steps:.1f}")
            print(f"  • Avg Tool Calls/Trial:      {avg_tools:.1f}")
            print(f"  • Avg Total Tokens/Trial:    {avg_tokens:.0f}")
            print(f"  • Avg Would-Have-Cost/Trial: ${avg_cost:.4f}")

            # Re-estimate days for Option 2 (20 tasks x 2 = 40 trials)
            opt2_calls = 40 * avg_steps
            opt2_days = opt2_calls / 400.0
            print(f"\nOption 2 Re-estimation (20 tasks x 2 = 40 trials):")
            print(f"  • Est. Total Agent Calls:   {opt2_calls:.0f}")
            print(f"  • Est. Daily Quota Usage:   {min(400.0, opt2_calls):.0f} / 400")
            print(f"  • Est. Days Required:       {opt2_days:.2f} days")

            # Re-estimate days for 40 tasks x 2 = 80 trials
            opt1_calls = 80 * avg_steps
            opt1_days = opt1_calls / 400.0
            print(f"\n40-Task Re-estimation (40 tasks x 2 = 80 trials):")
            print(f"  • Est. Total Agent Calls:   {opt1_calls:.0f}")
            print(f"  • Est. Days Required:       {opt1_days:.2f} days (spread across Pacific midnight resets)")


if __name__ == "__main__":
    asyncio.run(main())
