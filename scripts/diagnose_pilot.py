import json
import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tau_bench.envs.retail.tasks_test import TASKS_TEST

print("==================================================")
print("             PILOT DIAGNOSTIC REPORT              ")
print("==================================================")

conn = sqlite3.connect("results/agentgate.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT task_id, trial_id, reward, success, done, steps, tool_calls_count, error, step_records_json, trajectory_json
    FROM trial_results
    WHERE run_id = 'pilot_5x2'
    ORDER BY task_id, trial_id
""")
rows = cursor.fetchall()

for row in rows:
    task_id, trial_id, reward, success, done, steps, tool_calls_count, error, step_records_j, traj_j = row
    task = TASKS_TEST[task_id]
    gt_actions = [a.model_dump() for a in task.actions]
    step_records = json.loads(step_records_j) if step_records_j else []

    tools_called = []
    for s in step_records:
        for tc in s.get("tool_calls", []):
            tools_called.append({"name": tc.get("name"), "args": tc.get("arguments")})

    last_step = step_records[-1] if step_records else {}
    last_obs = (last_step.get("observations") or [""])[-1] if last_step else ""
    last_content = last_step.get("content", "")

    # End reason determination
    if error:
        if "503" in error or "UNAVAILABLE" in error or "high demand" in error:
            end_reason = "Error (HTTP 503: Service Unavailable / High Demand spike)"
        elif "429" in error or "quota" in error:
            end_reason = "Error (HTTP 429: Rate Limit / Quota Exceeded)"
        else:
            end_reason = f"Error ({error[:80]})"
    elif "###STOP###" in last_obs:
        end_reason = "Simulator ###STOP### (User ended conversation)"
    elif steps >= 15:
        end_reason = "Max steps reached (15/15)"
    else:
        end_reason = "Agent final reply without ###STOP###"

    gt_names = [a["name"] for a in gt_actions]
    called_names = [t["name"] for t in tools_called]
    
    # Check if any ground-truth state-changing action was called
    gt_called = []
    for gta in gt_actions:
        name = gta["name"]
        was_called = name in called_names
        gt_called.append((name, was_called))

    print(f"\n--- Task {task_id}, Trial {trial_id} ---")
    print(f"Status:             {'SUCCESS' if success else 'FAILED'}")
    print(f"End Reason:         {end_reason}")
    print(f"Agent Turns/Steps:  {steps}")
    print(f"Tool Calls Count:   {tool_calls_count}")
    print(f"Tools Called:       {called_names}")
    print(f"Ground Truth Req:   {gt_names}")
    for name, was_called in gt_called:
        print(f"  • {name}: {'CALL MADE' if was_called else 'NOT MADE'}")
    if error:
        print(f"Error Details:      {error.strip()[:160]}...")
