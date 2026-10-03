import json
import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("results/agentgate.db")
cursor = conn.cursor()

def dump_trial(task_id, trial_id):
    cursor.execute("""
        SELECT step_records_json, error
        FROM trial_results
        WHERE run_id = 'pilot_5x2' AND task_id = ? AND trial_id = ?
    """, (task_id, trial_id))
    row = cursor.fetchone()
    if not row:
        print(f"Task {task_id}, Trial {trial_id} not found.")
        return
    step_records = json.loads(row[0]) if row[0] else []
    error = row[1]
    print(f"==================================================")
    print(f"    FULL TRANSCRIPT: Task {task_id}, Trial {trial_id}")
    print(f"==================================================")
    for s in step_records:
        idx = s.get("step_index")
        content = s.get("content", "")
        tool_calls = s.get("tool_calls", [])
        obs = s.get("observations", [])
        print(f"\n[Turn {idx}]")
        if content:
            print(f"🤖 Agent: {content.strip()}")
        for tc in tool_calls:
            tname = tc.get("name")
            targs = tc.get("arguments")
            tobs = tc.get("observation", "")
            print(f"⚙️ Tool Call: {tname}({json.dumps(targs)})")
            print(f"   Observation: {tobs[:300]}...")
        for o in obs:
            print(f"👤 Customer: {o.strip()}")
    if error:
        print(f"\n❌ [TERMINATION REASON / ERROR]:\n{error}")

dump_trial(1, 1)
print("\n" + "="*60 + "\n")
dump_trial(2, 1)
