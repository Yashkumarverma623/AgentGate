import sqlite3

conn = sqlite3.connect("results/agentgate.db")
c = conn.cursor()
c.execute("SELECT COUNT(*), SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), SUM(CASE WHEN status='infra_error' THEN 1 ELSE 0 END) FROM trial_results WHERE run_id='baseline_v1'")
total, succ, infra = c.fetchone()
print(f"Trials completed: {total}/30 (Successes: {succ or 0}, Infra errors: {infra or 0})")
