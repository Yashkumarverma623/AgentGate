import sqlite3

conn = sqlite3.connect("results/agentgate.db")
c = conn.cursor()

# Get columns of trial_results
c.execute("PRAGMA table_info(trial_results)")
columns = [row[1] for row in c.fetchall()]
print("Columns:", columns)

# Counts by status
c.execute("SELECT status, COUNT(*) FROM trial_results WHERE run_id='baseline_v1' GROUP BY status")
status_counts = c.fetchall()
print("\nTrial counts by status for baseline_v1:")
for s, count in status_counts:
    print(f"  {s}: {count}")

# Fetch all trials for baseline_v1 to inspect error fields
c.execute("SELECT task_id, trial_id, status, success, error, info_json FROM trial_results WHERE run_id='baseline_v1'")
rows = c.fetchall()

print(f"\nTotal rows found: {len(rows)}")
network_keywords = ["connection", "network", "timeout", "dns", "getaddrinfo", "econnreset", "socket", "ssl", "sslcontext"]

to_update = []
for tid, trid, status, succ, err_msg, info_json in rows:
    err_text = (err_msg or "") + " " + (info_json or "")
    err_lower = err_text.lower()
    matched = [kw for kw in network_keywords if kw in err_lower]
    if matched:
        print(f"Trial (task={tid}, trial={trid}) [current status={status}, success={succ}]: matched keywords {matched}")
        print(f"  error: {err_msg}")
        if status != "infra_error":
            to_update.append((tid, trid, err_msg))

if to_update:
    print(f"\nUpdating {len(to_update)} trials to 'infra_error':")
    for tid, trid, msg in to_update:
        c.execute("UPDATE trial_results SET status='infra_error' WHERE run_id='baseline_v1' AND task_id=? AND trial_id=?", (tid, trid))
        print(f"  Updated task_id={tid}, trial_id={trid}")
    conn.commit()
else:
    print("\nNo completed trials with connection/network/timeout/DNS errors needed updating.")

# Final counts by status
c.execute("SELECT status, COUNT(*) FROM trial_results WHERE run_id='baseline_v1' GROUP BY status")
print("\nFinal trial counts by status for baseline_v1:")
for s, count in c.fetchall():
    print(f"  {s}: {count}")

conn.close()
