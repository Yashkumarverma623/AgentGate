"""Forced kill-and-resume test: simulates SIGINT/termination and verifies clean resumption."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_forced_kill_and_resume(tmp_path: Path) -> None:
    db_file = tmp_path / "test_kill_resume.db"
    quota_file = tmp_path / "quota.json"
    script_file = tmp_path / "run_killable.py"

    # Write a self-contained runner script using MockAdapter
    script_code = f"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from agentgate.adapter.base import BenchmarkAdapter, BenchmarkTask, GradeResult
from agentgate.agent.models import AgentConfig
from agentgate.runner.runner import EvaluationRunner
from agentgate.runner.pacer import QuotaPacer, ModelQuotaConfig

class SlowMockAdapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "slow-mock"

    def load_tasks(self, split="test", task_ids=None):
        return [BenchmarkTask(task_id=i, user_id=f"u{{i}}", instruction=f"Task {{i}}") for i in [1, 2, 3]]

    def create_env(self, task_id, user_model, user_strategy="llm"):
        return {{"task_id": task_id}}

    def reset(self, env, task_id):
        return f"Start task {{task_id}}", {{}}

    def step(self, env, action_name, kwargs):
        # Add small delay so SIGINT can hit while running
        import time
        time.sleep(0.4)
        return "Done ###STOP###", 1.0, True, {{}}

    def get_tools_info(self, env):
        return []

    def get_domain_policy(self, env):
        return "Policy"

    def grade(self, env):
        return GradeResult(reward=1.0, success=True, info={{}})

async def main():
    pacer = QuotaPacer(state_path=Path(r"{quota_file}"))
    pacer.default_config = ModelQuotaConfig(model="default", min_request_interval_seconds=0.01, daily_request_cap=1000)
    runner = EvaluationRunner(db_path=Path(r"{db_file}"), pacer=pacer)
    adapter = SlowMockAdapter()
    config = AgentConfig(model="test-model", max_steps=2)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="Done", tool_calls=None, model_dump=lambda: {{"role": "assistant", "content": "Done"}}))
    ]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("litellm.acompletion", return_value=mock_resp):
        await runner.run(
            run_id="kill_resume_test",
            config=config,
            task_ids=[1, 2, 3],
            num_trials=1,
            adapter=adapter,
            concurrency=1,
        )

if __name__ == "__main__":
    asyncio.run(main())
"""
    script_file.write_text(script_code, encoding="utf-8")

    # Step 1: Start runner process
    proc = subprocess.Popen(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll until at least 1 trial is saved in SQLite, then KILL the process
    killed = False
    start_wait = time.time()
    while time.time() - start_wait < 15:
        if db_file.exists():
            try:
                with sqlite3.connect(db_file) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM trial_results WHERE run_id = 'kill_resume_test'")
                    count = cursor.fetchone()[0]
                    if count >= 1:
                        # Kill the running process forcefully
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        killed = True
                        break
            except Exception:
                pass
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"Process exited prematurely with code {proc.returncode}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        time.sleep(0.1)

    assert killed, "Failed to kill process after 1st trial was written"

    # Verify that exactly 1 or 2 trials exist in DB
    with sqlite3.connect(db_file) as conn:
        cursor = conn.execute("SELECT task_id FROM trial_results WHERE run_id = 'kill_resume_test'")
        completed_before_resume = [row[0] for row in cursor.fetchall()]
    assert len(completed_before_resume) in (1, 2), f"Expected 1 or 2 completed before kill, got {completed_before_resume}"

    # Step 2: Restart the runner on the same DB and run_id
    resume_proc = subprocess.run(
        [sys.executable, str(script_file)],
        capture_output=True,
        text=True,
        check=True,
    )

    # Step 3: Assert all 3 tasks are now completed
    with sqlite3.connect(db_file) as conn:
        cursor = conn.execute("SELECT task_id, success FROM trial_results WHERE run_id = 'kill_resume_test' ORDER BY task_id")
        final_rows = cursor.fetchall()

    assert len(final_rows) == 3, f"Expected 3 completed tasks after resume, got {len(final_rows)}"
    assert [r[0] for r in final_rows] == [1, 2, 3]
    print(f"✅ Forced kill-and-resume verified: completed {completed_before_resume} before kill, resumed to all {final_rows}")
