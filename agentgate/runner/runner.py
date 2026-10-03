"""Resumable async runner with per-model rate limiting, quota enforcement, and SQLite persistence."""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from agentgate.adapter.base import BenchmarkAdapter
from agentgate.adapter.tau_bench import TauBenchRetailAdapter
from agentgate.agent.loop import AgentLoop
from agentgate.agent.models import AgentConfig, RunManifest, TrialResult
from agentgate.agent.prompt_manager import PromptManager
from agentgate.pricing import PricingRegistry
from agentgate.runner.pacer import QuotaExceededError, QuotaPacer
from agentgate.tracing.tracer import LocalTracer

logger = logging.getLogger("agentgate.runner")


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process with given PID is currently active."""
    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x00100000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_current_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_missing"


class EvaluationRunner:
    """Manages the execution of benchmark task trials with concurrency, resumability, and rate-limiting."""

    def __init__(
        self,
        db_path: Path | str = "results/agentgate.db",
        pacer: Optional[QuotaPacer] = None,
        pricing: Optional[PricingRegistry] = None,
        prompt_manager: Optional[PromptManager] = None,
        tracer: Optional[LocalTracer] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pacer = pacer or QuotaPacer()
        self.pricing = pricing or PricingRegistry()
        self.prompt_manager = prompt_manager or PromptManager()
        self.tracer = tracer or LocalTracer(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trial_results (
                    run_id TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    trial_id INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    success INTEGER NOT NULL,
                    done INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    tool_calls_count INTEGER NOT NULL,
                    total_prompt_tokens INTEGER NOT NULL,
                    total_completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    would_have_cost REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    trajectory_json TEXT,
                    step_records_json TEXT,
                    info_json TEXT,
                    error TEXT,
                    status TEXT DEFAULT 'completed',
                    finished_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, task_id, trial_id)
                )
                """
            )
            try:
                conn.execute(
                    "ALTER TABLE trial_results ADD COLUMN status TEXT DEFAULT 'completed'"
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_completed_pairs(self, run_id: str) -> Set[Tuple[int, int]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT task_id, trial_id FROM trial_results WHERE run_id = ? AND (status != 'infra_error' OR status IS NULL)",
                (run_id,),
            )
            return {(row[0], row[1]) for row in cursor.fetchall()}

    def save_trial_result(self, res: TrialResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trial_results (
                    run_id, task_id, trial_id, reward, success, done, steps,
                    tool_calls_count, total_prompt_tokens, total_completion_tokens,
                    total_tokens, would_have_cost, duration_seconds, trajectory_json,
                    step_records_json, info_json, error, status, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    res.run_id,
                    res.task_id,
                    res.trial_id,
                    res.reward,
                    1 if res.success else 0,
                    1 if res.done else 0,
                    res.steps,
                    res.tool_calls_count,
                    res.total_prompt_tokens,
                    res.total_completion_tokens,
                    res.total_tokens,
                    res.would_have_cost,
                    res.duration_seconds,
                    json.dumps(res.trajectory),
                    json.dumps([s.model_dump() for s in res.step_records]),
                    json.dumps(res.info),
                    res.error,
                    res.status,
                    res.finished_at,
                ),
            )
            conn.commit()

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest_dir = Path("results/manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / f"{manifest.run_id}.json"
        manifest_json = manifest.model_dump_json(indent=2)
        manifest_file.write_text(manifest_json, encoding="utf-8")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, manifest_json, created_at, status) VALUES (?, ?, ?, ?)",
                (manifest.run_id, manifest_json, manifest.created_at, manifest.status),
            )
            conn.commit()

    async def run(
        self,
        run_id: str,
        config: AgentConfig,
        task_ids: List[int],
        num_trials: int = 2,
        adapter: Optional[BenchmarkAdapter] = None,
        concurrency: int = 1,
    ) -> RunManifest:
        lock_file = Path("results/agentgate.lock")
        if lock_file.exists():
            try:
                lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                existing_pid = lock_data.get("pid")
                if existing_pid and is_pid_alive(existing_pid) and existing_pid != os.getpid():
                    raise RuntimeError(
                        f"Another EvaluationRunner process (PID {existing_pid}, run_id '{lock_data.get('run_id')}') "
                        f"is already active. Please terminate it before starting a new run."
                    )
            except (json.JSONDecodeError, OSError):
                pass

        lock_file.write_text(
            json.dumps({"pid": os.getpid(), "run_id": run_id, "start_time": time.time()}),
            encoding="utf-8",
        )

        try:
            return await self._run_internal(
                run_id=run_id,
                config=config,
                task_ids=task_ids,
                num_trials=num_trials,
                adapter=adapter,
                concurrency=concurrency,
            )
        finally:
            if lock_file.exists():
                try:
                    lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                    if lock_data.get("pid") == os.getpid():
                        lock_file.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _run_internal(
        self,
        run_id: str,
        config: AgentConfig,
        task_ids: List[int],
        num_trials: int = 2,
        adapter: Optional[BenchmarkAdapter] = None,
        concurrency: int = 1,
    ) -> RunManifest:
        if adapter is None:
            adapter = TauBenchRetailAdapter()

        _, prompt_hash = self.prompt_manager.load_prompt("retail", config.system_prompt_version)
        _, tool_hash = self.prompt_manager.load_tool_variant(config.tool_description_variant)

        manifest = RunManifest(
            run_id=run_id,
            git_sha=get_current_git_sha(),
            benchmark_name=adapter.name,
            benchmark_commit_sha="59a200c6d575d595120f1cb70fea53cef0632f6b",
            config_hash=config.compute_hash(),
            agent_config=config,
            prompt_version=config.system_prompt_version,
            prompt_hash=prompt_hash,
            tool_variant=config.tool_description_variant,
            tool_variant_hash=tool_hash,
            task_ids=task_ids,
            num_trials=num_trials,
            total_trials=len(task_ids) * num_trials,
            status="running",
        )
        self.save_manifest(manifest)

        completed_pairs = self.get_completed_pairs(run_id)
        logger.info(
            "Run %s: %d total trials scheduled, %d already completed.",
            run_id,
            manifest.total_trials,
            len(completed_pairs),
        )

        agent_loop = AgentLoop(
            config=config,
            prompt_manager=self.prompt_manager,
            pricing=self.pricing,
            pacer=self.pacer,
            tracer=self.tracer,
        )

        semaphore = asyncio.Semaphore(concurrency)
        quota_interrupted = False

        async def _run_single_trial(task_id: int, trial_id: int) -> Optional[TrialResult]:
            nonlocal quota_interrupted
            if quota_interrupted:
                return None

            if (task_id, trial_id) in completed_pairs:
                logger.info("Skipping completed trial: task=%d, trial=%d", task_id, trial_id)
                return None

            async with semaphore:
                if quota_interrupted:
                    return None
                try:
                    # Create isolated environment for this worker
                    env = adapter.create_env(
                        task_id=task_id,
                        user_model=config.user_model,
                        user_strategy=config.user_strategy,
                    )
                    res = await agent_loop.solve(
                        adapter=adapter,
                        env=env,
                        task_id=task_id,
                        trial_id=trial_id,
                        run_id=run_id,
                    )
                    self.save_trial_result(res)
                    logger.info(
                        "Finished task %d trial %d: reward=%.1f steps=%d tokens=%d cost=$%.4f",
                        task_id,
                        trial_id,
                        res.reward,
                        res.steps,
                        res.total_tokens,
                        res.would_have_cost,
                    )
                    return res
                except QuotaExceededError as qe:
                    logger.warning("Daily quota reached: %s", qe)
                    quota_interrupted = True
                    return None
                except Exception as e:
                    logger.error(
                        "Unhandled error in trial (task=%d, trial=%d): %s", task_id, trial_id, e
                    )
                    return None

        # Build list of pending jobs
        jobs = []
        for trial_id in range(num_trials):
            for task_id in task_ids:
                if (task_id, trial_id) not in completed_pairs:
                    jobs.append((task_id, trial_id))

        for task_id, trial_id in jobs:
            if quota_interrupted:
                break
            await _run_single_trial(task_id, trial_id)

        # Update final manifest
        all_completed = self.get_completed_pairs(run_id)
        manifest.completed_trials = len(all_completed)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN (status != 'infra_error' OR status IS NULL) AND success = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'infra_error' THEN 1 ELSE 0 END),
                    SUM(total_tokens),
                    SUM(would_have_cost)
                FROM trial_results WHERE run_id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            if row:
                manifest.successful_trials = int(row[1] or 0)
                manifest.infra_error_trials = int(row[2] or 0)
                manifest.total_tokens = int(row[3] or 0)
                manifest.total_would_have_cost = float(row[4] or 0.0)

        if quota_interrupted:
            manifest.status = "paused_quota"
        elif manifest.completed_trials >= manifest.total_trials:
            manifest.status = "completed"
        else:
            manifest.status = "incomplete"

        self.save_manifest(manifest)
        return manifest
