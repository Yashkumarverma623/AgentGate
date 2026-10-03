"""Unit tests for the CI regression gate (agentgate gate).

Validates paired bootstrap comparisons, McNemar tests, tolerance threshold rules,
and CLI exit codes using synthetic / fake trial data (100% offline, 0 API calls).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from agentgate.cli.main import app

runner = CliRunner()


def init_mock_db(db_path: Path, runs_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Creates a mock SQLite database populated with synthetic trial results."""
    with sqlite3.connect(db_path) as conn:
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                span_type TEXT NOT NULL
            )
            """
        )
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

        for run_id, trials in runs_data.items():
            for t in trials:
                conn.execute(
                    """
                    INSERT INTO trial_results (
                        run_id, task_id, trial_id, reward, success, done, steps,
                        tool_calls_count, total_prompt_tokens, total_completion_tokens,
                        total_tokens, would_have_cost, duration_seconds,
                        trajectory_json, step_records_json, info_json, error,
                        status, finished_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, 2, 1000, 100, 1100, 0.001, 2.0, ?, '[]', '{}', ?, ?, '2026-10-01T12:00:00Z')
                    """,
                    (
                        run_id,
                        t["task_id"],
                        t["trial_id"],
                        float(t["success"]),
                        int(t["success"]),
                        t.get("steps", 10),
                        json.dumps(t.get("trajectory", [{"content": "Done"}])),
                        t.get("error"),
                        t.get("status", "completed"),
                    ),
                )
                # Add mock span
                conn.execute(
                    """
                    INSERT INTO spans (span_id, run_id, task_id, trial_id, span_type)
                    VALUES (?, ?, ?, ?, 'llm_call')
                    """,
                    (f"{run_id}_{t['task_id']}_{t['trial_id']}_span", run_id, t["task_id"], t["trial_id"]),
                )
        conn.commit()


def test_gate_identical_runs_passes(tmp_path: Path) -> None:
    """Candidate identical to baseline must pass the gate with exit code 0."""
    db_file = tmp_path / "test_agentgate.db"
    trials = [
        {"task_id": tid, "trial_id": tr, "success": 1, "steps": 8}
        for tid in range(10)
        for tr in [0, 1]
    ]
    init_mock_db(db_file, {"baseline": trials, "candidate": trials})

    result = runner.invoke(
        app,
        ["gate", "--baseline", "baseline", "--candidate", "candidate", "--db-path", str(db_file)],
    )

    assert result.exit_code == 0
    assert "GATE CRITERIA MET" in result.output
    assert "+0.00%" in result.output


def test_gate_significant_regression_fails(tmp_path: Path) -> None:
    """Candidate with massive drop across 20 tasks has CI entirely below zero and must fail with exit code 1."""
    db_file = tmp_path / "test_agentgate.db"
    base_trials = [
        {"task_id": tid, "trial_id": tr, "success": 1, "steps": 5}
        for tid in range(20)
        for tr in [0, 1]
    ]
    # Candidate fails almost all tasks (only 2 pass)
    cand_trials = [
        {"task_id": tid, "trial_id": tr, "success": 1 if tid < 2 else 0, "steps": 30}
        for tid in range(20)
        for tr in [0, 1]
    ]
    init_mock_db(db_file, {"baseline": base_trials, "candidate": cand_trials})

    result = runner.invoke(
        app,
        ["gate", "--baseline", "baseline", "--candidate", "candidate", "--db-path", str(db_file)],
    )

    assert result.exit_code == 1
    assert "REGRESSION DETECTED" in result.output


def test_gate_tolerance_exceeded_rule(tmp_path: Path) -> None:
    """When drop exceeds tolerance (e.g. -15% > 10% tolerance), gate must fail even if CI touches zero."""
    db_file = tmp_path / "test_agentgate.db"
    # 10 tasks, k=2: baseline 100% (20/20)
    base_trials = [
        {"task_id": tid, "trial_id": tr, "success": 1, "steps": 10}
        for tid in range(10)
        for tr in [0, 1]
    ]
    # Candidate: 3 trials fail across distinct tasks => 17/20 pass (85.0%), delta = -15.0%
    cand_trials = []
    for tid in range(10):
        for tr in [0, 1]:
            # Task 0 tr 1, Task 1 tr 1, Task 2 tr 1 fail
            succ = 0 if (tid in [0, 1, 2] and tr == 1) else 1
            cand_trials.append({"task_id": tid, "trial_id": tr, "success": succ, "steps": 15})

    init_mock_db(db_file, {"baseline": base_trials, "candidate": cand_trials})

    # Default tolerance is 0.10 (10 points)
    result = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
            "--tolerance",
            "0.10",
            "--db-path",
            str(db_file),
        ],
    )

    assert result.exit_code == 1
    assert "TOLERANCE EXCEEDED" in result.output
    assert "-15.00%" in result.output


def test_gate_drop_within_tolerance_passes(tmp_path: Path) -> None:
    """Candidate with a small drop (e.g. -5.0%) within tolerance (10%) and CI touching zero must pass."""
    db_file = tmp_path / "test_agentgate.db"
    # 10 tasks, k=2: baseline 100%
    base_trials = [
        {"task_id": tid, "trial_id": tr, "success": 1, "steps": 10}
        for tid in range(10)
        for tr in [0, 1]
    ]
    # Candidate: only 1 trial fails (task 0, tr 1) => pass^1 is 95.0%, delta = -5.0%
    cand_trials = []
    for tid in range(10):
        for tr in [0, 1]:
            succ = 0 if (tid == 0 and tr == 1) else 1
            cand_trials.append({"task_id": tid, "trial_id": tr, "success": succ, "steps": 12})

    init_mock_db(db_file, {"baseline": base_trials, "candidate": cand_trials})

    result = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
            "--tolerance",
            "0.10",
            "--db-path",
            str(db_file),
        ],
    )

    assert result.exit_code == 0
    assert "GATE CRITERIA MET" in result.output
    assert "-5.00%" in result.output


def test_gate_mcnemar_and_failed_trials_audit(tmp_path: Path) -> None:
    """Ensures McNemar contingency counts and candidate failed trial details are printed."""
    db_file = tmp_path / "test_agentgate.db"
    base_trials = [
        {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10},
        {"task_id": 1, "trial_id": 1, "success": 1, "steps": 12},
    ]
    cand_trials = [
        {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10},
        {
            "task_id": 1,
            "trial_id": 1,
            "success": 0,
            "steps": 30,
            "trajectory": [{"content": "user"}, {"content": "###STOP###"}],
        },
    ]
    init_mock_db(db_file, {"base": base_trials, "cand": cand_trials})

    result = runner.invoke(
        app,
        ["gate", "--baseline", "base", "--candidate", "cand", "--tolerance", "0.60", "--db-path", str(db_file)],
    )

    # McNemar table check
    assert "McNemar Paired Trial Contingency" in result.output
    assert "Candidate Failed Trials Audit" in result.output
    assert "customer_simulator_stopped" in result.output
    assert "###STOP###" in result.output


def test_gate_no_overlapping_tasks_fails(tmp_path: Path) -> None:
    """When baseline and candidate share no task IDs, gate should exit non-zero with error."""
    db_file = tmp_path / "test_agentgate.db"
    base_trials = [{"task_id": 1, "trial_id": 0, "success": 1}]
    cand_trials = [{"task_id": 2, "trial_id": 0, "success": 1}]
    init_mock_db(db_file, {"base": base_trials, "cand": cand_trials})

    result = runner.invoke(
        app,
        ["gate", "--baseline", "base", "--candidate", "cand", "--db-path", str(db_file)],
    )

    assert result.exit_code == 1
    assert "No common tasks found" in result.output


def test_gate_end_reason_reads_manifest_max_steps(tmp_path: Path) -> None:
    """Candidate with max_steps=15 in its manifest must label 15-step failures as max_steps_reached (15)."""
    db_file = tmp_path / "test_agentgate.db"
    base_trials = [
        {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10},
        {"task_id": 1, "trial_id": 1, "success": 1, "steps": 12},
    ]
    cand_trials = [
        {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10},
        {"task_id": 1, "trial_id": 1, "success": 0, "steps": 15, "trajectory": [{"content": "user"}, {"content": "thinking"}]},
    ]
    init_mock_db(db_file, {"base": base_trials, "cand_15": cand_trials})

    # Insert candidate manifest with max_steps=15
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, manifest_json, created_at, status)
            VALUES (?, ?, '2026-10-01T12:00:00Z', 'completed')
            """,
            (
                "cand_15",
                json.dumps({
                    "run_id": "cand_15",
                    "agent_config": {"model": "test-model", "max_steps": 15},
                }),
            ),
        )
        conn.commit()

    result = runner.invoke(
        app,
        ["gate", "--baseline", "base", "--candidate", "cand_15", "--tolerance", "0.60", "--db-path", str(db_file)],
    )

    assert "max_steps_reached (15)" in result.output


def test_gate_baseline_json_without_db(tmp_path: Path) -> None:
    """Baseline loaded from committed JSON file works even when baseline is absent from DB."""
    db_file = tmp_path / "test_agentgate.db"
    json_file = tmp_path / "baseline_committed.json"

    # Save baseline to JSON
    baseline_json_data = {
        "run_id": "baseline_committed",
        "agent_config": {"model": "test-model", "max_steps": 30},
        "trials": [
            {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10, "would_have_cost": 0.001, "total_tokens": 1000},
            {"task_id": 1, "trial_id": 1, "success": 1, "steps": 12, "would_have_cost": 0.001, "total_tokens": 1200},
        ],
    }
    json_file.write_text(json.dumps(baseline_json_data), encoding="utf-8")

    # DB only has candidate, baseline is completely absent from DB
    cand_trials = [
        {"task_id": 1, "trial_id": 0, "success": 1, "steps": 10},
        {"task_id": 1, "trial_id": 1, "success": 1, "steps": 12},
    ]
    init_mock_db(db_file, {"candidate_run": cand_trials})

    result = runner.invoke(
        app,
        ["gate", "--baseline", str(json_file), "--candidate", "candidate_run", "--db-path", str(db_file)],
    )

    assert result.exit_code == 0
    assert "GATE CRITERIA MET" in result.output
    assert "+0.00%" in result.output


def test_gate_tolerance_exact_boundary_floating_point(tmp_path: Path) -> None:
    """When drop is exactly equal to tolerance (e.g. -0.10 drop with 0.10 tolerance),
    it must NOT fail due to floating-point representation (e.g. -0.10000000000000002 < -0.10)."""
    db_file = tmp_path / "test_agentgate.db"
    # 10 tasks, k=2: baseline 100% (20/20)
    base_trials = [
        {"task_id": tid, "trial_id": tr, "success": 1, "steps": 10}
        for tid in range(10)
        for tr in [0, 1]
    ]
    # Candidate: exactly 2 trials fail (e.g. task 0 tr 1 and task 1 tr 1) => 18/20 = 90.0%, delta = -10.0%
    cand_trials = []
    for tid in range(10):
        for tr in [0, 1]:
            succ = 0 if (tid in [0, 1] and tr == 1) else 1
            cand_trials.append({"task_id": tid, "trial_id": tr, "success": succ, "steps": 12})

    init_mock_db(db_file, {"baseline": base_trials, "candidate": cand_trials})

    # Exact tolerance of 0.10
    result = runner.invoke(
        app,
        [
            "gate",
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
            "--tolerance",
            "0.10",
            "--db-path",
            str(db_file),
        ],
    )

    # Must pass because -0.10 is NOT strictly less than -0.10
    assert result.exit_code == 0
    assert "GATE CRITERIA MET" in result.output
    assert "-10.00%" in result.output


