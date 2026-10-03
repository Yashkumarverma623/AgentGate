"""Unit tests for Phase 3 Failure Taxonomy using mock LLM and deterministic rule evaluation."""

import json
from pathlib import Path
import pytest

from agentgate.taxonomy.exporter import export_to_csv, export_to_json, summarize_taxonomy
from agentgate.taxonomy.judge import JUDGE_JSON_SCHEMA, JUDGE_SYSTEM_PROMPT, TaxonomyJudge
from agentgate.taxonomy.models import FailureCategory, FailureRecord
from agentgate.taxonomy.rules import (
    check_infra_error,
    check_max_steps,
    check_simulator_early_stop,
    check_tool_execution_error,
    check_tool_repetition_loop,
    check_transfer_to_human,
    evaluate_deterministic_rules,
)


def test_check_max_steps():
    trial_data = {"steps": 30, "done": False}
    res = check_max_steps(trial_data, max_steps=30)
    assert res is not None
    assert res.category == FailureCategory.MAX_STEPS_REACHED

    # Completed trial should not trigger max_steps
    res_done = check_max_steps({"steps": 30, "done": True}, max_steps=30)
    assert res_done is None


def test_check_simulator_early_stop():
    traj = [
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "Never mind, I need to go. ###STOP###"},
    ]
    res = check_simulator_early_stop(traj)
    assert res is not None
    assert res.category == FailureCategory.SIMULATOR_EARLY_STOP


def test_check_tool_repetition_loop():
    # 3 consecutive identical tool calls
    step_records = [
        {"tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "123"}}]},
        {"tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "123"}}]},
        {"tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "123"}}]},
    ]
    res = check_tool_repetition_loop(step_records)
    assert res is not None
    assert res.category == FailureCategory.TOOL_CALL_LOOP
    assert res.details["consecutive_invocations"] == 3


def test_check_transfer_to_human():
    step_records = [
        {"tool_calls": [{"name": "find_user_id_by_email", "arguments": {"email": "test@example.com"}}]},
        {"tool_calls": [{"name": "transfer_to_human_agents", "arguments": {"summary": "Cannot help"}}]},
    ]
    res = check_transfer_to_human(step_records)
    assert res is not None
    assert res.category == FailureCategory.TRANSFER_TO_HUMAN


def test_check_tool_execution_error():
    step_records = [
        {"tool_calls": [{"name": "modify_order", "arguments": {}, "is_error": True, "observation": "Error: Malformed JSON arguments schema failure"}]}
    ]
    res = check_tool_execution_error(step_records)
    assert res is not None
    assert res.category == FailureCategory.TOOL_EXECUTION_ERROR


def test_check_infra_error():
    trial_data = {"status": "infra_error", "error": "litellm.RateLimitError: 429 quota exhausted"}
    res = check_infra_error(trial_data)
    assert res is not None
    assert res.category == FailureCategory.INFRASTRUCTURE_ERROR


def test_evaluate_deterministic_priority():
    # If trial has tool repetition loop and also hit max_steps, loop takes priority
    step_records = [
        {"tool_calls": [{"name": "lookup", "arguments": {"id": 1}}]},
        {"tool_calls": [{"name": "lookup", "arguments": {"id": 1}}]},
        {"tool_calls": [{"name": "lookup", "arguments": {"id": 1}}]},
    ]
    trial_data = {
        "steps": 30,
        "done": False,
        "step_records": step_records,
    }
    res = evaluate_deterministic_rules(trial_data, max_steps=30)
    assert res is not None
    assert res.category == FailureCategory.TOOL_CALL_LOOP


@pytest.mark.asyncio
async def test_taxonomy_judge_with_fake_llm():
    fake_response = json.dumps({
        "primary_category": "POLICY_VIOLATION_CONFIRMATION",
        "secondary_category": "INCORRECT_TOOL_SELECTION",
        "confidence": 0.95,
        "root_cause": "The agent cancelled order #982 without asking the user for confirmation.",
        "culprit_turn": 6,
        "recommended_fix": "Add explicit confirmation gate before cancel_order tool call.",
        "evidence_snippet": "cancel_order(order_id='982')",
    })

    async def mock_caller(messages):
        return fake_response

    judge = TaxonomyJudge(llm_caller=mock_caller)
    trial_data = {
        "task_id": 4,
        "steps": 7,
        "trajectory": [
            {"role": "user", "content": "Cancel my order please"},
            {"role": "assistant", "content": "Sure", "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "982"}}]},
        ],
    }
    res = await judge.classify(trial_data)
    assert res.primary_category == FailureCategory.POLICY_VIOLATION_CONFIRMATION
    assert res.confidence == 0.95
    assert res.culprit_turn == 6


@pytest.mark.asyncio
async def test_taxonomy_judge_malformed_fallback():
    async def mock_caller(messages):
        return "Not JSON at all!"

    judge = TaxonomyJudge(llm_caller=mock_caller)
    res = await judge.classify({"task_id": 1})
    assert res.primary_category == FailureCategory.UNKNOWN_FAILURE
    assert res.confidence == 0.5


def test_export_and_summarize(tmp_path: Path):
    records = [
        FailureRecord(
            run_id="test_run",
            task_id=2,
            trial_id=0,
            primary_category=FailureCategory.MAX_STEPS_REACHED,
            source="deterministic",
            confidence=1.0,
            root_cause="Reached 30 steps",
            steps_used=30,
            total_tokens=50000,
            would_have_cost=0.005,
        ),
        FailureRecord(
            run_id="test_run",
            task_id=4,
            trial_id=0,
            primary_category=FailureCategory.POLICY_VIOLATION_CONFIRMATION,
            source="llm_judge",
            confidence=0.9,
            root_cause="No confirmation",
            steps_used=12,
            total_tokens=25000,
            would_have_cost=0.0025,
        ),
    ]

    json_path = tmp_path / "failures.json"
    csv_path = tmp_path / "failures.csv"

    export_to_json(records, json_path)
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded) == 2

    export_to_csv(records, csv_path)
    assert csv_path.exists()

    summary = summarize_taxonomy(records)
    assert summary.total_failed_trials == 2
    assert summary.category_counts["MAX_STEPS_REACHED"] == 1
    assert summary.category_counts["POLICY_VIOLATION_CONFIRMATION"] == 1
    assert summary.cost_by_category["MAX_STEPS_REACHED"] == 0.005
