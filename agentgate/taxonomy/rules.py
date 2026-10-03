"""Deterministic rule evaluation for agent failure trajectories."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentgate.taxonomy.models import DeterministicCheckResult, FailureCategory


def check_infra_error(trial_data: Dict[str, Any]) -> Optional[DeterministicCheckResult]:
    """Detects infrastructure failures such as rate limits or network connection drops."""
    status = trial_data.get("status")
    error = trial_data.get("error") or ""
    if status == "infra_error":
        return DeterministicCheckResult(
            category=FailureCategory.INFRASTRUCTURE_ERROR,
            reason=f"Infrastructure error: {error[:200]}",
            details={"error": error},
        )
    err_lower = error.lower()
    if any(k in err_lower for k in ["429", "503", "quota", "resource_exhausted", "connection", "getaddrinfo"]):
        return DeterministicCheckResult(
            category=FailureCategory.INFRASTRUCTURE_ERROR,
            reason=f"Infrastructure error in error field: {error[:200]}",
            details={"error": error},
        )
    return None


def check_tool_repetition_loop(step_records: List[Dict[str, Any]]) -> Optional[DeterministicCheckResult]:
    """Detects if an agent repeated the exact same tool call and arguments consecutively >= 3 times."""
    consecutive_count = 1
    last_call: Optional[tuple[str, str]] = None
    first_loop_step = None

    for step_idx, step in enumerate(step_records, 1):
        tool_calls = step.get("tool_calls", [])
        for tc in tool_calls:
            name = tc.get("name") or ""
            args_str = json.dumps(tc.get("arguments", {}), sort_keys=True)
            current_call = (name, args_str)

            if current_call == last_call:
                consecutive_count += 1
                if consecutive_count >= 3:
                    return DeterministicCheckResult(
                        category=FailureCategory.TOOL_CALL_LOOP,
                        confidence=1.0,
                        reason=f"Agent invoked tool '{name}' with identical arguments {consecutive_count} consecutive times.",
                        step_index=first_loop_step or step_idx,
                        details={"tool_name": name, "consecutive_invocations": consecutive_count},
                    )
            else:
                consecutive_count = 1
                last_call = current_call
                first_loop_step = step_idx
    return None


def check_transfer_to_human(step_records: List[Dict[str, Any]]) -> Optional[DeterministicCheckResult]:
    """Detects if agent transferred the conversation to a human representative."""
    for step_idx, step in enumerate(step_records, 1):
        for tc in step.get("tool_calls", []):
            name = tc.get("name", "")
            if "transfer" in name.lower() or name == "transfer_to_human_agents":
                return DeterministicCheckResult(
                    category=FailureCategory.TRANSFER_TO_HUMAN,
                    confidence=1.0,
                    reason="Agent invoked transfer_to_human_agents.",
                    step_index=step_idx,
                    details={"tool_name": name},
                )
    return None


def check_tool_execution_error(step_records: List[Dict[str, Any]]) -> Optional[DeterministicCheckResult]:
    """Detects if an unrecovered tool schema error or runtime exception occurred."""
    for step_idx, step in enumerate(step_records, 1):
        for tc in step.get("tool_calls", []):
            if tc.get("is_error"):
                obs = tc.get("observation", "")
                if "malformed json" in obs.lower() or "schema" in obs.lower():
                    return DeterministicCheckResult(
                        category=FailureCategory.TOOL_EXECUTION_ERROR,
                        confidence=0.9,
                        reason=f"Malformed arguments or schema failure at step {step_idx}: {obs[:120]}",
                        step_index=step_idx,
                        details={"observation": obs},
                    )
    return None


def check_simulator_early_stop(trajectory: List[Dict[str, Any]]) -> Optional[DeterministicCheckResult]:
    """Detects if customer simulator stopped the dialog with ###STOP###."""
    if not trajectory:
        return None
    last_msg = trajectory[-1]
    content = str(last_msg.get("content", ""))
    if "###STOP###" in content and last_msg.get("role") == "user":
        return DeterministicCheckResult(
            category=FailureCategory.SIMULATOR_EARLY_STOP,
            confidence=1.0,
            reason="Customer simulator issued ###STOP### before task requirements were fulfilled.",
            details={"last_content": content},
        )
    return None


def check_max_steps(trial_data: Dict[str, Any], max_steps: int = 30) -> Optional[DeterministicCheckResult]:
    """Detects if trial terminated simply because the step cap was reached."""
    steps = trial_data.get("steps", 0)
    done = trial_data.get("done", False)
    if steps >= max_steps and not done:
        return DeterministicCheckResult(
            category=FailureCategory.MAX_STEPS_REACHED,
            confidence=1.0,
            reason=f"Trial reached maximum step ceiling ({steps}/{max_steps}) without finishing.",
            step_index=steps,
            details={"steps": steps, "max_steps": max_steps},
        )
    return None


def evaluate_deterministic_rules(
    trial_data: Dict[str, Any],
    max_steps: int = 30,
) -> Optional[DeterministicCheckResult]:
    """Evaluates deterministic failure rules in priority order."""
    # 1. Infra error
    res = check_infra_error(trial_data)
    if res:
        return res

    # 2. Tool repetition loop
    step_records = trial_data.get("step_records") or []
    res = check_tool_repetition_loop(step_records)
    if res:
        return res

    # 3. Transfer to human
    res = check_transfer_to_human(step_records)
    if res:
        return res

    # 4. Tool execution error
    res = check_tool_execution_error(step_records)
    if res:
        return res

    # 5. Simulator early stop
    trajectory = trial_data.get("trajectory") or []
    res = check_simulator_early_stop(trajectory)
    if res:
        return res

    # 6. Max steps reached
    res = check_max_steps(trial_data, max_steps=max_steps)
    if res:
        return res

    return None
