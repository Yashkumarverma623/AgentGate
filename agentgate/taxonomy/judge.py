"""LLM-as-a-judge classification engine for ambiguous failure modes."""

from __future__ import annotations

import json
from typing import Any, Callable, Coroutine, Dict, List, Optional
from agentgate.taxonomy.models import FailureCategory, JudgeClassification

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator and taxonomist analyzing failed execution traces of an autonomous retail customer service agent.
Your objective is to diagnose the root cause of the agent's failure from the conversation trajectory, tool calls, and ground truth intent.

Taxonomy Categories:
1. POLICY_VIOLATION_AUTH: Agent failed to authenticate user identity at the start using `find_user_id_by_email` or `find_user_id_by_name_zip`, even if user provided their user_id in plain text.
2. POLICY_VIOLATION_CONFIRMATION: Agent executed a database-modifying action (cancel, modify, return, exchange) WITHOUT obtaining explicit user confirmation (e.g. "yes", "confirm") beforehand.
3. GROUNDING_HALLUCINATION: Agent claimed an action succeeded (e.g., "I have cancelled your order") when no such tool was called, or made up non-existent items/orders.
4. INCORRECT_TOOL_SELECTION: Agent selected the wrong tool for the user's objective (e.g., modify_pending_order instead of return_delivered_order).
5. REASONING_LOOP: Agent cycled repeatedly between the same questions or thoughts without making forward progress.
6. INCOMPLETE_GOAL: Agent terminated the conversation claiming task was complete, but omitted one or more required actions.
7. UNKNOWN_FAILURE: Failure does not fit other categories.

Respond strictly in JSON matching the specified schema.
"""

JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_category": {
            "type": "string",
            "enum": [
                "POLICY_VIOLATION_AUTH",
                "POLICY_VIOLATION_CONFIRMATION",
                "GROUNDING_HALLUCINATION",
                "INCORRECT_TOOL_SELECTION",
                "REASONING_LOOP",
                "INCOMPLETE_GOAL",
                "UNKNOWN_FAILURE",
            ],
            "description": "The primary root-cause category of the agent failure.",
        },
        "secondary_category": {
            "type": ["string", "null"],
            "enum": [
                "POLICY_VIOLATION_AUTH",
                "POLICY_VIOLATION_CONFIRMATION",
                "GROUNDING_HALLUCINATION",
                "INCORRECT_TOOL_SELECTION",
                "REASONING_LOOP",
                "INCOMPLETE_GOAL",
                "UNKNOWN_FAILURE",
                None,
            ],
            "description": "An optional secondary contributing factor.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score in this classification.",
        },
        "root_cause": {
            "type": "string",
            "description": "Clear explanation of why the agent failed.",
        },
        "culprit_turn": {
            "type": ["integer", "null"],
            "description": "The turn index (1-indexed) where the primary error occurred.",
        },
        "recommended_fix": {
            "type": "string",
            "description": "Specific recommendation to prevent this failure (prompt, tool definition, or model).",
        },
        "evidence_snippet": {
            "type": ["string", "null"],
            "description": "A brief quote from the transcript demonstrating the error.",
        },
    },
    "required": ["primary_category", "confidence", "root_cause", "recommended_fix"],
}


class TaxonomyJudge:
    """Classifies agent failure trajectories using LLM-as-a-judge with deterministic fallbacks."""

    def __init__(
        self,
        judge_model: str = "gemini/gemini-2.5-flash",
        llm_caller: Optional[Callable[[List[Dict[str, Any]]], Coroutine[Any, Any, str]]] = None,
    ) -> None:
        self.judge_model = judge_model
        self.llm_caller = llm_caller

    def format_judge_prompt(self, trial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        trajectory = trial_data.get("trajectory") or []
        task_id = trial_data.get("task_id")
        steps = trial_data.get("steps")
        info = trial_data.get("info") or {}

        # Format transcript
        formatted_turns = []
        for idx, msg in enumerate(trajectory, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            turn_str = f"[{idx}] Role: {role}\nContent: {content}"
            if tool_calls:
                turn_str += f"\nTool Calls: {json.dumps(tool_calls)}"
            formatted_turns.append(turn_str)

        transcript_text = "\n---\n".join(formatted_turns)
        user_prompt = f"""Task ID: {task_id}
Total Steps: {steps}
Ground Truth Evaluation Info: {json.dumps(info)}

Conversation Transcript:
{transcript_text}

Analyze the trajectory and return the failure diagnosis JSON."""

        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    async def classify(self, trial_data: Dict[str, Any]) -> JudgeClassification:
        """Classifies a failed trial trajectory using the provided or mock LLM caller."""
        messages = self.format_judge_prompt(trial_data)

        if not self.llm_caller:
            raise RuntimeError(
                "No LLM caller provided. Real LLM judge calls are restricted until Phase 3 execution."
            )

        raw_response = await self.llm_caller(messages)

        # Parse JSON output
        try:
            # Handle potential markdown code fencing in output
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            return JudgeClassification(
                primary_category=FailureCategory(parsed["primary_category"]),
                secondary_category=FailureCategory(parsed["secondary_category"]) if parsed.get("secondary_category") else None,
                confidence=float(parsed.get("confidence", 0.9)),
                root_cause=parsed.get("root_cause", ""),
                culprit_turn=parsed.get("culprit_turn"),
                recommended_fix=parsed.get("recommended_fix", ""),
                evidence_snippet=parsed.get("evidence_snippet"),
            )
        except Exception as e:
            return JudgeClassification(
                primary_category=FailureCategory.UNKNOWN_FAILURE,
                confidence=0.5,
                root_cause=f"Judge output parsing failed: {e}. Raw: {raw_response[:100]}",
                recommended_fix="Review judge output schema adherence.",
            )
