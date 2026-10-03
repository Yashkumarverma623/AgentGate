"""Data models and schemas for trace-based failure taxonomy."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    # Deterministic categories
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    SIMULATOR_EARLY_STOP = "SIMULATOR_EARLY_STOP"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    TOOL_CALL_LOOP = "TOOL_CALL_LOOP"
    TRANSFER_TO_HUMAN = "TRANSFER_TO_HUMAN"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"

    # LLM Judge categories
    POLICY_VIOLATION_AUTH = "POLICY_VIOLATION_AUTH"
    POLICY_VIOLATION_CONFIRMATION = "POLICY_VIOLATION_CONFIRMATION"
    GROUNDING_HALLUCINATION = "GROUNDING_HALLUCINATION"
    INCORRECT_TOOL_SELECTION = "INCORRECT_TOOL_SELECTION"
    REASONING_LOOP = "REASONING_LOOP"
    INCOMPLETE_GOAL = "INCOMPLETE_GOAL"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class DeterministicCheckResult(BaseModel):
    category: FailureCategory
    confidence: float = 1.0
    reason: str
    step_index: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class JudgeClassification(BaseModel):
    primary_category: FailureCategory
    secondary_category: Optional[FailureCategory] = None
    confidence: float = Field(ge=0.0, le=1.0)
    root_cause: str
    culprit_turn: Optional[int] = None
    recommended_fix: str
    evidence_snippet: Optional[str] = None


class FailureRecord(BaseModel):
    run_id: str
    task_id: int
    trial_id: int
    primary_category: FailureCategory
    secondary_category: Optional[FailureCategory] = None
    source: str = "deterministic"  # "deterministic" or "llm_judge"
    confidence: float
    root_cause: str
    steps_used: int
    total_tokens: int
    would_have_cost: float
    culprit_turn: Optional[int] = None
    recommended_fix: Optional[str] = None


class TaxonomySummary(BaseModel):
    total_failed_trials: int
    category_counts: Dict[str, int]
    category_proportions: Dict[str, float]
    cost_by_category: Dict[str, float]
    tokens_by_category: Dict[str, int]
