"""Pydantic models for AgentGate agent configuration, execution, and manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model: str = "gemini/gemini-3.1-flash-lite"
    temperature: float = 0.0
    max_steps: int = 30  # Canonical reference baseline cap (tau-bench default)
    system_prompt_version: str = "v1"
    tool_description_variant: str = "standard"
    reasoning_mode: Literal["none", "plan-then-act"] = "none"
    user_model: str = "gemini/gemma-4-26b"
    user_strategy: str = "llm"
    timeout_seconds: float = 60.0

    def compute_hash(self) -> str:
        dumped = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


class ToolCallRecord(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments: Optional[str] = None
    observation: Optional[str] = None
    is_error: bool = False


class StepRecord(BaseModel):
    step_index: int
    duration_seconds: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    would_have_cost: float = 0.0
    content: Optional[str] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)


class TrialResult(BaseModel):
    run_id: str
    task_id: int
    trial_id: int
    reward: float  # 1.0 or 0.0
    success: bool
    done: bool
    steps: int
    tool_calls_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    would_have_cost: float
    duration_seconds: float
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    step_records: List[StepRecord] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    status: Literal["completed", "infra_error"] = "completed"
    finished_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RunManifest(BaseModel):
    run_id: str
    git_sha: str
    benchmark_name: str = "tau-bench-retail"
    benchmark_commit_sha: str = "59a200c6d575d595120f1cb70fea53cef0632f6b"
    config_hash: str
    agent_config: AgentConfig
    prompt_version: str
    prompt_hash: str
    tool_variant: str
    tool_variant_hash: str
    task_ids: List[int]
    num_trials: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: Literal["running", "completed", "incomplete", "paused_quota"] = "running"
    total_trials: int = 0
    completed_trials: int = 0
    successful_trials: int = 0
    infra_error_trials: int = 0
    total_tokens: int = 0
    total_would_have_cost: float = 0.0
