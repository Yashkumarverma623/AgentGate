"""Abstract BenchmarkAdapter interface for AgentGate."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchmarkTask:
    task_id: int
    user_id: str
    instruction: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradeResult:
    reward: float  # 1.0 or 0.0
    success: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(abc.ABC):
    """Abstract interface that decouples AgentGate from specific evaluation benchmarks."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the benchmark adapter (e.g. 'tau-bench-retail')."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_tasks(
        self,
        split: str = "test",
        task_ids: Optional[List[int]] = None,
    ) -> List[BenchmarkTask]:
        """Loads tasks for the given split and optional task IDs."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_env(
        self,
        task_id: int,
        user_model: str,
        user_strategy: str = "llm",
    ) -> Any:
        """Creates an isolated environment instance for running a single trial."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, env: Any, task_id: int) -> Tuple[str, Dict[str, Any]]:
        """Resets the environment and returns the initial observation from the user."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        env: Any,
        action_name: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Steps the environment with an action and returns (observation, reward, done, info)."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_tools_info(self, env: Any) -> List[Dict[str, Any]]:
        """Returns the OpenAI-compatible tool specifications for the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain_policy(self, env: Any) -> str:
        """Returns the domain policy text / rules from the benchmark."""
        raise NotImplementedError

    @abc.abstractmethod
    def grade(self, env: Any) -> GradeResult:
        """Evaluates whether the agent successfully completed the task."""
        raise NotImplementedError
