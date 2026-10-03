"""Unit tests for EvaluationRunner resumability and persistence."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agentgate.adapter.base import BenchmarkAdapter, BenchmarkTask, GradeResult
from agentgate.agent.models import AgentConfig
from agentgate.runner.runner import EvaluationRunner


class MockAdapterMultiTask(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "mock-multi-task"

    def load_tasks(
        self, split: str = "test", task_ids: Optional[List[int]] = None
    ) -> List[BenchmarkTask]:
        return [
            BenchmarkTask(task_id=i, user_id=f"u{i}", instruction=f"Task {i}") for i in [1, 2, 3]
        ]

    def create_env(self, task_id: int, user_model: str, user_strategy: str = "llm") -> Any:
        return {"task_id": task_id}

    def reset(self, env: Any, task_id: int) -> tuple[str, Dict[str, Any]]:
        return f"Start task {task_id}", {}

    def step(
        self, env: Any, action_name: str, kwargs: Dict[str, Any]
    ) -> tuple[str, float, bool, Dict[str, Any]]:
        return "Done ###STOP###", 1.0, True, {}

    def get_tools_info(self, env: Any) -> List[Dict[str, Any]]:
        return []

    def get_domain_policy(self, env: Any) -> str:
        return "Policy"

    def grade(self, env: Any) -> GradeResult:
        return GradeResult(reward=1.0, success=True, info={})


@pytest.mark.asyncio
async def test_runner_resumability(tmp_path: Path) -> None:
    db_file = tmp_path / "test_resume.db"
    runner = EvaluationRunner(db_path=db_file)
    adapter = MockAdapterMultiTask()
    config = AgentConfig(model="test-model", max_steps=3)

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content="Resolving task",
                tool_calls=None,
                model_dump=lambda: {"role": "assistant", "content": "Resolving task"},
            )
        )
    ]
    mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

    with patch("litellm.acompletion", return_value=mock_resp):
        # Run 1: Run task 1 only
        manifest_1 = await runner.run(
            run_id="test_run",
            config=config,
            task_ids=[1],
            num_trials=1,
            adapter=adapter,
        )
        assert manifest_1.completed_trials == 1

        # Check DB has task 1
        completed = runner.get_completed_pairs("test_run")
        assert (1, 0) in completed
        assert len(completed) == 1

        # Run 2: Re-run with tasks [1, 2, 3] under same run_id
        # Task 1 must be skipped and only tasks 2 & 3 executed
        manifest_2 = await runner.run(
            run_id="test_run",
            config=config,
            task_ids=[1, 2, 3],
            num_trials=1,
            adapter=adapter,
        )
        assert manifest_2.completed_trials == 3
        all_completed = runner.get_completed_pairs("test_run")
        assert (1, 0) in all_completed
        assert (2, 0) in all_completed
        assert (3, 0) in all_completed
