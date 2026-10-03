"""Unit tests for the custom AgentLoop using a mock LLM client."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agentgate.adapter.base import BenchmarkAdapter, BenchmarkTask, GradeResult
from agentgate.agent.loop import AgentLoop
from agentgate.agent.models import AgentConfig
from agentgate.pricing import PricingRegistry
from agentgate.runner.pacer import QuotaPacer
from agentgate.tracing.tracer import LocalTracer


class MockAdapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "mock-adapter"

    def load_tasks(
        self, split: str = "test", task_ids: Optional[List[int]] = None
    ) -> List[BenchmarkTask]:
        return [BenchmarkTask(task_id=0, user_id="u1", instruction="Test instruction")]

    def create_env(self, task_id: int, user_model: str, user_strategy: str = "llm") -> Any:
        return {"state": "init"}

    def reset(self, env: Any, task_id: int) -> tuple[str, Dict[str, Any]]:
        return "Hello, I want to cancel order #123", {"user": "u1"}

    def step(
        self, env: Any, action_name: str, kwargs: Dict[str, Any]
    ) -> tuple[str, float, bool, Dict[str, Any]]:
        if action_name == "cancel_pending_order":
            return "Order #123 cancelled successfully", 1.0, False, {}
        elif action_name == "respond":
            return "Thank you! ###STOP###", 1.0, True, {}
        return "Unknown action", 0.0, False, {}

    def get_tools_info(self, env: Any) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "cancel_pending_order",
                    "description": "Cancel an order",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ]

    def get_domain_policy(self, env: Any) -> str:
        return "Mock domain policy"

    def grade(self, env: Any) -> GradeResult:
        return GradeResult(reward=1.0, success=True, info={"status": "passed"})


@pytest.mark.asyncio
async def test_agent_loop_successful_tool_call(tmp_path: Path) -> None:
    config = AgentConfig(model="test-model", max_steps=5)
    tracer = LocalTracer(db_path=tmp_path / "test.db")
    pacer = QuotaPacer(state_path=tmp_path / "quota.json")
    pricing = PricingRegistry()
    loop = AgentLoop(config=config, pricing=pricing, pacer=pacer, tracer=tracer)
    adapter = MockAdapter()

    # Step 1: Agent calls cancel_pending_order tool
    # Step 2: Agent responds to user with plain text
    mock_resp_1 = MagicMock()
    mock_resp_1.choices = [
        MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="call_1",
                        function=MagicMock(
                            name="cancel_pending_order",
                            arguments='{"order_id": "#123"}',
                        ),
                    )
                ],
                model_dump=lambda: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "cancel_pending_order",
                                "arguments": '{"order_id": "#123"}',
                            },
                        }
                    ],
                },
            )
        )
    ]
    mock_resp_1.usage = MagicMock(prompt_tokens=100, completion_tokens=20)

    mock_resp_2 = MagicMock()
    mock_resp_2.choices = [
        MagicMock(
            message=MagicMock(
                content="I have cancelled your order #123.",
                tool_calls=None,
                model_dump=lambda: {
                    "role": "assistant",
                    "content": "I have cancelled your order #123.",
                },
            )
        )
    ]
    mock_resp_2.usage = MagicMock(prompt_tokens=150, completion_tokens=30)

    with patch("litellm.acompletion", side_effect=[mock_resp_1, mock_resp_2]):
        env = adapter.create_env(0, "user-model")
        res = await loop.solve(adapter=adapter, env=env, task_id=0, trial_id=0, run_id="unit_run")

        assert res.success is True
        assert res.reward == 1.0
        assert res.steps == 2
        assert res.tool_calls_count == 1
        assert res.total_prompt_tokens == 250
        assert res.total_completion_tokens == 50


@pytest.mark.asyncio
async def test_agent_loop_malformed_json_recovery(tmp_path: Path) -> None:
    config = AgentConfig(model="test-model", max_steps=5)
    tracer = LocalTracer(db_path=tmp_path / "test.db")
    pacer = QuotaPacer(state_path=tmp_path / "quota.json")
    pricing = PricingRegistry()
    loop = AgentLoop(config=config, pricing=pricing, pacer=pacer, tracer=tracer)
    adapter = MockAdapter()

    # Turn 1: Malformed JSON arguments
    mock_resp_bad = MagicMock()
    mock_resp_bad.choices = [
        MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="bad_call",
                        function=MagicMock(
                            name="cancel_pending_order",
                            arguments="{order_id: INVALID_JSON}",
                        ),
                    )
                ],
                model_dump=lambda: {"role": "assistant", "content": None, "tool_calls": []},
            )
        )
    ]
    mock_resp_bad.usage = MagicMock(prompt_tokens=100, completion_tokens=20)

    # Turn 2: Recovered valid JSON arguments
    mock_resp_good = MagicMock()
    mock_resp_good.choices = [
        MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="good_call",
                        function=MagicMock(
                            name="cancel_pending_order",
                            arguments='{"order_id": "#123"}',
                        ),
                    )
                ],
                model_dump=lambda: {"role": "assistant", "content": None, "tool_calls": []},
            )
        )
    ]
    mock_resp_good.usage = MagicMock(prompt_tokens=150, completion_tokens=25)

    # Turn 3: Final message to user
    mock_resp_done = MagicMock()
    mock_resp_done.choices = [
        MagicMock(
            message=MagicMock(
                content="Cancelled!",
                tool_calls=None,
                model_dump=lambda: {"role": "assistant", "content": "Cancelled!"},
            )
        )
    ]
    mock_resp_done.usage = MagicMock(prompt_tokens=200, completion_tokens=15)

    with patch("litellm.acompletion", side_effect=[mock_resp_bad, mock_resp_good, mock_resp_done]):
        env = adapter.create_env(0, "user-model")
        res = await loop.solve(adapter=adapter, env=env, task_id=0, trial_id=0, run_id="unit_run")

        assert res.success is True
        assert res.steps == 3
        # First step had malformed error observation
        assert res.step_records[0].tool_calls[0].is_error is True
        assert "Malformed JSON" in str(res.step_records[0].tool_calls[0].observation)
