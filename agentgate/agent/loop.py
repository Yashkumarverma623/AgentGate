"""Custom tool-calling agent loop with parallel execution, malformed-call recovery, and tracing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import litellm

from agentgate.adapter.base import BenchmarkAdapter
from agentgate.agent.models import AgentConfig, StepRecord, ToolCallRecord, TrialResult
from agentgate.agent.prompt_manager import PromptManager
from agentgate.pricing import PricingRegistry
from agentgate.runner.pacer import QuotaPacer
from agentgate.tracing.tracer import LocalTracer

logger = logging.getLogger("agentgate.agent")


class AgentLoop:
    """Executes a single benchmark task trial with custom tool-calling, recovery, and tracing."""

    def __init__(
        self,
        config: AgentConfig,
        prompt_manager: Optional[PromptManager] = None,
        pricing: Optional[PricingRegistry] = None,
        pacer: Optional[QuotaPacer] = None,
        tracer: Optional[LocalTracer] = None,
    ) -> None:
        self.config = config
        self.prompt_manager = prompt_manager or PromptManager()
        self.pricing = pricing or PricingRegistry()
        self.pacer = pacer or QuotaPacer()
        self.tracer = tracer or LocalTracer()

    async def solve(
        self,
        adapter: BenchmarkAdapter,
        env: Any,
        task_id: int,
        trial_id: int = 0,
        run_id: str = "run_default",
    ) -> TrialResult:
        start_time = time.time()
        task_span_id = f"task_{run_id}_{task_id}_{trial_id}"

        # 1. Reset environment to obtain initial observation
        obs, initial_info = adapter.reset(env, task_id)
        self.tracer.record_span(
            span_id=task_span_id,
            name=f"Task {task_id} (Trial {trial_id})",
            span_type="task",
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            start_time=start_time,
            end_time=start_time,
            attributes={"instruction": obs, "model": self.config.model},
        )
        self.tracer.record_span(
            span_id=f"user_init_{task_span_id}",
            parent_id=task_span_id,
            name="Initial User Message",
            span_type="user_message",
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            start_time=start_time,
            end_time=start_time,
            attributes={"content": obs, "turn": 0},
        )

        # 2. Build system prompt & load tool definitions
        base_prompt, _ = self.prompt_manager.load_prompt(
            "retail", self.config.system_prompt_version
        )
        system_content = base_prompt
        if self.config.reasoning_mode == "plan-then-act":
            system_content += (
                "\n\n## Reasoning Requirement\n"
                "Before taking any tool action or communicating with the user, you MUST include a "
                "<plan>...</plan> block outlining your thoughts, current step, and immediate intent."
            )

        tools_info = adapter.get_tools_info(env)
        variant_overrides, _ = self.prompt_manager.load_tool_variant(
            self.config.tool_description_variant
        )
        tools_info = self.prompt_manager.apply_tool_variant(tools_info, variant_overrides)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": obs},
        ]

        step_records: List[StepRecord] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_would_have_cost = 0.0
        total_tool_calls_count = 0
        done = False
        error_msg: Optional[str] = None
        malformed_retry_count = 0

        # 3. Multi-turn step loop
        for step_idx in range(1, self.config.max_steps + 1):
            step_start = time.time()
            step_span_id = f"step_{task_span_id}_{step_idx}"

            # Prepare async LiteLLM completion call
            async def _call_llm() -> Any:
                return await litellm.acompletion(
                    model=self.config.model,
                    messages=messages,
                    tools=tools_info if tools_info else None,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout_seconds,
                    num_retries=0,
                )

            try:
                response = await self.pacer.execute_with_backoff(self.config.model, _call_llm)
            except Exception as e:
                logger.error("LLM call failed at step %d: %s", step_idx, e)
                error_msg = str(e)
                break

            step_end = time.time()
            choice = response.choices[0]
            assistant_message = choice.message
            content = assistant_message.content or ""

            # Extract token usage and compute commercial would-have-cost
            usage = getattr(response, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", 0) or 0
            c_tokens = getattr(usage, "completion_tokens", 0) or 0
            tot_tokens = p_tokens + c_tokens
            cost = self.pricing.calculate_would_have_cost(self.config.model, p_tokens, c_tokens)

            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            total_would_have_cost += cost

            llm_span_id = f"llm_{step_span_id}"
            self.tracer.record_span(
                span_id=llm_span_id,
                parent_id=step_span_id,
                name=f"LLM Call ({self.config.model})",
                span_type="llm_call",
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                start_time=step_start,
                end_time=step_end,
                attributes={
                    "model": self.config.model,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "would_have_cost": cost,
                    "content": content,
                },
            )

            step_record = StepRecord(
                step_index=step_idx,
                duration_seconds=step_end - step_start,
                model=self.config.model,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=tot_tokens,
                would_have_cost=cost,
                content=content,
            )

            # Check for tool calls
            raw_tool_calls = getattr(assistant_message, "tool_calls", None)

            if raw_tool_calls:
                # Append assistant message with tool calls to conversation history
                messages.append(assistant_message.model_dump())

                malformed_found = False
                for tc in raw_tool_calls:
                    total_tool_calls_count += 1
                    if isinstance(tc, dict):
                        tc_id = str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}")
                        func_dict = tc.get("function", {})
                        func_name = str(func_dict.get("name", ""))
                        raw_args = func_dict.get("arguments", "{}")
                    else:
                        tc_id = str(getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}")
                        func_obj = getattr(tc, "function", None)
                        if isinstance(func_obj, dict):
                            func_name = str(func_obj.get("name", ""))
                            raw_args = func_obj.get("arguments", "{}")
                        else:
                            func_name = str(getattr(func_obj, "name", ""))
                            raw_args = getattr(func_obj, "arguments", "{}")

                    # Validate JSON arguments with 1 retry recovery
                    try:
                        if isinstance(raw_args, dict):
                            parsed_args = raw_args
                        else:
                            parsed_args = json.loads(raw_args or "{}")
                    except Exception as json_err:
                        malformed_found = True
                        obs = f"Error: Malformed JSON arguments for tool '{func_name}': {json_err}. Please check argument schema and retry."
                        tool_rec = ToolCallRecord(
                            id=tc_id,
                            name=func_name,
                            raw_arguments=str(raw_args),
                            observation=obs,
                            is_error=True,
                        )
                        step_record.tool_calls.append(tool_rec)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": func_name,
                                "content": obs,
                            }
                        )
                        continue

                    # Execute valid tool call against benchmark adapter
                    t_start = time.time()
                    obs, r, step_done, step_info = adapter.step(env, func_name, parsed_args)
                    t_end = time.time()

                    is_err = "Error" in obs or "error" in obs.lower()
                    tool_rec = ToolCallRecord(
                        id=tc_id,
                        name=func_name,
                        arguments=parsed_args,
                        observation=obs,
                        is_error=is_err,
                    )
                    step_record.tool_calls.append(tool_rec)
                    step_record.observations.append(obs)

                    self.tracer.record_span(
                        span_id=f"tc_{tc_id}",
                        parent_id=step_span_id,
                        name=f"Tool: {func_name}",
                        span_type="tool_call",
                        run_id=run_id,
                        task_id=task_id,
                        trial_id=trial_id,
                        start_time=t_start,
                        end_time=t_end,
                        attributes={
                            "tool_name": func_name,
                            "arguments": parsed_args,
                            "observation": obs,
                            "is_error": is_err,
                        },
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": func_name,
                            "content": obs,
                        }
                    )

                    if step_done:
                        done = True

                if malformed_found:
                    malformed_retry_count += 1
                    if malformed_retry_count > 1:
                        logger.warning(
                            "Multiple malformed tool calls encountered in task %d", task_id
                        )

            else:
                # Plain conversational response to the customer
                messages.append({"role": "assistant", "content": content})
                t_start = time.time()
                obs, r, step_done, step_info = adapter.step(env, "respond", {"content": content})
                t_end = time.time()

                step_record.observations.append(obs)
                messages.append({"role": "user", "content": obs})

                self.tracer.record_span(
                    span_id=f"user_{step_span_id}",
                    parent_id=step_span_id,
                    name=f"User Turn {step_idx}",
                    span_type="user_message",
                    run_id=run_id,
                    task_id=task_id,
                    trial_id=trial_id,
                    start_time=t_start,
                    end_time=t_end,
                    attributes={"content": obs, "turn": step_idx},
                )

                if step_done or "###STOP###" in obs:
                    done = True

            step_records.append(step_record)
            if done:
                break

        # 4. Final grading against benchmark ground truth
        grade_result = adapter.grade(env)
        total_duration = time.time() - start_time

        # Update root span with final outcome
        self.tracer.record_span(
            span_id=task_span_id,
            name=f"Task {task_id} (Trial {trial_id})",
            span_type="task",
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            start_time=start_time,
            end_time=time.time(),
            attributes={
                "reward": grade_result.reward,
                "success": grade_result.success,
                "steps": len(step_records),
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "would_have_cost": total_would_have_cost,
                "error": error_msg,
            },
            status="OK" if grade_result.success else "ERROR",
        )

        return TrialResult(
            run_id=run_id,
            task_id=task_id,
            trial_id=trial_id,
            reward=grade_result.reward,
            success=grade_result.success,
            done=done,
            steps=len(step_records),
            tool_calls_count=total_tool_calls_count,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            would_have_cost=total_would_have_cost,
            duration_seconds=total_duration,
            trajectory=messages,
            step_records=step_records,
            info=grade_result.info,
            error=error_msg,
            status="infra_error" if error_msg else "completed",
        )
