"""Tau-bench retail and airline benchmark adapter implementation for AgentGate."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tau_bench.envs.user import BaseUserSimulationEnv

from agentgate.adapter.base import BenchmarkAdapter, BenchmarkTask, GradeResult


class GemmaUserSimulationEnv(BaseUserSimulationEnv):  # type: ignore[misc]
    """Direct Google GenAI user simulation environment for Gemma models."""

    def __init__(self, model: str = "gemma-4-26b-a4b-it") -> None:
        super().__init__()
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            api_key = api_key.strip().strip("'\"")
        self.client = genai.Client(api_key=api_key)
        clean_model = model.replace("gemini/", "").replace("models/", "")
        if clean_model in ("gemma-4-26b", "gemma-26b"):
            clean_model = "gemma-4-26b-a4b-it"
        elif clean_model in ("gemma-4-31b", "gemma-31b"):
            clean_model = "gemma-4-31b-it"
        self.model = clean_model
        self.history: List[str] = []
        self.system_prompt: str = ""
        self.total_cost: float = 0.0

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            f"\n\nInstruction: {instruction}\n" if instruction is not None else ""
        )
        return (
            f"You are a customer interacting with an agent.{instruction_display}\n"
            "Rules:\n"
            "- Just generate one line at a time to simulate the customer's message.\n"
            "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n"
            "- If the instruction goal is satisfied, generate '###STOP###' as a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        )

    def _generate(self) -> str:
        full_prompt = (
            f"{self.system_prompt}\n\nConversation history:\n"
            + "\n".join(self.history)
            + "\n\nCustomer:"
        )
        text = ""
        for attempt in range(3):
            try:
                res = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                )
                text = res.text.strip() if res.text else ""
                break
            except Exception:
                if attempt == 2:
                    text = "I need help with my order."
                else:
                    time.sleep(2.0 * (attempt + 1))

        if text.startswith("Customer:"):
            text = text[len("Customer:") :].strip()
        elif text.startswith("User:"):
            text = text[len("User:") :].strip()
        self.history.append(f"Customer: {text}")
        return text

    def reset(self, instruction: Optional[str] = None) -> str:
        self.system_prompt = self.build_system_prompt(instruction=instruction)
        self.history = ["Agent: Hi! How can I help you today?"]
        return self._generate()

    def step(self, content: str) -> str:
        self.history.append(f"Agent: {content}")
        return self._generate()

    def get_total_cost(self) -> float:
        return self.total_cost


class TauBenchRetailAdapter(BenchmarkAdapter):
    """Adapter wrapping Sierra Research's tau-bench retail domain."""

    def __init__(self, task_split: str = "test") -> None:
        self.task_split = task_split
        self._raw_tasks: Optional[List[Any]] = None

    @property
    def name(self) -> str:
        return "tau-bench-retail"

    def _ensure_tasks_loaded(self) -> List[Any]:
        if self._raw_tasks is None:
            if self.task_split == "test":
                from tau_bench.envs.retail.tasks_test import TASKS_TEST

                self._raw_tasks = TASKS_TEST
            elif self.task_split == "dev":
                from tau_bench.envs.retail.tasks_dev import TASKS_DEV

                self._raw_tasks = TASKS_DEV
            elif self.task_split == "train":
                from tau_bench.envs.retail.tasks_train import TASKS_TRAIN

                self._raw_tasks = TASKS_TRAIN
            else:
                from tau_bench.envs.retail.tasks_test import TASKS_TEST

                self._raw_tasks = TASKS_TEST
        return self._raw_tasks

    def get_stratified_subset_ids(self, count: int = 20, seed: int = 42) -> List[int]:
        """Returns a deterministic, stratified subset of task indices based on action category.
        Preserves round-robin order so that any prefix [:N] (e.g. [:20]) is itself stratified.
        """
        tasks = self._ensure_tasks_loaded()
        total = len(tasks)
        if count >= total:
            return list(range(total))

        _ = seed

        def _get_action_category(task: Any) -> str:
            action_names = [
                a.name
                for a in getattr(task, "actions", [])
                if any(k in a.name for k in ["exchange", "return", "modify", "cancel"])
            ]
            return action_names[0] if action_names else "info_only"

        buckets: Dict[str, List[int]] = {}
        for idx, task in enumerate(tasks):
            cat = _get_action_category(task)
            buckets.setdefault(cat, []).append(idx)

        selected: List[int] = []
        action_keys = sorted(buckets.keys())
        while len(selected) < count:
            added_any = False
            for k in action_keys:
                if buckets[k] and len(selected) < count:
                    idx = buckets[k].pop(0)
                    selected.append(idx)
                    added_any = True
            if not added_any:
                break
        return selected

    def load_tasks(
        self,
        split: str = "test",
        task_ids: Optional[List[int]] = None,
    ) -> List[BenchmarkTask]:
        self.task_split = split
        raw_tasks = self._ensure_tasks_loaded()

        if task_ids is None:
            indices = list(range(len(raw_tasks)))
        else:
            indices = [i for i in task_ids if 0 <= i < len(raw_tasks)]

        benchmark_tasks: List[BenchmarkTask] = []
        for idx in indices:
            t = raw_tasks[idx]
            actions_dump = [a.model_dump() for a in t.actions]
            benchmark_tasks.append(
                BenchmarkTask(
                    task_id=idx,
                    user_id=t.user_id,
                    instruction=t.instruction,
                    actions=actions_dump,
                    outputs=list(t.outputs),
                    metadata={"annotator": getattr(t, "annotator", "0")},
                )
            )
        return benchmark_tasks

    def create_env(
        self,
        task_id: int,
        user_model: str = "gemini/gemma-4-26b",
        user_strategy: str = "llm",
    ) -> Any:
        from tau_bench.envs import get_env

        is_gemma = "gemma" in user_model.lower()
        provider = None
        if "/" in user_model:
            provider, raw_model = user_model.split("/", 1)
        else:
            raw_model = user_model

        env = get_env(
            env_name="retail",
            user_strategy="human" if is_gemma else user_strategy,
            user_model=raw_model,
            user_provider=provider,
            task_split=self.task_split,
            task_index=task_id,
        )

        if is_gemma:
            env.user = GemmaUserSimulationEnv(model=raw_model)

        return env

    def reset(self, env: Any, task_id: int) -> Tuple[str, Dict[str, Any]]:
        res = env.reset(task_index=task_id)
        return res.observation, res.info.model_dump()

    def step(
        self,
        env: Any,
        action_name: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        from tau_bench.types import Action

        action = Action(name=action_name, kwargs=kwargs)
        res = env.step(action)
        return res.observation, res.reward, res.done, res.info.model_dump()

    def get_tools_info(self, env: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(env.tools_info)

    def get_domain_policy(self, env: Any) -> str:
        return str(env.wiki)

    def grade(self, env: Any) -> GradeResult:
        reward_res = env.calculate_reward()
        reward = float(reward_res.reward)
        success = reward >= (1.0 - 1e-6)
        return GradeResult(
            reward=reward,
            success=success,
            info=reward_res.model_dump(),
        )
