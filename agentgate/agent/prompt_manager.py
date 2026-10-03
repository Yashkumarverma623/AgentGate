"""Prompt and tool variant loader with deterministic SHA-256 hashing."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def hash_content(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:12]


class PromptManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        self.base_dir = Path(base_dir)

    def load_prompt(self, domain: str, version: str) -> Tuple[str, str]:
        prompt_file = self.base_dir / "prompts" / domain / f"{version}.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        content = prompt_file.read_text(encoding="utf-8")
        return content, hash_content(content)

    def load_tool_variant(self, variant_name: str) -> Tuple[Dict[str, Any], str]:
        variant_file = self.base_dir / "tool_variants" / f"{variant_name}.yaml"
        if not variant_file.exists():
            # Fallback to standard if missing
            return {}, "default"
        content = variant_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
        return data.get("tools", {}), hash_content(content)

    def apply_tool_variant(
        self,
        tools_info: List[Dict[str, Any]],
        overrides: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Applies tool variant descriptions or parameter hints to OpenAI-style tools schema."""
        if not overrides:
            return tools_info

        modified_tools: List[Dict[str, Any]] = []
        for tool in tools_info:
            tool_copy = copy.deepcopy(tool)
            func_name = tool_copy.get("function", {}).get("name")
            if func_name and func_name in overrides:
                override = overrides[func_name]
                if isinstance(override, str):
                    tool_copy["function"]["description"] = override
                elif isinstance(override, dict):
                    if "description" in override:
                        tool_copy["function"]["description"] = override["description"]
                    if "parameters" in override:
                        tool_copy["function"]["parameters"].update(override["parameters"])
            modified_tools.append(tool_copy)
        return modified_tools
