"""Pricing and cost computation engine for AgentGate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class ModelPricing:
    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float
    last_verified: str

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate commercial would-have-cost in USD from token counts."""
        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        cost = (
            (uncached_prompt * self.input_per_million / 1_000_000.0)
            + (cached_tokens * self.cached_input_per_million / 1_000_000.0)
            + (completion_tokens * self.output_per_million / 1_000_000.0)
        )
        return cost


class PricingRegistry:
    def __init__(self, pricing_path: Optional[str | Path] = None) -> None:
        if pricing_path is None:
            pricing_path = Path(__file__).resolve().parent.parent / "pricing.yaml"
        self.pricing_path = Path(pricing_path)
        self.version: str = "unknown"
        self.models: Dict[str, ModelPricing] = {}
        self.load()

    def load(self) -> None:
        if not self.pricing_path.exists():
            return
        with open(self.pricing_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.version = str(data.get("version", datetime.now().strftime("%Y-%m-%d")))
        models_data = data.get("models", {})
        for model_name, info in models_data.items():
            self.models[model_name] = ModelPricing(
                model=model_name,
                input_per_million=float(info.get("input_per_million", 0.0)),
                output_per_million=float(info.get("output_per_million", 0.0)),
                cached_input_per_million=float(
                    info.get("cached_input_per_million", info.get("input_per_million", 0.0) * 0.25)
                ),
                last_verified=str(info.get("last_verified", self.version)),
            )

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        # Exact match
        if model in self.models:
            return self.models[model]
        # Match without prefix (e.g. gemini-3.1-flash-lite vs gemini/gemini-3.1-flash-lite)
        for key, pricing in self.models.items():
            if key.split("/")[-1] == model.split("/")[-1]:
                return pricing
        # Fallback default low cost pricing for unlisted models
        return ModelPricing(
            model=model,
            input_per_million=0.100,
            output_per_million=0.400,
            cached_input_per_million=0.025,
            last_verified=self.version,
        )

    def calculate_would_have_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        pricing = self.get_pricing(model)
        if pricing is None:
            return 0.0
        return pricing.calculate_cost(prompt_tokens, completion_tokens, cached_tokens)
