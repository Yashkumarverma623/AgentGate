"""Export utilities for labeled failure taxonomies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union
from agentgate.taxonomy.models import FailureRecord, TaxonomySummary


def export_to_json(records: List[FailureRecord], output_path: Union[str, Path]) -> None:
    """Exports failure records to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [r.model_dump() for r in records]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def export_to_csv(records: List[FailureRecord], output_path: Union[str, Path]) -> None:
    """Exports failure records to a CSV file for analysis and reporting."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "task_id",
        "trial_id",
        "primary_category",
        "secondary_category",
        "source",
        "confidence",
        "steps_used",
        "total_tokens",
        "would_have_cost",
        "culprit_turn",
        "root_cause",
        "recommended_fix",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            d = r.model_dump()
            writer.writerow({k: d.get(k) for k in fieldnames})


def summarize_taxonomy(records: List[FailureRecord]) -> TaxonomySummary:
    """Computes distribution and cost aggregates by failure category."""
    counts: dict[str, int] = {}
    costs: dict[str, float] = {}
    tokens: dict[str, int] = {}
    total = len(records)

    for r in records:
        cat = r.primary_category.value
        counts[cat] = counts.get(cat, 0) + 1
        costs[cat] = round(costs.get(cat, 0.0) + r.would_have_cost, 4)
        tokens[cat] = tokens.get(cat, 0) + r.total_tokens

    proportions = {cat: round(cnt / total, 4) if total > 0 else 0.0 for cat, cnt in counts.items()}

    return TaxonomySummary(
        total_failed_trials=total,
        category_counts=counts,
        category_proportions=proportions,
        cost_by_category=costs,
        tokens_by_category=tokens,
    )
