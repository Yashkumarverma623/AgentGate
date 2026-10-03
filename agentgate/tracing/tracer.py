"""OpenTelemetry-compliant local tracing and span recording engine for AgentGate."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class LocalTracer:
    """Records hierarchical trace spans (Task > Step > LLM Call > Tool Call) into SQLite and JSONL."""

    def __init__(self, db_path: Path | str = "results/agentgate.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    run_id TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    trial_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    span_type TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration REAL,
                    attributes_json TEXT,
                    status TEXT DEFAULT 'OK'
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_spans_run_task
                ON spans(run_id, task_id, trial_id)
                """
            )
            conn.commit()

    def record_span(
        self,
        span_id: str,
        name: str,
        span_type: str,
        run_id: str,
        task_id: int,
        trial_id: int,
        start_time: float,
        end_time: float,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        status: str = "OK",
    ) -> None:
        duration = max(0.0, end_time - start_time)
        attrs = json.dumps(attributes or {}, default=str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO spans
                (span_id, parent_id, run_id, task_id, trial_id, name, span_type, start_time, end_time, duration, attributes_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span_id,
                    parent_id,
                    run_id,
                    task_id,
                    trial_id,
                    name,
                    span_type,
                    start_time,
                    end_time,
                    duration,
                    attrs,
                    status,
                ),
            )
            conn.commit()

    def get_task_spans(self, run_id: str, task_id: int, trial_id: int = 0) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM spans
                WHERE run_id = ? AND task_id = ? AND trial_id = ?
                ORDER BY start_time ASC
                """,
                (run_id, task_id, trial_id),
            )
            rows = cursor.fetchall()
            spans: List[Dict[str, Any]] = []
            for r in rows:
                span_dict = dict(r)
                span_dict["attributes"] = json.loads(span_dict["attributes_json"] or "{}")
                spans.append(span_dict)
            return spans

    def render_task_transcript(
        self,
        run_id: str,
        task_id: int,
        trial_id: int = 0,
        console: Optional[Console] = None,
    ) -> str:
        """Renders an interactive, readable terminal transcript of the trial."""
        if console is None:
            console = Console()

        spans = self.get_task_spans(run_id, task_id, trial_id)
        if not spans:
            return f"No trace records found for run={run_id}, task_id={task_id}, trial={trial_id}"

        # Fetch trial result from results table if available
        result_info = self.get_trial_result(run_id, task_id, trial_id)

        console.print("\n[bold cyan]================ TRIAL TRANSCRIPT ================[/bold cyan]")
        console.print(
            f"[bold]Run ID:[/bold] {run_id} | [bold]Task ID:[/bold] {task_id} | [bold]Trial:[/bold] {trial_id}"
        )
        if result_info:
            outcome = (
                "[bold green]PASSED (Reward 1.0)[/bold green]"
                if result_info.get("reward", 0) == 1
                else "[bold red]FAILED (Reward 0.0)[/bold red]"
            )
            console.print(
                f"[bold]Outcome:[/bold] {outcome} | [bold]Steps:[/bold] {result_info.get('steps')} | [bold]Duration:[/bold] {result_info.get('duration_seconds', 0):.2f}s | [bold]Tokens:[/bold] {result_info.get('total_tokens')}"
            )

        for span in spans:
            span_type = span["span_type"]
            attrs = span.get("attributes", {})

            if span_type == "user_message":
                console.print(
                    Panel(
                        Text(attrs.get("content", ""), style="yellow"),
                        title=f"[bold yellow]👤 Customer (Turn {attrs.get('turn', '')})[/bold yellow]",
                        border_style="yellow",
                    )
                )
            elif span_type == "llm_call":
                content = attrs.get("content")
                if content:
                    console.print(
                        Panel(
                            Text(content, style="cyan"),
                            title=f"[bold cyan]🤖 Agent ({attrs.get('model', '')})[/bold cyan]",
                            border_style="cyan",
                        )
                    )
            elif span_type == "tool_call":
                tool_name = attrs.get("tool_name", "")
                args_str = json.dumps(attrs.get("arguments", {}), indent=2)
                obs_str = attrs.get("observation", "")
                is_err = attrs.get("is_error", False)
                border = "red" if is_err else "magenta"

                body = f"[bold]Tool:[/bold] {tool_name}\n[bold]Arguments:[/bold]\n{args_str}\n\n[bold]Observation:[/bold]\n{obs_str}"
                console.print(
                    Panel(
                        body,
                        title=f"[bold {border}]⚙️ Tool Call: {tool_name}[/bold {border}]",
                        border_style=border,
                    )
                )

        return "Rendered successfully."

    def get_trial_result(
        self, run_id: str, task_id: int, trial_id: int
    ) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM trial_results
                WHERE run_id = ? AND task_id = ? AND trial_id = ?
                """,
                (run_id, task_id, trial_id),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
