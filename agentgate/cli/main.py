"""Command Line Interface for AgentGate."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import typer
import numpy as np
from scipy import stats
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentgate.adapter.tau_bench import TauBenchRetailAdapter
from agentgate.agent.models import AgentConfig
from agentgate.pricing import PricingRegistry
from agentgate.runner.runner import EvaluationRunner
from agentgate.stats.metrics import (
    paired_bootstrap_comparison,
)
from agentgate.tracing.tracer import LocalTracer

app = typer.Typer(
    name="agentgate",
    help="AgentGate: Evaluation and regression-testing harness for tool-using LLM agents",
    add_completion=False,
)
trace_app = typer.Typer(name="trace", help="Inspect and visualize execution traces")
app.add_typer(trace_app, name="trace")

console = Console()


@app.command()
def run(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", "-r", help="Unique identifier for the run"
    ),
    model: str = typer.Option("gemini/gemini-3.1-flash-lite", "--model", "-m", help="Agent model"),
    user_model: str = typer.Option(
        "gemini/gemma-4-26b", "--user-model", "-u", help="User simulator model"
    ),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated task IDs"),
    num_tasks: int = typer.Option(
        20, "--num-tasks", "-n", help="Number of stratified tasks to run"
    ),
    trials: int = typer.Option(2, "--trials", "-k", help="Trials per task (k)"),
    max_steps: int = typer.Option(30, "--max-steps", "-s", help="Max steps per trial (default 30)"),
    prompt_version: str = typer.Option(
        "v1", "--prompt-version", "-p", help="System prompt version"
    ),
    tool_variant: str = typer.Option("standard", "--tool-variant", help="Tool description variant"),
    reasoning_mode: str = typer.Option(
        "none", "--reasoning-mode", help="Reasoning mode (none | plan-then-act)"
    ),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Execution concurrency"),
    db_path: str = typer.Option("results/agentgate.db", "--db-path", help="SQLite database path"),
) -> None:
    """Execute a benchmark evaluation run."""
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    adapter = TauBenchRetailAdapter(task_split="test")
    if tasks:
        task_ids = [int(t.strip()) for t in tasks.split(",") if t.strip()]
    else:
        task_ids = adapter.get_stratified_subset_ids(count=num_tasks)

    config = AgentConfig(
        model=model,
        user_model=user_model,
        max_steps=max_steps,
        system_prompt_version=prompt_version,
        tool_description_variant=tool_variant,
        reasoning_mode=reasoning_mode,  # type: ignore
    )

    console.print(f"[bold cyan]Starting AgentGate Evaluation Run:[/bold cyan] {run_id}")
    console.print(f"• Model: [green]{model}[/green] | User Model: [green]{user_model}[/green]")
    console.print(
        f"• Tasks: {len(task_ids)} (stratified) | Trials: {trials} (total: {len(task_ids) * trials} trials)"
    )
    console.print(f"• Concurrency: {concurrency} | Pacer: Strict per-model rate limits")

    runner = EvaluationRunner(db_path=db_path)
    manifest = asyncio.run(
        runner.run(
            run_id=run_id,
            config=config,
            task_ids=task_ids,
            num_trials=trials,
            adapter=adapter,
            concurrency=concurrency,
        )
    )

    console.print(
        f"\n[bold green]Run {run_id} finished with status: {manifest.status}[/bold green]"
    )
    console.print(f"Completed: {manifest.completed_trials}/{manifest.total_trials} trials")
    console.print(f"Successful trials: {manifest.successful_trials}")
    if manifest.infra_error_trials > 0:
        console.print(f"[bold yellow]Infra-error trials (excluded): {manifest.infra_error_trials}[/bold yellow]")
    console.print(
        f"Total tokens: {manifest.total_tokens:,} | Would-have-cost: ${manifest.total_would_have_cost:.4f}"
    )


@trace_app.command("show")
def trace_show(
    run_id: str = typer.Argument(..., help="Run ID"),
    task_id: int = typer.Argument(..., help="Task ID"),
    trial_id: int = typer.Option(0, "--trial", "-t", help="Trial ID (0-indexed)"),
    db_path: str = typer.Option("results/agentgate.db", "--db-path", help="SQLite DB path"),
) -> None:
    """Print readable terminal transcript of a single task trial."""
    tracer = LocalTracer(db_path)
    tracer.render_task_transcript(run_id, task_id, trial_id, console=console)


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline run ID"),
    candidate: str = typer.Argument(..., help="Candidate run ID"),
    tolerance: float = typer.Option(0.0, "--tolerance", help="Regression tolerance delta"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level alpha"),
    db_path: str = typer.Option("results/agentgate.db", "--db-path", help="SQLite DB path"),
) -> None:
    """Statistically compare candidate run against baseline run on identical tasks."""
    with sqlite3.connect(db_path) as conn:

        def _get_outcomes(run_id: str) -> dict[int, list[int]]:
            cursor = conn.execute(
                "SELECT task_id, success FROM trial_results WHERE run_id = ? AND (status != 'infra_error' OR status IS NULL) ORDER BY task_id, trial_id",
                (run_id,),
            )
            outcomes: dict[int, list[int]] = {}
            for tid, succ in cursor.fetchall():
                outcomes.setdefault(tid, []).append(int(succ))
            return outcomes

        base_outcomes = _get_outcomes(baseline)
        cand_outcomes = _get_outcomes(candidate)

    if not base_outcomes or not cand_outcomes:
        console.print(
            "[bold red]Error: One or both runs have no recorded trial results in database.[/bold red]"
        )
        sys.exit(1)

    res = paired_bootstrap_comparison(
        baseline_outcomes=base_outcomes,
        candidate_outcomes=cand_outcomes,
        num_bootstrap=10000,
        alpha=alpha,
        tolerance=tolerance,
    )

    table = Table(title=f"Paired Comparison: {baseline} vs {candidate}")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta (95% CI)", justify="right")
    table.add_column("p-value", justify="right")

    delta_str = f"{res.delta_pass_1:+.4f} [{res.delta_ci[0]:.4f}, {res.delta_ci[1]:.4f}]"
    table.add_row(
        "pass^1",
        f"{res.baseline_pass_1:.4f}",
        f"{res.candidate_pass_1:.4f}",
        delta_str,
        f"{res.p_value:.4f}",
    )
    console.print(table)

    color = (
        "green"
        if res.verdict == "IMPROVEMENT"
        else ("red" if res.verdict == "REGRESSION" else "yellow")
    )
    console.print(
        Panel(
            f"[bold {color}]Verdict: {res.verdict}[/bold {color}]\n{res.details}",
            title="Gate Evaluation",
            border_style=color,
        )
    )


@app.command()
def gate(
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate run ID or JSON manifest path"),
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline run ID or JSON manifest path"),
    tolerance: float = typer.Option(0.10, "--tolerance", help="Allowed drop threshold in pass^1 points (default 0.10 / 10 pts)"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance alpha threshold (default 0.05)"),
    db_path: str = typer.Option("results/agentgate.db", "--db-path", help="SQLite database path"),
) -> None:
    """CI Regression Gate: Evaluates candidate against baseline with paired bootstrap and McNemar tests.
    Exits with non-zero code if significant regression (CI < 0) or drop exceeds tolerance."""

    def _load_run_data(run_id_or_path: str, db_file: str) -> tuple[str, int, list[dict[str, Any]], bool]:
        p = Path(run_id_or_path)
        if p.exists() and (run_id_or_path.endswith(".json") or p.is_file()):
            data = json.loads(p.read_text(encoding="utf-8"))
            rid = data.get("run_id", p.stem)
            agent_cfg = data.get("agent_config") or data.get("config") or {}
            m_steps = agent_cfg.get("max_steps", 30)
            raw_trials = data.get("trials", [])
            loaded_rows = []
            for t in raw_trials:
                row = dict(t)
                if "trajectory_json" not in row:
                    row["trajectory_json"] = json.dumps(row.get("trajectory", []))
                loaded_rows.append(row)
            return rid, m_steps, loaded_rows, True

        # Otherwise query SQLite
        if not Path(db_file).exists():
            return run_id_or_path, 30, [], False

        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            m_steps = 30
            try:
                cur.execute("SELECT manifest_json FROM runs WHERE run_id = ?", (run_id_or_path,))
                m_row = cur.fetchone()
                if m_row and m_row[0]:
                    m_data = json.loads(m_row[0])
                    agent_cfg = m_data.get("agent_config") or m_data.get("config") or {}
                    m_steps = agent_cfg.get("max_steps", 30)
            except sqlite3.OperationalError:
                pass

            cur.execute(
                """
                SELECT task_id, trial_id, success, reward, steps, would_have_cost,
                       total_prompt_tokens, total_completion_tokens, total_tokens,
                       duration_seconds, trajectory_json, error, status, finished_at
                FROM trial_results
                WHERE run_id = ? AND (status != 'infra_error' OR status IS NULL)
                ORDER BY task_id, trial_id
                """,
                (run_id_or_path,),
            )
            loaded_rows = [dict(r) for r in cur.fetchall()]
            return run_id_or_path, m_steps, loaded_rows, False

    base_id, base_max_steps, base_rows, base_is_json = _load_run_data(baseline, db_path)
    cand_id, cand_max_steps, cand_rows, cand_is_json = _load_run_data(candidate, db_path)

    if not base_rows:
        console.print(f"[bold red]Error: No completed trials found for baseline run '{baseline}'.[/bold red]")
        sys.exit(1)
    if not cand_rows:
        console.print(f"[bold red]Error: No completed trials found for candidate run '{candidate}'.[/bold red]")
        sys.exit(1)

    base_by_task: dict[int, list[int]] = {}
    for r in base_rows:
        base_by_task.setdefault(r["task_id"], []).append(int(r["success"]))

    cand_by_task: dict[int, list[int]] = {}
    for r in cand_rows:
        cand_by_task.setdefault(r["task_id"], []).append(int(r["success"]))

    shared_tasks = sorted(set(base_by_task.keys()) & set(cand_by_task.keys()))
    if not shared_tasks:
        console.print("[bold red]Error: No common tasks found between baseline and candidate runs.[/bold red]")
        sys.exit(1)

    # Filter rows to shared tasks for apples-to-apples comparison
    shared_base_rows = [r for r in base_rows if r["task_id"] in shared_tasks]
    shared_cand_rows = [r for r in cand_rows if r["task_id"] in shared_tasks]

    def _get_call_count(run_id: str, tasks: list[int], rows: list[dict[str, Any]], is_json: bool) -> int:
        if is_json or not Path(db_path).exists():
            return sum(r["steps"] for r in rows if r["task_id"] in tasks)
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in tasks)
            cur.execute(
                f"SELECT COUNT(*) FROM spans WHERE run_id = ? AND span_type = 'llm_call' AND task_id IN ({placeholders})",
                (run_id, *tasks),
            )
            count = cur.fetchone()[0]
            if count == 0:
                cur.execute(
                    f"SELECT SUM(steps) FROM trial_results WHERE run_id = ? AND task_id IN ({placeholders}) AND (status != 'infra_error' OR status IS NULL)",
                    (run_id, *tasks),
                )
                count = cur.fetchone()[0] or 0
            return count

    base_calls = _get_call_count(base_id, shared_tasks, shared_base_rows, base_is_json)
    cand_calls = _get_call_count(cand_id, shared_tasks, shared_cand_rows, cand_is_json)

    def _get_api_attempts(run_id: str, rows: list[dict[str, Any]]) -> Optional[int]:
        log_path = Path("results/api_calls.jsonl")
        if not log_path.exists():
            return None
        count = 0
        matched_by_id = False
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("run_id") == run_id:
                        count += 1
                        matched_by_id = True
                except Exception:
                    continue
        if matched_by_id:
            return count
        timestamps = [r["finished_at"] for r in rows if r.get("finished_at")]
        if not timestamps:
            return None
        t_min = min(timestamps)[:19]
        t_max = max(timestamps)[:19]
        count = 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ts = rec.get("timestamp", "")[:19]
                    if t_min <= ts <= t_max:
                        count += 1
                except Exception:
                    continue
        return count if count > 0 else None

    base_api_attempts = _get_api_attempts(base_id, shared_base_rows)
    cand_api_attempts = _get_api_attempts(cand_id, shared_cand_rows)

    N = len(shared_tasks)
    base_p1_per_task = np.array([np.mean(base_by_task[tid]) for tid in shared_tasks])
    cand_p1_per_task = np.array([np.mean(cand_by_task[tid]) for tid in shared_tasks])

    base_p1 = float(np.mean(base_p1_per_task))
    cand_p1 = float(np.mean(cand_p1_per_task))
    delta_p1 = cand_p1 - base_p1

    # Task-level paired bootstrap
    rng = np.random.RandomState(42)
    boot_indices = rng.choice(N, size=(10000, N), replace=True)
    boot_base = np.mean(base_p1_per_task[boot_indices], axis=1)
    boot_cand = np.mean(cand_p1_per_task[boot_indices], axis=1)
    boot_deltas = boot_cand - boot_base

    low_q = (alpha / 2.0) * 100.0
    high_q = (1.0 - alpha / 2.0) * 100.0
    delta_ci = (float(np.percentile(boot_deltas, low_q)), float(np.percentile(boot_deltas, high_q)))

    # Secondary: McNemar test on paired trials
    base_trial_map = {(r["task_id"], r["trial_id"]): r["success"] for r in shared_base_rows}
    cand_trial_map = {(r["task_id"], r["trial_id"]): r["success"] for r in shared_cand_rows}
    paired_keys = sorted(set(base_trial_map.keys()) & set(cand_trial_map.keys()))

    n00 = n01 = n10 = n11 = 0
    for k in paired_keys:
        b = base_trial_map[k]
        c = cand_trial_map[k]
        if b == 0 and c == 0:
            n00 += 1
        elif b == 0 and c == 1:
            n01 += 1
        elif b == 1 and c == 0:
            n10 += 1
        elif b == 1 and c == 1:
            n11 += 1

    discordant = n01 + n10
    if discordant == 0:
        p_val_mcnemar = 1.0
    elif discordant < 25:
        p_val_mcnemar = min(1.0, float(2.0 * stats.binom.cdf(min(n01, n10), discordant, 0.5)))
    else:
        chi2 = ((abs(n01 - n10) - 1.0) ** 2) / discordant
        p_val_mcnemar = float(1.0 - stats.chi2.cdf(chi2, df=1))

    # Bootstrap CIs for individual runs
    from agentgate.stats.metrics import bootstrap_ci_pass, compute_pass_metrics
    filtered_base_outcomes = {tid: base_by_task[tid] for tid in shared_tasks}
    filtered_cand_outcomes = {tid: cand_by_task[tid] for tid in shared_tasks}

    _, base_pk = compute_pass_metrics(filtered_base_outcomes, k=2)
    base_ci_p1, base_ci_pk = bootstrap_ci_pass(filtered_base_outcomes, k=2, num_bootstrap=10000)
    _, cand_pk = compute_pass_metrics(filtered_cand_outcomes, k=2)
    cand_ci_p1, cand_ci_pk = bootstrap_ci_pass(filtered_cand_outcomes, k=2, num_bootstrap=10000)

    # Resource metrics
    base_cost = sum(r["would_have_cost"] for r in shared_base_rows)
    cand_cost = sum(r["would_have_cost"] for r in shared_cand_rows)
    base_tokens = sum(r["total_tokens"] for r in shared_base_rows)
    cand_tokens = sum(r["total_tokens"] for r in shared_cand_rows)
    base_mean_steps = float(np.mean([r["steps"] for r in shared_base_rows]))
    cand_mean_steps = float(np.mean([r["steps"] for r in shared_cand_rows]))

    # Print Header & Delta Table
    console.print(f"\n[bold cyan]=== AgentGate CI Regression Gate ===[/bold cyan]")
    console.print(f"- Baseline:  [bold]{base_id}[/bold] ({len(shared_base_rows)} trials across {N} shared tasks)")
    console.print(f"- Candidate: [bold]{candidate}[/bold] ({len(shared_cand_rows)} trials across {N} shared tasks)")
    console.print(f"- Tolerance: [bold]{tolerance:.1%}[/bold] allowed drop | Alpha: [bold]{alpha}[/bold]")

    table = Table(title=f"Paired Comparison Summary ({N} Shared Tasks)")
    table.add_column("Evaluation Metric", style="bold")
    table.add_column(f"Baseline ({base_id})", justify="right")
    table.add_column(f"Candidate ({candidate})", justify="right")
    table.add_column("Delta (95% Bootstrap CI)", justify="right")

    table.add_row(
        "pass^1 (Task-Level)",
        f"{base_p1:.2%}",
        f"{cand_p1:.2%}",
        f"{delta_p1:+.2%} [{delta_ci[0]:+.2%}, {delta_ci[1]:+.2%}]",
    )
    table.add_row(
        "pass^2 (Task-Level Unbiased)",
        f"{base_pk:.2%}",
        f"{cand_pk:.2%}",
        f"{cand_pk - base_pk:+.2%}",
    )
    table.add_row(
        "Completed Trials",
        f"{len(shared_base_rows)}",
        f"{len(shared_cand_rows)}",
        f"{len(shared_cand_rows) - len(shared_base_rows):+d}",
    )
    table.add_row(
        "Mean Steps / Trial",
        f"{base_mean_steps:.1f}",
        f"{cand_mean_steps:.1f}",
        f"{cand_mean_steps - base_mean_steps:+.1f}",
    )
    table.add_row(
        "Agent steps (DB records)",
        f"{base_calls:,}",
        f"{cand_calls:,}",
        f"{cand_calls - base_calls:+d}",
    )
    if base_api_attempts is not None or cand_api_attempts is not None:
        b_att_str = f"{base_api_attempts:,}" if base_api_attempts is not None else "N/A"
        c_att_str = f"{cand_api_attempts:,}" if cand_api_attempts is not None else "N/A"
        d_att_str = (
            f"{cand_api_attempts - base_api_attempts:+d}"
            if (base_api_attempts is not None and cand_api_attempts is not None)
            else "N/A"
        )
        table.add_row(
            "API attempts (api_calls.jsonl)",
            b_att_str,
            c_att_str,
            d_att_str,
        )
    table.add_row(
        "Total Tokens",
        f"{base_tokens:,}",
        f"{cand_tokens:,}",
        f"{cand_tokens - base_tokens:+,}",
    )
    table.add_row(
        "Would-Have-Cost",
        f"${base_cost:.4f}",
        f"${cand_cost:.4f}",
        f"${cand_cost - base_cost:+.4f}",
    )
    console.print(table)

    # McNemar summary
    mcnemar_table = Table(title="Secondary Test: McNemar Paired Trial Contingency")
    mcnemar_table.add_column("Metric", style="bold")
    mcnemar_table.add_column("Value", justify="right")
    mcnemar_table.add_column("Description")
    mcnemar_table.add_row("Both Passed (n11)", str(n11), "Success maintained in both runs")
    mcnemar_table.add_row("Both Failed (n00)", str(n00), "Failure in both runs")
    mcnemar_table.add_row("Gains (n01)", str(n01), "Baseline failed -> Candidate passed")
    mcnemar_table.add_row("Regressions (n10)", str(n10), "Baseline passed -> Candidate failed")
    mcnemar_table.add_row("McNemar Exact p-value", f"{p_val_mcnemar:.4f}", "Two-tailed binomial on discordant pairs")
    console.print(mcnemar_table)

    # Failed trials audit
    cand_failed = [r for r in shared_cand_rows if not r["success"]]
    if cand_failed:
        fail_table = Table(title=f"Candidate Failed Trials Audit ({len(cand_failed)} total)")
        fail_table.add_column("Task ID", justify="center")
        fail_table.add_column("Trial ID", justify="center")
        fail_table.add_column("Steps", justify="right")
        fail_table.add_column("Reward", justify="right")
        fail_table.add_column("End Reason")

        for r in cand_failed:
            traj = json.loads(r["trajectory_json"]) if r["trajectory_json"] else []
            end_reason = "unknown"
            if traj and traj[-1].get("content") == "###STOP###":
                end_reason = "customer_simulator_stopped (###STOP###)"
            elif r["steps"] >= cand_max_steps:
                end_reason = f"max_steps_reached ({cand_max_steps})"
            elif r["error"]:
                end_reason = f"error: {r['error'][:50]}"
            else:
                end_reason = "agent_completed_goal_not_satisfied"

            fail_table.add_row(
                str(r["task_id"]),
                str(r["trial_id"]),
                str(r["steps"]),
                f"{r['reward']:.1f}",
                end_reason,
            )
        console.print(fail_table)

    # Gating Decision Logic (rounded to 1e-9 to prevent floating-point boundary errors)
    is_significant_regression = round(delta_ci[1], 9) < 0.0
    exceeds_tolerance = round(delta_p1, 9) < round(-tolerance, 9)

    if is_significant_regression:
        verdict_str = f"REGRESSION DETECTED: 95% Bootstrap CI [{delta_ci[0]:+.2%}, {delta_ci[1]:+.2%}] is entirely below zero."
        console.print(Panel(f"[bold red][FAILED] REGRESSION DETECTED[/bold red]\n{verdict_str}", title="Gate Decision", border_style="red"))
        sys.exit(1)
    elif exceeds_tolerance:
        verdict_str = f"TOLERANCE EXCEEDED: Delta pass^1 ({delta_p1:+.2%}) dropped by more than {tolerance:.1%} tolerance threshold."
        console.print(Panel(f"[bold red][FAILED] TOLERANCE EXCEEDED[/bold red]\n{verdict_str}", title="Gate Decision", border_style="red"))
        sys.exit(1)
    else:
        verdict_str = f"SUCCESS: Candidate meets gating criteria. Delta: {delta_p1:+.2%}, 95% CI: [{delta_ci[0]:+.2%}, {delta_ci[1]:+.2%}]."
        console.print(Panel(f"[bold green][PASSED] GATE CRITERIA MET[/bold green]\n{verdict_str}", title="Gate Decision", border_style="green"))
        sys.exit(0)


@app.command("cost-estimate")
def cost_estimate(
    num_tasks: int = typer.Option(20, "--num-tasks", "-n", help="Number of tasks"),
    trials: int = typer.Option(2, "--trials", "-k", help="Trials per task"),
    model: str = typer.Option("gemini/gemini-3.1-flash-lite", "--model", "-m", help="Agent model"),
    pricing_path: Optional[str] = typer.Option(None, "--pricing-path", help="pricing.yaml path"),
) -> None:
    """Estimates the dollar would-have-cost of a planned run before execution."""
    pricing = PricingRegistry(pricing_path)
    total_trials = num_tasks * trials

    # Averages from pilot: ~45,000 prompt tokens and ~1,000 completion tokens per trial
    est_prompt_tokens = total_trials * 45_000
    est_completion_tokens = total_trials * 1_000
    cost = pricing.calculate_would_have_cost(model, est_prompt_tokens, est_completion_tokens)

    table = Table(title="Pre-Run Cost & Quota Estimation")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Model", model)
    table.add_row("Total Tasks", str(num_tasks))
    table.add_row("Trials per Task (k)", str(trials))
    table.add_row("Total Task-Trials", str(total_trials))
    table.add_row("Est. Prompt Tokens", f"{est_prompt_tokens:,}")
    table.add_row("Est. Completion Tokens", f"{est_completion_tokens:,}")
    table.add_row("Actual Cash Spend", "$0.00 (Free Tier)")
    table.add_row("Commercial 'Would-Have-Cost'", f"${cost:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
