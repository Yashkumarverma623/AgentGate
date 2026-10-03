# AgentGate

AgentGate is an open-source evaluation and regression-testing harness for tool-using LLM agents. Built on Sierra's `tau-bench` retail benchmark, it provides statistical regression gating, trace-based failure taxonomy, and cost-weighted ablation tracking under realistic API constraints.

---

## Reference Baseline Evaluation (`baseline_v1_steps30`, `max_steps=30`)

The canonical reference baseline evaluates `gemini/gemini-3.1-flash-lite` against customer simulator `gemini/gemma-4-26b-a4b-it` across 15 stratified tasks $\times$ $k=2$ (30 trials total) using the tau-bench default step ceiling (`max_steps=30`).

### Empirical Results

| Metric | Point Estimate | 95% Bootstrap CI (10,000 resamples) | Interpretation |
| :--- | :---: | :---: | :--- |
| **$\text{pass}^1$ (Trial-Level)** | **93.33%** | **[83.33%, 100.00%]** | 28 successes out of 30 total trials |
| **$\text{pass}^1$ (Task-Level)** | **93.33%** | **[80.00%, 100.00%]** | Average task success rate over 15 tasks |
| **$\text{pass}^2$ (Task-Level Unbiased)** | **93.33%** | **[80.00%, 100.00%]** | Probability of passing both trials on a task (14/15 tasks) |

* **Step Ceiling:** `max_steps=30` (canonical tau-bench default).
* **Failure Modes:** Exactly 2 failures out of 30 trials (both on Task 34, labeled as an agent failure where required information was not provided before dialogue termination).
* **Cost Accounting:** 2,434,393 total tokens, commercial "would-have-cost" of $0.2530 ($0.00 actual cash spend on Google AI Studio Free Tier).

---

## Ablation B: Prompt System Degradation (`ablB_v3`)

Ablation B evaluates the candidate prompt `v3` ("degraded multi-factor: confirmation, authentication, policy strictness, domain rules removed") against the identical 10-task slice of the reference baseline `baseline_v1_steps30` ($k=2$, 20 trials total, `max_steps=30`):

| Evaluation Slice | $\text{pass}^1$ (Trial-Level) | $\text{pass}^1$ (Task-Level) | $\text{pass}^2$ (Unbiased) | Paired Bootstrap 95% CI | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Slice (`v1`, 10 tasks)** | 100.00% (20/20) | 100.00% | 100.00% | — | Reference Baseline |
| **Candidate `ablB_v3` (10 tasks)** | 85.00% (17/20) | 85.00% | 70.00% | [-30.00%, +0.00%] | **NO_SIGNIFICANT_CHANGE** |

*Note: At $N=10$ tasks, $\Delta \text{pass}^1 = -15.00\%$ with paired bootstrap 95% CI $[-30.00\%, +0.00\%]$ and McNemar exact $p=0.2500$ (verdict: `NO_SIGNIFICANT_CHANGE`). Failures observed: Task 2 trial 1, Task 31 trial 1, Task 40 trial 1.*

---

## Architecture & Features

- **Statistical Regression Gating:** Paired bootstrap hypothesis testing on task deltas and McNemar's exact test on paired trials.
- **Canonical Step Budget:** Aligned with tau-bench's default `max_steps=30`.
- **Trace-Based Failure Taxonomy:** Deterministic rule evaluation (step limits, tool loops, human transfers) combined with structured LLM-as-a-judge rubrics.
- **Per-Model Quota Pacing:** Leaky-bucket pacer enforcing model-specific RPM intervals and daily request quotas resetting at Pacific midnight.
- **Commercial Would-Have-Cost Accounting:** Accurate token-based pricing registry for tracking ROI and cost regressions.

---

## CI Regression Gate (`agentgate gate`)

The CI gate evaluates pull-request candidates against the canonical reference baseline committed at [`baselines/baseline_v1_steps30.json`](baselines/baseline_v1_steps30.json). CI workflows operate completely offline from git without needing `results/agentgate.db`.

### Decision Rules & Tolerance Thresholds
- **Significant Regression:** Candidate fails immediately if the 95% task-level paired bootstrap CI is strictly negative (`delta_ci[1] < 0.0`).
- **Tolerance Rule:** Candidate fails if $\text{pass}^1$ drops by more than the tolerance threshold (`delta_p1 < -tolerance`).
  - **Full 15-Task Benchmark:** Tolerance is set to **0.10 (10 points)** (each trial is only $3.33\%$).
  - **PR Smoke Subset (5 tasks $\times$ $k=2$):** Tolerance is set to **0.15 (15 points)**. Because each trial in a 10-trial smoke run represents a discrete $10.0\%$ drop, a 0.15 tolerance absorbs a single stochastic trial anomaly ($\Delta = -10.0\%$) while strictly rejecting any multi-trial or full-task regression ($\Delta \le -20.0\%$).
- **Floating-Point Precision:** Boundary comparisons round to $10^{-9}$ so that exact drops (e.g., exactly 1 trial dropping $10.0\%$ under a $10.0\%$ tolerance) are never rejected due to IEEE 754 precision artifacts.
- **Optional Retry Rule for PR Smoke Gate:** If a PR candidate fails on the tolerance rule, re-run the smoke subset once and fail only if both consecutive runs fail.

### Empirical Noise Check & Limitations
An empirical noise check was conducted on the unchanged baseline configuration (`prompt v1`, `gemini-3.1-flash-lite`, `max_steps=30`) across the 5 smoke tasks (`16, 0, 10, 17, 3`) with $k=2$:
- `baseline_v1_steps30` slice: 10/10 passed (100.0%)
- `smoke_rerun_1`: 10/10 passed (100.0%)
- `smoke_rerun_2`: 10/10 passed (100.0%)

*Noise-check limitation note:* Across all 30 trials on these 5 tasks, 0 failures were observed. Because no trial failures occurred under the unchanged configuration, the empirical false-alarm rate is **not measured, only bounded**: the 9.5% figure (via Rule of Three, $3/30$) is the **95% upper bound on the per-trial failure rate (0/30), not the gate's false-alarm probability**.

