"""Per-model request pacer and daily quota manager for Google AI Studio Free Tier."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

import pytz
import yaml

logger = logging.getLogger("agentgate.pacer")

T = TypeVar("T")


class QuotaExceededError(Exception):
    """Raised when a model has reached its daily request ceiling."""

    def __init__(self, model: str, daily_cap: int, resets_at: str) -> None:
        super().__init__(
            f"Model '{model}' exceeded daily request cap ({daily_cap}). "
            f"Quota resets at midnight Pacific Time ({resets_at})."
        )
        self.model = model
        self.daily_cap = daily_cap
        self.resets_at = resets_at


class ModelQuotaConfig:
    def __init__(
        self,
        model: str,
        rpm_limit: float = 15.0,
        min_request_interval_seconds: float = 4.5,
        daily_request_cap: int = 450,
        tpm_limit: Optional[int] = None,
        rpd_limit: Optional[int] = None,
    ) -> None:
        self.model = model
        self.rpm_limit = rpm_limit
        self.min_request_interval_seconds = min_request_interval_seconds
        self.daily_request_cap = daily_request_cap
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit


class QuotaPacer:
    """Tracks per-model request pacing and daily quotas resetting at midnight Pacific Time."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        state_path: Optional[str | Path] = None,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "quotas.yaml"
        self.config_path = Path(config_path)

        if state_path is None:
            state_path = Path("results/quota_state.json")
        self.state_path = Path(state_path)
        self.call_log_path = Path("results/api_calls.jsonl")

        self.timezone_name = "America/Los_Angeles"
        self.tz = pytz.timezone(self.timezone_name)
        self.model_configs: Dict[str, ModelQuotaConfig] = {}
        self.default_config = ModelQuotaConfig(
            model="default",
            rpm_limit=10.0,
            min_request_interval_seconds=6.0,
            daily_request_cap=300,
        )

        # In-memory tracking
        self._last_call_times: Dict[str, float] = {}
        self._daily_counts: Dict[str, int] = {}
        self._retry_counts: Dict[str, int] = {}
        self._current_date_pt: str = self.get_pacific_date_str()
        self._lock = asyncio.Lock()

        self.load_config()
        self.load_state()

    def log_call(
        self,
        model: str,
        attempt: int,
        outcome: str,
        duration_seconds: float,
        error: Optional[str] = None,
    ) -> None:
        """Appends a structured record for each individual API call attempt for AI Studio audit."""
        try:
            self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(pytz.utc).isoformat(),
                "model": model,
                "attempt": attempt,
                "outcome": outcome,
                "duration_seconds": round(duration_seconds, 3),
                "error": error[:300] if error else None,
            }
            with open(self.call_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("Failed to write to API call audit log: %s", e)

    def get_retry_count(self, model: Optional[str] = None) -> int:
        """Returns total retries executed across all models or for a specific model."""
        if model:
            return self._retry_counts.get(model, 0)
        return sum(self._retry_counts.values())

    def get_pacific_date_str(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def get_next_reset_str(self) -> str:
        now_pt = datetime.now(self.tz)
        tomorrow_pt = now_pt.replace(hour=23, minute=59, second=59, microsecond=999999)
        diff = tomorrow_pt - now_pt
        hours, remainder = divmod(int(diff.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def load_config(self) -> None:
        if not self.config_path.exists():
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.timezone_name = data.get("timezone", "America/Los_Angeles")
        self.tz = pytz.timezone(self.timezone_name)

        models_data = data.get("models", {})
        for model_name, info in models_data.items():
            self.model_configs[model_name] = ModelQuotaConfig(
                model=model_name,
                rpm_limit=float(info.get("rpm_limit", 15.0)),
                min_request_interval_seconds=float(info.get("min_request_interval_seconds", 4.5)),
                daily_request_cap=int(info.get("daily_request_cap", 400)),
                tpm_limit=info.get("tpm_limit"),
                rpd_limit=info.get("rpd_limit"),
            )

        defaults = data.get("defaults", {})
        if defaults:
            self.default_config = ModelQuotaConfig(
                model="default",
                rpm_limit=float(defaults.get("rpm_limit", 10.0)),
                min_request_interval_seconds=float(
                    defaults.get("min_request_interval_seconds", 6.0)
                ),
                daily_request_cap=int(defaults.get("daily_request_cap", 200)),
            )

    def load_state(self) -> None:
        current_today = self.get_pacific_date_str()
        if not self.state_path.exists():
            self._daily_counts = {}
            self._current_date_pt = current_today
            self._save_state()
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_date = data.get("current_date_pt", "")
            if saved_date != current_today:
                logger.info(
                    "Pacific midnight passed (saved: %s, now: %s). Resetting daily quotas.",
                    saved_date,
                    current_today,
                )
                self._daily_counts = {}
                self._current_date_pt = current_today
                self._save_state()
            else:
                self._daily_counts = data.get("counts", {})
                self._current_date_pt = saved_date
        except Exception as e:
            logger.warning("Failed to load quota state, initializing fresh: %s", e)
            self._daily_counts = {}
            self._current_date_pt = current_today

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_date_pt": self._current_date_pt,
            "timezone": self.timezone_name,
            "last_updated": datetime.now(self.tz).isoformat(),
            "counts": self._daily_counts,
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_config(self, model: str) -> ModelQuotaConfig:
        if model in self.model_configs:
            return self.model_configs[model]
        # Match by base name
        base_name = model.split("/")[-1]
        for key, config in self.model_configs.items():
            if key.split("/")[-1] == base_name:
                return config
        return self.default_config

    def get_remaining_quota(self, model: str) -> int:
        self._check_and_reset_date()
        config = self.get_config(model)
        used = self._daily_counts.get(model, 0)
        return max(0, config.daily_request_cap - used)

    def get_used_quota(self, model: str) -> int:
        self._check_and_reset_date()
        return self._daily_counts.get(model, 0)

    def _check_and_reset_date(self) -> None:
        today_pt = self.get_pacific_date_str()
        if today_pt != self._current_date_pt:
            logger.info("Resetting daily quota for new Pacific date: %s", today_pt)
            self._daily_counts = {}
            self._current_date_pt = today_pt
            self._save_state()

    async def acquire(self, model: str) -> None:
        """Paces the request and ensures daily quota is not exceeded."""
        async with self._lock:
            self._check_and_reset_date()
            config = self.get_config(model)
            used = self._daily_counts.get(model, 0)

            if used >= config.daily_request_cap:
                resets_in = self.get_next_reset_str()
                raise QuotaExceededError(
                    model=model,
                    daily_cap=config.daily_request_cap,
                    resets_at=f"in {resets_in}",
                )

            # Check pacing
            now = time.time()
            last_call = self._last_call_times.get(model, 0.0)
            elapsed = now - last_call
            required_interval = config.min_request_interval_seconds

            if elapsed < required_interval:
                delay = required_interval - elapsed
                logger.debug("Pacing model %s: sleeping for %.2fs", model, delay)
                await asyncio.sleep(delay)

            # Record call
            self._last_call_times[model] = time.time()
            self._daily_counts[model] = used + 1
            self._save_state()

    async def execute_with_backoff(
        self,
        model: str,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async provider call with pacer acquisition and exponential backoff on 429."""
        backoff_seconds = 6.0
        max_retries = 5

        for attempt in range(1, max_retries + 1):
            await self.acquire(model)
            t_start = time.time()
            try:
                res = await func(*args, **kwargs)
                duration = time.time() - t_start
                self.log_call(
                    model=model,
                    attempt=attempt,
                    outcome="SUCCESS",
                    duration_seconds=duration,
                )
                return res
            except Exception as e:
                duration = time.time() - t_start
                err_msg = str(e)
                self.log_call(
                    model=model,
                    attempt=attempt,
                    outcome="ERROR",
                    duration_seconds=duration,
                    error=err_msg,
                )
                err_lower = err_msg.lower()
                is_retryable = (
                    "429" in err_lower
                    or "503" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "rate limit" in err_lower
                    or "high demand" in err_lower
                    or "unavailable" in err_lower
                    or "internal error" in err_lower
                )

                if is_retryable and attempt < max_retries:
                    self._retry_counts[model] = self._retry_counts.get(model, 0) + 1
                    jitter = random.uniform(0.5, 2.0)
                    sleep_time = backoff_seconds + jitter
                    logger.warning(
                        "Rate limit or transient error (429/503) hit for model %s on attempt %d/%d (total retries: %d). Backing off for %.2fs...",
                        model,
                        attempt,
                        max_retries,
                        self._retry_counts[model],
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    backoff_seconds = min(60.0, backoff_seconds * 2.0)
                else:
                    raise
        raise RuntimeError(f"Exceeded max retries ({max_retries}) for model {model}")
