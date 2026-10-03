"""Unit tests for QuotaPacer and per-model pacing logic."""

import asyncio
from pathlib import Path

import pytest

from agentgate.runner.pacer import ModelQuotaConfig, QuotaExceededError, QuotaPacer


@pytest.mark.asyncio
async def test_pacer_pacing_delay(tmp_path: Path) -> None:
    pacer = QuotaPacer(state_path=tmp_path / "quota_state.json")
    # Set a tiny interval for testing
    pacer.model_configs["test/fast-model"] = ModelQuotaConfig(
        model="test/fast-model",
        rpm_limit=600,
        min_request_interval_seconds=0.05,
        daily_request_cap=10,
    )

    t0 = asyncio.get_event_loop().time()
    await pacer.acquire("test/fast-model")
    await pacer.acquire("test/fast-model")
    t1 = asyncio.get_event_loop().time()

    assert t1 - t0 >= 0.045
    assert pacer.get_used_quota("test/fast-model") == 2


@pytest.mark.asyncio
async def test_pacer_daily_quota_exceeded(tmp_path: Path) -> None:
    pacer = QuotaPacer(state_path=tmp_path / "quota_state.json")
    pacer.model_configs["test/capped-model"] = ModelQuotaConfig(
        model="test/capped-model",
        rpm_limit=600,
        min_request_interval_seconds=0.001,
        daily_request_cap=2,
    )

    await pacer.acquire("test/capped-model")
    await pacer.acquire("test/capped-model")

    with pytest.raises(QuotaExceededError) as exc_info:
        await pacer.acquire("test/capped-model")

    assert "exceeded daily request cap" in str(exc_info.value)
