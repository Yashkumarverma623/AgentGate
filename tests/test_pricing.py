"""Unit tests for pricing registry and token would-have-cost calculation."""

from agentgate.pricing import PricingRegistry


def test_pricing_calculation() -> None:
    registry = PricingRegistry()
    # Test gemini-3.1-flash-lite: input $0.10/M, output $0.40/M
    # 10,000 prompt tokens = $0.001
    # 1,000 completion tokens = $0.0004
    # Total = $0.0014
    cost = registry.calculate_would_have_cost(
        model="gemini/gemini-3.1-flash-lite",
        prompt_tokens=10000,
        completion_tokens=1000,
    )
    assert abs(cost - 0.0014) < 1e-6


def test_pricing_cached_tokens() -> None:
    registry = PricingRegistry()
    # 10,000 prompt tokens with 4,000 cached:
    # 6,000 uncached @ 0.10/M = $0.0006
    # 4,000 cached @ 0.025/M = $0.0001
    # 1,000 completion @ 0.40/M = $0.0004
    # Total = $0.0011
    cost = registry.calculate_would_have_cost(
        model="gemini/gemini-3.1-flash-lite",
        prompt_tokens=10000,
        completion_tokens=1000,
        cached_tokens=4000,
    )
    assert abs(cost - 0.0011) < 1e-6
