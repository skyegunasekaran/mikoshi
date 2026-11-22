"""Explicit deployment prices, in dollars per million reported tokens."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TokenPrices:
    """Base rates plus an optional multiplier above an inclusive-input threshold."""

    input_rate: float
    cache_read_rate: float
    output_rate: float
    thinking_rate: float
    cache_creation_rate: float | None = None
    cache_creation_5m_rate: float | None = None
    cache_creation_1h_rate: float | None = None
    long_context_threshold: int | None = None
    long_context_multiplier: float = 1.0

    def __post_init__(self):
        if self.long_context_threshold is not None and (
            type(self.long_context_threshold) is not int or self.long_context_threshold <= 0
        ):
            raise ValueError("pricing.long_context_threshold must be a positive integer or null")
        if (
            type(self.long_context_multiplier) not in (int, float)
            or not math.isfinite(self.long_context_multiplier)
            or self.long_context_multiplier <= 0
        ):
            raise ValueError("pricing.long_context_multiplier must be positive and finite")
        for name, value in asdict(self).items():
            if name in ("long_context_threshold", "long_context_multiplier"):
                continue
            if value is None and name.startswith("cache_creation"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"pricing.{name} must be nonnegative and finite")
        for name, multiplier in (
            ("cache_creation_rate", 1.25),
            ("cache_creation_5m_rate", 1.25),
            ("cache_creation_1h_rate", 2),
        ):
            if getattr(self, name) is None:
                value = self.input_rate * multiplier
                if not math.isfinite(value):
                    raise ValueError(f"pricing.{name} must be finite")
                object.__setattr__(self, name, value)

    def validate_provider(self, provider):
        from mikoshi.providers.registry import provider_adapter

        provider_adapter(provider).validate_prices(self)
