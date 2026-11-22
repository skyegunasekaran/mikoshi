"""Pure resolution of provider context and generation limits.

The configured context window is an operating limit, not necessarily the model's
capacity. Input, actor output, and compaction are kept separate
so callers do not have to infer one setting's meaning from another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mikoshi.providers.registry import provider_adapter


@dataclass(frozen=True)
class ResolvedModelLimits:
    """Validated per-run operating limits."""

    configured_context_window: int
    actor_output_ceiling: int
    input_ceiling: int
    compaction_trigger: int


def _positive_integer(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def resolve_model_limits(
    provider: str,
    model_id: str,
    *,
    configured_context_window: int | None = None,
    actor_output_ceiling: int | None = None,
    input_ceiling: int | None = None,
    compaction_trigger: int | None = None,
) -> ResolvedModelLimits:
    """Resolve requested limits for a provider deployment without I/O."""

    adapter = provider_adapter(provider)
    for name, value in (
        ("configured context window", configured_context_window),
        ("actor output ceiling", actor_output_ceiling),
        ("input ceiling", input_ceiling),
        ("compaction trigger", compaction_trigger),
    ):
        _positive_integer(value, name)

    output = actor_output_ceiling or adapter.default_output_tokens
    context = adapter.context_window(model_id, configured_context_window, input_ceiling)

    if context <= output:
        raise ValueError("Configured context window must exceed the actor output ceiling")

    allowance = context - output
    effective_input = input_ceiling if input_ceiling is not None else allowance
    if effective_input > allowance:
        raise ValueError("Input ceiling must fit the context minus actor output allowance")

    minimum = adapter.minimum_compaction_tokens
    margin = min(16384, max(1, context // 64))
    trigger = (
        compaction_trigger
        if compaction_trigger is not None
        else min(effective_input, max(1, allowance - margin))
    )
    if not minimum <= trigger <= effective_input:
        raise ValueError(f"Compaction trigger must be between {minimum:,} and the input ceiling")

    return ResolvedModelLimits(
        configured_context_window=context,
        actor_output_ceiling=output,
        input_ceiling=effective_input,
        compaction_trigger=trigger,
    )


@dataclass(frozen=True)
class ProviderDeployment:
    """Deployment choices; model and transport tuning stay with their owners."""

    openai_base_url: str | None = None
    tokenizer_path: str | None = None
    concurrency: int | None = None  # Omitted: one model request slot per peer.
    agent_models: dict[str, str] = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        from mikoshi.providers.openai import normalize_openai_base_url

        if not isinstance(self.options, dict):
            raise ValueError("deployment.provider.options must be an object")
        _positive_integer(self.concurrency, "provider concurrency")
        if not isinstance(self.agent_models, dict) or any(
            not isinstance(peer, str)
            or not peer.strip()
            or not isinstance(model, str)
            or not model.strip()
            for peer, model in self.agent_models.items()
        ):
            raise ValueError("deployment.provider.agent_models must map peer IDs to model IDs")
        if self.openai_base_url is not None:
            object.__setattr__(
                self, "openai_base_url", normalize_openai_base_url(self.openai_base_url)
            )
        if self.tokenizer_path is not None and (
            not isinstance(self.tokenizer_path, str) or not self.tokenizer_path.strip()
        ):
            raise ValueError("deployment.provider.tokenizer_path must be a nonempty path")
