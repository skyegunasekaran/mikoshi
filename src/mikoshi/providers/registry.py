"""Provider selection for the CLI, local setup checks and repeated trials."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Callable, Mapping

from mikoshi.decisions import DecisionProvider
from mikoshi.extensions import extension_names, load_extension
from mikoshi.providers.anthropic import AnthropicDecisionProvider
from mikoshi.providers.base import close_provider
from mikoshi.providers.google import GoogleDecisionProvider
from mikoshi.providers.openai import OpenAIDecisionProvider
from mikoshi.providers.settings import ProviderSettings

BUILTINS = {
    "openai": OpenAIDecisionProvider,
    "anthropic": AnthropicDecisionProvider,
    "google": GoogleDecisionProvider,
}
GROUP = "mikoshi.providers"


def provider_names():
    return extension_names(GROUP, BUILTINS)


def provider_adapter(name):
    adapter = load_extension(GROUP, name, BUILTINS)
    if not isinstance(adapter, type) or not issubclass(adapter, ProviderSettings):
        raise ValueError(f"provider {name!r} must be a ProviderSettings adapter class")
    return adapter


def build_decision_provider(
    args: argparse.Namespace,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> DecisionProvider:
    agent_models = args.provider_deployment.agent_models
    if agent_models:
        from mikoshi.providers.routing import RoutedDecisionProvider

        providers = {}
        try:
            for model in dict.fromkeys(agent_models.values()):
                child_args = argparse.Namespace(**vars(args))
                child_args.model = model
                child_args.model_limits = args.agent_model_limits[model]
                child_args.provider_deployment = replace(args.provider_deployment, agent_models={})
                providers[model] = build_decision_provider(child_args, progress_callback)
            return RoutedDecisionProvider(
                next(iter(providers.values())),
                providers,
                agent_models,
                concurrency=args.provider_concurrency,
            )
        except BaseException:
            for provider in providers.values():
                close_provider(provider)
            raise
    provider_type = provider_adapter(args.provider)
    return provider_type(
        model_id=args.model,
        max_tokens=args.model_limits.actor_output_ceiling,
        context_window_tokens=args.model_limits.configured_context_window,
        compaction_input_tokens=args.model_limits.compaction_trigger,
        concurrency=args.provider_concurrency,
        progress_callback=progress_callback,
        reasoning_effort=args.provider_reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        tokenizer_path=args.provider_deployment.tokenizer_path,
        budget=getattr(args, "execution_budget", None),
        **provider_type.deployment_options(args.provider_deployment, args.model_limits),
    )
