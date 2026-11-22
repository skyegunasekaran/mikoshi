"""Launch contract for a named provider. Adapters may override deployment-specific rules."""

import math
import os


class ProviderSettings:
    provider_name: str
    api_key_env: str | None = None
    api_key_aliases: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    default_context_window: int
    default_output_tokens: int
    minimum_compaction_tokens = 1
    max_temperature = 1
    base_url: str
    endpoint: str

    @classmethod
    def environment_api_key(cls):
        return next(
            (
                os.environ[name].strip()
                for name in (cls.api_key_env, *cls.api_key_aliases)
                if name and os.environ.get(name, "").strip()
            ),
            "",
        )

    @classmethod
    def context_window(cls, model_id, configured, input_ceiling):
        return configured or cls.default_context_window

    @classmethod
    def validate_settings(cls, args):
        cls.validate_configuration(args)
        if cls.api_key_env and not cls.environment_api_key():
            names = " or ".join((cls.api_key_env, *cls.api_key_aliases))
            raise ValueError(f"{names} must be set for {cls.provider_name}")

    @classmethod
    def validate_configuration(cls, args):
        """Validate operating choices without reading credentials or contacting a provider."""
        if not args.model or not args.model.strip():
            raise ValueError(f"--model is required for {cls.provider_name}")
        effort = args.provider_reasoning_effort
        if effort is not None and effort not in cls.reasoning_efforts:
            raise ValueError(f"--provider-reasoning-effort is unsupported for {cls.provider_name}")
        for name, maximum in (("temperature", cls.max_temperature), ("top_p", 1)):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= maximum):
                raise ValueError(
                    f"--{name.replace('_', '-')} must be finite and between 0 and {maximum}"
                )

    @classmethod
    def validate_prices(cls, prices):
        """Override if the provider's usage categories constrain prices."""

    @classmethod
    def adapter_options(cls, options):
        """Validate adapter-specific JSON options and return constructor keywords."""
        if options:
            raise ValueError(f"{cls.provider_name} does not support deployment.provider.options")
        return {}

    @classmethod
    def deployment_options(cls, deployment, limits):
        if deployment.openai_base_url is not None:
            raise ValueError("deployment.provider.openai_base_url requires --provider openai")
        return cls.adapter_options(deployment.options)

    @classmethod
    def provenance(cls, provider):
        base = getattr(provider, "base_url", cls.base_url)
        return {"provider_endpoint": f"{base.rstrip('/')}/{cls.endpoint}"}
