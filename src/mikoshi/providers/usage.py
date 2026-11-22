"""Token measurements and normalization of native usage reports for the shared ledger."""

import hashlib
import json
from pathlib import Path


class TokenMeter:
    """Attribute reported and locally measured tokens without conflating them."""

    def __init__(self, tokenizer_path=None):
        self.tokenizer = None
        self.sha256 = None
        if tokenizer_path:
            from tokenizers import Tokenizer

            path = Path(tokenizer_path)
            self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.tokenizer = Tokenizer.from_file(str(path))

    def count(self, content):
        if self.tokenizer is None or not isinstance(content, str):
            return None
        return len(self.tokenizer.encode(content, add_special_tokens=False).ids)

    def measure(self, body):
        usage = body.get("usage") or {}
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        completion_details = usage.get("completion_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        reasoning = completion_details.get(
            "reasoning_tokens",
            output_details.get(
                "reasoning_tokens",
                output_details.get("thinking_tokens", usage.get("reasoning_tokens")),
            ),
        )
        source = "provider" if reasoning is not None else "unavailable"
        if reasoning is None and isinstance(message.get("reasoning_content"), str):
            reasoning = self.count(message["reasoning_content"])
            if reasoning is not None:
                source = "retokenized_reasoning_content"
        tool_calls = message.get("tool_calls")
        final_content = (
            json.dumps(tool_calls, separators=(",", ":"), ensure_ascii=False)
            if tool_calls is not None
            else message.get("content")
        )
        final = self.count(final_content)
        provider_final = (
            usage.get("output_tokens") - reasoning
            if _is_token_count(usage.get("output_tokens"))
            and _is_token_count(reasoning)
            and reasoning <= usage["output_tokens"]
            else None
        )
        if provider_final is not None:
            final = provider_final
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        prompt_details = usage.get("prompt_tokens_details", usage.get("input_tokens_details")) or {}
        cache_read_tokens = prompt_details.get("cached_tokens")
        # Responses reports cache writes inside the inclusive input total;
        # Messages normalizes its disjoint creation total under a separate key.
        cache_creation_tokens = prompt_details.get(
            "cache_creation_tokens", prompt_details.get("cache_write_tokens", 0)
        )
        uncached_input_tokens = (
            prompt_tokens - cache_read_tokens - cache_creation_tokens
            if _is_token_count(prompt_tokens)
            and _is_token_count(cache_read_tokens)
            and _is_token_count(cache_creation_tokens)
            and cache_read_tokens + cache_creation_tokens <= prompt_tokens
            else None
        )
        return {
            "input_tokens": prompt_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "cache_read_tokens": cache_read_tokens if uncached_input_tokens is not None else None,
            "cache_creation_tokens": cache_creation_tokens
            if uncached_input_tokens is not None
            else None,
            "cache_measurement": "provider" if uncached_input_tokens is not None else "unavailable",
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "reasoning_tokens": reasoning,
            "reasoning_measurement": source,
            "final_output_tokens": final,
            "final_output_measurement": "provider"
            if provider_final is not None
            else ("retokenized_tool_calls" if tool_calls is not None else "retokenized_content")
            if final is not None
            else "unavailable",
            "raw_usage": body.get("native_usage", usage),
            "tokenizer_sha256": self.sha256,
        }


def _is_token_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _cache_creation_split(usage):
    created = usage.get("cache_creation_input_tokens", 0)
    if not _is_token_count(created):
        raise ValueError("invalid cache creation usage")
    details = usage.get("cache_creation")
    if details is None:
        return 0, 0, created
    if not isinstance(details, dict):
        raise ValueError("invalid cache creation details")
    five = details.get("ephemeral_5m_input_tokens", 0)
    hour = details.get("ephemeral_1h_input_tokens", 0)
    unknown = usage.get("cache_creation_unattributed_tokens", 0)
    if (
        not all(_is_token_count(n) for n in (five, hour, unknown))
        or five + hour + unknown != created
    ):
        raise ValueError("cache creation breakdown does not match total")
    return five, hour, unknown


def aggregate_message_usage(usage):
    """Include all sampling/compaction iterations and preserve cache TTL attribution."""
    if not isinstance(usage, dict) or "iterations" not in usage:
        return usage
    iterations = usage["iterations"]
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("invalid message usage iterations")
    totals = {
        key: 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    five = hour = unknown = 0
    for iteration in iterations:
        if not isinstance(iteration, dict):
            raise ValueError("invalid message usage iteration")
        for key in totals:
            value = iteration.get(key, 0 if key.startswith("cache_") else None)
            if not _is_token_count(value):
                raise ValueError("invalid message iteration token count")
            totals[key] += value
        part_five, part_hour, part_unknown = _cache_creation_split(iteration)
        five += part_five
        hour += part_hour
        unknown += part_unknown
    # Top-level output details describe the same total only when output counts
    # agree. Otherwise require a complete per-iteration breakdown; missing
    # compaction reasoning must not silently become zero.
    output_details = {}
    if usage.get("output_tokens") == totals["output_tokens"]:
        output_details = usage.get("output_tokens_details") or {}
        if not isinstance(output_details, dict):
            raise ValueError("invalid output token details")
    if not output_details:
        thinking = []
        for iteration in iterations:
            details = iteration.get("output_tokens_details") or {}
            if not isinstance(details, dict):
                raise ValueError("invalid iteration output token details")
            value = details.get("thinking_tokens")
            if value is not None and (
                not _is_token_count(value) or value > iteration["output_tokens"]
            ):
                raise ValueError("invalid iteration thinking token count")
            thinking.append(value)
        if all(_is_token_count(value) for value in thinking):
            output_details = {"thinking_tokens": sum(thinking)}
    return {
        **totals,
        **({"output_tokens_details": output_details} if output_details else {}),
        "cache_creation": {"ephemeral_5m_input_tokens": five, "ephemeral_1h_input_tokens": hour},
        "cache_creation_unattributed_tokens": unknown,
    }


def usage_breakdown(usage, *, cache_ttl=None):
    """Validate provider usage and derive mutually exclusive billing categories."""
    if not isinstance(usage, dict):
        raise ValueError("invalid usage")
    usage = aggregate_message_usage(usage)
    created = creation = five = hour = 0
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        # Messages reports three disjoint input categories.
        cached = usage.get("cache_read_input_tokens", 0)
        created = usage.get("cache_creation_input_tokens", 0)
        plain = usage.get("input_tokens")
        if not all(_is_token_count(n) for n in (plain, cached, created)):
            raise ValueError("invalid cache usage")
        five, hour, unknown = _cache_creation_split(usage)
        if unknown:
            if cache_ttl == "5m":
                five += unknown
            elif cache_ttl == "1h":
                hour += unknown
            else:
                raise ValueError("cache creation TTL is unavailable")
        usage = {
            **usage,
            "prompt_tokens": plain + cached + created,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if not _is_token_count(input_tokens) or not _is_token_count(completion_tokens):
        raise ValueError("invalid usage")

    prompt_details = usage.get("prompt_tokens_details", usage.get("input_tokens_details")) or {}
    if not isinstance(prompt_details, dict):
        raise ValueError("invalid prompt token details")
    # Responses cache writes have a separate price, without Messages TTLs.
    # Omission preserves providers that report only ordinary/cache-hit input.
    creation = prompt_details.get("cache_write_tokens", 0)
    if not _is_token_count(creation):
        raise ValueError("invalid cache write usage")
    created += creation
    cache_read_tokens = prompt_details.get("cached_tokens")
    if cache_read_tokens is not None and (
        not _is_token_count(cache_read_tokens) or cache_read_tokens > input_tokens
    ):
        raise ValueError("invalid cached token usage")
    if created and cache_read_tokens is None:
        raise ValueError("cached token usage is required with cache writes")
    if cache_read_tokens is not None and cache_read_tokens + created > input_tokens:
        raise ValueError("cache usage exceeds input tokens")
    uncached_input_tokens = (
        input_tokens - cache_read_tokens - created if cache_read_tokens is not None else None
    )

    completion_details = usage.get("completion_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    if not isinstance(completion_details, dict) or not isinstance(output_details, dict):
        raise ValueError("invalid completion token details")
    thinking_tokens = completion_details.get(
        "reasoning_tokens",
        output_details.get(
            "reasoning_tokens", output_details.get("thinking_tokens", usage.get("reasoning_tokens"))
        ),
    )
    if thinking_tokens is not None and (
        not _is_token_count(thinking_tokens) or thinking_tokens > completion_tokens
    ):
        raise ValueError("invalid reasoning token usage")
    output_tokens = completion_tokens - thinking_tokens if thinking_tokens is not None else None
    return {
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": creation,
        "cache_creation_5m_tokens": five,
        "cache_creation_1h_tokens": hour,
        "unattributed_input_tokens": input_tokens if cache_read_tokens is None else 0,
        "completion_tokens": completion_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "unattributed_completion_tokens": completion_tokens if thinking_tokens is None else 0,
    }
