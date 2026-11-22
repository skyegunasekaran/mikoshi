"""Route fixed peer identities to models while retaining owner-thread accounting."""

from __future__ import annotations

import math
import time

from mikoshi.decisions import dispatcher_for
from mikoshi.providers.base import close_provider


class RoutedDecisionProvider:
    """Multiplex native async handles; the lifecycle enforces shared concurrency."""

    def __init__(self, default, providers, agent_models, *, concurrency):
        self.default = default
        self.providers = dict(providers)
        self.agent_models = dict(agent_models)
        self.concurrency = concurrency
        self.context_window_tokens = min(p.context_window_tokens for p in self.providers.values())
        self.max_tokens = min(p.max_tokens for p in self.providers.values())
        self._dispatchers = {
            model: dispatcher_for(child) for model, child in self.providers.items()
        }
        self._handles = {}
        self._counter = 0

    def __getattr__(self, name):
        return getattr(self.default, name)

    def for_agent(self, agent_id):
        if agent_id not in self.agent_models:
            raise ValueError(f"no model configured for peer {agent_id!r}")
        return self.providers[self.agent_models[agent_id]]

    def set_evidence_callback(self, callback):
        for provider in self.providers.values():
            if callable(getattr(provider, "set_evidence_callback", None)):
                provider.set_evidence_callback(callback)

    def submit_request(self, request):
        model = self.agent_models[request.metadata["agent_id"]]
        provider = self._dispatchers[model]
        native_handle = provider.submit_request(request)
        self._counter += 1
        self._handles[self._counter] = (provider, native_handle)
        return self._counter

    def collect_ready(self, handles=None, timeout_seconds=0):
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")
        selected = list(self._handles) if handles is None else list(handles)
        grouped = {}
        for handle in selected:
            provider, native = self._handles[handle]
            grouped.setdefault(provider, {})[native] = handle
        deadline = time.monotonic() + timeout_seconds
        while True:
            ready = {}
            for provider, mapping in grouped.items():
                ready.update(
                    {
                        mapping[h]: result
                        for h, result in provider.collect_ready(
                            tuple(mapping), timeout_seconds=0
                        ).items()
                    }
                )
            remaining = deadline - time.monotonic()
            if ready or remaining <= 0 or not selected:
                return ready
            time.sleep(min(0.01, remaining))

    def acknowledge_result(self, handle):
        provider, native = self._handles[handle]
        provider.acknowledge_result(native)
        del self._handles[handle]

    def generate_batch(self, requests):
        handles = [self.submit_request(request) for request in requests]
        results = {}
        while len(results) < len(handles):
            results.update(
                self.collect_ready(
                    [handle for handle in handles if handle not in results], timeout_seconds=0.05
                )
            )
        for handle in handles:
            # Interrupted requests remain suspended in the native provider.
            from mikoshi.runtime.budgets import BudgetExhausted

            if not isinstance(results[handle], BudgetExhausted):
                self.acknowledge_result(handle)
        return [results[handle] for handle in handles]

    def cancel_pending(self):
        self._apply_all("cancel_pending")

    def close(self):
        self._apply_all("close")

    def _apply_all(self, method):
        errors = []
        targets = self.providers if method == "close" else self._dispatchers
        for provider in targets.values():
            try:
                if method == "close":
                    close_provider(provider)
                else:
                    provider.cancel_pending()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup(f"routed provider {method} failed", errors)

    @property
    def metrics(self):
        per_model = {
            model: getattr(provider, "metrics", {}) for model, provider in self.providers.items()
        }
        return {
            "by_model": per_model,
            "agent_models": self.agent_models,
            "requests": sum(metrics.get("requests", 0) for metrics in per_model.values()),
            "timing_events": [
                event
                for metrics in per_model.values()
                for event in metrics.get("timing_events", ())
            ],
            "timing_event_count": sum(
                metrics.get("timing_event_count", len(metrics.get("timing_events", ())))
                for metrics in per_model.values()
            ),
            "timing_events_truncated": any(
                metrics.get("timing_events_truncated", False) for metrics in per_model.values()
            ),
        }
