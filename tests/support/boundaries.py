"""Host-controlled test boundaries, separate from production resource policy."""

import time


def run_for(runtime, count=None, *, timeout_seconds=15, **kwargs):
    """Pause at a fixture boundary; catch scripted-provider loops instead of hanging pytest."""
    prior_stop = kwargs.pop("stop_requested", None)
    started = time.monotonic()
    baseline = sum(a["status"] != "running" for a in runtime._store.list_activations())

    def stop():
        if prior_stop:
            reason = prior_stop()
            if reason:
                return reason
        activations = list(runtime._store.list_activations())
        if (
            count is not None
            and sum(a["status"] != "running" for a in activations) >= baseline + count
        ):
            return "test_boundary"
        if time.monotonic() - started > timeout_seconds or any(
            (a.get("current_activation_step") or 0) > 80 for a in activations
        ):
            raise AssertionError("scripted provider did not reach its intended test boundary")
        return None

    return runtime.run(stop_requested=stop, **kwargs)
