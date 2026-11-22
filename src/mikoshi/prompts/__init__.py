"""Shared model-facing guidance, shipped with the runtime as an editable document."""

from importlib.resources import files

WORKER_ENVIRONMENT = (
    files("mikoshi.prompts").joinpath("worker_environment.md").read_text(encoding="utf-8").strip()
)


def worker_environment(*, disable_yield=False):
    """Describe the operator's completion policy without conflicting yield advice."""
    if not disable_yield:
        return WORKER_ENVIRONMENT
    before, section = WORKER_ENVIRONMENT.split("## Activations and yielding\n", 1)
    _, after = section.split("## Observations\n", 1)
    return (
        before
        + (
            "## Activations\n\n"
            "Voluntary yielding is disabled for this run. The peer_yield tool is unavailable. "
            "Each activation continues through tool calls until the time or spending limit "
            "runs out, the operator stops the run, or execution fails. "
            "Record progress and remaining work in the shared pages while continuing work "
            "toward the objective.\n\n## Observations\n"
        )
        + after
    )
