"""Check local documentation links and heading anchors without model calls."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"\[[^\]\n]+\]\(([^)\n]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
FENCE = re.compile(r"```.*?```", re.S)


def _body(document: Path) -> str:
    # Fenced examples may intentionally contain hypothetical repository paths.
    return FENCE.sub("", document.read_text(encoding="utf-8"))


def _anchors(document: Path) -> set[str]:
    """Return GitHub-style heading anchors for a Markdown document."""
    anchors, seen = set(), {}
    for heading in HEADING.findall(_body(document)):
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
        text = re.sub(r"[`*]", "", text).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def check_links(root: Path, documents: list[Path]) -> None:
    errors = []
    for document in documents:
        for match in LINK.finditer(_body(document)):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            path = Path(unquote(parsed.path)) if parsed.path else None
            linked = document if path is None else document.parent / path
            if (path is not None and path.is_absolute()) or not linked.exists():
                errors.append(f"{document.relative_to(root)}: invalid local link {target!r}")
            elif (
                parsed.fragment
                and linked.suffix == ".md"
                and unquote(parsed.fragment) not in _anchors(linked)
            ):
                errors.append(f"{document.relative_to(root)}: missing anchor {target!r}")
    if errors:
        raise ValueError("\n".join(errors))


def check_repository(root: Path) -> None:
    documents = [
        root / "README.md",
        root / "EXPERIMENTS.md",
        root / "ARCHITECTURE.md",
        root / "SCHEMA.md",
        root / "AGENTS.md",
        root / "THIRD_PARTY.md",
        root / "tests/README.md",
    ]
    # Examples are tracked but excluded from source distributions, so they may be absent.
    for example in ("examples/programbench/README.md", "examples/programbench/ffmpeg/README.md"):
        if (root / example).exists():
            documents.append(root / example)
    check_links(root, documents)
    print(f"Verified local links in {len(documents)} guides")


if __name__ == "__main__":
    check_repository(Path(__file__).resolve().parents[2])
