"""Text truncation and exact-text editing, ported from the Pi coding agent.

Source: badlogic/pi-mono, packages/coding-agent/src/core/tools/{truncate,edit-diff}.ts.
Truncation applies two independent limits, 200 lines and 16 KiB, and whichever is
hit first wins. Edits replace unique, non-overlapping regions of the original text.
Unlike Pi, uniqueness is judged on exact matches first, and a fuzzy fallback match
replaces only its original span, never rewriting normalized text outside it.
"""

from __future__ import annotations

import unicodedata

MAX_LINES = 200
MAX_BYTES = 16 * 1024


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def split_lines(text: str) -> list[str]:
    """Lines without LF/CRLF terminators; a final newline adds no empty line."""
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES):
    """Keep the first complete lines within both limits, like Pi's truncateHead.

    Returns (content, kept_lines, truncated_by). truncated_by is None, "lines" or
    "bytes"; "first_line" means the first line alone exceeds max_bytes.
    """
    lines = split_lines(text)
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return text, len(lines), None
    if lines and len(lines[0].encode()) > max_bytes:
        return "", 0, "first_line"
    kept, size, truncated_by = [], 0, "lines"
    for line in lines[:max_lines]:
        cost = len(line.encode()) + (1 if kept else 0)
        if size + cost > max_bytes:
            truncated_by = "bytes"
            break
        kept.append(line)
        size += cost
    return "\n".join(kept), len(kept), truncated_by


def truncate_tail(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES):
    """Keep the last complete lines within both limits, like Pi's truncateTail.

    Returns (content, kept_lines, truncated_by, last_line_partial). A partial line is
    returned only when the final line alone exceeds max_bytes.
    """
    lines = split_lines(text)
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return text, len(lines), None, False
    kept, size, truncated_by, partial = [], 0, "lines", False
    for line in reversed(lines):
        if len(kept) >= max_lines:
            break
        cost = len(line.encode()) + (1 if kept else 0)
        if size + cost > max_bytes:
            truncated_by = "bytes"
            if not kept:
                kept.append(line.encode()[-max_bytes:].decode("utf-8", errors="ignore"))
                partial = True
            break
        kept.append(line)
        size += cost
    return "\n".join(reversed(kept)), len(kept), truncated_by, partial


# --- Exact-text edits -------------------------------------------------------------


class EditError(ValueError):
    """An edit could not be applied; the message tells the model how to fix it."""


_FUZZY_CHARACTERS = str.maketrans(
    {
        **dict.fromkeys("‘’‚‛", "'"),
        **dict.fromkeys("“”„‟", '"'),
        **dict.fromkeys("‐‑‒–—―−", "-"),
        **dict.fromkeys("            　", " "),
    }
)


def _fuzzy_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text for fallback matching and map each result character to its source.

    Each character is NFKC-normalized, then quotes, dashes and special spaces become
    ASCII; trailing whitespace before a newline or the end is dropped. positions[i] is
    the index in `text` that produced normalized character i, so a match found in the
    normalized text maps back to an exact span of the original.
    """
    chars, positions = [], []
    for index, char in enumerate(text):
        for piece in unicodedata.normalize("NFKC", char).translate(_FUZZY_CHARACTERS):
            chars.append(piece)
            positions.append(index)
    kept_chars, kept_positions, pending = [], [], []
    for char, position in zip(chars, positions):
        if char == "\n":
            pending = []  # Trailing whitespace before a newline is dropped.
        if char != "\n" and char.isspace():
            pending.append((char, position))
            continue
        for item in pending:
            kept_chars.append(item[0])
            kept_positions.append(item[1])
        pending = []
        kept_chars.append(char)
        kept_positions.append(position)
    return "".join(kept_chars), kept_positions


def _occurrences(haystack: str, needle: str) -> list[int]:
    """Start offsets of every occurrence, including overlapping ones."""
    found, start = [], haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def _locate(content: str, old: str, fuzzy: tuple[str, list[int]] | None):
    """Return (start, end, occurrences) of old in content: exact first, fuzzy as fallback."""
    exact = _occurrences(content, old)
    if exact:
        return exact[0], exact[0] + len(old), len(exact)
    normalized, positions = fuzzy
    needle, _ = _fuzzy_with_map(old)
    found = _occurrences(normalized, needle) if needle else []
    if not found:
        return None, None, 0
    start = positions[found[0]]
    end = positions[found[0] + len(needle) - 1] + 1
    return start, end, len(found)


def _newline_map(text: str) -> tuple[str, list[int], list[int]]:
    """LF-only text, plus the original start and end of each normalized character."""
    if "\r" not in text:
        indexes = list(range(len(text)))
        return text, indexes, [index + 1 for index in indexes]
    chars, starts, ends, index = [], [], [], 0
    while index < len(text):
        width = 2 if text.startswith("\r\n", index) else 1
        chars.append("\n" if text[index] == "\r" else text[index])
        starts.append(index)
        ends.append(index + width)
        index += width
    return "".join(chars), starts, ends


def apply_edits(content: str, edits, path: str) -> str:
    """Apply {old_text, new_text} edits, each matched once against the original content.

    Each old_text must match exactly once. Only when it has no exact match is it matched
    again ignoring trailing whitespace and differences in Unicode quotes, dashes, spaces
    and compatibility forms. Either way, only the matched span of the original text is
    replaced; everything outside it keeps its exact bytes, including mixed line endings.
    Replacement text uses the file's first line ending. A UTF-8 BOM is preserved.
    """
    bom = "﻿" if content.startswith("﻿") else ""
    content = content[len(bom) :]
    crlf, lf = content.find("\r\n"), content.find("\n")
    ending = "\r\n" if crlf != -1 and crlf <= lf else "\n"
    normalized, starts, ends = _newline_map(content)
    edits = [
        (
            edit["old_text"].replace("\r\n", "\n").replace("\r", "\n"),
            edit["new_text"].replace("\r\n", "\n").replace("\r", "\n"),
        )
        for edit in edits
    ]
    total = len(edits)

    def name(index):
        return "the text" if total == 1 else f"edits[{index}]"

    fuzzy = None
    matches = []
    for index, (old, new) in enumerate(edits):
        if not old:
            field = "old_text" if total == 1 else f"edits[{index}].old_text"
            raise EditError(f"{field} must not be empty in {path}.")
        if fuzzy is None and old not in normalized:
            fuzzy = _fuzzy_with_map(normalized)
        start, end, occurrences = _locate(normalized, old, fuzzy)
        if occurrences == 0:
            raise EditError(
                f"Could not find {name(index)} in {path}. old_text must match the current "
                "text exactly, including whitespace and newlines."
            )
        if occurrences > 1:
            raise EditError(
                f"Found {occurrences} occurrences of {name(index)} in {path}. The text must "
                "match exactly once; more surrounding context makes it unique."
            )
        matches.append((start, end, new, index))
    matches.sort()
    for previous, current in zip(matches, matches[1:]):
        if previous[1] > current[0]:
            raise EditError(
                f"edits[{previous[3]}] and edits[{current[3]}] overlap in {path}. Overlapping "
                "edits must be combined into one."
            )
    # Splice into the original text so lines outside each match keep their endings.
    updated = content
    for start, end, new, _ in reversed(matches):
        if ending == "\r\n":
            new = new.replace("\n", "\r\n")
        updated = updated[: starts[start]] + new + updated[ends[end - 1] :]
    if updated == content:
        raise EditError(f"No changes made to {path}. The replacement produced identical content.")
    return bom + updated
