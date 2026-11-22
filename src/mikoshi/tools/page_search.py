"""Rank the caller's current, authorized page views with SQLite FTS5.

The short-lived index includes rendered discussions and runtime records. It cannot
outlive a permission change or return stale task state, and needs no index journal.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from typing import Any

PAGE_SIZE = 20
TITLE_CHARS = 200
EXCERPT_CHARS = 400


def search_pages(
    pages: Iterable[Mapping[str, Any]],
    render: Callable[[Mapping[str, Any]], str],
    query: str,
    offset: int,
) -> dict[str, Any]:
    """Search authorized pages, falling back to partial matches only when needed."""
    pages = list(pages)
    if not query.strip():
        raise ValueError("search query must not be blank; use list_pages to browse")

    terms = _query_terms(query)
    if not terms:
        return _result([], 0, offset, "none")

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    try:
        db.execute(
            "CREATE VIRTUAL TABLE search USING fts5("
            "page_id UNINDEXED, revision UNINDEXED, title, body, "
            "tokenize='porter unicode61')"
        )
        db.executemany(
            "INSERT INTO search VALUES (?, ?, ?, ?)",
            ((page["id"], page["revision"], page["title"], render(page)) for page in pages),
        )
        expression = " AND ".join(terms)
        total = db.execute(
            "SELECT count(*) FROM search WHERE search MATCH ?", (expression,)
        ).fetchone()[0]
        mode = "all_terms"
        if not total:
            expression = " OR ".join(terms)
            total = db.execute(
                "SELECT count(*) FROM search WHERE search MATCH ?", (expression,)
            ).fetchone()[0]
            mode = "partial" if total else "none"
        matches = [
            dict(row)
            for row in db.execute(
                "SELECT page_id, revision, title, "
                "snippet(search, -1, '[', ']', '...', 32) AS excerpt "
                "FROM search WHERE search MATCH ? "
                "ORDER BY bm25(search, 0, 0, 5, 1), page_id LIMIT ? OFFSET ?",
                (expression, PAGE_SIZE, offset),
            )
        ]
        for match in matches:
            match["title"] = match["title"][:TITLE_CHARS]
            match["excerpt"] = match["excerpt"][:EXCERPT_CHARS]
        return _result(matches, total, offset, mode)
    finally:
        db.close()


def _query_terms(query: str) -> list[str]:
    """Quote plain-text terms safely while preserving explicitly quoted phrases."""
    terms = []
    for part in re.findall(r'"[^"]*"|[^\s"]+', query):
        # Split identifiers such as command-1 into independent searchable words.
        words = re.findall(r"[^\W_]+", part, flags=re.UNICODE)
        if part.startswith('"') and words:
            terms.append('"' + " ".join(words) + '"')
        else:
            terms.extend('"' + word + '"' for word in words)
    return list(dict.fromkeys(terms))


def _result(matches: list[dict[str, Any]], total: int, offset: int, mode: str) -> dict[str, Any]:
    next_offset = offset + PAGE_SIZE
    return {
        "matches": matches,
        "total": total,
        "next_offset": next_offset if next_offset < total else None,
        "match_mode": mode,
    }
