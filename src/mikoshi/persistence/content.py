"""Content-addressed JSON values shared by public evidence records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_reference(data):
    digest = hashlib.sha256(data).hexdigest()
    return {
        "content_id": f"sha256:{digest}",
        "path": f"content/{digest[:2]}/{digest}.json",
        "sha256": digest,
        "bytes": len(data),
        "media_type": "application/json",
    }


class ContentStore:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, value):
        data = content_bytes(value)
        reference = content_reference(data)
        path = self.root / reference["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as target:
                target.write(data)
        except FileExistsError:
            read_content(self.root, reference)
        return reference


def read_content(root, reference):
    """Read a referenced JSON value after verifying its identity and byte count."""
    digest = reference.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid content digest")
    expected_path = f"content/{digest[:2]}/{digest}.json"
    if (
        reference.get("path") != expected_path
        or reference.get("content_id") != f"sha256:{digest}"
        or reference.get("media_type") != "application/json"
    ):
        raise ValueError("invalid content reference")
    root = Path(root).resolve()
    path = (root / expected_path).resolve()
    if root not in path.parents:
        raise ValueError("content reference escapes its bundle")
    data = path.read_bytes()
    if len(data) != reference.get("bytes") or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"content fingerprint mismatch: {reference['content_id']}")
    return json.loads(data)
