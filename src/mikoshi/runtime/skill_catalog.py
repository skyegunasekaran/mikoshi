"""Bounded skill metadata discovery; skill bodies remain explicit file reads."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from mikoshi.tools.workspace import WorkspaceAccessError, WorkspaceChangedError

PATTERNS = (".agents/skills/*/SKILL.md", ".codex/skills/*/SKILL.md")
PLUGIN_PATTERN = ".agents/plugins/*/skills/*/SKILL.md"


def _read(workspace, actor, path, warnings, limit=12000):
    try:
        return workspace.read_text(actor, path, limit)["content"]
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, WorkspaceAccessError, WorkspaceChangedError) as error:
        warnings.append({"path": path, "error": str(error)})
        return None


def _config(workspace, actor, warnings):
    configs = []
    content = _read(workspace, actor, ".codex/config.toml", warnings)
    if content is not None:
        try:
            configs.append((".codex/config.toml", tomllib.loads(content)))
        except ValueError as error:
            warnings.append({"path": ".codex/config.toml", "error": str(error)})
    plugins, disabled = {}, set()
    for source, config in configs:
        plugin_settings = config.get("plugins", {})
        if not isinstance(plugin_settings, dict):
            warnings.append({"path": source, "error": "plugins must be a table"})
            plugin_settings = {}
        for name, options in plugin_settings.items():
            if isinstance(options, dict) and isinstance(options.get("enabled"), bool):
                plugins[name.split("@", 1)[0]] = options["enabled"]
        skills = config.get("skills", {})
        settings = skills.get("config", []) if isinstance(skills, dict) else []
        if not isinstance(settings, list):
            warnings.append({"path": source, "error": "skills.config must be an array"})
            settings = []
        for setting in settings:
            if isinstance(setting, dict) and setting.get("enabled") is False:
                path = setting.get("path")
                if isinstance(path, str):
                    candidate = Path(path).expanduser()
                    if not candidate.is_absolute():
                        candidate = workspace.canonical_repository / candidate
                    disabled.add(str(candidate.resolve()))
    return plugins, disabled, [path for path, _ in configs]


def _scalar(header, key):
    match = re.search(rf"(?m)^{key}:[ \t]*([^\n]*(?:\n[ \t]+[^\n]+)*)", header)
    if not match:
        return ""
    value = match[1].strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            pass
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return " ".join(value.lstrip(">|+-").split())


def _discover(workspace, actor):
    warnings = []
    plugins, disabled, config_sources = _config(workspace, actor, warnings)
    root = workspace.canonical_repository
    entries, manifests = [], {}
    for pattern in (*PATTERNS, PLUGIN_PATTERN):
        for file in sorted(root.glob(pattern)):
            path = file.relative_to(root).as_posix()
            if str(file.resolve()) in disabled or str(file.parent.resolve()) in disabled:
                continue
            plugin = None
            if pattern == PLUGIN_PATTERN:
                directory = file.parents[2]
                if directory not in manifests:
                    manifest_path = (
                        (directory / ".codex-plugin/plugin.json").relative_to(root).as_posix()
                    )
                    content = _read(workspace, actor, manifest_path, warnings)
                    try:
                        manifest = json.loads(content) if content else {}
                        name = manifest.get("name") if isinstance(manifest, dict) else None
                        manifests[directory] = name if isinstance(name, str) else None
                    except ValueError as error:
                        warnings.append({"path": manifest_path, "error": str(error)})
                        manifests[directory] = None
                plugin = manifests[directory]
                if not plugins.get(plugin, False):
                    continue
            content = _read(workspace, actor, path, warnings)
            if content is None:
                continue
            header = content.split("---", 2)[1] if content.startswith("---\n") else ""
            policy_path = (file.parent / "agents/openai.yaml").relative_to(root).as_posix()
            policy = _read(workspace, actor, policy_path, warnings) or ""
            policy_block = re.search(r"(?m)^policy:\s*\n((?:[ \t]+[^\n]*\n?)*)", policy)
            explicit_only = bool(
                policy_block
                and re.search(
                    r"(?m)^\s+allow_implicit_invocation:\s*false\s*(?:#.*)?$", policy_block[1]
                )
            )
            item = {
                "name": _scalar(header, "name") or file.parent.name,
                "description": _scalar(header, "description"),
                "path": path,
                "allow_implicit_invocation": not explicit_only,
            }
            if plugin:
                item["plugin"] = plugin
            entries.append(item)
    return entries, warnings, config_sources


def build_skill_catalog(workspace, actor, context_window_tokens=None, count_tokens=None):
    """Expose descriptions upfront, shortening them before omitting whole entries.

    Without a tokenizer a byte ceiling is deliberately conservative, rather than
    claiming an estimated token count is an exact two-percent token budget.
    """
    entries, warnings, sources = _discover(workspace, actor)
    known_window = isinstance(context_window_tokens, int) and context_window_tokens > 0
    budget = max(1, context_window_tokens // 50) if known_window else 8000
    measured = callable(count_tokens) and isinstance(count_tokens("[]"), int)
    unit = "tokens" if known_window and measured else "utf8_bytes" if known_window else "characters"

    def size(items):
        content = json.dumps(items, ensure_ascii=False, sort_keys=True)
        if unit == "tokens":
            count = count_tokens(content)
            if not isinstance(count, int):
                raise ValueError("token counter stopped returning token counts")
            return count
        return len(content.encode("utf-8")) if unit == "utf8_bytes" else len(content)

    def capped(length):
        return [{**item, "description": item["description"][:length]} for item in entries]

    selected = entries
    if size(selected) > budget:
        low, high = 0, max((len(item["description"]) for item in entries), default=0)
        while low < high:
            middle = (low + high + 1) // 2
            if size(capped(middle)) <= budget:
                low = middle
            else:
                high = middle - 1
        selected = capped(low)
        while selected and size(selected) > budget:
            selected.pop()
    return {
        "skills": selected,
        "total": len(entries),
        "omitted": len(entries) - len(selected),
        "shortened_descriptions": sum(
            len(item["description"]) < len(original["description"])
            for item, original in zip(selected, entries)
        ),
        "budget": budget,
        "budget_unit": unit,
        "used": size(selected),
        "config_sources": sources,
        "warnings": warnings[:10],
        "warning_count": len(warnings),
        "instruction": "Each entry has a skill's description and the path to its SKILL.md. "
        "Skills with allow_implicit_invocation=false apply only when explicitly requested. "
        "A listed skill doesn't install plugins or add tools.",
    }
