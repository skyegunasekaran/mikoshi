"""Resolve operator-installed extensions by name without importing unselected plugins."""

from importlib.metadata import entry_points


def extension_names(group, builtins):
    return sorted(set(builtins) | {entry.name for entry in entry_points(group=group)})


def load_extension(group, name, builtins):
    matches = list(entry_points(group=group, name=name))
    if not matches and name in builtins:
        return builtins[name]
    if len(matches) == 1 and name not in builtins:
        try:
            return matches[0].load()
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot load {group} extension {name!r}: {error}") from error
    if matches:
        raise ValueError(f"Ambiguous {group} extension {name!r}")
    raise ValueError(f"Unknown {group} extension {name!r}")


def installed_extension(group, name):
    """Package provenance for a selected extension, absent for bundled implementations."""
    matches = list(entry_points(group=group, name=name))
    if not matches:
        return None
    (entry,) = matches
    return {
        "name": name,
        "entry_point": entry.value,
        "distribution": entry.dist.name,
        "version": entry.dist.version,
    }
