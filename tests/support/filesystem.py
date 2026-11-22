"""Filesystem capabilities used only by tests that exercise those features."""

import os

import pytest


def symlink_or_skip(target, link, *, target_is_directory=False):
    """Skip symlink coverage when an unprivileged Windows host cannot create links."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("symbolic links require Windows Developer Mode or link privileges")
        raise
