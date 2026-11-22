"""Prevent one test's process-wide cwd change from affecting later tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def mikoshi_restore_working_directory():
    directory = os.open(".", os.O_RDONLY)
    try:
        yield
    finally:
        try:
            os.fchdir(directory)
        finally:
            os.close(directory)
