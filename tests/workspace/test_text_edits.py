"""Exact-text edits change only the matched span of the original bytes."""

import pytest

from mikoshi.tools.text_tools import EditError, apply_edits


def edit(content, old, new):
    return apply_edits(content, [{"old_text": old, "new_text": new}], "file.txt")


@pytest.mark.parametrize(
    ("content", "old", "new", "expected"),
    [
        ("a\r\nb\nc\n", "b", "B", "a\r\nB\nc\n"),
        ("a\nb\r\nc\n", "a", "A", "A\nb\r\nc\n"),
        ("progress\r50%\nx\n", "x", "y", "progress\r50%\ny\n"),
        ("﻿a\r\nb\r\n", "a\nb", "a\nB", "﻿a\r\nB\r\n"),
    ],
)
def test_line_endings_outside_the_match_are_preserved(content, old, new, expected):
    assert edit(content, old, new) == expected


def test_replacement_lines_use_the_files_first_line_ending():
    assert edit("one\r\ntwo\n", "one", "1\n2") == "1\r\n2\r\ntwo\n"


def test_fuzzy_match_maps_back_to_the_original_span():
    assert edit("say “hi”\r\nend\n", 'say "hi"', "say bye") == "say bye\r\nend\n"


def test_an_edit_that_changes_nothing_is_rejected():
    with pytest.raises(EditError, match="identical"):
        edit("a\r\nb\r\n", "a\nb", "a\nb")
