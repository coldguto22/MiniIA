import pytest

from dante.core.text import normalize_text


@pytest.mark.unit
def test_normalize_text_removes_terminal_sequences_and_controls():
    assert normalize_text("texto\x1b[2D\x1b[K   com\nruído") == "texto com ruído"


@pytest.mark.unit
def test_normalize_text_limits_embedding_input():
    assert len(normalize_text("x" * 50, max_chars=12)) == 12