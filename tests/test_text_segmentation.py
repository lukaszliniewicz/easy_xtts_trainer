from __future__ import annotations

from easy_xtts_trainer.text.segmentation import find_real_sentence_end, is_abbreviation


def test_is_abbreviation_detects_common_abbrev() -> None:
    text = "Dr."
    assert is_abbreviation(text, len(text) - 1)


def test_is_abbreviation_rejects_normal_sentence_end() -> None:
    text = "Hello."
    assert not is_abbreviation(text, len(text) - 1)


def test_find_real_sentence_end_out_of_bounds_returns_input() -> None:
    assert find_real_sentence_end("Hello", 99) == 99


def test_find_real_sentence_end_handles_simple_sentence() -> None:
    text = "Hello. Next"
    dot_pos = text.index(".")
    assert find_real_sentence_end(text, dot_pos) == dot_pos + 1
