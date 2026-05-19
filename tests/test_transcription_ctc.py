from __future__ import annotations

import json
from pathlib import Path

from easy_xtts_trainer.transcription.ctc import (
    build_ctc_command,
    build_source_text_candidates,
    extract_whisper_query_words,
    map_ctc_language,
    score_ctc_alignment,
    should_use_ctc_result,
)


def test_map_ctc_language_uses_known_mapping() -> None:
    assert map_ctc_language("en") == "eng"
    assert map_ctc_language("nl") == "nld"


def test_map_ctc_language_falls_back_to_input() -> None:
    assert map_ctc_language("de") == "de"


def test_build_ctc_command_includes_alignment_model_when_present() -> None:
    command = build_ctc_command(
        audio_path=Path("audio.wav"),
        text_path=Path("text.txt"),
        ctc_language="eng",
        align_model="custom-model",
    )

    assert command[:2] == ["ctc-forced-aligner", "--audio_path"]
    assert "--alignment_model" in command
    assert "custom-model" in command


def test_extract_whisper_query_words_reads_word_segments(tmp_path: Path) -> None:
    whisper_json = tmp_path / "transcription.json"
    whisper_json.write_text(
        json.dumps(
            {
                "word_segments": [
                    {"word": "Hello,"},
                    {"word": "Captain"},
                    {"word": "Nemo."},
                ]
            }
        ),
        encoding="utf-8",
    )

    words = extract_whisper_query_words(whisper_json)

    assert words == ["hello", "captain", "nemo"]


def test_build_source_text_candidates_returns_matching_book_window() -> None:
    source_text = (
        "Opening material that does not match. "
        "Captain Nemo looked across the sea and spoke quietly to his crew about the voyage. "
        "Additional chapter text follows after the matching passage."
    )
    query_words = ["captain", "nemo", "spoke", "quietly", "crew", "voyage"]

    candidates = build_source_text_candidates(source_text, query_words, top_k=2)

    assert candidates
    assert "Captain Nemo looked across the sea" in candidates[0].text
    assert candidates[0].retrieval_score > 0


def test_score_ctc_alignment_averages_segment_scores() -> None:
    score = score_ctc_alignment(
        {
            "segments": [
                {"text": "hello", "score": 0.4},
                {"text": "world", "score": 0.8},
            ]
        }
    )

    assert 0.59 <= score <= 0.61


def test_should_use_ctc_result_applies_confidence_gate() -> None:
    assert should_use_ctc_result(0.7, 0.8, min_alignment_score=0.5, min_word_coverage=0.35)
    assert not should_use_ctc_result(0.2, 0.8, min_alignment_score=0.5, min_word_coverage=0.35)
    assert not should_use_ctc_result(0.7, 0.1, min_alignment_score=0.5, min_word_coverage=0.35)
