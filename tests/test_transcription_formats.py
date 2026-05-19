from __future__ import annotations

from easy_xtts_trainer.transcription.formats import convert_ctc_to_whisperx_format


def test_convert_ctc_to_whisperx_format_preserves_word_segments() -> None:
    ctc_data = {
        "segments": [
            {"text": "Hello", "start": 0.1, "end": 0.3, "score": 0.95},
            {"text": "world", "start": 0.4, "end": 0.7, "score": 0.91},
        ]
    }

    converted = convert_ctc_to_whisperx_format(ctc_data)

    assert converted["text"] == "Hello world"
    assert converted["word_segments"] == [
        {"word": "Hello", "start": 0.1, "end": 0.3, "score": 0.95},
        {"word": "world", "start": 0.4, "end": 0.7, "score": 0.91},
    ]
