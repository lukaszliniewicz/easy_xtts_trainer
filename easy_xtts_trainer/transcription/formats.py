from __future__ import annotations


def convert_ctc_to_whisperx_format(ctc_data: dict) -> dict:
    """Convert CTC forced aligner output into WhisperX-compatible JSON shape."""
    whisperx_format = {"word_segments": []}

    for segment in ctc_data["segments"]:
        whisperx_format["word_segments"].append(
            {
                "word": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "score": segment["score"],
            }
        )

    whisperx_format["text"] = " ".join(segment["text"] for segment in ctc_data["segments"])
    return whisperx_format
