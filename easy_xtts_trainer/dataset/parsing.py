from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from easy_xtts_trainer.config import DatasetRuntimeConfig
from easy_xtts_trainer.text.segmentation import is_abbreviation


class SegmentProcessor(Protocol):
    def process_segments(
        self,
        audio_file: str,
        segments: list[dict[str, Any]],
        processed_dir: Path,
        dataset_config: DatasetRuntimeConfig,
    ) -> list[dict[str, Any]]: ...


def parse_transcription(
    json_file: str | Path,
    audio_file: str | Path,
    processed_dir: Path,
    session_data: dict[str, Any],
    dataset_config: DatasetRuntimeConfig,
    *,
    processor_factory: Callable[[int], SegmentProcessor],
) -> int:
    """Parse transcription and extract training segments with clean cuts."""
    with open(json_file, "r", encoding="utf-8") as handle:
        transcription = json.load(handle)

    processor = processor_factory(dataset_config.sample_rate)
    qualifying_segments = session_data["qualifying_segments"]

    # Extract words directly from word_segments
    all_words = transcription.get("word_segments", [])

    # Initialize containers for segment information
    pending_segments = []

    def is_valid_timestamps(word: dict[str, Any]) -> bool:
        """Check if word has valid timestamps."""
        if not all(key in word for key in ["start", "end", "word"]):
            return False
        try:
            float(word["start"])
            float(word["end"])
            return True
        except (ValueError, TypeError):
            return False

    def find_next_valid_break(words: list[dict[str, Any]], start_idx: int) -> int | None:
        """Find next valid punctuation break with timestamps."""
        for index in range(start_idx, len(words)):
            word = words[index]
            if is_valid_timestamps(word) and any(p in word["word"] for p in ".!?;,-"):
                if not is_abbreviation(word["word"], len(word["word"]) - 1):
                    return index
        return None

    def calculate_segment_duration(words: list[dict[str, Any]]) -> float:
        """Calculate duration between first and last word with valid timestamps."""
        first_timestamp = None
        last_timestamp = None

        for word in words:
            if is_valid_timestamps(word):
                if first_timestamp is None:
                    first_timestamp = float(word["start"])
                last_timestamp = float(word["end"])

        if first_timestamp is None or last_timestamp is None:
            return 0

        return last_timestamp - first_timestamp

    def add_pending_segment(words: list[dict[str, Any]], text: str | None = None) -> bool:
        if not words:
            return False

        segment_text = text
        if segment_text is None:
            segment_text = " ".join(
                str(word.get("word", "")).strip() for word in words if str(word.get("word", "")).strip()
            )

        segment_text = segment_text.strip()
        if not segment_text:
            return False

        pending_segments.append({"words": words.copy(), "text": segment_text})
        return True

    def process_maximise_punctuation(words_to_process: list[dict[str, Any]] | None = None) -> int:
        segments_processed = 0
        try:
            words = all_words if words_to_process is None else list(words_to_process)
            current_segment = {
                "words": [],
                "text": "",
                "break_points": [],
            }

            for word in words:
                word_text = str(word.get("word", ""))

                # Always add words without timestamps to current segment.
                if not is_valid_timestamps(word):
                    if current_segment["words"]:
                        current_segment["words"].append(word)
                        if word_text.strip():
                            current_segment["text"] = (
                                f"{current_segment['text']} {word_text}".strip()
                                if current_segment["text"]
                                else word_text.strip()
                            )
                    continue

                # Before adding the next valid word, flush oversized segment if possible.
                if current_segment["words"]:
                    current_duration = calculate_segment_duration(current_segment["words"])
                    if current_duration > dataset_config.max_audio_time:
                        if current_segment["break_points"]:
                            optimal_break = current_segment["break_points"][-1]
                            if add_pending_segment(optimal_break["words"], optimal_break["text"]):
                                segments_processed += 1
                        current_segment = {
                            "words": [],
                            "text": "",
                            "break_points": [],
                        }

                # Add word to current segment.
                current_segment["words"].append(word)
                if word_text.strip():
                    current_segment["text"] = (
                        f"{current_segment['text']} {word_text}".strip()
                        if current_segment["text"]
                        else word_text.strip()
                    )

                duration = calculate_segment_duration(current_segment["words"])
                text_length = len(current_segment["text"])

                # Only consider words with timestamps as break points.
                if any(punctuation in word_text for punctuation in ".!?;,-"):
                    if not is_abbreviation(word_text, len(word_text) - 1):
                        if duration <= dataset_config.max_audio_time and text_length <= dataset_config.max_text_length:
                            current_segment["break_points"].append(
                                {
                                    "words": current_segment["words"].copy(),
                                    "text": current_segment["text"],
                                    "duration": duration,
                                }
                            )

                # Check if we've exceeded limits.
                if duration > dataset_config.max_audio_time or text_length > dataset_config.max_text_length:
                    if current_segment["break_points"]:
                        optimal_break = current_segment["break_points"][-1]
                        if add_pending_segment(optimal_break["words"], optimal_break["text"]):
                            segments_processed += 1

                        break_word_index = len(optimal_break["words"]) - 1
                        current_segment["words"] = current_segment["words"][break_word_index + 1 :]
                        current_segment["text"] = " ".join(
                            str(item.get("word", "")).strip()
                            for item in current_segment["words"]
                            if str(item.get("word", "")).strip()
                        )
                        current_segment["break_points"] = []
                    else:
                        current_segment = {
                            "words": [],
                            "text": "",
                            "break_points": [],
                        }

            # Process any remaining segment.
            if current_segment["words"] and current_segment["break_points"]:
                optimal_break = current_segment["break_points"][-1]
                if add_pending_segment(optimal_break["words"], optimal_break["text"]):
                    segments_processed += 1

            return segments_processed
        except Exception as exc:
            print(f"Error in maximise_punctuation: {str(exc)}")
            return 0

    def process_punctuation_only(words_to_process: list[dict[str, Any]] | None = None) -> int:
        segments_processed = 0
        try:
            words = all_words if words_to_process is None else list(words_to_process)
            current_words: list[dict[str, Any]] = []

            for word in words:
                word_text = str(word.get("word", ""))

                # Always add words without timestamps to current segment.
                if not is_valid_timestamps(word):
                    if current_words:
                        current_words.append(word)
                    continue

                current_words.append(word)

                if len(current_words) >= 2:
                    duration = calculate_segment_duration(current_words)
                    current_text = " ".join(
                        str(item.get("word", "")).strip() for item in current_words if str(item.get("word", "")).strip()
                    ).strip()

                    if duration > dataset_config.max_audio_time or len(current_text) > dataset_config.max_text_length:
                        current_words = [word]
                        continue

                    if duration >= 2.0 and any(punctuation in word_text for punctuation in ".!?;,-"):
                        if not is_abbreviation(word_text, len(word_text) - 1):
                            if add_pending_segment(current_words, current_text):
                                segments_processed += 1
                            current_words = []

            return segments_processed
        except Exception as exc:
            print(f"Error in process_punctuation_only: {str(exc)}")
            return 0

    def process_mixed(method_proportion: float = 0.6) -> int:
        if not all_words:
            return 0

        total_duration = calculate_segment_duration(all_words)
        split_target = total_duration * method_proportion

        # Find best split point
        split_index = None
        best_diff = float("inf")

        for index, word in enumerate(all_words):
            if is_valid_timestamps(word) and any(p in word["word"] for p in ".!?"):
                if not is_abbreviation(word["word"], len(word["word"]) - 1):
                    duration = calculate_segment_duration(all_words[: index + 1])
                    diff = abs(duration - split_target)
                    if diff < best_diff:
                        best_diff = diff
                        split_index = index

        segments_count = 0

        if split_index is not None:
            # Split the words
            first_part = all_words[: split_index + 1]
            second_part = all_words[split_index + 1 :]

            # Process first part with maximise-punctuation
            segments_count += process_maximise_punctuation(first_part)

            # Process second part with punctuation-only
            segments_count += process_punctuation_only(second_part)
        else:
            # If no good split point, use maximise-punctuation for whole file
            segments_count += process_maximise_punctuation()

        return segments_count

    # Main processing logic
    segments_count = 0
    if dataset_config.sample_method == "maximise-punctuation":
        segments_count = process_maximise_punctuation()
    elif dataset_config.sample_method == "punctuation-only":
        segments_count = process_punctuation_only()
    elif dataset_config.sample_method == "mixed":
        method_prop = dataset_config.method_proportion
        segments_count = process_mixed(method_prop)

    # Process all pending segments with optimal cut points
    if pending_segments:
        processed_segments = processor.process_segments(
            str(audio_file),
            pending_segments,
            processed_dir,
            dataset_config,
        )

        # Add processed segments to session data
        qualifying_segments.extend(processed_segments)

    return segments_count
