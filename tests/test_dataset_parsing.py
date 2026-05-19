from __future__ import annotations

import json
from pathlib import Path

from easy_xtts_trainer.config import DatasetRuntimeConfig
from easy_xtts_trainer.dataset.parsing import parse_transcription


def _dataset_config(**overrides: object) -> DatasetRuntimeConfig:
    values: dict[str, object] = {
        "input_path": "input",
        "sample_rate": 22050,
        "breath": False,
        "sample_method": "punctuation-only",
        "method_proportion": 0.6,
        "max_audio_time": 11.0,
        "max_text_length": 200,
        "negative_offset_last_word": 50,
        "min_gap": None,
        "discard_abrupt": False,
        "normalize": None,
        "dess": False,
        "denoise": False,
        "compress": None,
        "fade_in": 30,
        "fade_out": 40,
        "trim": False,
        "training_proportion": 0.8,
        "prepare_dataset": False,
    }
    values.update(overrides)
    return DatasetRuntimeConfig(**values)


def test_parse_transcription_appends_segments_from_processor(tmp_path: Path) -> None:
    json_file = tmp_path / "sample.json"
    json_file.write_text(
        json.dumps(
            {
                "word_segments": [
                    {"word": "Hello", "start": 0.0, "end": 1.2},
                    {"word": "world.", "start": 1.2, "end": 2.6},
                ]
            }
        ),
        encoding="utf-8",
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    audio_file = tmp_path / "audio.wav"
    session_data = {"qualifying_segments": []}
    created_processors: list[StubProcessor] = []

    class StubProcessor:
        def __init__(self, target_sr: int) -> None:
            self.target_sr = target_sr
            self.calls: list[tuple[str, list[dict[str, object]], Path, DatasetRuntimeConfig]] = []

        def process_segments(
            self,
            audio_path: str,
            segments: list[dict[str, object]],
            output_dir: Path,
            dataset_config: DatasetRuntimeConfig,
        ) -> list[dict[str, str]]:
            self.calls.append((audio_path, segments, output_dir, dataset_config))
            return [
                {
                    "audio_file": str(output_dir / "segment_001.wav"),
                    "text": str(segments[0]["text"]),
                    "speaker_name": "coqui",
                }
            ]

    def processor_factory(target_sr: int) -> StubProcessor:
        processor = StubProcessor(target_sr)
        created_processors.append(processor)
        return processor

    config = _dataset_config(sample_method="mixed")
    segments_count = parse_transcription(
        json_file=json_file,
        audio_file=audio_file,
        processed_dir=processed_dir,
        session_data=session_data,
        dataset_config=config,
        processor_factory=processor_factory,
    )

    assert segments_count == 1
    assert len(session_data["qualifying_segments"]) == 1
    assert "Hello world." in session_data["qualifying_segments"][0]["text"]
    assert len(created_processors) == 1
    assert created_processors[0].target_sr == 22050
    assert created_processors[0].calls[0][0] == str(audio_file)


def test_parse_transcription_skips_processor_when_no_pending_segments(tmp_path: Path) -> None:
    json_file = tmp_path / "sample.json"
    json_file.write_text(
        json.dumps(
            {
                "word_segments": [
                    {"word": "Hello", "start": 0.0, "end": 1.2},
                    {"word": "world", "start": 1.2, "end": 2.6},
                ]
            }
        ),
        encoding="utf-8",
    )

    call_count = 0

    class StubProcessor:
        def process_segments(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return []

    config = _dataset_config(sample_method="mixed")
    segments_count = parse_transcription(
        json_file=json_file,
        audio_file=tmp_path / "audio.wav",
        processed_dir=tmp_path / "processed",
        session_data={"qualifying_segments": []},
        dataset_config=config,
        processor_factory=lambda _: StubProcessor(),
    )

    assert segments_count == 0
    assert call_count == 0
