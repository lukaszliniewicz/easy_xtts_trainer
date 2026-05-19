from __future__ import annotations

from pathlib import Path

from easy_xtts_trainer.audio import preprocessing
from easy_xtts_trainer.config import DatasetRuntimeConfig


def _dataset_config(**overrides: object) -> DatasetRuntimeConfig:
    values: dict[str, object] = {
        "input_path": "input",
        "sample_rate": 22050,
        "breath": False,
        "sample_method": "maximise-punctuation",
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


def test_collect_input_audio_files_filters_supported_extensions(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)

    (input_dir / "sample.mp3").write_bytes(b"a")
    (input_dir / "voice.wav").write_bytes(b"b")
    (nested / "clip.flac").write_bytes(b"c")
    (input_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    discovered = preprocessing.collect_input_audio_files(input_dir)

    assert {path.name for path in discovered} == {"sample.mp3", "voice.wav", "clip.flac"}


def test_process_audio_uses_converter_for_each_discovered_file(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.mp3").write_bytes(b"x")
    (input_dir / "two.ogg").write_bytes(b"y")
    (input_dir / "skip.txt").write_text("not audio", encoding="utf-8")

    converted: list[tuple[str, str, int]] = []

    def fake_convert(input_path: Path, output_path: Path, target_sample_rate: int) -> None:
        converted.append((input_path.name, output_path.name, target_sample_rate))
        output_path.write_bytes(b"wav")

    monkeypatch.setattr(preprocessing, "convert_to_wav", fake_convert)

    session_path = tmp_path / "session"
    config = _dataset_config(sample_rate=44100)
    audio_sources_dir = preprocessing.process_audio(str(input_dir), session_path, config)

    assert audio_sources_dir == session_path / "audio_sources"
    assert {name for name, _, _ in converted} == {"one.mp3", "two.ogg"}
    assert all(sample_rate == 44100 for _, _, sample_rate in converted)
    assert (session_path / "audio_sources" / "one.wav").exists()
    assert (session_path / "audio_sources" / "two.wav").exists()
