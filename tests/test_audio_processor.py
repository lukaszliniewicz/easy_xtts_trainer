from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from easy_xtts_trainer.audio.processor import AudioProcessor


def test_normalize_applies_true_peak_limit() -> None:
    processor = AudioProcessor(target_sr=22050)
    audio = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

    normalized = processor._normalize(audio, target_lufs=0.0)

    assert np.max(np.abs(normalized)) <= 0.99 + 1e-6


def test_calculate_zcr_variance_is_zero_for_constant_signal() -> None:
    processor = AudioProcessor(target_sr=22050)
    audio = np.ones(400, dtype=np.float32)

    variance = processor._calculate_zcr_variance(audio)

    assert variance == 0.0


def test_process_audio_soft_fails_denoise_when_deepfilter_missing(monkeypatch, tmp_path: Path, capsys) -> None:
    processor = AudioProcessor(target_sr=22050)
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"in")

    fallback_reads: list[str] = []

    def fake_import_deepfilter() -> bool:
        processor._warn_denoise_unavailable("DeepFilterNet is not installed")
        return False

    def fake_read(path: str):
        fallback_reads.append(path)
        return np.array([0.1, -0.1, 0.05], dtype=np.float32), 22050

    def fake_write(path: str, audio: np.ndarray, sr: int) -> None:
        Path(path).write_bytes(b"ok")

    monkeypatch.setattr(processor, "_import_deepfilter", fake_import_deepfilter)
    monkeypatch.setattr("easy_xtts_trainer.audio.processor.sf.read", fake_read)
    monkeypatch.setattr("easy_xtts_trainer.audio.processor.sf.write", fake_write)

    result = processor.process_audio(str(input_path), str(output_path), denoise=True)

    assert result is True
    assert output_path.exists()
    assert fallback_reads == [str(input_path)]
    assert "Continuing without denoise" in capsys.readouterr().out
