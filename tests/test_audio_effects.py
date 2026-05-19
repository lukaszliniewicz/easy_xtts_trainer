from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from easy_xtts_trainer.audio import effects


def test_process_audio_files_maps_cli_args_to_processor(tmp_path: Path) -> None:
    class StubProcessor:
        created: list["StubProcessor"] = []

        def __init__(self, target_sr: int) -> None:
            self.target_sr = target_sr
            self.calls: list[dict[str, object]] = []
            StubProcessor.created.append(self)

        def process_audio(
            self,
            input_path: str,
            output_path: str,
            normalize_target=None,
            dess_profile=None,
            denoise: bool = False,
            compress_profile=None,
        ) -> bool:
            self.calls.append(
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "normalize_target": normalize_target,
                    "dess_profile": dess_profile,
                    "denoise": denoise,
                    "compress_profile": compress_profile,
                }
            )
            return True

    input_file = tmp_path / "voice.wav"
    output_dir = tmp_path / "processed"
    args = SimpleNamespace(sample_rate=44100, normalize=16.0, dess=True, denoise=True, compress="female")

    effects.process_audio_files([input_file], output_dir, args, processor_class=StubProcessor)

    assert len(StubProcessor.created) == 1
    processor = StubProcessor.created[0]
    assert processor.target_sr == 44100
    assert processor.calls[0]["input_path"] == str(input_file)
    assert processor.calls[0]["output_path"] == str(output_dir / "voice.wav")
    assert processor.calls[0]["normalize_target"] == -16.0
    assert processor.calls[0]["dess_profile"] == "high"
    assert processor.calls[0]["denoise"] is True
    assert processor.calls[0]["compress_profile"] == "female"


def test_process_audio_files_handles_per_file_exceptions(capsys, tmp_path: Path) -> None:
    class StubProcessor:
        def __init__(self, target_sr: int) -> None:
            self.target_sr = target_sr

        def process_audio(
            self,
            input_path: str,
            output_path: str,
            normalize_target=None,
            dess_profile=None,
            denoise: bool = False,
            compress_profile=None,
        ) -> bool:
            if input_path.endswith("bad.wav"):
                raise RuntimeError("boom")
            return False

    args = SimpleNamespace(sample_rate=22050, normalize=None, dess=False, denoise=False, compress=None)
    input_files = [tmp_path / "bad.wav", tmp_path / "good.wav"]

    effects.process_audio_files(input_files, tmp_path / "out", args, processor_class=StubProcessor)

    output = capsys.readouterr().out
    assert "Error processing" in output
    assert "boom" in output
    assert "Failed to process: good.wav" in output
