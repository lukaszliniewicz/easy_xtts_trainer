from __future__ import annotations

from pathlib import Path

from easy_xtts_trainer.transcription import whisperx as whisperx_module
from easy_xtts_trainer.transcription.whisperx import (
    build_whisperx_pixi_command,
    build_whisperx_command,
    discover_pandrator_pixi_runtime,
    get_conda_executable,
    get_pixi_runtime,
    is_pascal_like_gpu_name,
    run_whisperx_with_fallback,
)


def test_build_whisperx_command_includes_optional_flags() -> None:
    command = build_whisperx_command(
        conda_executable="C:/conda/conda.exe",
        env_name="whisperx",
        audio_file="audio.wav",
        source_language="en",
        whisper_model="large-v3",
        output_dir="out",
        align_model="model-x",
        use_int8=True,
    )

    assert "--align_model" in command
    assert "model-x" in command
    assert "--compute_type" in command
    assert "int8" in command


def test_get_conda_executable_prefers_existing_possible_path(tmp_path: Path) -> None:
    existing = tmp_path / "conda.exe"
    existing.write_text("stub", encoding="utf-8")

    executable = get_conda_executable(
        conda_path_arg=None,
        conda_env_arg="whisperx",
        fallback_conda_path="fallback/conda.exe",
        possible_paths=[str(tmp_path / "missing.exe"), str(existing)],
    )

    assert executable == str(existing)


def test_build_whisperx_pixi_command_includes_optional_flags() -> None:
    command = build_whisperx_pixi_command(
        pixi_executable="C:/pandrator/bin/pixi.exe",
        pixi_manifest="C:/pandrator/envs/whisperx_installer/pixi.toml",
        audio_file="audio.wav",
        source_language="en",
        whisper_model="large-v3",
        output_dir="out",
        align_model="model-x",
        use_int8=True,
    )

    assert "--manifest-path" in command
    assert "C:/pandrator/envs/whisperx_installer/pixi.toml" in command
    assert "--align_model" in command
    assert "model-x" in command
    assert "--compute_type" in command
    assert "int8" in command


def test_discover_pandrator_pixi_runtime_finds_standard_layout(tmp_path: Path) -> None:
    pixi_executable = tmp_path / "bin" / "pixi.exe"
    pixi_manifest = tmp_path / "envs" / "whisperx_installer" / "pixi.toml"
    pixi_executable.parent.mkdir(parents=True)
    pixi_manifest.parent.mkdir(parents=True)
    pixi_executable.write_text("stub", encoding="utf-8")
    pixi_manifest.write_text("[project]\nname='whisperx'", encoding="utf-8")

    runtime = discover_pandrator_pixi_runtime(search_roots=[tmp_path])

    assert runtime == (str(pixi_executable), str(pixi_manifest))


def test_get_pixi_runtime_prefers_explicit_arguments(tmp_path: Path) -> None:
    explicit_exe = tmp_path / "explicit.exe"
    explicit_manifest = tmp_path / "explicit.toml"
    explicit_exe.write_text("stub", encoding="utf-8")
    explicit_manifest.write_text("[project]\nname='explicit'", encoding="utf-8")

    env_exe = tmp_path / "env.exe"
    env_manifest = tmp_path / "env.toml"
    env_exe.write_text("stub", encoding="utf-8")
    env_manifest.write_text("[project]\nname='env'", encoding="utf-8")

    runtime = get_pixi_runtime(
        pixi_executable_arg=str(explicit_exe),
        pixi_manifest_arg=str(explicit_manifest),
        env={
            "WHISPERX_PIXI_EXE": str(env_exe),
            "WHISPERX_PIXI_MANIFEST": str(env_manifest),
        },
        discovery_roots=[tmp_path],
    )

    assert runtime == (str(explicit_exe), str(explicit_manifest))


def test_run_whisperx_with_fallback_uses_pixi_in_auto_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        whisperx_module,
        "get_pixi_runtime",
        lambda **_: ("C:/pandrator/bin/pixi.exe", "C:/pandrator/envs/whisperx_installer/pixi.toml"),
    )

    def fake_run_whisperx_in_pixi(**kwargs):
        calls.append((kwargs["pixi_executable"], kwargs["pixi_manifest"]))
        return True

    monkeypatch.setattr(whisperx_module, "run_whisperx_in_pixi", fake_run_whisperx_in_pixi)
    monkeypatch.setattr(
        whisperx_module,
        "run_whisperx_in_env",
        lambda **_: (_ for _ in ()).throw(AssertionError("Conda fallback should not run")),
    )

    run_whisperx_with_fallback(
        audio_file="audio.wav",
        output_dir="out",
        source_language="en",
        whisper_model="large-v3",
        align_model=None,
        conda_env=None,
        conda_path=None,
        fallback_conda_path="C:/conda/conda.exe",
        whisperx_runner="auto",
    )

    assert calls == [("C:/pandrator/bin/pixi.exe", "C:/pandrator/envs/whisperx_installer/pixi.toml")]


def test_is_pascal_like_gpu_name_detects_known_cards() -> None:
    assert is_pascal_like_gpu_name("NVIDIA GeForce GTX 1080 Ti")
    assert not is_pascal_like_gpu_name("NVIDIA GeForce RTX 4090")
