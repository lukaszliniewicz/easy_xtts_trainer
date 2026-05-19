from __future__ import annotations

import pytest

from easy_xtts_trainer.config import (
    build_dataset_runtime_config,
    build_training_runtime_config,
    build_transcription_runtime_config,
    build_voice_sample_runtime_config,
    parse_arguments,
)


def test_parse_arguments_sets_default_align_model() -> None:
    args = parse_arguments(["--source-language", "en"])
    assert args.align_model == "jonatasgrosman/wav2vec2-xls-r-1b-english"


def test_parse_arguments_converts_proportions() -> None:
    args = parse_arguments(
        [
            "--source-language",
            "en",
            "--method-proportion",
            "7_3",
            "--training-proportion",
            "9_1",
        ]
    )

    assert args.method_proportion == 0.7
    assert args.training_proportion == 0.9


def test_parse_arguments_rejects_invalid_method_proportion() -> None:
    with pytest.raises(ValueError, match="Method proportion"):
        parse_arguments(["--source-language", "en", "--method-proportion", "3_3"])


def test_build_transcription_runtime_config_maps_namespace_fields() -> None:
    args = parse_arguments(["--source-language", "en", "--source-text", "book.txt"])
    runtime_config = build_transcription_runtime_config(args)

    assert runtime_config.source_language == "en"
    assert runtime_config.whisper_model == "large-v3"
    assert runtime_config.source_text == "book.txt"
    assert runtime_config.whisperx_runner == "auto"
    assert runtime_config.whisperx_pixi_exe is None
    assert runtime_config.whisperx_pixi_manifest is None


def test_build_transcription_runtime_config_maps_pixi_runner_fields() -> None:
    args = parse_arguments(
        [
            "--source-language",
            "en",
            "--whisperx-runner",
            "pixi",
            "--whisperx-pixi-exe",
            "C:/pandrator/bin/pixi.exe",
            "--whisperx-pixi-manifest",
            "C:/pandrator/envs/whisperx_installer/pixi.toml",
        ]
    )
    runtime_config = build_transcription_runtime_config(args)

    assert runtime_config.whisperx_runner == "pixi"
    assert runtime_config.whisperx_pixi_exe == "C:/pandrator/bin/pixi.exe"
    assert runtime_config.whisperx_pixi_manifest == "C:/pandrator/envs/whisperx_installer/pixi.toml"


def test_build_dataset_runtime_config_maps_namespace_fields() -> None:
    args = parse_arguments(
        [
            "--source-language",
            "en",
            "--input",
            "audio",
            "--sample-method",
            "mixed",
            "--method-proportion",
            "6_4",
            "--training-proportion",
            "8_2",
            "--breath",
            "--trim",
            "--fade-in",
            "15",
            "--fade-out",
            "25",
        ]
    )
    runtime_config = build_dataset_runtime_config(args)

    assert runtime_config.input_path == "audio"
    assert runtime_config.sample_method == "mixed"
    assert runtime_config.method_proportion == 0.6
    assert runtime_config.training_proportion == 0.8
    assert runtime_config.breath is True
    assert runtime_config.trim is True
    assert runtime_config.fade_in == 15
    assert runtime_config.fade_out == 25


def test_build_training_runtime_config_maps_namespace_fields() -> None:
    args = parse_arguments(
        [
            "--source-language",
            "en",
            "--epochs",
            "9",
            "--batch",
            "4",
            "--gradient",
            "2",
            "--xtts-base-model",
            "v2.0.2",
            "--xtts-model-name",
            "my_model",
            "--learning-rate",
            "0.00001",
            "--scheduler",
            "cosine",
        ]
    )
    runtime_config = build_training_runtime_config(args)

    assert runtime_config.source_language == "en"
    assert runtime_config.epochs == 9
    assert runtime_config.batch == 4
    assert runtime_config.gradient == 2
    assert runtime_config.xtts_base_model == "v2.0.2"
    assert runtime_config.xtts_model_name == "my_model"
    assert runtime_config.scheduler == "cosine"


def test_build_voice_sample_runtime_config_maps_namespace_fields() -> None:
    args = parse_arguments(
        [
            "--source-language",
            "en",
            "--voice-sample-mode",
            "dynamic",
            "--voice-samples",
            "4",
            "--voice-sample-only-sentence",
        ]
    )
    runtime_config = build_voice_sample_runtime_config(args)

    assert runtime_config.voice_sample_mode == "dynamic"
    assert runtime_config.voice_samples == 4
    assert runtime_config.voice_sample_only_sentence is True
