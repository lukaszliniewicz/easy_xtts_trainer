from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

SOURCE_LANGUAGE_CHOICES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "ko",
    "hu",
]

WHISPER_MODEL_CHOICES = ["medium", "medium.en", "large-v2", "large-v3"]

WHISPERX_RUNNER_CHOICES = ["auto", "pixi", "conda"]

SAMPLE_METHOD_CHOICES = ["maximise-punctuation", "punctuation-only", "mixed"]

VOICE_SAMPLE_MODE_CHOICES = ["basic", "extended", "dynamic"]

COMPRESS_PROFILE_CHOICES = ["male", "female", "neutral"]

SCHEDULER_CHOICES = ["multistep", "cosine"]

LANGUAGE_ALIGN_MODELS = {
    "pl": "jonatasgrosman/wav2vec2-xls-r-1b-polish",
    "nl": "GroNLP/wav2vec2-dutch-large-ft-cgn",
    "de": "aware-ai/wav2vec2-xls-r-1b-german",
    "en": "jonatasgrosman/wav2vec2-xls-r-1b-english",
    "fr": "jonatasgrosman/wav2vec2-xls-r-1b-french",
    "it": "jonatasgrosman/wav2vec2-xls-r-1b-italian",
    "ru": "jonatasgrosman/wav2vec2-xls-r-1b-russian",
    "es": "jonatasgrosman/wav2vec2-xls-r-1b-spanish",
}


@dataclass(frozen=True)
class TranscriptionRuntimeConfig:
    source_language: str
    whisper_model: str
    align_model: Optional[str]
    conda_env: Optional[str]
    conda_path: Optional[str]
    source_text: Optional[str] = None
    chapter_per_audio: int = 1
    whisperx_runner: str = "auto"
    whisperx_pixi_exe: Optional[str] = None
    whisperx_pixi_manifest: Optional[str] = None


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    input_path: Optional[str]
    sample_rate: int
    breath: bool
    sample_method: str
    method_proportion: float
    max_audio_time: float
    max_text_length: int
    negative_offset_last_word: int
    min_gap: Optional[int]
    discard_abrupt: bool
    normalize: Optional[float]
    dess: bool
    denoise: bool
    compress: Optional[str]
    fade_in: int
    fade_out: int
    trim: bool
    training_proportion: float
    prepare_dataset: bool


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    source_language: str
    xtts_base_model: str
    xtts_model_name: Optional[str]
    epochs: int
    batch: int
    gradient: int
    sample_rate: int
    max_audio_time: float
    max_text_length: int
    learning_rate: float
    scheduler: Optional[str]
    prepare_dataset: bool


@dataclass(frozen=True)
class VoiceSampleRuntimeConfig:
    voice_sample_mode: str
    voice_samples: int
    voice_sample_only_sentence: bool


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XTTS Model Training App")
    parser.add_argument(
        "--source-language",
        choices=SOURCE_LANGUAGE_CHOICES,
        required=True,
        help="Source language for XTTS",
    )
    parser.add_argument(
        "--whisper-model",
        choices=WHISPER_MODEL_CHOICES,
        default="large-v3",
        help="Whisper model to use for transcription",
    )
    parser.add_argument("--enhance", action="store_true", help="Enable audio enhancement")
    parser.add_argument("--input", help="Input folder or single file (required if not reusing session)")
    parser.add_argument("--session", help="Name for the session folder or existing session to reuse")
    parser.add_argument("--separate", action="store_true", help="Enable speech separation")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--xtts-base-model", default="v2.0.2", help="XTTS base model version")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient", type=int, default=1, help="Gradient accumulation levels")
    parser.add_argument("--xtts-model-name", help="Name for the trained model")
    parser.add_argument(
        "--sample-method",
        choices=SAMPLE_METHOD_CHOICES,
        default="maximise-punctuation",
        help="Method for preparing training samples",
    )
    parser.add_argument("-conda_env", help="Name of the Conda environment to use")
    parser.add_argument("-conda_path", help="Path to the Conda installation folder")
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[22050, 44100],
        default=22050,
        help="Sample rate for WAV files (default: 22050, recommended for XTTS training)",
    )
    parser.add_argument(
        "--max-audio-time",
        type=float,
        default=11,
        help="Maximum audio duration in seconds (default: 11.6)",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=200,
        help="Maximum text length in characters (default: 200)",
    )
    parser.add_argument(
        "--align-model",
        help=(
            "Model to use for phoneme-level alignment. Common options for English include:\n"
            "- WAV2VEC2_ASR_LARGE_LV60K_960H (largest, most accurate)\n"
            "- WAV2VEC2_ASR_BASE_960H (medium size)\n"
            "- WAV2VEC2_ASR_BASE_100H (smallest, fastest)"
        ),
    )
    parser.add_argument(
        "--whisperx-runner",
        choices=WHISPERX_RUNNER_CHOICES,
        default="auto",
        help="WhisperX runtime mode: auto (Pixi first), pixi, or conda",
    )
    parser.add_argument(
        "--whisperx-pixi-exe",
        help="Path to the Pixi executable used for WhisperX runtime",
    )
    parser.add_argument(
        "--whisperx-pixi-manifest",
        help="Path to whisperx_installer pixi.toml manifest",
    )
    parser.add_argument(
        "--normalize",
        type=float,
        nargs="?",
        const=16.0,
        help="Normalize audio to target LUFS (default: -16.0 if no value provided)",
    )
    parser.add_argument("--dess", action="store_true", help="Apply de-essing to reduce sibilance")
    parser.add_argument("--denoise", action="store_true", help="Apply DeepFilterNet noise reduction")
    parser.add_argument(
        "--compress",
        choices=COMPRESS_PROFILE_CHOICES,
        help="Apply dynamic range compression with voice-specific profile",
    )
    parser.add_argument(
        "--method-proportion",
        default="6_4",
        help="For mixed method, proportion of maximise-punctuation to punctuation-only (e.g., '6_4' for 60-40 split)",
    )
    parser.add_argument(
        "--training-proportion",
        default="8_2",
        help="Proportion of training to validation data (e.g., '8_2' for 80-20 split)",
    )
    parser.add_argument(
        "--negative-offset-last-word",
        type=int,
        default=50,
        help="Subtract this many milliseconds from the end time of the last word in each segment (default: 50)",
    )
    parser.add_argument("--breath", action="store_true", help="Apply breath removal preprocessing")
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Automatically trim trailing silence from segments while preserving word endings",
    )
    parser.add_argument(
        "--fade-in",
        type=int,
        metavar="MS",
        default=30,
        help="Apply fade-in effect for specified milliseconds from start (default: 30ms)",
    )
    parser.add_argument(
        "--fade-out",
        type=int,
        metavar="MS",
        default=40,
        help="Apply fade-out effect for specified milliseconds from end (default: 40ms)",
    )
    parser.add_argument(
        "--discard-abrupt",
        action="store_true",
        help="Detect and discard segments with abrupt endings",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-06,
        help="Learning rate for training (default: 5e-06)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=SCHEDULER_CHOICES,
        help="Learning rate scheduler (if not specified, uses constant learning rate)",
    )
    parser.add_argument(
        "--source-text",
        type=str,
        help="Source text file (.txt or .epub) for audiobook alignment",
    )
    parser.add_argument(
        "--chapter-per-audio",
        type=int,
        default=1,
        help="Number of chapters to combine per audio file (default: 1)",
    )
    parser.add_argument(
        "--prepare_dataset",
        action="store_true",
        help="Only prepare the dataset without starting training",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        metavar="MS",
        help="Minimum gap in milliseconds required between segments",
    )
    parser.add_argument(
        "--voice-sample-mode",
        choices=VOICE_SAMPLE_MODE_CHOICES,
        default="basic",
        help=(
            "Mode for organizing voice samples (basic: 2 files in main directory, "
            "extended: specialized samples by characteristic, dynamic: samples with internal variation)"
        ),
    )
    parser.add_argument(
        "--voice-samples",
        type=int,
        choices=[3, 4],
        default=3,
        help="Number of reference voice samples to save in extended/dynamic mode (default: 3)",
    )
    parser.add_argument(
        "--voice-sample-only-sentence",
        action="store_true",
        help="Only use complete sentences (starting with capital letter and ending with sentence punctuation) for voice samples",
    )

    return parser


def _parse_proportion(value: str, label: str) -> float:
    try:
        first, second = map(int, value.split("_"))
        if first + second != 10:
            raise ValueError(f"{label} proportions must sum to 10")
        return first / 10
    except (ValueError, AttributeError):
        example = "6_4" if label == "Method" else "8_2"
        raise ValueError(
            f"{label} proportion must be in format 'N_M' where N+M=10 (e.g., '{example}')"
        )


def normalize_parsed_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if not args.align_model:
        args.align_model = LANGUAGE_ALIGN_MODELS.get(args.source_language)

    if args.method_proportion:
        args.method_proportion = _parse_proportion(args.method_proportion, "Method")

    if args.training_proportion:
        args.training_proportion = _parse_proportion(args.training_proportion, "Training")

    if args.negative_offset_last_word is None:
        args.negative_offset_last_word = 0 if args.source_language == "en" else 50

    return args


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return normalize_parsed_arguments(args)


def build_transcription_runtime_config(args: argparse.Namespace) -> TranscriptionRuntimeConfig:
    return TranscriptionRuntimeConfig(
        source_language=args.source_language,
        whisper_model=args.whisper_model,
        align_model=getattr(args, "align_model", None),
        conda_env=getattr(args, "conda_env", None),
        conda_path=getattr(args, "conda_path", None),
        source_text=getattr(args, "source_text", None),
        chapter_per_audio=getattr(args, "chapter_per_audio", 1),
        whisperx_runner=getattr(args, "whisperx_runner", "auto"),
        whisperx_pixi_exe=getattr(args, "whisperx_pixi_exe", None),
        whisperx_pixi_manifest=getattr(args, "whisperx_pixi_manifest", None),
    )


def build_dataset_runtime_config(args: argparse.Namespace) -> DatasetRuntimeConfig:
    return DatasetRuntimeConfig(
        input_path=getattr(args, "input", None),
        sample_rate=args.sample_rate,
        breath=args.breath,
        sample_method=args.sample_method,
        method_proportion=args.method_proportion,
        max_audio_time=args.max_audio_time,
        max_text_length=args.max_text_length,
        negative_offset_last_word=args.negative_offset_last_word,
        min_gap=getattr(args, "min_gap", None),
        discard_abrupt=args.discard_abrupt,
        normalize=getattr(args, "normalize", None),
        dess=args.dess,
        denoise=args.denoise,
        compress=getattr(args, "compress", None),
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        trim=args.trim,
        training_proportion=args.training_proportion,
        prepare_dataset=args.prepare_dataset,
    )


def build_training_runtime_config(args: argparse.Namespace) -> TrainingRuntimeConfig:
    return TrainingRuntimeConfig(
        source_language=args.source_language,
        xtts_base_model=args.xtts_base_model,
        xtts_model_name=getattr(args, "xtts_model_name", None),
        epochs=args.epochs,
        batch=args.batch,
        gradient=args.gradient,
        sample_rate=args.sample_rate,
        max_audio_time=args.max_audio_time,
        max_text_length=args.max_text_length,
        learning_rate=args.learning_rate,
        scheduler=getattr(args, "scheduler", None),
        prepare_dataset=args.prepare_dataset,
    )


def build_voice_sample_runtime_config(args: argparse.Namespace) -> VoiceSampleRuntimeConfig:
    return VoiceSampleRuntimeConfig(
        voice_sample_mode=args.voice_sample_mode,
        voice_samples=args.voice_samples,
        voice_sample_only_sentence=args.voice_sample_only_sentence,
    )
