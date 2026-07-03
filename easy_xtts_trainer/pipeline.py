from __future__ import annotations

import csv
import gc
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from easy_xtts_trainer.config import (
    DatasetRuntimeConfig,
    TrainingRuntimeConfig,
    TranscriptionRuntimeConfig,
    VoiceSampleRuntimeConfig,
)
from easy_xtts_trainer.session.reuse import copy_session_files, create_new_session_name, is_session_reusable


@dataclass(frozen=True)
class PipelineHooks:
    create_session_folder: Callable[[Optional[str]], Path]
    create_log_file: Callable[[Path], Path]
    create_json_file: Callable[[Path], Path]
    process_audio: Callable[[Optional[str], Path, DatasetRuntimeConfig], Optional[Path]]
    transcribe_audio: Callable[[Path, TranscriptionRuntimeConfig, Path], Path]
    parse_transcription: Callable[[Path, Path, Path, dict, DatasetRuntimeConfig], int]
    create_metadata_files: Callable[[dict, Path, float], tuple[Optional[str], Optional[str]]]
    download_models: Callable[[Path, str], None]
    train_gpt: Callable[..., tuple[Any, Any, Any, Any, Any]]
    optimize_model: Callable[[Path, Path], tuple[str, str]]
    copy_model_to_xtts_server: Callable[[Path, str], Optional[Path]]
    copy_reference_samples: Callable[[Path, Path, dict, VoiceSampleRuntimeConfig], None]


def _resolve_session_audio_path(raw_audio_path: str, session_path: Path, metadata_csv_path: Path) -> Path:
    audio_path = Path(raw_audio_path)
    if audio_path.is_absolute():
        return audio_path

    candidates = [
        session_path / audio_path,
        session_path / "audio_sources" / "processed" / audio_path.name,
        metadata_csv_path.parent / audio_path,
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def run_training_pipeline(
    args: Any,
    transcription_config: TranscriptionRuntimeConfig,
    dataset_config: DatasetRuntimeConfig,
    training_config: TrainingRuntimeConfig,
    voice_sample_config: VoiceSampleRuntimeConfig,
    hooks: PipelineHooks,
    logger: logging.Logger,
) -> bool:
    try:
        # Check if session exists and is reusable
        original_session_path = Path(args.session) if args.session else None
        reusing_session = False

        logger.info(f"Original session path: {original_session_path}")

        try:
            if original_session_path and original_session_path.exists():
                logger.info("Found existing session directory")
                reusable, message = is_session_reusable(original_session_path)
                logger.info(f"Session reusability check: {message}")

                if reusable:
                    reusing_session = True
                    logger.info(f"Will reuse existing session: {original_session_path}")
                elif not dataset_config.input_path:
                    raise ValueError("--input is required when not reusing a session")
            elif not dataset_config.input_path:
                raise ValueError("Either --session (for reuse) or --input is required")
        except Exception as exc:
            logger.error(f"Error checking session reusability: {exc}")
            raise

        # Create new session path
        try:
            if reusing_session:
                session_path = create_new_session_name(
                    original_session_path,
                    training_config.epochs,
                    training_config.gradient,
                )
                logger.info(f"Creating new session based on existing one: {session_path}")
            else:
                session_path = hooks.create_session_folder(args.session)
                logger.info(f"Creating new session from scratch: {session_path}")
        except Exception as exc:
            logger.error(f"Error creating session directory: {exc}")
            raise

        # Initialize session data
        session_data = {"qualifying_segments": []}
        logger.info("Initialized empty session data")

        if reusing_session:
            logger.info("Beginning session reuse process...")
            try:
                # Copy files
                if not copy_session_files(original_session_path, session_path):
                    raise RuntimeError("Failed to copy session files")
                logger.info("Successfully copied session files")

                # Load metadata files
                train_csv_path = session_path / "databases" / "train_metadata.csv"
                eval_csv_path = session_path / "databases" / "eval_metadata.csv"

                if not train_csv_path.exists() or not eval_csv_path.exists():
                    raise FileNotFoundError("Missing CSV files after copy")

                logger.info("Loading metadata from CSV files...")
                loaded_segments = 0

                # Load train CSV
                try:
                    with open(train_csv_path, "r", encoding="utf-8") as handle:
                        reader = csv.reader(handle, delimiter="|")
                        next(reader)  # Skip header
                        for row in reader:
                            if len(row) >= 3:
                                audio_file = _resolve_session_audio_path(row[0], session_path, train_csv_path)
                                if not audio_file.exists():
                                    logger.warning(f"Audio file not found: {audio_file}")
                                    continue
                                session_data["qualifying_segments"].append(
                                    {
                                        "audio_file": str(audio_file),
                                        "text": row[1],
                                        "speaker_name": row[2],
                                    }
                                )
                                loaded_segments += 1

                    logger.info(f"Loaded {loaded_segments} segments from train CSV")

                except Exception as exc:
                    logger.error(f"Error reading train CSV: {exc}")
                    raise

                # Create new session files
                logger.info("Creating new session files...")
                hooks.create_log_file(session_path)
                json_file = hooks.create_json_file(session_path)

                with open(json_file, "w") as handle:
                    json.dump(session_data, handle)
                logger.info("Created new session files")

                if dataset_config.prepare_dataset:
                    logger.info(
                        "Dataset preparation completed. "
                        "Skipping training as --prepare_dataset was specified."
                    )
                    return True

                train_csv = str(train_csv_path)
                eval_csv = str(eval_csv_path)

            except Exception as exc:
                logger.error(f"Error during session reuse: {exc}")
                raise
        else:
            logger.info("Processing new session...")
            try:
                # Process new audio files
                hooks.create_log_file(session_path)
                json_file = hooks.create_json_file(session_path)

                with open(json_file, "w") as handle:
                    json.dump(session_data, handle)

                logger.info("Processing audio files...")
                audio_sources_dir = hooks.process_audio(dataset_config.input_path, session_path, dataset_config)
                if not audio_sources_dir:
                    raise RuntimeError("Audio processing failed")

                # Process each audio file
                total_segments = 0
                audio_files = list(audio_sources_dir.glob("*.wav"))
                logger.info(f"Found {len(audio_files)} audio files to process")

                for audio_file in audio_files:
                    logger.info(f"Processing {audio_file.name}...")
                    try:
                        json_file = hooks.transcribe_audio(audio_file, transcription_config, session_path)
                        segments_count = hooks.parse_transcription(
                            json_file,
                            audio_file,
                            audio_sources_dir / "processed",
                            session_data,
                            dataset_config,
                        )
                        total_segments += segments_count
                        logger.info(f"Added {segments_count} segments from {audio_file.name}")
                    except Exception as exc:
                        logger.error(f"Error processing {audio_file.name}: {exc}")
                        continue

                logger.info(f"Total segments processed: {total_segments}")

                # Create metadata files
                logger.info("Creating metadata files...")
                train_csv, eval_csv = hooks.create_metadata_files(
                    session_data,
                    session_path,
                    dataset_config.training_proportion,
                )

                if train_csv is None or eval_csv is None:
                    raise RuntimeError("Failed to create metadata files")

                if dataset_config.prepare_dataset:
                    logger.info(
                        "Dataset preparation completed. "
                        "Skipping training as --prepare_dataset was specified."
                    )
                    return True

            except Exception as exc:
                logger.error(f"Error processing new session: {exc}")
                raise

        # Prepare for training
        logger.info("Beginning training preparation...")
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()

                logger.info(
                    f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB, "
                    f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB"
                )

            # Create model directory
            models_dir = session_path / "models"
            models_dir.mkdir(exist_ok=True)

            model_name = training_config.xtts_model_name or f"xtts_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model_output_path = models_dir / model_name
            logger.info(f"Model will be saved to: {model_output_path}")

            # Download base models
            logger.info("Downloading base models...")
            base_model_path = Path.cwd() / "base_models" / training_config.xtts_base_model
            hooks.download_models(base_model_path, training_config.xtts_base_model)

            # Start training
            logger.info("Starting model training...")
            hooks.train_gpt(
                custom_model="",
                version=training_config.xtts_base_model,
                language=training_config.source_language,
                num_epochs=training_config.epochs,
                batch_size=training_config.batch,
                grad_acumm=training_config.gradient,
                train_csv=str(train_csv),
                eval_csv=str(eval_csv),
                output_path=str(model_output_path),
                sample_rate=training_config.sample_rate,
                model_name=model_name,
                max_audio_time=training_config.max_audio_time,
                max_text_length=training_config.max_text_length,
                learning_rate=training_config.learning_rate,
                scheduler=training_config.scheduler,
            )

            logger.info("Optimizing trained model...")
            optimization_message, optimized_model_path = hooks.optimize_model(model_output_path, base_model_path)

            logger.info(f"Training completed. Model saved at: {model_output_path}")
            logger.info(optimization_message)

            if optimized_model_path:
                logger.info("Copying model to XTTS API server...")
                xtts_server_model_path = hooks.copy_model_to_xtts_server(model_output_path, session_path.name)

                if xtts_server_model_path:
                    logger.info(f"Model successfully copied to XTTS server at: {xtts_server_model_path}")
                else:
                    logger.warning("Failed to copy model to XTTS server. Please copy manually.")

                # Then proceed with copying reference samples
                logger.info("Copying reference samples...")
                hooks.copy_reference_samples(
                    session_path / "audio_sources" / "processed",
                    session_path,
                    session_data,
                    voice_sample_config,
                )
                logger.info("Reference samples copied successfully")
            else:
                logger.warning("Optimized model path not found. Reference samples not copied.")

            logger.info("=== Training Process Completed Successfully ===")
            return True

        except Exception as exc:
            logger.error(f"Error during training phase: {exc}")
            logger.error("Stack trace:", exc_info=True)
            raise

    except Exception as exc:
        logger.error("=== Training Process Failed ===")
        logger.error(f"Error: {exc}")
        logger.error("Stack trace:", exc_info=True)
        return False
