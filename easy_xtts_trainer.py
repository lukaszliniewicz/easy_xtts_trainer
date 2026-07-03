import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from pydub import AudioSegment
import csv
import torch
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs
import gc
from datetime import datetime
import requests
from tqdm import tqdm
import re
import logging
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import librosa
import traceback
from easy_xtts_trainer.audio.effects import process_audio_files
from easy_xtts_trainer.audio.preprocessing import process_audio
from easy_xtts_trainer.audio.processor import AudioProcessor
from easy_xtts_trainer.config import (
    DatasetRuntimeConfig,
    TranscriptionRuntimeConfig,
    VoiceSampleRuntimeConfig,
    build_dataset_runtime_config,
    build_transcription_runtime_config,
    build_training_runtime_config,
    build_voice_sample_runtime_config,
    parse_arguments as parse_cli_arguments,
)
from easy_xtts_trainer.dataset.parsing import parse_transcription as parse_transcription_impl
from easy_xtts_trainer.pipeline import PipelineHooks, run_training_pipeline
from easy_xtts_trainer.text.epub import prepare_source_text_for_audio
from easy_xtts_trainer.transcription.ctc import CTCAlignmentRequest, run_ctc_alignment
from easy_xtts_trainer.transcription.whisperx import is_pascal_like_gpu_name, run_whisperx_with_fallback

conda_path = Path("../conda/Scripts/conda.exe")

@dataclass
class TrainingMetrics:
    model_name: str
    start_time: float
    end_time: float = 0
    total_audio_duration_minutes: float = 0  # in minutes
    total_training_minutes: float = 0
    num_samples: int = 0
    num_batches: int = 0
    num_epochs: int = 0
    gradient_accumulation: int = 0
    language: str = ""
    sample_rate: int = 0
    text_losses: List[float] = None
    mel_losses: List[float] = None
    
    def __post_init__(self):
        self.text_losses = self.text_losses or []
        self.mel_losses = self.mel_losses or []
    
    def update_training_time(self):
        """Calculate training time in minutes"""
        if self.end_time:
            self.total_training_minutes = round((self.end_time - self.start_time) / 60, 2)
    
    def parse_loss_values(self, log_text: str, loss_type: str) -> List[float]:
        losses = []
        lines = log_text.split('\n')
        for line in lines:
            if f"avg_loss_{loss_type}_ce:" in line:
                log_position = log_text.find(line)
                context_before = log_text[max(0, log_position-1000):log_position]
                
                if '> EVALUATION' in context_before:
                    number_match = re.search(r'\x1b$$\d+m\s*([\d.]+)', line)
                    if not number_match:
                        number_match = re.search(r'ce:\s*([\d.]+)', line)
                    
                    if number_match:
                        value = float(number_match.group(1))
                        losses.append(value)
        return losses
    
    def update_from_log(self, log_path: Path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
        
        self.text_losses = self.parse_loss_values(log_text, "text")
        self.mel_losses = self.parse_loss_values(log_text, "mel")
    
    def calculate_audio_stats(self, audio_segments: List[Dict]):
        """Calculate audio statistics from the training segments"""
        self.num_samples = len(audio_segments)
        self.num_batches = len(audio_segments) // 2  # assuming batch_size=2
        total_seconds = sum(
            AudioSegment.from_wav(seg['audio_file']).duration_seconds 
            for seg in audio_segments
        )
        self.total_audio_duration_minutes = round(total_seconds / 60, 2)
    
    # def save(self, output_dir: Path):
        # metrics_file = output_dir / "training_metrics.json"
        # self.update_training_time()  # Update training time before saving
        
        # # Create a clean dict without internal tracking fields
        # metrics_dict = {
            # "model_name": self.model_name,
            # "total_training_minutes": self.total_training_minutes,
            # "total_audio_duration_minutes": self.total_audio_duration_minutes,
            # "num_samples": self.num_samples,
            # "num_batches": self.num_batches,
            # "num_epochs": self.num_epochs,
            # "gradient_accumulation": self.gradient_accumulation,
            # "language": self.language,
            # "sample_rate": self.sample_rate,
            # "text_losses": self.text_losses,
            # "mel_losses": self.mel_losses
        # }
        
        # with open(metrics_file, 'w', encoding='utf-8') as f:
            # json.dump(metrics_dict, f, indent=2)

def parse_arguments():
    return parse_cli_arguments()

def create_session_folder(session_name):
    if not session_name:
        session_name = f"xtts-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    session_path = Path(session_name)
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def transcribe_audio(audio_file: Path, config: TranscriptionRuntimeConfig, session_path: Path):
    try:
        print("\n=== Starting Audio Transcription ===")
        print(f"Processing file: {audio_file}")
        print(f"Language: {config.source_language}")
        
        output_dir = (session_path / "transcriptions").resolve()
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory created/verified: {output_dir}")
        
        audio_file_absolute = audio_file.resolve()
        print(f"Absolute audio path: {audio_file_absolute}")
        
        use_int8 = False
        if torch.cuda.is_available():
            use_int8 = is_pascal_like_gpu_name(torch.cuda.get_device_name(0))

        whisper_json_file = output_dir / f"{audio_file.stem}.json"

        # If source text is provided, run WhisperX first, then CTC validate
        # top source candidates and only apply correction when confidence is high.
        if config.source_text:
            print("\n--- Running WhisperX baseline for source correction ---")
            run_whisperx_with_fallback(
                audio_file=audio_file_absolute,
                output_dir=output_dir,
                source_language=config.source_language,
                whisper_model=config.whisper_model,
                align_model=config.align_model,
                conda_env=config.conda_env,
                conda_path=config.conda_path,
                fallback_conda_path=conda_path,
                use_int8=use_int8,
                whisperx_runner=config.whisperx_runner,
                pixi_executable=config.whisperx_pixi_exe,
                pixi_manifest=config.whisperx_pixi_manifest,
            )

            print("\n--- Running CTC Forced Aligner candidate validation ---")
            try:
                source_text_path = Path(config.source_text).resolve()
                if not source_text_path.exists():
                    raise FileNotFoundError(f"Source text file not found: {source_text_path}")
                source_text_path = prepare_source_text_for_audio(
                    source_text_path=source_text_path,
                    audio_file=audio_file,
                    session_path=session_path,
                    chapter_per_audio=config.chapter_per_audio,
                )

                json_file = run_ctc_alignment(
                    CTCAlignmentRequest(
                        audio_file=audio_file_absolute,
                        source_text_path=source_text_path,
                        output_dir=output_dir,
                        source_language=config.source_language,
                        align_model=config.align_model,
                        whisper_json_path=whisper_json_file,
                        candidate_workspace_dir=session_path / "ctc_candidates",
                    )
                )

                if json_file == whisper_json_file:
                    print("\n=== Using WhisperX Output (CTC confidence below threshold) ===")
                else:
                    print("\n=== CTC Alignment Completed Successfully ===")
                return json_file

            except Exception as e:
                print("\n!!! CTC Alignment Failed !!!")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                traceback.print_exc()
                print("\nUsing WhisperX transcript fallback.")
                return whisper_json_file

        run_whisperx_with_fallback(
            audio_file=audio_file_absolute,
            output_dir=output_dir,
            source_language=config.source_language,
            whisper_model=config.whisper_model,
            align_model=config.align_model,
            conda_env=config.conda_env,
            conda_path=config.conda_path,
            fallback_conda_path=conda_path,
            use_int8=use_int8,
            whisperx_runner=config.whisperx_runner,
            pixi_executable=config.whisperx_pixi_exe,
            pixi_manifest=config.whisperx_pixi_manifest,
        )

        return whisper_json_file
        
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        raise

def create_log_file(session_path):
    log_file = session_path / f"session-log--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.txt"
    return log_file

def create_json_file(session_path):
    json_file = session_path / f"session-data--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.json"
    return json_file

def parse_transcription(
    json_file,
    audio_file,
    processed_dir,
    session_data,
    dataset_config: DatasetRuntimeConfig,
):
    return parse_transcription_impl(
        json_file=json_file,
        audio_file=audio_file,
        processed_dir=processed_dir,
        session_data=session_data,
        dataset_config=dataset_config,
        processor_factory=AudioProcessor,
    )

def create_metadata_files(session_data, session_path, training_proportion=0.8):
    databases_dir = session_path / "databases"
    databases_dir.mkdir(exist_ok=True)

    all_samples = session_data['qualifying_segments']
    valid_samples = []

    print(f"Total samples in session_data: {len(all_samples)}")

    for sample in all_samples:
        audio_path = Path(sample['audio_file'])
        if audio_path.exists() and sample['text'].strip():
            valid_samples.append({
                "audio_file": str(audio_path.absolute()),
                "text": sample['text'].strip(),
                "speaker_name": sample['speaker_name']
            })
        else:
            print(f"Skipping invalid sample: file exists: {audio_path.exists()}, text: '{sample['text']}'")

    print(f"Valid samples: {len(valid_samples)}")

    if not valid_samples:
        print("No valid samples found. Check audio processing and transcription steps.")
        return None, None

    random.shuffle(valid_samples)

    train_size = int(training_proportion * len(valid_samples))  # Use the provided proportion
    train_samples = valid_samples[:train_size]
    eval_samples = valid_samples[train_size:]

    train_csv = databases_dir / "train_metadata.csv"
    eval_csv = databases_dir / "eval_metadata.csv"

    for csv_file, samples in [(train_csv, train_samples), (eval_csv, eval_samples)]:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')  # Use '|' as delimiter
            writer.writerow(["audio_file", "text", "speaker_name"])  # Write header
            for sample in samples:
                writer.writerow([sample['audio_file'], sample['text'], sample['speaker_name']])
        
        print(f"Created {csv_file} with {len(samples)} samples")

    return str(train_csv.absolute()), str(eval_csv.absolute())

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as file, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_models(base_path, xtts_base_model):
    os.makedirs(base_path, exist_ok=True)
    
    files_to_download = [
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth", "dvae.pth"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth", "mel_stats.pth"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/vocab.json", "vocab.json"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/model.pth", "model.pth"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/config.json", "config.json"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth", "speakers_xtts.pth")
    ]
    
    for url, filename in files_to_download:
        local_path = os.path.join(base_path, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            download_file(url, local_path)
        else:
            print(f"{filename} already exists. Skipping download.")
        
        # Verify file
        if os.path.exists(local_path):
            print(f"File {filename} size: {os.path.getsize(local_path)} bytes")
            if filename.endswith('.json'):
                with open(local_path, 'r') as f:
                    print(f"First 100 characters of {filename}: {f.read(100)}")
        else:
            print(f"Error: {filename} does not exist after download attempt!")

def get_dataset_size(train_csv: str) -> int:
    """Get number of samples from training CSV file."""
    try:
        with open(train_csv, 'r', encoding='utf-8') as f:
            # Subtract 1 for header row
            return sum(1 for line in f) - 1
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0


def _resolve_metadata_audio_path(raw_audio_path: str, metadata_csv_path: Path, dataset_root: Path | None = None) -> Path:
    audio_path = Path(raw_audio_path)
    if audio_path.is_absolute():
        return audio_path.resolve()

    candidates = []
    if dataset_root is not None:
        candidates.append((dataset_root / audio_path).resolve())
    if metadata_csv_path.parent.name.lower() == "databases":
        candidates.append((metadata_csv_path.parent.parent / audio_path).resolve())
    candidates.append((metadata_csv_path.parent / audio_path).resolve())

    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _peek_first_audio_path(metadata_csv_path: Path, dataset_root: Path | None = None) -> Optional[Path]:
    """Read the first audio path from a metadata CSV file."""
    try:
        with open(metadata_csv_path, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='|')
            for row in reader:
                raw_audio_path = (row.get('audio_file') or '').strip()
                if not raw_audio_path:
                    continue

                return _resolve_metadata_audio_path(raw_audio_path, metadata_csv_path, dataset_root)
    except Exception:
        return None

    return None


def _build_dataset_config_paths(train_csv: str, eval_csv: str) -> tuple[Path, str, str]:
    """Build dataset root and metadata paths expected by the Coqui formatter."""
    train_csv_path = Path(train_csv).resolve()
    eval_csv_path = Path(eval_csv).resolve()

    # Metadata files are normally stored in <session>/databases.
    if train_csv_path.parent.name.lower() == "databases":
        dataset_root = train_csv_path.parent.parent
    else:
        dataset_root = train_csv_path.parent

    first_audio_path = _peek_first_audio_path(train_csv_path, dataset_root) or _peek_first_audio_path(eval_csv_path, dataset_root)
    if first_audio_path is not None:
        try:
            first_audio_path.relative_to(dataset_root)
        except ValueError:
            try:
                dataset_root = Path(os.path.commonpath([str(dataset_root), str(first_audio_path)]))
            except ValueError:
                pass

    try:
        meta_file_train = str(train_csv_path.relative_to(dataset_root))
    except ValueError:
        meta_file_train = train_csv_path.name

    try:
        meta_file_val = str(eval_csv_path.relative_to(dataset_root))
    except ValueError:
        meta_file_val = eval_csv_path.name

    return dataset_root, meta_file_train, meta_file_val

def train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, 
              train_csv, eval_csv, output_path, sample_rate, model_name, max_audio_time,
              max_text_length, learning_rate=5e-06, scheduler=None):
    """
    Train the GPT model.
    """
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    OUT_PATH = os.path.join(output_path, "run", "training")
    CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "base_models", f"{version}")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    dataset_root, meta_file_train, meta_file_val = _build_dataset_config_paths(train_csv, eval_csv)

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=str(dataset_root),
        meta_file_train=meta_file_train,
        meta_file_val=meta_file_val,
        language=language,
    )
    DATASETS_CONFIG_LIST = [config_dataset]

    # Use local paths
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "config.json")
    XTTS_SPEAKER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "speakers_xtts.pth")

    if custom_model and os.path.exists(custom_model) and custom_model.endswith('.pth'):
        XTTS_CHECKPOINT = custom_model

    # Calculate max lengths from max_audio_time and sample_rate
    max_wav_length = int((max_audio_time + 0.5) * sample_rate) # Convert seconds to samples
    max_conditioning_length = max_wav_length  # Same as max_wav_length for XTTS

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=6615,
        max_wav_length=max_wav_length,
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
        debug_loading_failures=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=sample_rate,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )

    # Calculate steps based on actual dataset
    samples_per_epoch = get_dataset_size(train_csv)
    steps_per_epoch = samples_per_epoch // (batch_size * grad_acumm)
    total_steps = steps_per_epoch * num_epochs

    print(f"\nTraining schedule configuration:")
    print(f"Samples per epoch: {samples_per_epoch}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {grad_acumm}")
    print(f"Effective batch size: {batch_size * grad_acumm}")
    print(f"Initial learning rate: {learning_rate}")

    # Configure scheduler based on type and training parameters
    if scheduler == 'multistep':
        # Calculate optimal milestone steps based on training setup
        if num_epochs <= 10:
            milestone_percents = [0.7, 0.9]
            milestones = [int(total_steps * p) for p in milestone_percents]
        else:
            milestone_percents = [0.25, 0.6, 0.8]
            milestones = [int(total_steps * p) for p in milestone_percents]

        scheduler_config = {
            'scheduler': "MultiStepLR",
            'params': {
                "milestones": milestones,
                "gamma": 0.5,
                "last_epoch": -1
            }
        }
        print(f"\nMultiStep LR Schedule:")
        print(f"Milestones at steps: {milestones}")
        print(f"Learning rates: {learning_rate} -> {learning_rate*0.5} -> {learning_rate*0.25}")

    elif scheduler == 'cosine':
        # Calculate optimal cycle length based on training setup
        if num_epochs <= 10:
            # For short training, one full cycle
            t0 = total_steps
            t_mult = 1
        else:
            # For longer training, multiple cycles
            t0 = steps_per_epoch * 3  # Initial cycle of 3 epochs
            t_mult = 2

        eta_min = learning_rate * 0.1  # Minimum LR is 10% of initial LR
        
        scheduler_config = {
            'scheduler': "CosineAnnealingWarmRestarts",
            'params': {
                "T_0": t0,
                "T_mult": t_mult,
                "eta_min": eta_min
            }
        }
        print(f"\nCosine Annealing Schedule:")
        print(f"Initial cycle length (T_0): {t0} steps ({t0/steps_per_epoch:.1f} epochs)")
        print(f"Cycle multiplier (T_mult): {t_mult}")
        print(f"Learning rate range: {learning_rate} -> {eta_min}")

    else:
        # No scheduler - constant learning rate
        scheduler_config = None
        print("\nConstant learning rate schedule")
        print(f"Learning rate: {learning_rate}")

    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        audio=audio_config,
        batch_group_size=48,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_loader_workers=8 if language != "ja" else 0,
        print_step=50,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer_wd_only_on_weights=True,
        optimizer="AdamW",
        optimizer_params={
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2
        },
        lr=learning_rate,
        scheduler_after_epoch=False,  # Always use step-based scheduling
        lr_scheduler=scheduler_config['scheduler'] if scheduler_config else None,
        lr_scheduler_params=scheduler_config['params'] if scheduler_config else None,
    )

    print("\nScheduler configuration:")
    if scheduler_config:
        print(f"Type: {scheduler_config['scheduler']}")
        print(f"Parameters: {scheduler_config['params']}")
    else:
        print("None (constant learning rate)")

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            grad_accum_steps=grad_acumm,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Train the model
    trainer.fit()

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_SPEAKER_FILE, XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, OUT_PATH

def optimize_model(out_path, base_model_path):
    run_dir = Path(out_path) / "run" / "training"
    best_model = None

    print(f"Searching for best model in: {run_dir}")

    for model_dir in run_dir.glob("*"):
        if model_dir.is_dir():
            print(f"Checking directory: {model_dir}")
            model_files = list(model_dir.glob("best_model_*.pth"))
            print(f"Found {len(model_files)} model files")
            if model_files:
                best_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"Selected best model: {best_model}")
                break

    if best_model is None:
        return "Best model not found in training output", ""

    print(f"Best model found: {best_model}")

    # Load the checkpoint and remove unnecessary parts
    checkpoint = torch.load(best_model, map_location=torch.device("cpu"))
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Save the optimized model directly in out_path
    optimized_model = out_path / "model.pth"
    torch.save(checkpoint, optimized_model)

    # Copy config, speakers, and vocab files from base model
    files_to_copy = ["config.json", "speakers_xtts.pth", "vocab.json"]
    for file in files_to_copy:
        src = Path(base_model_path) / file
        dst = out_path / file
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"Warning: {file} not found in base model path.")

    # Remove the run directory and its contents
    run_dir = Path(out_path) / "run"
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError:
            logging.warning(f"Unable to delete the run directory at {run_dir} due to a permission error. You may want to manually delete this directory to save disk space.")

    return f"Model optimized and saved at {optimized_model}!", str(optimized_model)

def copy_model_to_xtts_server(model_path, session_name):
    """
    Copy model files to the XTTS API server directory.
    """
    try:
        # Check if XTTS API server models directory exists
        xtts_server_dir = Path("../xtts-api-server/xtts_models")
        if not xtts_server_dir.exists():
            print(f"XTTS API server directory not found: {xtts_server_dir}")
            return None
        
        # Create a directory for this model using the session name
        target_dir = xtts_server_dir / session_name
        target_dir.mkdir(exist_ok=True)
        
        print(f"Copying model files to XTTS server directory: {target_dir}")
        
        # Files to copy (skipping the 'run' directory)
        files_to_copy = [
            "model.pth",
            "speakers_xtts.pth", 
            "config.json", 
            "vocab.json"
        ]
        
        for file in files_to_copy:
            src = model_path / file
            if src.exists():
                dst = target_dir / file
                print(f"Copying {file}...")
                shutil.copy2(src, dst)
                if dst.exists():
                    print(f"  Success: {dst} ({dst.stat().st_size} bytes)")
                else:
                    print(f"  Failed to copy {file}")
            else:
                print(f"Warning: Source file not found: {src}")
        
        # Verify the copy
        copied_files = list(target_dir.glob("*"))
        print(f"Copied {len(copied_files)} of {len(files_to_copy)} required files to {target_dir}")
        
        return target_dir
    
    except Exception as e:
        print(f"Error copying model to XTTS server: {str(e)}")
        traceback.print_exc()
        return None

def copy_reference_samples(
    processed_dir,
    session_path,
    session_data,
    config: VoiceSampleRuntimeConfig,
):
    """Copy reference voice samples according to the specified mode."""
    # Import required libraries
    import numpy as np
    import librosa
    import shutil
    from pathlib import Path
    import random
    
    # Get parameters
    voice_sample_mode = config.voice_sample_mode
    voice_samples = config.voice_samples
    only_sentences = config.voice_sample_only_sentence
    
    pandrator_voices_dir = Path("../Pandrator/tts_voices")
    if not pandrator_voices_dir.exists():
        print("Pandrator tts_voices directory not found. Skipping reference sample copying.")
        return

    # Get session name for file naming - always use session name for consistency
    session_name = session_path.name
    
    print(f"\nSelecting voice samples with mode: {voice_sample_mode}, count: {voice_samples if voice_sample_mode != 'basic' else 2}")
    if only_sentences:
        print("Filtering for complete sentences (capital letter + ending punctuation)")

    # Get all wav files and their corresponding segments
    wav_files = []
    skipped_files = 0
    
    for wav_file in processed_dir.glob('*.wav'):
        segment = next((seg for seg in session_data['qualifying_segments'] 
                       if Path(seg['audio_file']).name == wav_file.name), None)
        if segment:
            # Check if we should only use complete sentences
            if only_sentences:
                text = segment['text'].strip()
                # Check if text starts with capital letter
                starts_with_capital = text and text[0].isupper()
                
                # Check for sentence-ending punctuation, including when followed by closing quotes/parentheses
                ends_with_punctuation = False
                if text:
                    if text[-1] in '.!?':
                        ends_with_punctuation = True
                    elif len(text) >= 2 and text[-2] in '.!?' and text[-1] in '"\')':
                        ends_with_punctuation = True
                    elif len(text) >= 3 and text[-3] in '.!?' and text[-2:] in [')"', ')\'', '")', '\')', ').', '").']:
                        ends_with_punctuation = True
                
                if not (starts_with_capital and ends_with_punctuation):
                    skipped_files += 1
                    continue  # Skip this segment
            
            try:
                # Calculate basic audio properties
                y, sr = librosa.load(wav_file, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Calculate pitch 
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_indices = np.argmax(magnitudes, axis=0)
                pitches_by_frame = pitches[pitch_indices, range(len(pitch_indices))]
                pitches_by_frame = pitches_by_frame[pitches_by_frame > 0]
                avg_pitch = np.mean(pitches_by_frame) if len(pitches_by_frame) > 0 else 0
                
                wav_files.append({
                    'path': wav_file,
                    'size': wav_file.stat().st_size,
                    'text': segment['text'],
                    'duration': duration,
                    'avg_pitch': avg_pitch,
                    'speech_rate': len(segment['text']) / duration if duration > 0 else 0
                })
            except Exception as e:
                print(f"Error analyzing {wav_file}: {e}")
                continue

    if only_sentences:
        print(f"Sentence filtering: skipped {skipped_files} segments, kept {len(wav_files)} eligible segments")
    
    if not wav_files:
        print("No valid audio files found for reference samples.")
        return
    
    print(f"Found {len(wav_files)} valid audio files for analysis.")
    
    # Handle based on selected mode
    if voice_sample_mode == 'basic':
        # Original basic mode logic (always 2 samples)
        # Sort files by size in descending order
        wav_files.sort(key=lambda x: x['size'], reverse=True)

        # Select one random file from top 10%
        top_10_percent = wav_files[:max(1, len(wav_files) // 10)]
        random_long = random.choice(top_10_percent)
        
        # Copy the random long sample
        shutil.copy2(random_long['path'], pandrator_voices_dir / f"{session_name}.wav")

        # Get top 70% by length for speech rate calculation
        top_70_percent = wav_files[:int(len(wav_files) * 0.7)]
        
        # Find the file with highest speech rate
        fastest = max(top_70_percent, key=lambda x: x['speech_rate'])
        
        # Copy the fastest sample
        shutil.copy2(fastest['path'], pandrator_voices_dir / f"{session_name}_fastest.wav")

        print(f"Copied reference samples to {pandrator_voices_dir} (basic mode):")
        print(f"Long sample: {random_long['path'].name} ({random_long['duration']:.2f}s)")
        print(f"Fastest sample: {fastest['path'].name} (speech rate: {fastest['speech_rate']:.2f} chars/sec)")
        
    elif voice_sample_mode == 'extended':
        # Extended mode logic for 3-4 specialized samples
        # Create folder for this model's samples
        model_samples_dir = pandrator_voices_dir / session_name
        model_samples_dir.mkdir(exist_ok=True)
        
        # Sort by size for length-based selections
        wav_files.sort(key=lambda x: x['size'], reverse=True)
        
        # Get top 70% by length for advanced selection
        top_70_percent = wav_files[:int(len(wav_files) * 0.7)]
        selected_samples = []
        
        # 1. Fast sample - highest speech rate from top 70%
        fastest = max(top_70_percent, key=lambda x: x['speech_rate'])
        shutil.copy2(fastest['path'], model_samples_dir / f"{session_name}_fast.wav")
        selected_samples.append(fastest['path'])
        
        # 2. Low pitch sample - from top 70% length files, avoiding bottom 10% pitch
        pitch_sorted = sorted(top_70_percent, key=lambda x: x['avg_pitch'])
        skip_count = max(0, int(len(pitch_sorted) * 0.1))
        
        # Find low pitch sample not already selected
        low_pitch_candidates = [s for s in pitch_sorted[skip_count:] if s['path'] not in selected_samples]
        if low_pitch_candidates:
            low_pitch_sample = low_pitch_candidates[0]  # First eligible low pitch sample
            shutil.copy2(low_pitch_sample['path'], model_samples_dir / f"{session_name}_low.wav")
            selected_samples.append(low_pitch_sample['path'])
        
        # 3. Slow sample (part of the minimum 3 samples in extended mode)
        # Sort by speech rate, exclude bottom 10% to avoid outliers
        speech_rates = sorted([f['speech_rate'] for f in top_70_percent])
        min_valid_rate = speech_rates[max(0, int(len(speech_rates) * 0.1))]
        
        valid_slow_samples = [f for f in top_70_percent 
                             if f['speech_rate'] >= min_valid_rate and f['path'] not in selected_samples]
        if valid_slow_samples:
            slowest = min(valid_slow_samples, key=lambda x: x['speech_rate'])
            shutil.copy2(slowest['path'], model_samples_dir / f"{session_name}_slow.wav")
            selected_samples.append(slowest['path'])
        
        # 4. High pitch sample if requested (for 4 samples mode)
        if voice_samples >= 4:
            # Reverse pitch sort, skip top 10% to avoid distortion
            pitch_sorted.reverse()
            skip_count = max(0, int(len(pitch_sorted) * 0.1))
            
            # Find high pitch sample not already selected
            high_pitch_candidates = [s for s in pitch_sorted[skip_count:] if s['path'] not in selected_samples]
            if high_pitch_candidates:
                high_pitch_sample = high_pitch_candidates[0]
                shutil.copy2(high_pitch_sample['path'], model_samples_dir / f"{session_name}_high.wav")
                selected_samples.append(high_pitch_sample['path'])
        
        print(f"Copied {len(selected_samples)} reference samples to {model_samples_dir} (extended mode):")
        for sample_path in selected_samples:
            sample = next(s for s in wav_files if s['path'] == sample_path)
            sample_type = ""
            if sample_path == fastest['path']:
                sample_type = "fast"
            elif 'low_pitch_sample' in locals() and sample_path == low_pitch_sample['path']:
                sample_type = "low pitch"
            elif 'slowest' in locals() and sample_path == slowest['path']:
                sample_type = "slow"
            elif 'high_pitch_sample' in locals() and sample_path == high_pitch_sample['path']:
                sample_type = "high pitch"
                
            print(f"- {sample_path.name} ({sample_type}): {sample['duration']:.2f}s, {sample['speech_rate']:.2f} chars/sec")
        
    else:  # dynamic mode
        # Create folder for this model's samples
        model_samples_dir = pandrator_voices_dir / session_name
        model_samples_dir.mkdir(exist_ok=True)
        
        # Analyze dynamic variation
        dynamic_samples = analyze_dynamic_variation(wav_files)
        
        # Select samples with dynamic internal variation
        selected_dynamic_samples = select_dynamic_samples(dynamic_samples, voice_samples)
        
        # Copy the selected samples
        print(f"\nCopying {len(selected_dynamic_samples)} dynamic variation samples to {model_samples_dir}:")
        for i, sample in enumerate(selected_dynamic_samples, 1):
            output_file = model_samples_dir / f"{session_name}_dynamic_{i}.wav"
            shutil.copy2(sample['path'], output_file)
            
            # Print details about the selected sample
            pitch_var = sample.get('pitch_variation', 0)
            speed_var = sample.get('speed_variation', 0)
            rhythm_var = sample.get('rhythm_variation', 0)
            
            print(f"  {i}. {sample['path'].name}")
            print(f"     Duration: {sample['duration']:.2f}s, Text: \"{sample['text'][:50]}{'...' if len(sample['text']) > 50 else ''}\"")
            print(f"     Variation scores - Pitch: {pitch_var:.3f}, Speed: {speed_var:.3f}, Rhythm: {rhythm_var:.3f}")
            print(f"     Dynamic score: {sample.get('dynamic_score', 0):.3f}")

def analyze_dynamic_variation(samples):
    """Analyze internal variation within each sample and mark outliers."""
    # Calculate variation metrics for each sample
    for sample in samples:
        try:
            y, sr = librosa.load(sample['path'], sr=None)
            
            # 1. Pitch Variation (how much pitch changes within the sample)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_indices = np.argmax(magnitudes, axis=0)
            pitches = pitches[pitch_indices, range(len(pitch_indices))]
            pitches = pitches[pitches > 0]  # Remove zero values
            
            # Calculate how much pitch changes over time
            sample['pitch_variation'] = np.std(pitches) / np.mean(pitches) if len(pitches) > 0 and np.mean(pitches) > 0 else 0
            
            # 2. Speed Variation (changes in speaking rate)
            # Divide audio into segments and analyze energy patterns
            num_chunks = max(3, int(sample['duration'] / 2))  # ~2 second chunks
            if len(y) > num_chunks and num_chunks > 1:
                chunk_size = len(y) // num_chunks
                energy_per_chunk = []
                
                for i in range(num_chunks):
                    chunk = y[i*chunk_size:min((i+1)*chunk_size, len(y))]
                    # Calculate local energy in this segment
                    energy_per_chunk.append(np.sum(np.abs(chunk)))
                
                # How much energy (proxy for speaking rate) varies
                sample['speed_variation'] = np.std(energy_per_chunk) / np.mean(energy_per_chunk) if np.mean(energy_per_chunk) > 0 else 0
            else:
                sample['speed_variation'] = 0
            
            # 3. Rhythm/Prosody Variation
            # Detect onsets (beginnings of syllables/words)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Analyze how regularly spaced these onsets are
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            sample['rhythm_variation'] = np.std(np.mean(tempogram, axis=0)) if tempogram.size > 0 else 0
            
            # 4. Text-based indications of variation
            text = sample['text']
            # Count different punctuation types (proxy for different speaking styles)
            question_marks = text.count('?')
            exclamations = text.count('!')
            periods = text.count('.')
            commas = text.count(',')
            
            # Does the sample contain different sentence types?
            has_variety = (question_marks > 0) + (exclamations > 0) + (periods > 0) > 1
            punctuation_density = (question_marks + exclamations + periods + commas) / len(text) if len(text) > 0 else 0
            
            # 5. Dynamic composite score
            sample['dynamic_score'] = (
                0.30 * sample['pitch_variation'] + 
                0.30 * sample['speed_variation'] + 
                0.25 * sample['rhythm_variation'] + 
                0.15 * (punctuation_density * 10 + (0.5 if has_variety else 0))  # Scale up text factors
            )
            
        except Exception as e:
            print(f"Error analyzing sample {sample['path'].name}: {e}")
            # Set default values if analysis fails
            sample['pitch_variation'] = 0
            sample['speed_variation'] = 0 
            sample['rhythm_variation'] = 0
            sample['dynamic_score'] = 0
    
    # Now identify outliers for each metric
    for metric in ['pitch_variation', 'speed_variation', 'rhythm_variation', 'dynamic_score']:
        if not samples:
            continue
            
        values = [s[metric] for s in samples if metric in s]
        if not values:
            continue
            
        # Sort values to identify percentile thresholds
        values.sort()
        
        # Get 10th and 90th percentile values
        low_threshold_idx = max(0, int(len(values) * 0.1))
        high_threshold_idx = min(len(values) - 1, int(len(values) * 0.9))
        
        low_threshold = values[low_threshold_idx]
        high_threshold = values[high_threshold_idx]
        
        # Mark outliers
        for sample in samples:
            if metric in sample:
                # Samples are outliers if they're in the bottom or top 10%
                if sample[metric] < low_threshold or sample[metric] > high_threshold:
                    sample[f'{metric}_outlier'] = True
                else:
                    sample[f'{metric}_outlier'] = False
    
    return samples

def select_dynamic_samples(samples, target_count=3):
    """Select samples with internal variation while avoiding outliers."""
    if not samples:
        return []
        
    # Filter to quality samples (top 70% by size/duration)
    samples.sort(key=lambda x: x['size'], reverse=True)
    quality_samples = samples[:max(5, int(len(samples) * 0.7))]
    
    # Remove samples that are outliers in multiple dimensions
    normal_samples = []
    mild_outliers = []  # Outliers in just one dimension
    extreme_outliers = []  # Outliers in multiple dimensions
    
    for sample in quality_samples:
        # Count how many dimensions this sample is an outlier in
        outlier_count = sum(1 for metric in ['pitch_variation', 'speed_variation', 
                                          'rhythm_variation', 'dynamic_score'] 
                          if sample.get(f'{metric}_outlier', False))
        
        if outlier_count == 0:
            normal_samples.append(sample)
        elif outlier_count == 1:
            mild_outliers.append(sample)
        else:
            extreme_outliers.append(sample)
    
    # Prioritize samples with the best dynamic scores among non-outliers
    normal_samples.sort(key=lambda x: x['dynamic_score'], reverse=True)
    
    # Start with best normal samples
    selected = normal_samples[:target_count]
    
    # If we don't have enough, add mild outliers
    if len(selected) < target_count:
        mild_outliers.sort(key=lambda x: x['dynamic_score'], reverse=True)
        selected.extend(mild_outliers[:target_count - len(selected)])
    
    # If we still don't have enough, reluctantly add extreme outliers
    if len(selected) < target_count and extreme_outliers:
        extreme_outliers.sort(key=lambda x: x['dynamic_score'], reverse=True)
        selected.extend(extreme_outliers[:target_count - len(selected)])
    
    # Final check - if we somehow ended up with too few, add any remaining samples
    if len(selected) < target_count:
        remaining = [s for s in samples if s not in selected]
        remaining.sort(key=lambda x: x['dynamic_score'], reverse=True)
        selected.extend(remaining[:target_count - len(selected)])
    
    return selected[:target_count]

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    try:
        logger.info("=== Starting XTTS Training Process ===")
        args = parse_arguments()
        logger.info("Arguments parsed successfully")

        transcription_config = build_transcription_runtime_config(args)
        dataset_config = build_dataset_runtime_config(args)
        training_config = build_training_runtime_config(args)
        voice_sample_config = build_voice_sample_runtime_config(args)

        hooks = PipelineHooks(
            create_session_folder=create_session_folder,
            create_log_file=create_log_file,
            create_json_file=create_json_file,
            process_audio=process_audio,
            transcribe_audio=transcribe_audio,
            parse_transcription=parse_transcription,
            create_metadata_files=create_metadata_files,
            download_models=download_models,
            train_gpt=train_gpt,
            optimize_model=optimize_model,
            copy_model_to_xtts_server=copy_model_to_xtts_server,
            copy_reference_samples=copy_reference_samples,
        )

        return run_training_pipeline(
            args=args,
            transcription_config=transcription_config,
            dataset_config=dataset_config,
            training_config=training_config,
            voice_sample_config=voice_sample_config,
            hooks=hooks,
            logger=logger,
        )
    except Exception as e:
        logger.error("=== Training Process Failed ===")
        logger.error(f"Error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("Training process completed successfully.")
            sys.exit(0)
        else:
            print("Training process failed.")
            sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
