import argparse
import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from pydub import AudioSegment
import csv
import torch
import torchaudio
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs
import gc
from datetime import datetime
import requests
from tqdm import tqdm
import re
import logging
import sys

conda_path = Path("../conda/Scripts/conda.exe")

def parse_arguments():
    parser = argparse.ArgumentParser(description="XTTS Model Training App")
    parser.add_argument("--source-language", choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"], required=True, help="Source language for XTTS")
    parser.add_argument("--whisper-model", choices=["medium", "medium.en", "large-v2", "large-v3"], default="medium", help="Whisper model to use for transcription")
    parser.add_argument("--denoise", action="store_true", help="Enable denoising")
    parser.add_argument("--enhance", action="store_true", help="Enable audio enhancement")
    parser.add_argument("-i", "--input", required=True, help="Input folder or single file")
    parser.add_argument("--session", help="Name for the session folder")
    parser.add_argument("--separate", action="store_true", help="Enable speech separation")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--xtts-base-model", default="v2.0.2", help="XTTS base model version")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient", type=int, default=1, help="Gradient accumulation levels")
    parser.add_argument("--xtts-model-name", help="Name for the trained model")
    parser.add_argument("--sample-method", choices=["maximise-punctuation", "punctuation-only", "mixed"], default="maximise-punctuation", help="Method for preparing training samples")
    parser.add_argument("-conda_env", help="Name of the Conda environment to use")
    parser.add_argument("-conda_path", help="Path to the Conda installation folder")
    
    return parser.parse_args()

def create_session_folder(session_name):
    if not session_name:
        session_name = f"xtts-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    session_path = Path(session_name)
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def transcribe_audio(audio_file, args, session_path):
    output_dir = (session_path / "transcriptions").resolve()
    output_dir.mkdir(exist_ok=True)
    
    audio_file_absolute = audio_file.resolve()
    
    def get_conda_path():
        if args.conda_path:
            return os.path.join(args.conda_path, "conda.exe")
        
        possible_paths = [
            os.path.join(os.environ.get('USERPROFILE', ''), "anaconda3", "Scripts", "conda.exe"),
            os.path.join(os.environ.get('USERPROFILE', ''), "miniconda3", "Scripts", "conda.exe"),
            r"C:\ProgramData\Anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\Miniconda3\Scripts\conda.exe"
        ]
        
        if args.conda_env:
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return str(conda_path)
    
    conda_executable = get_conda_path()
    
    def run_whisperx(env):
        command = [
            conda_executable,
            "run",
            "-n",
            env,
            "python",
            "-m",
            "whisperx",
            str(audio_file_absolute),
            "--language", args.source_language,
            "--model", args.whisper_model,
            "--output_dir", str(output_dir),
            "--output_format", "json"
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running WhisperX with environment {env}: {e}")
            print(f"Standard output: {e.stdout}")
            print(f"Standard error: {e.stderr}")
            return False
    
    if args.conda_env:
        if not run_whisperx(args.conda_env):
            raise Exception(f"Failed to run WhisperX with provided environment: {args.conda_env}")
    else:
        # Try with hardcoded path and whisperx_installer
        if not run_whisperx("whisperx_installer"):
            # If it fails, try with "whisperx" and one of the usual Windows paths
            conda_executable = next((path for path in possible_paths if os.path.exists(path)), None)
            if conda_executable:
                if not run_whisperx("whisperx"):
                    raise Exception("Failed to run WhisperX with both whisperx_installer and whisperx environments")
            else:
                raise Exception("Could not find a valid conda path")
    
    json_file = output_dir / f"{audio_file.stem}.json"
    return json_file

def create_log_file(session_path):
    log_file = session_path / f"session-log--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.txt"
    return log_file

def create_json_file(session_path):
    json_file = session_path / f"session-data--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.json"
    return json_file

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")

def process_audio(input_path, session_path, args):
    audio_sources_dir = session_path / "audio_sources"
    audio_sources_dir.mkdir(exist_ok=True)
    processed_dir = audio_sources_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [f for f in os.listdir(input_path) if f.endswith(('.mp3', '.wav', '.flac', '.ogg'))]

    for file in files:
        input_file = Path(input_path) / file if os.path.isdir(input_path) else Path(file)
        output_file = audio_sources_dir / f"{input_file.stem}.wav"
        convert_to_wav(input_file, output_file)

        if args.separate:
            # Placeholder for speech separation
            pass

        if args.denoise or args.enhance:
            # Placeholder for denoising/enhancement
            pass

    return audio_sources_dir

def parse_transcription(json_file, audio_file, processed_dir, session_data, sample_method):
    with open(json_file, 'r', encoding='utf-8') as f:  
        transcription = json.load(f)

    audio = AudioSegment.from_wav(audio_file)
    qualifying_segments = []

    def save_segment(start, end, text):
        # Add 50ms buffer to the start time, but ensure we don't go below 0
        start_with_buffer = max(start - 0.04, 0)
        
        # Add 60ms to the end time, but ensure we don't exceed audio length or 11s limit
        end_with_buffer = min(end + 0.04, len(audio) / 1000, start_with_buffer + 11)
        
        duration = end_with_buffer - start_with_buffer
        if duration <= 11 and len(text) <= 200:
            audio_segment = audio[int(start_with_buffer * 1000):int(end_with_buffer * 1000)]
            output_file = processed_dir / f"{audio_file.stem}_segment_{len(qualifying_segments)}.wav"
            audio_segment.export(output_file, format="wav")

            if output_file.exists() and text.strip():
                qualifying_segments.append({
                    "audio_file": str(output_file.absolute()),
                    "text": text.strip(),
                    "speaker_name": "001"
                })
                return True
        return False

    if sample_method == "maximise-punctuation":
        process_maximise_punctuation(transcription['segments'], save_segment)
    elif sample_method == "punctuation-only":
        process_punctuation_only(transcription['segments'], save_segment)
    elif sample_method == "mixed":
        process_mixed(transcription['segments'], save_segment)

    print(f"Processed {audio_file.name}: {len(qualifying_segments)} qualifying segments")
    
    session_data['qualifying_segments'].extend(qualifying_segments)
    return len(qualifying_segments)

def process_maximise_punctuation(segments, save_segment):
    current_segment = {
        "start": None,
        "end": None,
        "text": "",
        "last_punctuation": None
    }

    for segment in segments:
        for word in segment.get('words', []):
            if 'start' not in word or 'end' not in word:
                continue

            if current_segment["start"] is None:
                current_segment["start"] = word['start']

            current_segment["end"] = word['end']
            current_segment["text"] += " " + word['word']

            if re.search(r'[.!?,;:]', word['word']):
                current_segment["last_punctuation"] = {
                    "end": word['end'],
                    "text": current_segment["text"]
                }

            duration = current_segment["end"] - current_segment["start"]
            if duration > 11 or len(current_segment["text"]) > 200:
                if current_segment["last_punctuation"]:
                    save_segment(current_segment["start"], 
                                 current_segment["last_punctuation"]["end"], 
                                 current_segment["last_punctuation"]["text"])
                    current_segment = {"start": None, "end": None, "text": "", "last_punctuation": None}
                else:
                    current_segment = {"start": None, "end": None, "text": "", "last_punctuation": None}

    if current_segment["start"] is not None and current_segment["last_punctuation"]:
        save_segment(current_segment["start"], 
                     current_segment["last_punctuation"]["end"], 
                     current_segment["last_punctuation"]["text"])

def process_punctuation_only(segments, save_segment):
    current_segment = {
        "start": None,
        "end": None,
        "text": ""
    }

    for segment in segments:
        for word in segment.get('words', []):
            if 'start' not in word or 'end' not in word:
                continue

            if current_segment["start"] is None:
                current_segment["start"] = word['start']

            current_segment["end"] = word['end']
            current_segment["text"] += " " + word['word']

            if re.search(r'[.!?]', word['word']) or (re.search(r'[,;:]', word['word']) and 
                                                     (current_segment["end"] - current_segment["start"] > 10 or 
                                                      len(current_segment["text"]) > 190)):
                save_segment(current_segment["start"], current_segment["end"], current_segment["text"])
                current_segment = {"start": None, "end": None, "text": ""}

    if current_segment["start"] is not None:
        save_segment(current_segment["start"], current_segment["end"], current_segment["text"])

def process_mixed(segments, save_segment):
    max_punct_segments = []
    process_maximise_punctuation(segments, lambda start, end, text: max_punct_segments.append((start, end, text)))

    for i, segment in enumerate(max_punct_segments):
        if i % 5 < 3:  # 60% of segments
            sub_segments = [s for s in segments if segment[0] <= s['start'] and s['end'] <= segment[1]]
            process_punctuation_only(sub_segments, save_segment)
        else:
            save_segment(*segment)

def create_metadata_files(session_data, session_path):
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

    train_size = int(0.8 * len(valid_samples))
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

def train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    OUT_PATH = os.path.join(output_path, "run", "training")
    CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "base_models", f"{version}")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
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

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        max_wav_length=max_audio_length,
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        audio=audio_config,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_loader_workers=8 if language != "ja" else 0,
        print_step=50,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    )

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

def copy_reference_samples(processed_dir, session_path):
    pandrator_voices_dir = Path("../Pandrator/tts_voices")
    if not pandrator_voices_dir.exists():
        print("Pandrator tts_voices directory does not exist. Skipping reference sample copying.")
        return

    # Use the session folder name for the reference directory
    session_name = session_path.name
    reference_dir = pandrator_voices_dir / session_name
    reference_dir.mkdir(exist_ok=True)

    # Get all wav files from the processed directory
    wav_files = list(processed_dir.glob('*.wav'))
    
    # Sort files by size in descending order
    wav_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    # Select the top 10% largest files
    top_10_percent = wav_files[:max(3, len(wav_files) // 10)]
    
    # Randomly select 3 files from the top 10%
    selected_files = random.sample(top_10_percent, min(3, len(top_10_percent)))
    
    # Copy selected files to the reference directory with new names
    for i, file in enumerate(selected_files, start=1):
        new_filename = f"{session_name}_{i}.wav"
        shutil.copy2(file, reference_dir / new_filename)
    
    print(f"Copied {len(selected_files)} reference samples to {reference_dir}")

def main():
    args = parse_arguments()
    session_path = create_session_folder(args.session)
    log_file = create_log_file(session_path)
    json_file = create_json_file(session_path)

    session_data = {'qualifying_segments': []}

    with open(json_file, 'w') as f:
        json.dump(session_data, f)

    audio_sources_dir = process_audio(args.input, session_path, args)

    total_segments = 0
    for audio_file in audio_sources_dir.glob('*.wav'):
        json_file = transcribe_audio(audio_file, args, session_path)
        segments_count = parse_transcription(json_file, audio_file, audio_sources_dir / "processed", session_data, args.sample_method)
        total_segments += segments_count
        print(f"Total qualifying segments after processing {audio_file.name}: {segments_count}")

        # Save updated session_data to JSON file after each audio file
        with open(json_file, 'w') as f:
            json.dump(session_data, f)

    print(f"Total qualifying segments across all files: {total_segments}")

    train_csv, eval_csv = create_metadata_files(session_data, session_path)

    if train_csv is None or eval_csv is None:
        print("Failed to create metadata files. Exiting.")
        return

    models_dir = session_path / "models"
    models_dir.mkdir(exist_ok=True)

    model_name = args.xtts_model_name or f"xtts_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_output_path = models_dir / model_name

    base_model_path = os.path.join(Path.cwd(), "base_models", args.xtts_base_model)
    download_models(base_model_path, args.xtts_base_model)

    speaker_file, config_file, checkpoint_file, tokenizer_file, training_output_path = train_gpt(
        custom_model="",
        version=args.xtts_base_model,
        language=args.source_language,
        num_epochs=args.epochs,
        batch_size=args.batch,
        grad_acumm=args.gradient,
        train_csv=str(train_csv),
        eval_csv=str(eval_csv),
        output_path=str(model_output_path)
    )

    optimization_message, optimized_model_path = optimize_model(model_output_path, base_model_path)

    print(f"Training completed. Model saved at: {model_output_path}")
    print(optimization_message)
    print(f"Optimized model path: {optimized_model_path}")

    if optimized_model_path:  # This ensures the optimization was successful
        copy_reference_samples(audio_sources_dir / "processed", session_path)
        print("Reference samples copied successfully.")
    else:
        print("Warning: Optimized model path not found. Reference samples not copied.")

    return True  # Indicate successful completion

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("Training process completed successfully.")
            sys.exit(0)  # Exit with success code
        else:
            print("Training process failed.")
            sys.exit(1)  # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)  # Exit with error code
