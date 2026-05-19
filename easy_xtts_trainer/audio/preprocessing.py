from __future__ import annotations

import shutil
import subprocess
import traceback
from pathlib import Path

from pydub import AudioSegment

from easy_xtts_trainer.config import DatasetRuntimeConfig

SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".webm")


def collect_input_audio_files(input_path: str | Path) -> list[Path]:
    resolved_path = Path(input_path)
    if resolved_path.is_file():
        return [resolved_path]

    return [
        file_path
        for file_path in resolved_path.rglob("*")
        if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]


def convert_to_wav(input_path: Path, output_path: Path, target_sample_rate: int) -> None:
    # If input is already a WAV file, check its properties
    if input_path.suffix.lower() == ".wav":
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(input_path)
            is_mono = waveform.shape[0] == 1

            # If the file already matches our requirements, just copy it
            if sample_rate == target_sample_rate and is_mono:
                shutil.copy2(input_path, output_path)
                return

        except Exception as exc:
            print(f"Error reading WAV file {input_path}: {exc}")
            # Continue to regular conversion if there's an error reading the WAV

    # Convert the file if it's not WAV or doesn't match requirements
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")


def process_audio(input_path: str | Path, session_path: Path, config: DatasetRuntimeConfig) -> Path | None:
    """Process audio files and prepare them for training."""
    # Create necessary directories
    audio_sources_dir = session_path / "audio_sources"
    audio_sources_dir.mkdir(exist_ok=True, parents=True)
    processed_dir = audio_sources_dir / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    files = collect_input_audio_files(input_path)
    breath_removal_enabled = config.breath

    # Check if breath removal is available if --breath is passed
    if breath_removal_enabled:
        try:
            subprocess.run(["breath-removal", "--help"], capture_output=True, text=True)
            breath_removal_available = True
            print("Breath removal tool found and available.")
        except FileNotFoundError:
            breath_removal_available = False
            print("Warning: breath-removal command not found. Continuing without breath removal.")
            breath_removal_enabled = False
        except subprocess.CalledProcessError:
            # Command exists but returned error - might be okay if --help isn't supported
            breath_removal_available = True
            print("Breath removal tool found.")
    else:
        breath_removal_available = False

    # Create a temporary directory for breath removal if needed
    breath_removal_dir = None
    if breath_removal_enabled and breath_removal_available:
        breath_removal_dir = audio_sources_dir / "breath_removal_temp"
        breath_removal_dir.mkdir(exist_ok=True)
        print(f"Created temporary directory for breath removal: {breath_removal_dir}")

    processed_files: list[Path] = []

    for file_path in files:
        try:
            if breath_removal_enabled and breath_removal_available:
                # Run breath removal
                print(f"Processing {file_path.name} with breath removal...")
                breath_output = breath_removal_dir / f"breath_removal_{file_path.name}"

                try:
                    command = [
                        "breath-removal",
                        "-i",
                        str(file_path.absolute()),
                        "-o",
                        str(breath_removal_dir),
                    ]
                    print(f"Running command: {' '.join(command)}")

                    subprocess.run(command, check=True, capture_output=True, text=True)

                    # Check if the breath removal output exists
                    if breath_output.exists():
                        print(f"Breath removal successful for {file_path.name}")
                        file_to_process = breath_output
                    else:
                        print(f"Warning: Breath removal output not found for {file_path.name}, using original file")
                        file_to_process = file_path
                except subprocess.CalledProcessError as exc:
                    print(f"Error during breath removal for {file_path.name}:")
                    print(f"Command output: {exc.stdout}")
                    print(f"Command error: {exc.stderr}")
                    print("Using original file instead")
                    file_to_process = file_path
            else:
                file_to_process = file_path

            print(f"Converting {file_to_process.name} to WAV format...")
            # Convert to WAV with target sample rate
            output_file = audio_sources_dir / f"{file_path.stem}.wav"
            convert_to_wav(file_to_process, output_file, config.sample_rate)
            processed_files.append(output_file)
            print(f"Successfully processed {file_path.name} -> {output_file.name}")

        except Exception as exc:
            print(f"Error processing {file_path.name}: {str(exc)}")
            print(f"Stack trace: {traceback.format_exc()}")
            continue

    # Clean up breath removal temporary directory if it was created
    if breath_removal_dir and breath_removal_dir.exists():
        try:
            shutil.rmtree(breath_removal_dir)
            print("Cleaned up breath removal temporary directory")
        except Exception as exc:
            print(f"Warning: Could not remove temporary breath removal directory: {exc}")

    if not processed_files:
        print("No files were successfully processed!")
        return None

    print(f"Successfully processed {len(processed_files)} files")
    return audio_sources_dir
