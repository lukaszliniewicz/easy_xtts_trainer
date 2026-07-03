from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Tuple


def is_session_reusable(session_path: Path) -> Tuple[bool, str]:
    """Check whether a session folder contains everything needed for reuse."""
    try:
        if not session_path.exists():
            return False, "Session folder does not exist"

        database_dir = session_path / "databases"
        train_csv = database_dir / "train_metadata.csv"
        eval_csv = database_dir / "eval_metadata.csv"

        if not all([database_dir.exists(), train_csv.exists(), eval_csv.exists()]):
            return False, "Missing database files"

        processed_dir = session_path / "audio_sources" / "processed"
        if not processed_dir.exists():
            return False, "Missing processed audio directory"

        wav_files = list(processed_dir.glob("*.wav"))
        if not wav_files:
            return False, "No processed audio files found"

        try:
            with open(train_csv, "r", encoding="utf-8") as handle:
                train_data = handle.readlines()
            with open(eval_csv, "r", encoding="utf-8") as handle:
                eval_data = handle.readlines()

            if len(train_data) < 2 or len(eval_data) < 2:
                return False, "Empty metadata files"
        except Exception as exc:
            return False, f"Error reading metadata files: {exc}"

        return True, "Session is reusable"
    except Exception as exc:
        return False, f"Error checking session: {exc}"


def create_new_session_name(original_session: Path, epochs: int, grads: int) -> Path:
    """Create a new session name with timestamp and training parameters."""
    timestamp = datetime.now().strftime("%y_%m_%d__%H_%M")
    new_name = f"{original_session.name}__{timestamp}__e{epochs}__g{grads}"
    return Path(new_name)


def update_csv_paths(csv_file: Path, old_session: Path, new_session: Path) -> bool:
    """Update audio file paths in CSV to point to a copied session."""
    try:
        with open(csv_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        new_lines = []
        header = True

        for line in lines:
            if header:
                new_lines.append(line)
                header = False
                continue

            if line.strip() and "|" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        raw_path = Path(parts[0])
                        old_session_abs = old_session.resolve()
                        new_session_abs = new_session.resolve()
                        if raw_path.is_absolute():
                            old_path = raw_path.resolve()
                        else:
                            candidates = [
                                (old_session_abs / raw_path).resolve(),
                                (old_session_abs / "audio_sources" / "processed" / raw_path.name).resolve(),
                            ]
                            old_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

                        if str(old_session_abs) in str(old_path):
                            rel_path = old_path.relative_to(old_session_abs)
                            new_path = new_session_abs / rel_path
                        else:
                            new_path = new_session_abs / "audio_sources" / "processed" / old_path.name

                        new_lines.append(f"{new_path}|{parts[1]}|{parts[2]}")
                    except Exception as exc:
                        print(f"Warning: Could not process path in line: {line.strip()}")
                        print(f"Error details: {exc}")
                        new_lines.append(line)
            else:
                new_lines.append(line)

        with open(csv_file, "w", encoding="utf-8") as handle:
            handle.writelines(new_lines)

        print(f"Updated paths in {csv_file.name}")
        return True
    except Exception as exc:
        print(f"Error updating paths in {csv_file}: {exc}")
        return False


def verify_copied_files(src_session: Path, dst_session: Path) -> bool:
    """Verify copied session files and audio payloads."""
    try:
        dst_train = dst_session / "databases" / "train_metadata.csv"
        dst_eval = dst_session / "databases" / "eval_metadata.csv"

        if not all(path.exists() for path in [dst_train, dst_eval]):
            return False

        src_audio_files = set((path.name for path in (src_session / "audio_sources" / "processed").glob("*.wav")))
        dst_audio_files = set((path.name for path in (dst_session / "audio_sources" / "processed").glob("*.wav")))

        if src_audio_files != dst_audio_files:
            print(
                "Audio files mismatch. "
                f"Source has {len(src_audio_files)} files, destination has {len(dst_audio_files)}"
            )
            return False

        for filename in src_audio_files:
            src_size = (src_session / "audio_sources" / "processed" / filename).stat().st_size
            dst_size = (dst_session / "audio_sources" / "processed" / filename).stat().st_size
            if src_size != dst_size:
                print(f"Size mismatch for {filename}")
                return False

        return True
    except Exception as exc:
        print(f"Error verifying copied files: {exc}")
        return False


def copy_session_files(src_session: Path, dst_session: Path) -> bool:
    """Copy reusable session files into a new destination session."""
    try:
        print("\nCopying session files...")

        src_session = src_session.resolve()
        dst_session = dst_session.resolve()
        print(f"Source session: {src_session}")
        print(f"Destination session: {dst_session}")

        dst_session.mkdir(parents=True, exist_ok=True)
        (dst_session / "databases").mkdir(exist_ok=True)
        (dst_session / "audio_sources" / "processed").mkdir(parents=True, exist_ok=True)

        print("Copying database files...")
        for csv_file in ["train_metadata.csv", "eval_metadata.csv"]:
            src = src_session / "databases" / csv_file
            dst = dst_session / "databases" / csv_file
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)

            print(f"Updating paths in {csv_file}...")
            if not update_csv_paths(dst, src_session, dst_session):
                raise RuntimeError(f"Failed to update paths in {csv_file}")

        print("Copying audio files...")
        src_processed = src_session / "audio_sources" / "processed"
        audio_files = list(src_processed.glob("*.wav"))
        total_files = len(audio_files)
        print(f"Found {total_files} audio files to copy")

        for index, wav_file in enumerate(audio_files, 1):
            dst_file = dst_session / "audio_sources" / "processed" / wav_file.name
            print(f"Copying {index}/{total_files}: {wav_file.name}")
            shutil.copy2(wav_file, dst_file)

        print("\nVerifying copied files...")
        if not verify_copied_files(src_session, dst_session):
            raise RuntimeError("File verification failed")

        print("Session files copied and verified successfully")
        return True
    except Exception as exc:
        print(f"Error copying session files: {exc}")
        return False
