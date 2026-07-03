from __future__ import annotations

from pathlib import Path

from easy_xtts_trainer.session.reuse import (
    copy_session_files,
    create_new_session_name,
    is_session_reusable,
)


def _create_reusable_session(base_dir: Path, session_name: str = "session") -> Path:
    session = base_dir / session_name
    databases = session / "databases"
    processed = session / "audio_sources" / "processed"
    databases.mkdir(parents=True)
    processed.mkdir(parents=True)

    wav_file = processed / "sample.wav"
    wav_file.write_bytes(b"fake wav bytes")

    train_csv = databases / "train_metadata.csv"
    eval_csv = databases / "eval_metadata.csv"
    header = "audio_file|text|speaker_name\n"
    row = f"{wav_file}|hello|001\n"

    train_csv.write_text(header + row, encoding="utf-8")
    eval_csv.write_text(header + row, encoding="utf-8")
    return session


def test_is_session_reusable_true_for_valid_layout(tmp_path: Path) -> None:
    session = _create_reusable_session(tmp_path)
    reusable, message = is_session_reusable(session)

    assert reusable is True
    assert message == "Session is reusable"


def test_create_new_session_name_includes_training_params() -> None:
    name = create_new_session_name(Path("base_session"), epochs=7, grads=3)
    assert name.name.startswith("base_session__")
    assert name.name.endswith("__e7__g3")


def test_copy_session_files_copies_audio_and_updates_csv(tmp_path: Path) -> None:
    src_session = _create_reusable_session(tmp_path, session_name="src_session")
    dst_session = tmp_path / "dst_session"

    assert copy_session_files(src_session, dst_session) is True

    copied_csv = dst_session / "databases" / "train_metadata.csv"
    copied_wav = dst_session / "audio_sources" / "processed" / "sample.wav"

    assert copied_csv.exists()
    assert copied_wav.exists()
    assert str(dst_session.resolve()) in copied_csv.read_text(encoding="utf-8")


def test_copy_session_files_accepts_exported_relative_audio_paths(tmp_path: Path) -> None:
    src_session = _create_reusable_session(tmp_path, session_name="relative_src")
    relative_row = "audio_sources/processed/sample.wav|hello|001\n"
    for csv_path in (src_session / "databases").glob("*_metadata.csv"):
        csv_path.write_text("audio_file|text|speaker_name\n" + relative_row, encoding="utf-8")

    dst_session = tmp_path / "relative_dst"

    assert copy_session_files(src_session, dst_session) is True

    copied_csv = dst_session / "databases" / "train_metadata.csv"
    copied_wav = dst_session / "audio_sources" / "processed" / "sample.wav"

    assert copied_wav.exists()
    copied_text = copied_csv.read_text(encoding="utf-8")
    assert str(copied_wav.resolve()) in copied_text
