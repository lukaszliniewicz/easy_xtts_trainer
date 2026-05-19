from __future__ import annotations

from pathlib import Path

from easy_xtts_trainer.text.epub import infer_audio_chapter_index, prepare_source_text_for_audio


def test_infer_audio_chapter_index_from_numbered_stem() -> None:
    assert infer_audio_chapter_index("chapter_03") == 2


def test_infer_audio_chapter_index_without_numbers() -> None:
    assert infer_audio_chapter_index("intro") == 0


def test_prepare_source_text_for_audio_non_epub_returns_original_path(tmp_path: Path) -> None:
    source_text = tmp_path / "book.txt"
    source_text.write_text("hello", encoding="utf-8")

    resolved = prepare_source_text_for_audio(
        source_text_path=source_text,
        audio_file=tmp_path / "audio_01.wav",
        session_path=tmp_path,
        chapter_per_audio=1,
    )

    assert resolved == source_text


def test_prepare_source_text_for_audio_epub_builds_full_source_file(monkeypatch, tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    call_count = 0

    def fake_process_epub(self, _: Path) -> None:
        nonlocal call_count
        call_count += 1
        self.chapters = ["Chapter one text.", "Chapter two text.", "Chapter three text."]

    monkeypatch.setattr("easy_xtts_trainer.text.epub.EpubProcessor.process_epub", fake_process_epub)

    resolved_first = prepare_source_text_for_audio(
        source_text_path=epub_path,
        audio_file=tmp_path / "audio_01.wav",
        session_path=tmp_path,
        chapter_per_audio=1,
    )
    resolved_second = prepare_source_text_for_audio(
        source_text_path=epub_path,
        audio_file=tmp_path / "audio_77.wav",
        session_path=tmp_path,
        chapter_per_audio=1,
    )

    assert resolved_first == resolved_second
    combined_text = resolved_first.read_text(encoding="utf-8")
    assert "Chapter one text." in combined_text
    assert "Chapter two text." in combined_text
    assert "Chapter three text." in combined_text
    assert call_count == 1
