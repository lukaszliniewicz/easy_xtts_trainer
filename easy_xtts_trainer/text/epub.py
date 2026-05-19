from __future__ import annotations

from pathlib import Path


def infer_audio_chapter_index(audio_stem: str) -> int:
    """Infer zero-based chapter group index from an audio filename stem."""
    audio_numbers = "".join(filter(str.isdigit, audio_stem))
    return (int(audio_numbers) - 1) if audio_numbers else 0


class EpubProcessor:
    def __init__(self) -> None:
        self.chapters: list[str] = []

    def extract_chapter_text(self, html_content: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        for element in soup.find_all(
            class_=lambda value: value and ("caption" in value.lower() or "footnote" in value.lower())
        ):
            element.decompress()

        text_parts: list[str] = []
        for paragraph in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            if not any(css_class in paragraph.get("class", []) for css_class in ["caption", "footnote"]):
                text_parts.append(paragraph.get_text().strip())

        return "\n\n".join(filter(None, text_parts))

    def process_epub(self, epub_path: Path) -> None:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        print("Processing EPUB file...")
        book = epub.read_epub(epub_path)
        chapter_count = 0

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                filename = item.get_name()
                if "cover" not in filename.lower() and "toc" not in filename.lower():
                    print(f"Processing document: {filename}")
                    content = item.get_content().decode("utf-8")
                    soup = BeautifulSoup(content, "html.parser")

                    chapter_markers = soup.find_all(["h2", "div"], class_=lambda value: value and "chapter" in value.lower())
                    if not chapter_markers:
                        chapter_markers = soup.find_all("h2")

                    if chapter_markers:
                        print(f"Found {len(chapter_markers)} potential chapter markers")
                        for marker in chapter_markers:
                            chapter_text: list[str] = []
                            current = marker

                            while current:
                                next_sibling = current.find_next_sibling()
                                if next_sibling and (
                                    next_sibling.name == "h2"
                                    or (
                                        next_sibling.get("class")
                                        and "chapter" in " ".join(next_sibling.get("class")).lower()
                                    )
                                ):
                                    break

                                if current.name not in ["h2"]:
                                    text = current.get_text().strip()
                                    if text:
                                        chapter_text.append(text)
                                current = next_sibling

                            if chapter_text:
                                chapter_count += 1
                                print(f"Extracted chapter {chapter_count} with {len(chapter_text)} paragraphs")
                                self.chapters.append("\n\n".join(chapter_text))
                    else:
                        text = self.extract_chapter_text(content)
                        if text.strip():
                            chapter_count += 1
                            print(f"Extracted chapter {chapter_count} from {filename}")
                            self.chapters.append(text)

        print(f"Total chapters extracted: {len(self.chapters)}")

    def get_combined_chapters(self, num_chapters: int) -> list[str]:
        if not self.chapters:
            print("Warning: No chapters available")
            return []

        print(f"Combining chapters with {num_chapters} chapters per group")
        combined_chapters: list[str] = []

        for index in range(0, len(self.chapters), num_chapters):
            chapters_to_combine = self.chapters[index : index + num_chapters]
            combined_text = "\n\n".join(chapters_to_combine)
            combined_chapters.append(combined_text)
            print(
                "Created combined chapter group "
                f"{len(combined_chapters)} with {len(chapters_to_combine)} chapters"
            )

        return combined_chapters


def prepare_source_text_for_audio(
    source_text_path: Path,
    audio_file: Path,
    session_path: Path,
    chapter_per_audio: int,
) -> Path:
    """Resolve source text for alignment, expanding EPUB into a full-text temp file when needed."""
    if source_text_path.suffix.lower() != ".epub":
        return source_text_path

    # Kept for signature/backward compatibility with existing call sites.
    _ = audio_file

    chapters_per_group = max(1, chapter_per_audio)
    temp_text_dir = session_path / "temp_source_text"
    temp_text_dir.mkdir(exist_ok=True)
    temp_text_path = temp_text_dir / f"{source_text_path.stem}_full_c{chapters_per_group}.txt"

    if temp_text_path.exists() and temp_text_path.stat().st_size > 0:
        print(f"Reusing cached full-source text file: {temp_text_path}")
        return temp_text_path

    print(f"Processing epub file: {source_text_path}")
    epub_processor = EpubProcessor()
    epub_processor.process_epub(source_text_path)

    combined_chapters = epub_processor.get_combined_chapters(chapters_per_group)
    if not combined_chapters:
        raise ValueError("No chapters were extracted from the epub file")

    full_source_text = "\n\n".join(combined_chapters).strip()
    if not full_source_text:
        raise ValueError("No usable text content was extracted from the epub file")

    print(f"Creating temporary full-source text file: {temp_text_path}")
    with open(temp_text_path, "w", encoding="utf-8") as handle:
        handle.write(full_source_text)

    print(f"Using temporary text file for alignment: {temp_text_path}")
    return temp_text_path
