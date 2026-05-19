from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Optional

from easy_xtts_trainer.transcription.formats import convert_ctc_to_whisperx_format

LANGUAGE_MAP = {
    "en": "eng",
    "nl": "nld",
}

TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)
DEFAULT_MAX_QUERY_WORDS = 320
DEFAULT_CANDIDATE_TOP_K = 3


@dataclass(frozen=True)
class CTCAlignmentRequest:
    audio_file: Path
    source_text_path: Path
    output_dir: Path
    source_language: str
    align_model: Optional[str] = None
    whisper_json_path: Optional[Path] = None
    candidate_workspace_dir: Optional[Path] = None
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    min_alignment_score: float = 0.5
    min_word_coverage: float = 0.35


@dataclass(frozen=True)
class SourceTextCandidate:
    text: str
    retrieval_score: float
    start_token: int
    end_token: int


@dataclass(frozen=True)
class CTCAlignmentAttempt:
    source_text_path: Path
    retrieval_score: float
    ctc_score: float
    word_coverage: float
    combined_score: float
    ctc_data: dict


@dataclass(frozen=True)
class SourceTextIndex:
    source_text: str
    source_tokens: list[str]
    source_spans: list[tuple[int, int]]
    source_positions: dict[str, list[int]]


_SOURCE_TEXT_INDEX_CACHE: dict[tuple[str, int, int], SourceTextIndex] = {}


def map_ctc_language(source_language: str) -> str:
    return LANGUAGE_MAP.get(source_language, source_language)


def build_ctc_command(
    audio_path: Path,
    text_path: Path,
    ctc_language: str,
    align_model: Optional[str] = None,
) -> list[str]:
    command = [
        "ctc-forced-aligner",
        "--audio_path",
        str(audio_path),
        "--text_path",
        str(text_path),
        "--language",
        ctc_language,
        "--romanize",
    ]

    if align_model:
        command.extend(["--alignment_model", align_model])

    return command


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _tokenize_words_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    spans: list[tuple[int, int]] = []

    for match in TOKEN_PATTERN.finditer(text):
        words.append(match.group(0).lower())
        spans.append((match.start(), match.end()))

    return words, spans


def _build_source_positions(source_tokens: list[str]) -> dict[str, list[int]]:
    source_positions: dict[str, list[int]] = {}
    for index, token in enumerate(source_tokens):
        source_positions.setdefault(token, []).append(index)
    return source_positions


def load_source_text_index(source_text_path: Path) -> SourceTextIndex:
    resolved_path = source_text_path.resolve()
    file_stat = resolved_path.stat()
    path_key = str(resolved_path).lower()
    cache_key = (path_key, file_stat.st_mtime_ns, file_stat.st_size)

    cached_index = _SOURCE_TEXT_INDEX_CACHE.get(cache_key)
    if cached_index is not None:
        return cached_index

    # Remove stale entries for the same path if the source text has changed.
    for existing_key in list(_SOURCE_TEXT_INDEX_CACHE):
        if existing_key[0] == path_key and existing_key != cache_key:
            del _SOURCE_TEXT_INDEX_CACHE[existing_key]

    with open(resolved_path, "r", encoding="utf-8") as handle:
        source_text = handle.read()

    source_tokens, source_spans = _tokenize_words_with_spans(source_text)
    source_index = SourceTextIndex(
        source_text=source_text,
        source_tokens=source_tokens,
        source_spans=source_spans,
        source_positions=_build_source_positions(source_tokens),
    )
    _SOURCE_TEXT_INDEX_CACHE[cache_key] = source_index
    return source_index


def extract_whisper_query_words(
    whisper_json_path: Path,
    max_words: int = DEFAULT_MAX_QUERY_WORDS,
) -> list[str]:
    if not whisper_json_path.exists():
        return []

    with open(whisper_json_path, "r", encoding="utf-8") as handle:
        whisper_data = json.load(handle)

    words: list[str] = []
    for segment in whisper_data.get("word_segments", []):
        words.extend(_tokenize_words(str(segment.get("word", ""))))

    if not words:
        for segment in whisper_data.get("segments", []):
            segment_words = segment.get("words")
            if isinstance(segment_words, list):
                for word in segment_words:
                    if isinstance(word, dict):
                        words.extend(_tokenize_words(str(word.get("word", ""))))
            elif isinstance(segment, dict):
                words.extend(_tokenize_words(str(segment.get("text", ""))))

    if not words:
        words.extend(_tokenize_words(str(whisper_data.get("text", ""))))

    if len(words) > max_words:
        first_half = max_words // 2
        second_half = max_words - first_half
        words = words[:first_half] + words[-second_half:]

    return words


def _expand_to_sentence_bounds(
    source_text: str,
    start_char: int,
    end_char: int,
    max_extension: int = 320,
) -> tuple[int, int]:
    expanded_start = start_char
    extension_count = 0

    while expanded_start > 0 and extension_count < max_extension:
        if source_text[expanded_start - 1] in ".!?\n":
            break
        expanded_start -= 1
        extension_count += 1

    expanded_end = end_char
    extension_count = 0
    text_length = len(source_text)

    while expanded_end < text_length and extension_count < max_extension:
        if source_text[expanded_end] in ".!?\n":
            expanded_end += 1
            break
        expanded_end += 1
        extension_count += 1

    return expanded_start, expanded_end


def build_source_text_candidates(
    source_text: str,
    query_words: list[str],
    top_k: int = DEFAULT_CANDIDATE_TOP_K,
    *,
    source_tokens: Optional[list[str]] = None,
    source_spans: Optional[list[tuple[int, int]]] = None,
    source_positions: Optional[dict[str, list[int]]] = None,
) -> list[SourceTextCandidate]:
    if top_k < 1:
        return []

    if source_tokens is None or source_spans is None:
        source_tokens, source_spans = _tokenize_words_with_spans(source_text)

    if not source_tokens:
        return []

    normalized_query = [word.lower() for word in query_words if word]
    if not normalized_query:
        return [
            SourceTextCandidate(
                text=source_text.strip(),
                retrieval_score=0.0,
                start_token=0,
                end_token=len(source_tokens),
            )
        ]

    if source_positions is None:
        source_positions = _build_source_positions(source_tokens)

    vote_counter: dict[int, float] = defaultdict(float)
    for query_index, token in enumerate(normalized_query):
        if len(token) < 3:
            continue

        matches = source_positions.get(token, [])
        if not matches or len(matches) > 400:
            continue

        token_weight = 1.0 + min(0.5, len(token) / 12.0)
        for source_index in matches:
            start_index = max(0, source_index - query_index)
            vote_counter[start_index] += token_weight

    if not vote_counter:
        window_token_length = max(120, min(len(normalized_query) + 120, 1800))
        if len(source_tokens) <= window_token_length:
            return [
                SourceTextCandidate(
                    text=source_text.strip(),
                    retrieval_score=0.0,
                    start_token=0,
                    end_token=len(source_tokens),
                )
            ]

        max_start = max(0, len(source_tokens) - window_token_length)
        starts: list[int] = [0]
        if top_k > 1:
            step = max(1, max_start // (top_k - 1))
            starts = [min(max_start, index * step) for index in range(top_k)]

        fallback_candidates: list[SourceTextCandidate] = []
        for start_index in starts:
            end_index = min(len(source_tokens), start_index + window_token_length)
            char_start = source_spans[start_index][0]
            char_end = source_spans[end_index - 1][1]
            expanded_start, expanded_end = _expand_to_sentence_bounds(source_text, char_start, char_end)
            candidate_text = source_text[expanded_start:expanded_end].strip()
            if candidate_text:
                fallback_candidates.append(
                    SourceTextCandidate(
                        text=candidate_text,
                        retrieval_score=0.0,
                        start_token=start_index,
                        end_token=end_index,
                    )
                )

        if fallback_candidates:
            return fallback_candidates

        return [
            SourceTextCandidate(
                text=source_text.strip(),
                retrieval_score=0.0,
                start_token=0,
                end_token=len(source_tokens),
            )
        ]

    window_token_length = max(120, min(len(normalized_query) + 120, 1800))
    dedupe_distance = max(40, window_token_length // 3)
    max_candidates = max(top_k * 6, 10)

    sorted_votes = sorted(vote_counter.items(), key=lambda item: item[1], reverse=True)
    candidate_starts: list[int] = []
    for start_index, _ in sorted_votes:
        if any(abs(start_index - existing_start) < dedupe_distance for existing_start in candidate_starts):
            continue
        candidate_starts.append(start_index)
        if len(candidate_starts) >= max_candidates:
            break

    query_counter = Counter(normalized_query)
    query_word_count = len(normalized_query)
    max_vote = sorted_votes[0][1]

    candidates: list[SourceTextCandidate] = []
    for start_index in candidate_starts:
        end_index = min(len(source_tokens), start_index + window_token_length)
        if end_index <= start_index:
            continue

        candidate_tokens = source_tokens[start_index:end_index]
        if not candidate_tokens:
            continue

        candidate_counter = Counter(candidate_tokens)
        overlap = sum(min(candidate_counter[token], count) for token, count in query_counter.items())
        precision = overlap / max(1, len(candidate_tokens))
        recall = overlap / max(1, query_word_count)
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        vote_score = vote_counter[start_index] / max_vote if max_vote > 0 else 0.0
        retrieval_score = (0.65 * vote_score) + (0.35 * f1_score)

        char_start = source_spans[start_index][0]
        char_end = source_spans[end_index - 1][1]
        expanded_start, expanded_end = _expand_to_sentence_bounds(source_text, char_start, char_end)
        candidate_text = source_text[expanded_start:expanded_end].strip()
        if not candidate_text:
            continue

        candidates.append(
            SourceTextCandidate(
                text=candidate_text,
                retrieval_score=round(retrieval_score, 5),
                start_token=start_index,
                end_token=end_index,
            )
        )

    deduped_candidates: list[SourceTextCandidate] = []
    seen_texts: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.retrieval_score, reverse=True):
        dedupe_key = candidate.text[:1200]
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        deduped_candidates.append(candidate)
        if len(deduped_candidates) >= top_k:
            break

    if deduped_candidates:
        return deduped_candidates

    return [
        SourceTextCandidate(
            text=source_text.strip(),
            retrieval_score=0.0,
            start_token=0,
            end_token=len(source_tokens),
        )
    ]


def score_ctc_alignment(ctc_data: dict) -> float:
    scores: list[float] = []
    for segment in ctc_data.get("segments", []):
        try:
            scores.append(float(segment.get("score", 0.0)))
        except (TypeError, ValueError):
            continue

    if not scores:
        return 0.0

    scores.sort()
    middle = len(scores) // 2
    if len(scores) % 2 == 0:
        median = (scores[middle - 1] + scores[middle]) / 2
    else:
        median = scores[middle]

    mean = sum(scores) / len(scores)
    combined = (mean + median) / 2
    return max(0.0, min(1.0, combined))


def should_use_ctc_result(
    alignment_score: float,
    word_coverage: float,
    min_alignment_score: float,
    min_word_coverage: float,
) -> bool:
    return alignment_score >= min_alignment_score and word_coverage >= min_word_coverage


def _compute_word_coverage(ctc_data: dict, reference_word_count: int) -> float:
    if reference_word_count <= 0:
        return 1.0

    aligned_words = 0
    for segment in ctc_data.get("segments", []):
        aligned_words += len(_tokenize_words(str(segment.get("text", ""))))

    if aligned_words <= 0:
        return 0.0

    return min(aligned_words, reference_word_count) / reference_word_count


def _run_single_ctc_candidate(
    request: CTCAlignmentRequest,
    text_path: Path,
    retrieval_score: float,
    ctc_language: str,
    reference_word_count: int,
) -> CTCAlignmentAttempt:
    command = build_ctc_command(
        audio_path=request.audio_file,
        text_path=text_path,
        ctc_language=ctc_language,
        align_model=request.align_model,
    )

    print("\nExecuting CTC Forced Aligner command:")
    print(f"Command: {' '.join(command)}")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("\nCTC Aligner Output:")
    print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")

    ctc_json = request.audio_file.parent / f"{request.audio_file.stem}.json"
    print(f"Looking for CTC output at: {ctc_json}")
    if not ctc_json.exists():
        print(f"Warning: Expected JSON file not found at {ctc_json}")
        print("Checking directory contents:")
        print(list(request.audio_file.parent.glob("*.json")))
        raise FileNotFoundError(f"CTC alignment output not found: {ctc_json}")

    with open(ctc_json, "r", encoding="utf-8") as handle:
        ctc_data = json.load(handle)

    ctc_score = score_ctc_alignment(ctc_data)
    word_coverage = _compute_word_coverage(ctc_data, reference_word_count)
    combined_score = (0.75 * ctc_score) + (0.15 * retrieval_score) + (0.10 * word_coverage)

    return CTCAlignmentAttempt(
        source_text_path=text_path,
        retrieval_score=retrieval_score,
        ctc_score=ctc_score,
        word_coverage=word_coverage,
        combined_score=combined_score,
        ctc_data=ctc_data,
    )


def _write_json_file(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_ctc_alignment(request: CTCAlignmentRequest) -> Path:
    print(f"Using source text file: {request.source_text_path}")
    source_index = load_source_text_index(request.source_text_path)
    text_content = source_index.source_text
    print(f"Text content length: {len(text_content)} characters")
    print(f"Source token count: {len(source_index.source_tokens)}")

    ctc_language = map_ctc_language(request.source_language)
    print(f"Mapped language code: {request.source_language} -> {ctc_language}")

    if request.align_model:
        print(f"Using custom alignment model: {request.align_model}")

    query_words: list[str] = []
    if request.whisper_json_path:
        query_words = extract_whisper_query_words(request.whisper_json_path)
        print(f"Loaded {len(query_words)} WhisperX words for source retrieval")

    candidate_files: list[tuple[Path, float]] = []
    if query_words:
        top_k = max(1, request.candidate_top_k)
        source_candidates = build_source_text_candidates(
            text_content,
            query_words,
            top_k=top_k,
            source_tokens=source_index.source_tokens,
            source_spans=source_index.source_spans,
            source_positions=source_index.source_positions,
        )
        candidate_workspace_dir = (
            request.candidate_workspace_dir
            if request.candidate_workspace_dir is not None
            else (request.output_dir / "ctc_candidates")
        )
        candidate_workspace_dir.mkdir(parents=True, exist_ok=True)

        for index, candidate in enumerate(source_candidates, start=1):
            candidate_file = candidate_workspace_dir / f"{request.audio_file.stem}_candidate_{index:02d}.txt"
            with open(candidate_file, "w", encoding="utf-8") as handle:
                handle.write(candidate.text)
            candidate_files.append((candidate_file, candidate.retrieval_score))

        print(f"Prepared {len(candidate_files)} source-text candidates for CTC validation")
    else:
        candidate_files = [(request.source_text_path, 0.0)]
        print("Using provided source text as a single CTC candidate")

    reference_word_count = len(query_words)
    attempts: list[CTCAlignmentAttempt] = []
    for index, (candidate_file, retrieval_score) in enumerate(candidate_files, start=1):
        print(f"\n--- CTC candidate {index}/{len(candidate_files)}: {candidate_file} ---")
        try:
            attempt = _run_single_ctc_candidate(
                request=request,
                text_path=candidate_file,
                retrieval_score=retrieval_score,
                ctc_language=ctc_language,
                reference_word_count=reference_word_count,
            )
            attempts.append(attempt)

            if len(candidate_files) > 1:
                candidate_ctc_path = request.output_dir / f"{request.audio_file.stem}_candidate_{index:02d}_ctc.json"
                _write_json_file(candidate_ctc_path, attempt.ctc_data)

            print(
                "Candidate scores: "
                f"retrieval={attempt.retrieval_score:.3f}, "
                f"ctc={attempt.ctc_score:.3f}, "
                f"coverage={attempt.word_coverage:.3f}, "
                f"combined={attempt.combined_score:.3f}"
            )
        except subprocess.CalledProcessError as exc:
            print(f"CTC candidate failed: {candidate_file}")
            print(f"Error: {exc}")
            print(f"stdout: {exc.stdout}")
            print(f"stderr: {exc.stderr}")
        except Exception as exc:
            print(f"CTC candidate failed: {candidate_file}")
            print(f"Error: {exc}")

    if not attempts:
        raise RuntimeError("All CTC candidate alignment attempts failed")

    best_attempt = max(attempts, key=lambda attempt: attempt.combined_score)
    print(
        "\nSelected best CTC candidate: "
        f"{best_attempt.source_text_path} "
        f"(combined={best_attempt.combined_score:.3f}, ctc={best_attempt.ctc_score:.3f}, "
        f"coverage={best_attempt.word_coverage:.3f})"
    )

    if request.whisper_json_path:
        use_coverage_gate = bool(query_words)
        word_coverage = best_attempt.word_coverage if use_coverage_gate else 1.0
        min_word_coverage = request.min_word_coverage if use_coverage_gate else 0.0

        if not should_use_ctc_result(
            alignment_score=best_attempt.ctc_score,
            word_coverage=word_coverage,
            min_alignment_score=request.min_alignment_score,
            min_word_coverage=min_word_coverage,
        ):
            print(
                "Warning: CTC confidence below threshold; "
                "falling back to WhisperX transcript for this audio file."
            )
            return request.whisper_json_path

    original_ctc_path = request.output_dir / f"{request.audio_file.stem}_original_ctc.json"
    _write_json_file(original_ctc_path, best_attempt.ctc_data)
    print(f"Saved original CTC output to: {original_ctc_path}")

    whisperx_data = convert_ctc_to_whisperx_format(best_attempt.ctc_data)
    print(f"Converted {len(whisperx_data['word_segments'])} word segments")

    json_file = request.output_dir / f"{request.audio_file.stem}.json"
    _write_json_file(json_file, whisperx_data)
    print(f"Saved WhisperX format JSON to: {json_file}")
    print(f"File size: {json_file.stat().st_size} bytes")

    return json_file
