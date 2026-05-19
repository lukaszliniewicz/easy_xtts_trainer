from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def process_audio_files(
    input_files: Iterable[str | Path],
    output_dir: str | Path,
    args: Any,
    *,
    processor_class: Any | None = None,
) -> None:
    """Process audio files with the selected enhancement effects."""
    resolved_processor_class = processor_class
    if resolved_processor_class is None:
        from easy_xtts_trainer.audio.processor import AudioProcessor

        resolved_processor_class = AudioProcessor

    processor = resolved_processor_class(target_sr=args.sample_rate)
    output_dir_path = Path(output_dir)

    for input_file in input_files:
        try:
            input_path = Path(input_file)
            output_path = output_dir_path / input_path.name

            success = processor.process_audio(
                str(input_path),
                str(output_path),
                normalize_target=-float(args.normalize) if args.normalize else None,
                dess_profile="high" if args.dess else None,
                denoise=args.denoise,
                compress_profile=args.compress if args.compress else None,
            )

            if success:
                print(f"Successfully processed: {input_path.name}")
            else:
                print(f"Failed to process: {input_path.name}")

        except Exception as exc:
            print(f"Error processing {input_file}: {str(exc)}")
