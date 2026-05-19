# Easy XTTS Trainer

Easy XTTS Trainer is a CLI for preparing speech datasets and fine-tuning XTTS models.
It is designed to run inside the Pandrator ecosystem, while still supporting standalone usage.

## What is new

- Pixi-first WhisperX runtime (`auto`, `pixi`, `conda`) with legacy Conda fallback.
- Optional/lazy DeepFilterNet (`--denoise` soft-fails when DeepFilter is not installed).
- Robust `--source-text` correction flow for full-book `.txt` and `.epub` sources with partial-audio inputs.
- Python baseline moved to `3.13` with updated Torch/Coqui compatibility pins.

## Requirements

- Windows (primary target) with NVIDIA GPU recommended for training.
- Python `>=3.13,<3.14`.
- For source-text alignment: optional `ctc-forced-aligner` dependencies (ffmpeg on `PATH`; MSVC/Build Tools if building from source on Windows).

## Installation

### Pandrator-first (recommended)

Install and launch through Pandrator. Pandrator provides the Pixi environment layout used by this trainer.

### Standalone

```bash
pip install -r requirements.txt
```

Optional features:

```bash
# Optional: denoise support (only on supported Python/platform wheels)
pip install DeepFilterNet==0.5.6 DeepFilterLib==0.5.6

# Optional: source-text alignment (requires ffmpeg on PATH; prefer vendored prebuilt wheel)
pip install ./vendor/ctc_forced_aligner-0.3.0-cp313-cp313-win_amd64.whl

# Optional fallback: build latest MahmoudAshraf97 source from Git
pip install "ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@264e7a1f81bff9ff5e787a5537020c2ad0b0df02"
```

## Usage

Run help:

```bash
python -m easy_xtts_trainer --help
```

Or, if installed as a package:

```bash
easy-xtts-trainer --help
```

Example:

```bash
python -m easy_xtts_trainer \
  --source-language en \
  --input "D:\\audiobook" \
  --session "xtts-finetune-mybook" \
  --sample-method mixed \
  --method-proportion 6_4
```

## WhisperX runtime modes

`--whisperx-runner` controls transcription execution:

- `auto` (default):
  1. explicit CLI Pixi args,
  2. `WHISPERX_PIXI_EXE` + `WHISPERX_PIXI_MANIFEST`,
  3. Pandrator layout discovery (`bin/pixi.exe` + `envs/whisperx_installer/pixi.toml`),
  4. Conda fallback.
- `pixi`: require Pixi runtime.
- `conda`: use Conda flow only.

Related options:

- `--whisperx-pixi-exe`
- `--whisperx-pixi-manifest`
- legacy fallback options: `-conda_env`, `-conda_path`

## Source-text correction (`--source-text`)

When source text is provided, the pipeline is:

1. Run WhisperX baseline transcription.
2. Build query words from Whisper output.
3. Retrieve top text candidates from the full source (works with whole-book `.txt`/`.epub`).
4. Validate candidates with CTC alignment.
5. Keep the best candidate only when confidence is high; otherwise keep WhisperX output.

This prevents destructive over-correction while still improving proper nouns and wording when good source context exists.

## Key CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--source-language` | Source language code for transcription/training | required |
| `--input` | Input folder or single audio file | optional when reusing session |
| `--session` | Session folder name/path | `xtts-finetune-YYYY-MM-DD-HH-MM` |
| `--whisper-model` | Whisper model | `large-v3` |
| `--whisperx-runner` | WhisperX runtime mode (`auto`, `pixi`, `conda`) | `auto` |
| `--source-text` | `.txt` or `.epub` source text for correction/alignment | off |
| `--chapter-per-audio` | EPUB chapter grouping size before full-source merge | `1` |
| `--sample-method` | Segmentation strategy (`maximise-punctuation`, `punctuation-only`, `mixed`) | `maximise-punctuation` |
| `--method-proportion` | Mixed segmentation split (`N_M`) | `6_4` |
| `--training-proportion` | Train/validation split (`N_M`) | `8_2` |
| `--max-audio-time` | Max segment duration (seconds) | `11` |
| `--max-text-length` | Max segment text length (chars) | `200` |
| `--denoise` | Enable DeepFilterNet denoise (optional dependency) | off |
| `--dess` | De-esser | off |
| `--normalize` | Target LUFS normalization | unset (or `-16.0` when flag is used without value) |
| `--compress` | Compression profile (`male`, `female`, `neutral`) | off |
| `--prepare_dataset` | Prepare dataset only, skip training | off |

## Testing

```bash
python -m pytest
```

## Notes

- `ctc-forced-aligner` is optional by design.
- If DeepFilter is unavailable, `--denoise` prints a warning and processing continues.
- For Pandrator installations, prefer leaving WhisperX runner mode on `auto`.
