
# Easy XTTS Trainer

This command-line app simplifies the process of training custom XTTS models. While designed to work seamlessly with [Pandrator](https://github.com/lukaszliniewicz/Pandrator) for a GUI-driven experience, it can also be used standalone.  Pandrator offers immediate model loading, testing, and convenient installers/portable packages that bundle this trainer and its dependencies.

## Table of Contents
- [Installation](#installation)
  - [Setting up Conda Environments](#setting-up-conda-environments)
  - [Installing the App](#installing-the-app)
- [System Requirements](#system-requirements)
- [Usage](#usage)
  - [Arguments](#arguments)
- [Segment Methods Explanation](#segment-methods-explanation)
- [Audio Preprocessing and Refinement](#audio-preprocessing-and-refinement)


## Installation

### Setting up Conda Environments

Before installing the app, you need to set up a Conda environment named `xtts_training` for the app itself and `whisperx` for the WhisperX transcription tool.

**Create the `xtts_training` environment:**

```bash
conda create --name xtts_training python=3.10
```

**Create the `whisperx` environment:**

```bash
conda create --name whisperx python=3.10
```

**Activate the `whisperx` environment and install WhisperX:**

```bash
conda activate whisperx
conda run pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda install cudnn=8.9.7.29 -c conda-forge -y
pip install git+https://github.com/m-bain/whisperx.git
conda deactivate
```


### Installing the App

**Clone the repository:**

```bash
git clone https://github.com/your-repo/xtts-training-app.git
```

**Activate the `xtts_training` environment:**

```bash
conda activate xtts_training
```

**Install the requirements:**

```bash
pip install -r requirements.txt
```

**Install PyTorch and Torchaudio:**
Ensure these versions are compatible with your CUDA setup.

```bash
pip install torch torchaudio
```

## System Requirements

* Nvidia GPU with at least 8GB of VRAM (11GB recommended for larger models/batch sizes)
* Python 3.10
* Conda environments: `xtts_training` and `whisperx`

## Usage

**Run the app with the `--help` flag to see all options:**

```bash
python easy-xtts-trainer.py --help
```

### Arguments

| Argument | Description | Required | Options/Defaults |
|---|---|---|---|
| `--source-language` | Source language for the model. | Yes |  en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu |
| `--whisper-model` | Whisper model for transcription. | No | medium, medium.en, large-v2, large-v3 (default: `large-v3`) |
| `--denoise` | Apply DeepFilterNet noise reduction to audio. | No | False |
| `--enhance` | Placeholder for future audio enhancement features. Not currently implemented. | No | False (not functional) |
| `-i`, `--input` | Input folder containing audio files or a single audio file. | Yes | - |
| `--session` | Name of the output folder for the training session. | No | `xtts-finetune-YYYY-MM-DD-HH-MM` |
| `--separate` | Placeholder for future speech separation features. Not currently implemented. | No | False (not functional) |
| `--epochs` | Number of training epochs. | No | 6 |
| `--xtts-base-model` | XTTS base model version. | No | v2.0.2 |
| `--batch` | Training batch size. | No | 2 |
| `--gradient` | Gradient accumulation steps. | No | 1 |
| `--xtts-model-name` | Name for the trained model. | No | `xtts_model_YYYYMMDD_HHMMSS` |
| `--sample-method` | Method for segmenting audio and text for training. | No | maximise-punctuation, punctuation-only, mixed (default: `maximise-punctuation`) |
| `-conda_env` | Name of the Conda environment to use (overrides automatic detection). | No | - |
| `-conda_path` | Path to the Conda installation (overrides automatic detection). | No | - |
| `--sample-rate` | Sample rate for audio. | No | 22050, 44100 (default: 22050) |
| `--max-audio-time` | Maximum audio segment duration in seconds. | No | 11 |
| `--max-text-length` | Maximum text segment length in characters. | No | 200 |
| `--align-model` | Model used for phoneme alignment during transcription (for WhisperX).  | No | -  |
| `--normalize` | Normalize audio to target LUFS. | No | -16.0 (if flag is used without a value) |
| `--dess` | Apply de-essing to audio. | No | False |
| `--compress` | Apply dynamic range compression. | No | male, female, neutral |
| `--method-proportion` | Proportion of `maximise-punctuation` to `punctuation-only` when using `mixed` sample method. Format: `N_M` where N+M=10 (e.g., `6_4` for 60/40 split). | No | 6_4 (60% maximise, 40% punctuation) |
| `--training-proportion` | Proportion of data for training vs. validation. Format: `N_M` where N+M=10 (e.g., `8_2` for 80/20 split). | No | 8_2 (80% train, 20% validation) |



## Segment Methods Explanation

The app employs these methods to prepare training segments:

* **`maximise-punctuation`**:  Prioritizes creating longer segments within the `--max-audio-time` (default 11 seconds) and `--max-text-length` (default 200 characters) limits. Segments are split at sentence-ending or clause-ending punctuation marks (`.!?;,-`). This method aims for fewer, longer segments, potentially capturing more context.

* **`punctuation-only`**: Segments are strictly split at every sentence-ending or clause-ending punctuation mark, regardless of the resulting segment length. This can create many shorter segments.

* **`mixed`**: Combines both methods. The `--method-proportion` argument (default `6_4`) controls the ratio. For example, `6_4` means 60% of the audio duration will be segmented using `maximise-punctuation` and 40% using `punctuation-only`.


## Audio Preprocessing and Refinement

The app now integrates audio preprocessing steps to improve training data quality. These options can be used individually or combined:

* **`--normalize <target_lufs>`**: Normalizes audio to a target LUFS value (Loudness Unit Full Scale). This helps ensure consistent loudness across training samples. The default target is -16.0 LUFS if the flag is used without specifying a value.
* **`--dess`**: Applies de-essing to reduce harsh sibilant sounds ("s", "sh", "ch").
* **`--denoise`**: Uses DeepFilterNet for noise reduction.
* **`--compress <profile>`**: Applies dynamic range compression using profiles optimized for `male`, `female`, or `neutral` voices. This reduces the difference between the loudest and quietest parts of the audio.
* **`--sample-rate <rate>`**: Sets the target sample rate for audio.  Default is 22050 Hz, which is recommended for XTTS.  Optionally use 44100 Hz.


**Segment Refinement:**

To minimize crossovers between segments and create clean transitions, the app refines the segment boundaries by analyzing the audio *between* adjacent segments:

1. **Boundary Zone:**  The app identifies the boundary zone between the end of one segment and the beginning of the next. This zone is determined by the timestamps of the last word in the preceding segment and the first word in the following segment. A small amount of padding (up to 200ms) is added to either side of this zone if the adjacent words do not have valid timestamp information.
2. **Lowest Energy Point:** Within this boundary zone, the app searches for the point with the lowest RMS energy (the quietest point), using 2ms windows for analysis with 1ms overlap. This dynamic approach helps find the optimal cut point, even if the WhisperX timestamps weren't perfect.
