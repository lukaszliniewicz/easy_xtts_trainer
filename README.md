# Easy XTTS Trainer

This command line app was designed to work with [Pandrator](https://github.com/lukaszliniewicz/Pandrator), but can be used on its own. Pandrator provides a GUI and a way to load and test the models immediately.

## Table of Contents
- [Installation](#installation)
  - [Setting up Conda Environments](#setting-up-conda-environments)
  - [Installing the App](#installing-the-app)
- [System Requirements](#system-requirements)
- [Usage](#usage)
  - [Arguments](#arguments)
- [Segment Methods Explanation](#segment-methods-explanation)

## Installation

### Setting up Conda Environments

Before installing the app, you need to set up two Conda environments: `xtts_training` for the app itself and `whisperx` for the WhisperX transcription tool.

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

```bash
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## System Requirements

* Nvidia GPU with at least 8GB of VRAM (11GB recommended)
* Python 3.10
* Conda environments: `xtts_training` and `whisperx`

## Usage

**Run the app:**

```bash
python easy-xtts-trainer.py --help
```

### Arguments

| Argument | Description | Required | Options |
| --- | --- | --- | --- |
| `--source-language` | Source language | Yes | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu |
| `--whisper-model` | Whisper model to use for transcription | No | medium, medium.en, large-v2, large-v3 (default: `medium`) |
| `--denoise` | Enable denoising | No | Not implemented yet |
| `--enhance` | Enable audio enhancement | No | Not implemented yet |
| `-i`, `--input` | Input folder or single file | Yes | - |
| `--session` | Name for the model folder | No | - |
| `--separate` | Enable speech separation | No | Not implemented yet |
| `--epochs` | Number of training epochs | No | Default: 6 |
| `--xtts-base-model` | XTTS base model version | No | (default: `v2.0.2`) |
| `--batch` | Batch size | No | (default: 2) |
| `--gradient` | Gradient accumulation levels | No | Default: 1 |
| `--xtts-model-name` | Name for the trained model | No | - |
| `--sample-method` | Method for preparing training samples | No | maximise-punctuation, punctuation-only, mixed (default: `mixed`) |
| `-conda_env` | Name of the Conda environment to use | No | - |
| `-conda_path` | Path to the Conda installation folder | No | - |

## Segment Methods Explanation

The app uses three segment methods to prepare training samples:

* `maximise-punctuation`: This method tries to maximise the lenght of segments (audio/text pairs) within the 11s and 200 characters limit and is guided by punctuation (segments must end with a clause or sentence-ending punctuation mark). It produces fewer but longer training samples.
* `punctuation-only`: This method is guided by punctuation marks, but doesn't try to maximise the lenght of the samples within the limit. It will cut and move to create the next segment when it encounters a sentence-ending or clause punctuation mark.
* `mixed`: This method combines the `maximise-punctuation` and `punctuation-only` methods. 60% of samples are produced using the punctuation method, and the rest using the maximise punctiation method. 
