# Compatibility Notes (Windows + Coqui focus)

## Verified metadata (May 2026)

- `coqui-tts==0.25.3` requires Python `>=3.9,<3.13` and ships a universal wheel.
- `coqui-tts-trainer==0.2.2` requires Python `>=3.10,<3.13` and ships a universal wheel.
- `DeepFilterLib==0.5.6` has Windows wheels for CPython 3.10 and 3.11, but not 3.12.

## Build-tools risk points

- `ctc-forced-aligner` is installed from Git and builds a `pybind11` C++ extension (`forced_align_impl.cpp`).
- On Windows this typically needs MSVC/Build Tools.
- Recent upstream commits also require the `ffmpeg` CLI on `PATH` at runtime for audio decoding.

## Practical baseline recommendation

- Target Python 3.11 as the project baseline.
- Keep `ctc-forced-aligner` optional, not part of baseline installation.
- Baseline install should remain wheel-first to reduce Windows setup friction.
