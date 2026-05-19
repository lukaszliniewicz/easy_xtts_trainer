## Vendored wheels

This directory contains prebuilt optional wheels used by Easy XTTS Trainer.

- `ctc_forced_aligner-0.3.0-cp313-cp313-win_amd64.whl`
  - Built from commit `264e7a1f81bff9ff5e787a5537020c2ad0b0df02`
  - Target: CPython 3.13, Windows x86_64
  - Runtime requirement: `ffmpeg` executable must be on PATH

Rebuild command used on this machine:

```powershell
& "E:\Pandrator\envs\whisperx_installer\.pixi\envs\default\python.exe" -m pip wheel --no-deps --wheel-dir "E:\easy_xtts_trainer\vendor" "ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@264e7a1f81bff9ff5e787a5537020c2ad0b0df02"
```
