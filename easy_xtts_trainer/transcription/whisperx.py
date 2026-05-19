from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Optional, Sequence

DEFAULT_PASCAL_GPU_MARKERS = ("1060", "1070", "1080", "1660", "1650")
WHISPERX_RUNNER_CHOICES = ("auto", "pixi", "conda")
WHISPERX_PIXI_EXE_ENV = "WHISPERX_PIXI_EXE"
WHISPERX_PIXI_MANIFEST_ENV = "WHISPERX_PIXI_MANIFEST"


def get_possible_conda_paths(userprofile: Optional[str] = None) -> list[str]:
    profile = userprofile if userprofile is not None else os.environ.get("USERPROFILE", "")
    return [
        os.path.join(profile, "anaconda3", "Scripts", "conda.exe"),
        os.path.join(profile, "miniconda3", "Scripts", "conda.exe"),
        r"C:\ProgramData\Anaconda3\Scripts\conda.exe",
        r"C:\ProgramData\Miniconda3\Scripts\conda.exe",
    ]


def get_conda_executable(
    conda_path_arg: Optional[str],
    conda_env_arg: Optional[str],
    fallback_conda_path: str | Path,
    possible_paths: Optional[Sequence[str]] = None,
) -> str:
    if conda_path_arg:
        return os.path.join(conda_path_arg, "conda.exe")

    if conda_env_arg:
        paths = list(possible_paths) if possible_paths is not None else get_possible_conda_paths()
        for path in paths:
            if os.path.exists(path):
                return path

    return str(fallback_conda_path)


def build_whisperx_command(
    conda_executable: str,
    env_name: str,
    audio_file: str | Path,
    source_language: str,
    whisper_model: str,
    output_dir: str | Path,
    align_model: Optional[str] = None,
    use_int8: bool = False,
) -> list[str]:
    command = [
        conda_executable,
        "run",
        "-n",
        env_name,
        "--no-capture-output",
        "python",
        "-m",
        "whisperx",
        str(audio_file),
        "--language",
        source_language,
        "--model",
        whisper_model,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
    ]

    if align_model:
        command.extend(["--align_model", align_model])

    if use_int8:
        command.extend(["--compute_type", "int8"])

    return command


def build_whisperx_pixi_command(
    pixi_executable: str,
    pixi_manifest: str | Path,
    audio_file: str | Path,
    source_language: str,
    whisper_model: str,
    output_dir: str | Path,
    align_model: Optional[str] = None,
    use_int8: bool = False,
) -> list[str]:
    command = [
        pixi_executable,
        "run",
        "--manifest-path",
        str(pixi_manifest),
        "python",
        "-m",
        "whisperx",
        str(audio_file),
        "--language",
        source_language,
        "--model",
        whisper_model,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
    ]

    if align_model:
        command.extend(["--align_model", align_model])

    if use_int8:
        command.extend(["--compute_type", "int8"])

    return command


def is_pascal_like_gpu_name(gpu_name: str, markers: Sequence[str] = DEFAULT_PASCAL_GPU_MARKERS) -> bool:
    lowered = gpu_name.lower()
    return any(marker in lowered for marker in markers)


def _run_whisperx_command(command: Sequence[str], runner_label: str) -> bool:
    try:
        subprocess.run(list(command), check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error running WhisperX via {runner_label}: {exc}")
        print(f"Standard output: {exc.stdout}")
        print(f"Standard error: {exc.stderr}")
        return False


def run_whisperx_in_env(
    conda_executable: str,
    env_name: str,
    audio_file: str | Path,
    source_language: str,
    whisper_model: str,
    output_dir: str | Path,
    align_model: Optional[str] = None,
    use_int8: bool = False,
) -> bool:
    command = build_whisperx_command(
        conda_executable=conda_executable,
        env_name=env_name,
        audio_file=audio_file,
        source_language=source_language,
        whisper_model=whisper_model,
        output_dir=output_dir,
        align_model=align_model,
        use_int8=use_int8,
    )
    return _run_whisperx_command(command, f"Conda environment '{env_name}'")


def run_whisperx_in_pixi(
    pixi_executable: str,
    pixi_manifest: str | Path,
    audio_file: str | Path,
    source_language: str,
    whisper_model: str,
    output_dir: str | Path,
    align_model: Optional[str] = None,
    use_int8: bool = False,
) -> bool:
    command = build_whisperx_pixi_command(
        pixi_executable=pixi_executable,
        pixi_manifest=pixi_manifest,
        audio_file=audio_file,
        source_language=source_language,
        whisper_model=whisper_model,
        output_dir=output_dir,
        align_model=align_model,
        use_int8=use_int8,
    )
    return _run_whisperx_command(command, "Pixi")


def _is_available_pixi_executable(pixi_executable: Optional[str]) -> bool:
    if not pixi_executable:
        return False

    return bool(Path(pixi_executable).exists() or shutil.which(pixi_executable))


def _normalize_pixi_runtime(
    pixi_executable: Optional[str],
    pixi_manifest: Optional[str],
) -> Optional[tuple[str, str]]:
    if not pixi_executable or not pixi_manifest:
        return None

    if not _is_available_pixi_executable(pixi_executable):
        return None

    manifest_path = Path(pixi_manifest)
    if not manifest_path.exists():
        return None

    return pixi_executable, str(manifest_path)


def discover_pandrator_pixi_runtime(
    search_roots: Optional[Sequence[str | Path]] = None,
) -> Optional[tuple[str, str]]:
    roots: list[Path] = []
    if search_roots:
        roots.extend(Path(path).resolve() for path in search_roots)
    else:
        roots.extend([Path.cwd().resolve(), Path(__file__).resolve().parent])

    seen: set[str] = set()

    for root in roots:
        for candidate in (root, *root.parents):
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)

            pixi_executable = candidate / "bin" / "pixi.exe"
            pixi_manifest = candidate / "envs" / "whisperx_installer" / "pixi.toml"
            runtime = _normalize_pixi_runtime(str(pixi_executable), str(pixi_manifest))
            if runtime:
                return runtime

    return None


def get_pixi_runtime(
    pixi_executable_arg: Optional[str],
    pixi_manifest_arg: Optional[str],
    env: Optional[Mapping[str, str]] = None,
    discovery_roots: Optional[Sequence[str | Path]] = None,
) -> Optional[tuple[str, str]]:
    runtime = _normalize_pixi_runtime(pixi_executable_arg, pixi_manifest_arg)
    if runtime:
        return runtime

    active_env = env if env is not None else os.environ
    runtime = _normalize_pixi_runtime(
        active_env.get(WHISPERX_PIXI_EXE_ENV),
        active_env.get(WHISPERX_PIXI_MANIFEST_ENV),
    )
    if runtime:
        return runtime

    return discover_pandrator_pixi_runtime(search_roots=discovery_roots)


def _run_whisperx_with_conda_fallback(
    audio_file: str | Path,
    output_dir: str | Path,
    source_language: str,
    whisper_model: str,
    align_model: Optional[str],
    conda_env: Optional[str],
    conda_path: Optional[str],
    fallback_conda_path: str | Path,
    use_int8: bool,
) -> None:
    possible_paths = get_possible_conda_paths()
    conda_executable = get_conda_executable(
        conda_path_arg=conda_path,
        conda_env_arg=conda_env,
        fallback_conda_path=fallback_conda_path,
        possible_paths=possible_paths,
    )

    if conda_env:
        success = run_whisperx_in_env(
            conda_executable=conda_executable,
            env_name=conda_env,
            audio_file=audio_file,
            source_language=source_language,
            whisper_model=whisper_model,
            output_dir=output_dir,
            align_model=align_model,
            use_int8=use_int8,
        )
        if not success:
            raise Exception(f"Failed to run WhisperX with provided environment: {conda_env}")
        return

    success = run_whisperx_in_env(
        conda_executable=conda_executable,
        env_name="whisperx_installer",
        audio_file=audio_file,
        source_language=source_language,
        whisper_model=whisper_model,
        output_dir=output_dir,
        align_model=align_model,
        use_int8=use_int8,
    )
    if success:
        return

    secondary_conda_executable = next((path for path in possible_paths if os.path.exists(path)), None)
    if not secondary_conda_executable:
        raise Exception("Could not find a valid conda path")

    success = run_whisperx_in_env(
        conda_executable=secondary_conda_executable,
        env_name="whisperx",
        audio_file=audio_file,
        source_language=source_language,
        whisper_model=whisper_model,
        output_dir=output_dir,
        align_model=align_model,
        use_int8=use_int8,
    )
    if not success:
        raise Exception("Failed to run WhisperX with both whisperx_installer and whisperx environments")


def run_whisperx_with_fallback(
    audio_file: str | Path,
    output_dir: str | Path,
    source_language: str,
    whisper_model: str,
    align_model: Optional[str],
    conda_env: Optional[str],
    conda_path: Optional[str],
    fallback_conda_path: str | Path,
    use_int8: bool = False,
    whisperx_runner: str = "auto",
    pixi_executable: Optional[str] = None,
    pixi_manifest: Optional[str] = None,
) -> None:
    runner_mode = (whisperx_runner or "auto").lower()
    if runner_mode not in WHISPERX_RUNNER_CHOICES:
        valid_modes = ", ".join(WHISPERX_RUNNER_CHOICES)
        raise ValueError(f"Invalid WhisperX runner '{whisperx_runner}'. Expected one of: {valid_modes}")

    if runner_mode in {"auto", "pixi"}:
        pixi_runtime = get_pixi_runtime(
            pixi_executable_arg=pixi_executable,
            pixi_manifest_arg=pixi_manifest,
        )

        if pixi_runtime:
            resolved_pixi_executable, resolved_pixi_manifest = pixi_runtime
            success = run_whisperx_in_pixi(
                pixi_executable=resolved_pixi_executable,
                pixi_manifest=resolved_pixi_manifest,
                audio_file=audio_file,
                source_language=source_language,
                whisper_model=whisper_model,
                output_dir=output_dir,
                align_model=align_model,
                use_int8=use_int8,
            )
            if success:
                return

            if runner_mode == "pixi":
                raise Exception("Failed to run WhisperX in Pixi mode")

            print("Warning: WhisperX Pixi execution failed. Falling back to Conda.")
        elif runner_mode == "pixi":
            raise Exception(
                "WhisperX runner is set to pixi, but no valid Pixi runtime was found. "
                "Set --whisperx-pixi-exe and --whisperx-pixi-manifest or provide "
                "WHISPERX_PIXI_EXE and WHISPERX_PIXI_MANIFEST."
            )

    _run_whisperx_with_conda_fallback(
        audio_file=audio_file,
        output_dir=output_dir,
        source_language=source_language,
        whisper_model=whisper_model,
        align_model=align_model,
        conda_env=conda_env,
        conda_path=conda_path,
        fallback_conda_path=fallback_conda_path,
        use_int8=use_int8,
    )
