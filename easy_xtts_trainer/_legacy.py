from __future__ import annotations

import importlib.util
from pathlib import Path
import traceback
from types import ModuleType


def _load_legacy_script_module() -> ModuleType:
    legacy_script = Path(__file__).resolve().parent.parent / "easy_xtts_trainer.py"
    spec = importlib.util.spec_from_file_location("_easy_xtts_trainer_legacy", legacy_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy script from {legacy_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_legacy_main() -> int:
    legacy_module = _load_legacy_script_module()
    try:
        success = legacy_module.main()
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        traceback.print_exc()
        return 1

    if success:
        print("Training process completed successfully.")
        return 0

    print("Training process failed.")
    return 1
