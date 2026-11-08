from typing import Any
import torch

from src.config_parameters import SetConfig
from src.repo_paths import PATHS

class ConfigProxy:
    """Expose validated settings and runtime-derived values through one object."""

    def __init__(self, settings: SetConfig) -> None:
        """Store the immutable settings object used by this process."""
        self._settings: SetConfig = settings

    def __getattr__(self, name: str) -> Any:
        """Resolve delegated path or settings attributes when directly accessed."""
        if name.startswith("_"):
            # Avoid infinite recursion if ``_settings`` is looked up before it is set (e.g. when unpickling).
            raise AttributeError(name)

        if name in PATHS._path_map:
            return getattr(PATHS, name)

        settings_fields = type(self._settings).model_fields
        if name in settings_fields:
            return getattr(self._settings, name)

        for section_name in settings_fields:
            section = getattr(self._settings, section_name)
            section_fields = getattr(type(section), "model_fields", {})
            if name in section_fields:
                return getattr(section, name)

        raise AttributeError(f"'ConfigProxy' object has no attribute {name!r}")

    def resolve_torch_device(self) -> torch.device:
        """Choose CPU/CUDA from configuration and reject unavailable CUDA requests."""
        target: str = self._settings.hardware.device
        
        if target == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        device_obj = torch.device(target)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA backend requested via configurations target '{target}', "
                f"but no hardware acceleration nodes were discovered on this environment runtime."
            )
        return device_obj

    def __repr__(self) -> str:
        """Return a compact description of the active runtime configuration."""
        return f"RuntimeContext(root='{PATHS.root}', state=OPERATIONAL)"
