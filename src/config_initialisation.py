import json
from typing import Any, Optional, Dict

from src.config_parameters import SetConfig
from src.config_runtime_proxy import ConfigProxy
from src.repo_paths import PATHS

_active_config: Optional[ConfigProxy] = None

def initialize_global_config(cli_overrides: Optional[Dict[str, Any]] = None) -> ConfigProxy:
    """Load validated settings once and expose them through the global proxy."""

    global _active_config
    if _active_config is not None:
        raise RuntimeError(
            "Error: Configuration infrastructure has already been initialised. "
            "Modifying configurations mid-process run is strictly forbidden."
        )

    target_json = PATHS.config_json
    file_kwargs: dict[str, Any] = {}

    if target_json.is_file():
        try:
            with open(target_json, "r", encoding="utf-8") as config_file:
                file_kwargs = json.load(config_file)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"Malformed JSON configuration layout discovered inside config file: {target_json}"
            ) from err
        if not isinstance(file_kwargs, dict):
            raise ValueError(f"Configuration file must contain a JSON object: {target_json}")

    if cli_overrides:
        file_kwargs.update(cli_overrides)

    normalized_kwargs = dict(file_kwargs)
    for section_name, section_field in SetConfig.model_fields.items():
        section_model = section_field.annotation
        section_fields = getattr(section_model, "model_fields", {})
        if not section_fields:
            continue

        section_values = normalized_kwargs.get(section_name, {})
        if not isinstance(section_values, dict):
            continue
        section_values = dict(section_values)

        for field_name in section_fields:
            if field_name in normalized_kwargs:
                section_values[field_name] = normalized_kwargs.pop(field_name)

        if section_values or section_name in normalized_kwargs:
            normalized_kwargs[section_name] = section_values

    base_values = SetConfig(**normalized_kwargs)
    _active_config = ConfigProxy(settings=base_values)
    return _active_config


def __getattr__(name: str) -> Any:
    """Provide lazy access to ``CONFIG`` after configuration initialization."""

    if name == "CONFIG":
        if _active_config is None:
            raise RuntimeError(
                "The configuration engine context has not been initialised for this process tree yet.\n"
                "Run 'initialize_global_config()' explicitly to initialise configuration"
            )
        return _active_config
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
