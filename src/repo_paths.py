import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

PATHS_CONFIG_RELATIVE: Final[Path] = Path("data") / "Configuration" / "paths_config.json"


class ProjectPaths:
    """Resolve the repository-relative paths named in ``paths_config.json``.

    Tokens ending in ``_dir`` or ``_directory`` are created on first access.
    """

    _root: Final[Path]
    _path_map: Final[Mapping[str, str]]

    def __init__(self, starting_point: str | Path = __file__) -> None:
        """Find the repository root (the folder containing ``paths_config.json``) and load the path map."""
        candidate = Path(starting_point).resolve()
        search_directory = candidate if candidate.is_dir() else candidate.parent
        root = next(
            (parent for parent in (search_directory, *search_directory.parents)
             if (parent / PATHS_CONFIG_RELATIVE).is_file()),
            None,
        )
        if root is None:
            raise FileNotFoundError(
                f"Could not find the repository root from {candidate}; expected "
                f"{PATHS_CONFIG_RELATIVE.as_posix()} in this directory or a parent."
            )
        self._root = root
        self._path_map = MappingProxyType(self._load_path_map(root / PATHS_CONFIG_RELATIVE))

    @staticmethod
    def _load_path_map(paths_file: Path) -> dict[str, str]:
        """Read path tokens and their relative locations."""
        try:
            with open(paths_file, "r", encoding="utf-8") as file:
                path_config = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"{paths_file.name} contains invalid JSON: {error}") from error
        if not isinstance(path_config, dict) or not all(isinstance(value, str) for value in path_config.values()):
            raise ValueError(f"{paths_file.name} must be a flat JSON object mapping names to relative paths.")
        return path_config

    @property
    def root(self) -> Path:
        """Return the resolved repository root directory."""
        return self._root

    def __getattr__(self, name: str) -> Path:
        """Resolve a named path token, creating it if it names a directory."""
        if name.startswith("_") or name not in self._path_map:
            raise AttributeError(f"The path token {name!r} is not present in 'paths_config.json'.")

        absolute_path = (self._root / self._path_map[name]).resolve()
        if name.endswith(("_dir", "_directory")):
            absolute_path.mkdir(parents=True, exist_ok=True)
        return absolute_path

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the initialised root and path map."""
        if name in ("_root", "_path_map") and name in self.__dict__:
            raise AttributeError("Project paths are immutable at runtime.")
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"ProjectPaths(root='{self._root}')"


PATHS = ProjectPaths()
