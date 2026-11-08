import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Generator, Literal

REPLACE_ATTEMPTS = 3  # short retries cover brief locks from antivirus or cloud-sync tools
RETRY_DELAY_SECONDS = 1.0


def _try_replace(source: Path, target: Path) -> bool:
    """Atomically move ``source`` over ``target``; False if the target stays locked."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return True
        except PermissionError:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS)
    return False


def _user_wants_retry(target: Path) -> bool:
    """Ask, in an interactive terminal, whether to retry after the user closes the file."""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    try:
        answer = input(
            f"Close '{target.name}' (for example in Excel), then press Enter to try again, "
            "or type 's' to save the new version as a separate copy instead: "
        )
    except EOFError:
        return False
    return answer.strip().lower() != "s"


def _commit(temporary_path: Path, target_path: Path) -> None:
    """Replace the target, or keep the new version in a separate file if the target is locked."""
    while not _try_replace(temporary_path, target_path):
        print(
            f"\nWARNING: '{target_path}' could not be overwritten. It is probably open in another program.",
            flush=True,
        )
        if _user_wants_retry(target_path):
            continue
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fallback_path = target_path.with_name(f"{target_path.stem}_unsaved_{timestamp}{target_path.suffix}")
        os.replace(temporary_path, fallback_path)
        print(
            f"WARNING: The new version was saved to '{fallback_path}' instead. '{target_path.name}' still "
            "holds the previous version; close it and copy the new file over it, or re-run this step.",
            flush=True,
        )
        return


@contextmanager
def safe_writer(path: Path | str, mode: Literal["w", "wb", "a", "ab"] = "w") -> Generator[IO[Any], None, None]:
    """Write through a temporary sibling file, then atomically replace the target.

    The target is never left half-written. If it is locked by another program (common on
    Windows when a CSV is open in Excel), the write is retried briefly. In an interactive
    terminal the user is then asked to close the file and retry. Otherwise the new version is
    saved alongside as ``<name>_unsaved_<timestamp>`` with a warning, instead of crashing.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    is_binary = "b" in mode

    # Text files are always UTF-8 with no newline translation, so a file written on Windows
    # reads back identically on Linux (and the other way round). Without this, the temporary
    # file would take the machine's locale encoding, which pandas then misreads, and Windows
    # would turn the line terminators pandas writes into blank lines between every row.
    text_options: dict[str, str] = {} if is_binary else {"encoding": "utf-8", "newline": ""}

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if is_binary else "w",
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            delete=False,
            **text_options,
        ) as temp:
            temporary_path = Path(temp.name)

            if mode in ("a", "ab") and target_path.exists():
                with open(target_path, "rb" if is_binary else "r", **text_options) as source:
                    shutil.copyfileobj(source, temp)
                temp.flush()

            yield temp
            temp.flush()
            os.fsync(temp.fileno())

        _commit(temporary_path, target_path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
