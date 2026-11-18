"""Saving files so they survive the computer shutting down or restarting.

Windows (and other systems) keep what a program writes in memory for a while before it reaches the disk. If the
computer switches off in that moment, a file can come back the right size but full of zeros, and PeakCut then
cannot read the recording it belongs to, so its clips stop showing. Here a file is written to a temporary name,
pushed all the way to the disk, and only then put in place, so the old version or the new one is always whole."""
import os
from pathlib import Path


def write_text(path, text: str) -> None:
    """Replace a small text file (a recording's notes, its transcript) in a way a sudden shutdown cannot spoil."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _sync_folder(path.parent)


def flush(path) -> None:
    """Push a file that is already written (a finished clip) all the way to the disk."""
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
    _sync_folder(Path(path).parent)


def _sync_folder(folder) -> None:
    # On Linux and macOS the folder itself remembers the new name and is pushed too. Windows has no such step
    # (opening a folder like a file is not allowed there); the file's own push is what matters on it.
    if os.name == "nt":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
