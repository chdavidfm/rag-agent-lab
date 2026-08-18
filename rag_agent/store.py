"""Persisting a built index to disk.

Embedding a corpus is the slowest part of the pipeline, and it produces the
same vectors every time the documents are unchanged. Caching that result turns
a cold start from minutes into milliseconds.

The cache is content-addressed: its key is a fingerprint of the documents and
of the settings that shaped the index. Any change to either produces a
different key, so a stale index is never served — the cache misses and the
index is rebuilt.

Nothing is stored with `pickle`. Loading a pickle executes arbitrary code, and
a cache file is exactly the kind of artefact an attacker would try to replace.
Passages travel as JSON and vectors as a NumPy archive, both inert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = "1"


def fingerprint(paths: list[Path], **parts: Any) -> str:
    """Compute the cache key for a corpus and a set of index settings.

    The digest covers each document's path and byte content, so an edit that
    leaves the size and timestamp untouched still invalidates the cache.

    Args:
        paths: Documents that make up the corpus.
        **parts: Any setting that changes the resulting index, such as the
            backend name or the chunk size.

    Returns:
        A short hexadecimal digest.
    """
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode())
    for key in sorted(parts):
        digest.update(f"{key}={parts[key]}".encode())
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


class IndexCache:
    """Reads and writes index state under a content-addressed key."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _paths(self, key: str) -> tuple[Path, Path]:
        """Return the metadata and array files backing ``key``."""
        return self.directory / f"{key}.json", self.directory / f"{key}.npz"

    def load(self, key: str) -> dict[str, Any] | None:
        """Return the cached state for ``key``, or None when unavailable.

        A cache is an optimisation, never a source of truth: if the files are
        missing, truncated or unreadable, this reports a miss so the caller
        rebuilds the index instead of failing.
        """
        meta_path, array_path = self._paths(key)
        if not meta_path.exists():
            return None
        try:
            state = json.loads(meta_path.read_text(encoding="utf-8"))
            if array_path.exists():
                with np.load(array_path) as archive:
                    _restore_arrays(state, _read_archive(archive))
            else:
                # No archive: any placeholder left in the metadata cannot be
                # resolved, and returning it would hand back a corrupt index.
                # _restore_arrays raises, and this reports a miss instead.
                _restore_arrays(state, {})
            return state
        except (OSError, ValueError, KeyError):
            return None

    def save(self, key: str, state: dict[str, Any]) -> None:
        """Store ``state`` under ``key``.

        Files are written to a temporary name and then moved into place, so a
        process interrupted mid-write never leaves a half-written cache that a
        later run would try to read.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path, array_path = self._paths(key)

        arrays: dict[str, np.ndarray] = {}
        skeleton = _extract_arrays(state, arrays)

        tmp_meta = meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(skeleton), encoding="utf-8")
        tmp_meta.replace(meta_path)

        if arrays:
            # The temporary name must also end in .npz: NumPy appends that
            # suffix itself when the path lacks it, which would leave the
            # archive under a name this code never moves into place.
            tmp_array = array_path.with_name(f"{key}.tmp.npz")
            _write_archive(tmp_array, arrays)
            tmp_array.replace(array_path)


def _write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write named arrays to a compressed archive.

    The names are passed through a keyword mapping, which NumPy also uses for
    its own options such as ``allow_pickle``. Prefixing every entry keeps a
    passage key from ever colliding with one of them.
    """
    prefixed = {f"a_{name}": array for name, array in arrays.items()}
    # NumPy types its second parameter as ``allow_pickle``, so its stubs reject
    # any keyword mapping of arrays. The call is correct at runtime; the prefix
    # above guarantees no entry can shadow that option.
    np.savez_compressed(path, **prefixed)  # type: ignore[arg-type]


def _read_archive(archive: Any) -> dict[str, np.ndarray]:
    """Read back the arrays written by :func:`_write_archive`."""
    return {name[2:]: archive[name] for name in archive.files if name.startswith("a_")}


def _extract_arrays(node: Any, arrays: dict[str, np.ndarray], prefix: str = "") -> Any:
    """Replace every array in ``node`` with a placeholder, collecting them.

    JSON cannot hold a NumPy array, so arrays are pulled out into a separate
    archive and the structure keeps a reference in their place.
    """
    if isinstance(node, np.ndarray):
        name = prefix or "array"
        arrays[name] = node
        return {"__array__": name}
    if isinstance(node, dict):
        return {
            key: _extract_arrays(value, arrays, f"{prefix}.{key}" if prefix else key)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_extract_arrays(value, arrays, f"{prefix}.{i}") for i, value in enumerate(node)]
    return node


def _restore_arrays(node: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Put the arrays back where the placeholders are, in place.

    Raises:
        KeyError: If a placeholder has no matching array, which means the
            cache entry is incomplete and must not be used.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and "__array__" in value:
                node[key] = arrays[value["__array__"]]
            else:
                _restore_arrays(value, arrays)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, dict) and "__array__" in value:
                node[i] = arrays[value["__array__"]]
            else:
                _restore_arrays(value, arrays)
    return node
