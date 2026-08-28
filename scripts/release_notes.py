"""Extract one version's section from CHANGELOG.md, for release notes.

Auto-generated notes list commit subjects, which describe what was typed
rather than what changed for whoever installs it. The changelog already says
that in prose, so the release quotes it instead of inventing a second version
of the truth.

    python scripts/release_notes.py 0.5.0
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")
PYPROJECT = Path("pyproject.toml")


def declared_version() -> str:
    """The version the package claims to be."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def section(version: str, text: str) -> str:
    """Return the changelog body for ``version``.

    Raises:
        SystemExit: If the version has no section, which means the tag was
            pushed before the changelog was written.
    """
    heading = re.compile(rf"^##\s*\[{re.escape(version)}\]", re.M)
    match = heading.search(text)
    if match is None:
        sys.exit(f"CHANGELOG.md has no section for {version}. Write it before tagging.")

    rest = text[match.end() :]
    following = re.search(r"^##\s*\[", rest, re.M)
    body = rest[: following.start()] if following else rest
    return body.strip("\n").lstrip("\n").strip()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit("usage: release_notes.py <version>")
    version = argv[1].lstrip("v")

    declared = declared_version()
    if declared != version:
        sys.exit(
            f"Tag says {version}, pyproject.toml says {declared}. "
            "A release whose version disagrees with its package is worse than no release."
        )

    print(section(version, CHANGELOG.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
