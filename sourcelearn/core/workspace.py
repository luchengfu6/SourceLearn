"""Workspace: a checked-out heterogeneous source tree, pinned to a version."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sourcelearn.core.schema import VersionStamp

# Directories never treated as sources.
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
                ".mypy_cache", ".pytest_cache", ".eggs", "build", "dist"}


class Workspace:
    """Read access to one workspace snapshot; all paths are root-relative."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"workspace root not found: {self.root}")
        self.version = self._resolve_version()

    # -- version ------------------------------------------------------------
    def _resolve_version(self) -> VersionStamp:
        sha = self.git("rev-parse", "HEAD")
        ts = self.git("show", "-s", "--format=%cI", "HEAD") if sha else None
        return VersionStamp(commit_sha=sha, timestamp=ts)

    def git(self, *args: str) -> str | None:
        """Run git in the workspace; None when not a repo or command fails."""
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), *args],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    # -- files --------------------------------------------------------------
    def list_files(self, suffixes: tuple[str, ...] | None = None) -> list[str]:
        files = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            if suffixes and p.suffix not in suffixes:
                continue
            files.append(str(rel))
        return files

    def resolve(self, rel_path: str) -> Path:
        """Resolve a workspace-relative path, refusing escapes from the root."""
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"path escapes workspace: {rel_path}")
        return p

    def read_text(self, rel_path: str, max_bytes: int = 2_000_000) -> str:
        p = self.resolve(rel_path)
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")

