"""Shared data contracts: source elements, workspace versions, and the coercion every
LLM-written list field passes through before entering a pydantic model."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


def coerce_str_list(value: Any) -> list[str]:
    """Sanitize an LLM-written list-of-strings field. Models routinely ignore
    the prompt schema (e.g. conditions as [{"condition": ...}]); every LLM
    output must pass through coercion before entering a pydantic contract —
    off-schema shapes degrade, never raise."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return [str(value)]
    out = []
    for v in value:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):  # take the values: {"condition": "x"} -> "x"
            out.extend(str(x) for x in v.values())
        elif v is not None:
            out.append(str(v))
    return out


class VersionStamp(BaseModel):
    """The version (commit and time) of a workspace snapshot."""

    commit_sha: str | None = None
    timestamp: str | None = None


class SourceElement(BaseModel):
    """A stable, fingerprinted unit of a source (section, symbol, file...);
    the anchor granularity of every unit in M."""

    element_id: str  # stable across line shifts, e.g. "docs/api.md#retention"
    node_id: str
    kind: Literal["section", "symbol", "file", "log"]
    name: str  # heading, dotted symbol name, or file path
    file: str
    start_line: int | None = None
    end_line: int | None = None
    fingerprint: str  # whitespace-insensitive content hash
    excerpt: str = ""  # bounded text snapshot
