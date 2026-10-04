"""CompileBundle: the artifacts of one initial reading (brief -> partition ->
induce), loaded together so self-study and refinement consume the compiler's
output through one interface.

A compile directory holds `m0_union.json` (partition-then-compile) or
`m0_induced.json` (single compile), optionally `partition.json` and
`brief/brief.md`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sourcelearn.source_model import SourceModelUnit


@dataclass
class CompileBundle:
    model_path: Path
    partition_path: Path | None = None
    brief_path: Path | None = None
    model: list[SourceModelUnit] = field(default_factory=list)
    partition: dict | None = None
    brief: str = ""

    @classmethod
    def load(cls, model: str | Path, partition: str | Path | None = None,
             brief: str | Path | None = None) -> "CompileBundle":
        b = cls(Path(model), Path(partition) if partition else None, Path(brief) if brief else None)
        b.model = [SourceModelUnit(**u) for u in json.loads(b.model_path.read_text())]
        # a union of per-block compiles repeats region-level ids (map:<region>,
        # l2:<region>:abs:n); ids must be unique for lineage and replacement
        seen: dict[str, int] = {}
        for u in b.model:
            n = seen.get(u.unit_id, 0)
            seen[u.unit_id] = n + 1
            if n:
                u.unit_id = f"{u.unit_id}#{n + 1}"
        if b.partition_path:
            b.partition = json.loads(b.partition_path.read_text())
        if b.brief_path:
            b.brief = b.brief_path.read_text()
        return b
