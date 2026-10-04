"""Representation policy Pi: what deserves representation in this source model.

    M = C(S; Pi)  — the same source, read through a policy.

Pi_0 is generic. Task experience adds task-derived preferences, each carried
by the tasks whose lessons it consolidates (aggregate.aggregate). The
rendered policy enters every writer prompt (reconstruction.reconstruct: observe,
rewrite_region, reconstruct_region): it changes the lens through which the
source is represented, never the facts that may be written — grounding,
preservation and dedup stay as they are.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("preserve", "granularity", "distinguish", "relate", "condition")
GENERIC = ("Preserve the major entities of the source, their roles, the important relations between them, "
           "procedures, governing conditions and representative details, while avoiding redundant low-level content.")


@dataclass
class Policy:
    generic: str = GENERIC
    items: list[dict] = field(default_factory=list)   # {text, kind, support: [qids], round, from_lessons: [ids]}
    round: int = 0

    def render(self) -> str:
        out = [self.generic]
        if self.items:
            out.append("Task-derived representation preferences (learned from how this source is used; numbered):")
            out += [f"{i + 1}. {it['text']}" for i, it in enumerate(self.items)]
        return "\n".join(out)

    def texts(self) -> list[str]:
        return [it["text"] for it in self.items]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls(**json.loads(Path(path).read_text()))
