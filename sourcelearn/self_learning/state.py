"""StudyState: the training state of self-study, kept OUTSIDE the model so
the unit schema never turns into an experiment log."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StudyState:
    step: int = 0
    question_history: list[dict] = field(default_factory=list)   # {step, target, question, verdict, applied}
    notes: dict[str, list[dict]] = field(default_factory=dict)      # block -> pending study notes (observations)
    reorganized: dict[str, int] = field(default_factory=dict)       # region -> rewrites committed
    cycles: list[dict] = field(default_factory=list)                 # per-cycle statistics (observations, actions, rewrites, tokens)

    def record(self, target: dict, question: str, tele: dict) -> None:
        self.question_history.append({"step": self.step, "target": target, "question": question,
                                      "verdict": tele["verdict"], "applied": tele["applied"]})
        self.step += 1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "StudyState":
        return cls(**json.loads(Path(path).read_text()))
