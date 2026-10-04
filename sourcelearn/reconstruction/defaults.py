"""Every numeric knob of SourceLearn in ONE place, each tagged with the
principle it serves and its class:

  budget     — a resource constraint (context, calls, parallelism); the method is
               defined without it and only its *existence* matters;
  default    — an implementation default standing in for a principled rule;
  evaluation — a scoring convention of the evaluation harness, not of the method.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def workers(default: int) -> int:
    """Concurrent LLM calls for a loop; `SOURCELEARN_WORKERS` overrides every default."""
    return int(os.environ.get("SOURCELEARN_WORKERS", default))


@dataclass(frozen=True)
class Knob:
    name: str
    value: object
    kind: str        # budget | default | evaluation
    stage: str
    principle: str   # the rule this value stands in for


KNOBS: list[Knob] = [
    # ---- access to the source and to M
    Knob("M_BUDGET", 24000, "budget", "access", "Retrieve_M loads at most this many M tokens (whole M when it fits, routed blocks otherwise)"),
    Knob("AUTO_BLOCK_MAX_FILES", 40, "default", "access", "mechanical blocks follow the directory tree; a block over this many files splits deeper"),
    Knob("AUTO_BLOCK_MAX_TOKENS", 16000, "budget", "access", "a block must fit the routing budget with room for a second block; flat directories split into file chunks"),
    Knob("RAW_TOP_K", 8, "budget", "access", "raw excerpts the answerer sees next to M"),
    Knob("INSPECT_MIN_HITS", 2, "default", "access", "a scoped retrieval tier (hint -> M-scoped -> global) is trusted when it hits at least this many elements"),
    Knob("INSPECT_CAND_MULT", 4, "budget", "access", "candidates fetched per tier = this x k"),
    Knob("CODE_EXPAND_HITS", 3, "budget", "access", "code symbol hits expanded into full windows"),
    Knob("CODE_WINDOW_CHARS", 24000, "budget", "access", "cap on expanded window text per inspection"),
    # ---- reading and observing (self-study inspection, task-guided refinement)
    Knob("OBSERVATIONS_PER_QUESTION", 3, "budget", "study", "observations kept per study question"),
    Knob("STUDY_READ_CHARS", 48000, "budget", "study", "full-region reading budget; larger regions are read in source-coherent chunks"),
    # ---- region rewrite (grounded, preserving, structurally improved)
    Knob("EVIDENCE_PER_ELEMENT", 12000, "budget", "reconstruct", "chars of one evidence element shown to the writer/grounder (full-text study elements keep a whole function)"),
    Knob("EVIDENCE_TOTAL", 60000, "budget", "reconstruct", "evidence chars per rewrite; priority task evidence > study evidence > existing anchors"),
    Knob("GROUND_CHARS", 6000, "budget", "reconstruct", "evidence chars per element shown to the grounder"),
    Knob("MAX_REWRITE_UNITS", 60, "budget", "reconstruct", "units a rewrite proposal may contain"),
    Knob("GROWTH_ALLOWANCE", 300, "budget", "reconstruct", "tokens a region may always gain under a growth cap"),
    Knob("STATEMENT_CHARS", 600, "budget", "reconstruct", "length of one unit statement"),
    Knob("REWRITE_STATEMENT_CHARS", 400, "budget", "reconstruct", "guardrail on a rewritten unit's statement (one semantic commitment, not one excerpt; the prompt asks for it, this cuts)"),
    Knob("DUP_OVERLAP", 0.6, "default", "reconstruct", "content overlap at which two units are logged as duplicates (telemetry)"),
    # ---- self-directed source learning: inspection -> adaptive study -> consolidation
    Knob("STUDY_K", 20, "budget", "self-study", "adaptive study actions (DEEPEN / CONNECT) per cycle, planned on the inspection observations; 0 = inspection + consolidation only"),
    Knob("STUDY_ROUND", 5, "budget", "self-study", "actions one planner call chooses; the next call sees the observations the previous round added"),
    Knob("DEEPEN_K", 12, "budget", "self-study", "evidence elements retrieved for a DEEPEN question, scoped to the entity's files"),
    Knob("STUDY_DIGEST_CHARS", 60000, "budget", "self-study", "chars of the observation digest shown to the study planner (entities with the most observations first)"),
    # ---- task-guided source learning: failure-guided local refinement + task-induced representation calibration
    Knob("TASK_EVIDENCE_K", 8, "budget", "tasklearn", "source elements retrieved around (question + reference) as candidates for the evidence a task needs"),
    Knob("REFERENCE_QUERY_CHARS", 600, "default", "tasklearn", "reference-answer prefix joined to the question to retrieve the evidence a task needs"),
    Knob("NEEDED_MAX", 4, "budget", "tasklearn", "needed claims (evidence element + what it establishes) kept per task"),
    Knob("LESSON_MIN_SUPPORT", 2, "default", "tasklearn", "distinct tasks whose lessons a representation preference must consolidate before it enters the policy"),
    Knob("LESSON_MAX_OVERLAP", 0.35, "default", "tasklearn", "share of a lesson's content words that may repeat the task's question / needed claims; above it the lesson restates the task instead of naming a category of content (one retry)"),
    Knob("POLICY_MAX_ITEMS", 8, "budget", "tasklearn", "task-derived preferences a policy may hold (lowest support dropped first)"),
    Knob("CALIBRATE_GROWTH", 0, "budget", "tasklearn", "recalibration growth cap, per entity and on the whole model (0 = off; 1.15 = fixed capacity, the policy decides what fills it)"),
    Knob("CALIBRATE_ALLOWANCE", 80, "budget", "tasklearn", "tokens an entity may always gain under recalibration"),
    Knob("TASK_WORKERS", 4, "budget", "tasklearn", "training tasks and recalibrated entities processed concurrently"),
    # ---- evaluation harness
    Knob("EVAL_WORKERS", 6, "budget", "eval", "questions answered concurrently"),
    Knob("EVAL_REPEATS", 1, "evaluation", "eval", "times each final model answers the test set (report mean and majority)"),
]

_BY_NAME = {k.name: k.value for k in KNOBS}
globals().update(_BY_NAME)   # importable as constants: from sourcelearn.reconstruction.defaults import STUDY_K
