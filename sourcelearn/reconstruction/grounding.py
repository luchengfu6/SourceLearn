"""Grounding gate of the shared reconstruction operator: a reconstructed statement is kept only when
its cited source excerpts entail it (the *grounding* invariant of the source model)."""
from __future__ import annotations

from sourcelearn.core.llm import LLMClient
from sourcelearn.core.schema import coerce_str_list

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"},
                               "description": "claims in the statement "
                                              "that the excerpts do not "
                                              "state or directly imply"},
    },
    "required": ["supported"],
}

SUPPORT_RULES = (
    "Judge whether a revised source-model statement is SUPPORTED by the "
    "shown source excerpts. Every causal, ordering, conditional, or "
    "behavioral claim in the statement must be stated by, or follow "
    "directly from, the excerpts. Plausible-sounding mechanisms that the "
    "excerpts do not establish are NOT supported, even if they are "
    "consistent with them. supported=false if ANY claim is unsupported.")


def supported_by_excerpts(statement: str, gold: dict, llm_strong: LLMClient
                          ) -> tuple[bool, list[str]]:
    """Content-support gate: anchor existence != content support. A
    revision must be entailed by the evidence it cites, not merely
    point at it."""
    text = "\n\n".join(f"[{eid}]\n{e.excerpt[:2000]}" for eid, e in gold.items())
    out = llm_strong.complete_json(
        system=SUPPORT_RULES,
        user=f"Statement:\n{statement}\n\n## Source excerpts\n{text}",
        schema=SUPPORT_SCHEMA, purpose="train_support")
    return bool(out.get("supported")), coerce_str_list(
        out.get("unsupported_claims"))
