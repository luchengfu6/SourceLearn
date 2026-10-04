"""Shared learning primitives of every stage that changes M after the draft
(self-directed and task-guided source learning):

    blocks.py       partition blocks as views over M; entity keys; bounded evidence
    inspect.py      target -> source evidence (hint -> M-scoped -> global; code hits
                    expand to entity-centred windows); full-text reading of an entity
    reconstruct.py  observe (no edit); region rewrite and local reconstruction,
                    grounded first, applied atomically
    grounding.py    the entailment check behind every grounding gate
    update.py       grounding / scope / novelty gates, M size
    defaults.py     every numeric knob in one table
"""
