"""Unit ids must identify units.

A compile composes an id from (role, entity, local index): `card:<entity>`, `mech:<entity>:3`,
`l2:<region>:abs:7`. Nothing guarantees the result is unique — one entity read in several batches
restarts the local index, a `card:` id carries no index at all, and every block of a partitioned
source numbers inside the same region. Every id-keyed step downstream (retrieval of a unit,
REPLACE_GROUP bookkeeping, E* coverage, `derived_from` lineage) would conflate such units.

`unique_ids` is a relabelling, never a content change: ids that are already unique are kept exactly
as they are, later duplicates get a `#2`, `#3` suffix, and `derived_from` references are remapped
to the unit they actually came from (the nearest preceding unit carrying that id). It runs at the
end of a compile and on the union of a partitioned source.
"""
from __future__ import annotations


def unique_ids(units: list[dict]) -> dict[str, int]:
    """Rename duplicate ids in place (order-preserving). Returns {original id: copies renamed}."""
    seen: dict[str, int] = {}
    renamed: dict[str, int] = {}
    latest: dict[str, str] = {}          # original id -> the id its most recent occurrence now has
    for u in units:
        old = u.get("unit_id")
        if old is None:
            continue
        n = seen.get(old, 0) + 1
        seen[old] = n
        if n > 1:
            new = f"{old}#{n}"
            while new in seen:           # a source that already contains the suffixed form
                n += 1
                new = f"{old}#{n}"
            seen[new] = 1
            u["unit_id"] = new
            renamed[old] = renamed.get(old, 0) + 1
        latest[old] = u["unit_id"]
        if u.get("derived_from"):        # a reference means the unit with that id seen so far
            u["derived_from"] = [latest.get(d, d) for d in u["derived_from"]]
    return renamed
