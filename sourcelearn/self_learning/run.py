"""Self-Directed Source Learning: M0 (the draft) -> M_S, without any task.

One self-study cycle, read many / write once:
  1. INSPECTION     every entity of the source read in full against the
                    model's units about it -> observations (no edit);
  2. ADAPTIVE STUDY STUDY_K DEEPEN / CONNECT actions the planner chooses on
                    the observation digest (STUDY_ROUND per planner call);
                    a question is an instrument for resolving one identified
                    gap: evidence retrieved, more observations (no edit);
  3. CONSOLIDATION  every entity with observations rewritten once
                    (reconstruction.reconstruct.rewrite_region) under
                    set-level gates (every new unit grounds, old meanings the
                    rewrite drops are kept, the diff is mechanical), then one
                    relation pass per block and per CONNECT pair.

Outputs: m_self.json, study_trace.jsonl (one row per step),
reorganize_trace.jsonl (one row per rewrite), state.json, study_summary.json,
FINISHED when the cycle completed. An interrupted run resumes from
state.json + m_self.json on rerun.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sourcelearn.core.llm import LLMClient, load_project_env
from sourcelearn.core.schema import SourceElement
from sourcelearn.core.trace import Trace
from sourcelearn.core.workspace import Workspace
from sourcelearn.draft.artifacts import CompileBundle
from sourcelearn.reconstruction.blocks import auto_partition, block_view, bounded_evidence, entity_key, entity_key_of_file
from sourcelearn.reconstruction.defaults import DEEPEN_K, STUDY_K, STUDY_ROUND
from sourcelearn.reconstruction.inspect import full_text_evidence, inspect_source
from sourcelearn.reconstruction.reconstruct import apply_rewrite, observe, prepare_rewrite, relevant_units, rewrite_region
from sourcelearn.reconstruction.update import m_tokens
from sourcelearn.retrieval import build_retriever
from sourcelearn.self_learning.planner import entity_index, observation_digest, study_actions
from sourcelearn.self_learning.state import StudyState
from sourcelearn.source_model import SourceModelSession, SourceModelUnit

REWRITE_EVIDENCE_CHARS = 1500     # per element shown to the region rewrite and its grounder
REWRITE_EVIDENCE_TOTAL = 60000    # total evidence chars per rewrite (~15k tokens)
CONNECT_K = 8                     # evidence elements retrieved for a CONNECT question


RELATIONAL = ("mechanism", "distinction", "procedure", "pattern", "exception", "relation")


def rewrite_metrics(reorgs: list[dict]) -> dict:
    """Over committed rewrites: how much the writer covered on its own
    (subsumed) vs what the preservation gate had to rescue (kept), per role."""
    by: dict[str, dict[str, int]] = {}
    for r in reorgs:
        if not r.get("applied"):
            continue
        for role, c in (r.get("preserve_by_role") or {}).items():
            d = by.setdefault(role, {"subsumed": 0, "kept": 0})
            d["subsumed"] += c["subsumed"]; d["kept"] += c["kept"]
    old = sum(c["subsumed"] + c["kept"] for c in by.values())
    kept = sum(c["kept"] for c in by.values())
    rel = {k: v for k, v in by.items() if k in RELATIONAL}
    rel_old = sum(c["subsumed"] + c["kept"] for c in rel.values())
    return {"rewrites": sum(1 for r in reorgs if r.get("applied")), "attempted": len(reorgs),
            "old_units": old, "subsumed": old - kept, "kept": kept,
            "rescue_rate": round(kept / max(1, old), 3),
            "writer_coverage_by_role": {k: round(c["subsumed"] / max(1, c["subsumed"] + c["kept"]), 2) for k, c in sorted(by.items())},
            "relational_old_units": rel_old,
            "relational_writer_coverage": round(sum(c["subsumed"] for c in rel.values()) / max(1, rel_old), 3)}


class SelfStudy:
    def __init__(self, model: list[SourceModelUnit], partition: dict, ws: Workspace, sm: SourceModelSession,
                 retriever, llm: LLMClient, brief: str = "", k: int = 8, max_growth: float = 0,
                 state: StudyState | None = None, trace_path: str | Path | None = None,
                 m0: list[SourceModelUnit] | None = None, reorg_path: str | Path | None = None,
                 workers: int = 6, checkpoint=None):
        """`m0` = the draft the study started from (differs from `model` only
        when resuming); rates are relative to it. `max_growth`: growth cap of
        one region rewrite (0 = off). `checkpoint`: called after every
        inspected block and study round (the driver saves state.json +
        m_self.json so a rerun resumes)."""
        self.checkpoint = checkpoint
        self._ents: dict[str, dict] = {}                  # entity -> {block, units}, refreshed by plan_study
        self._cycle = 1                                   # the current cycle (targets are tagged with it)
        self._chars_by_entity: dict[str, int] | None = None
        self.model, self.partition, self.ws, self.sm, self.retriever, self.llm = model, partition, ws, sm, retriever, llm
        self.brief, self.k, self.growth, self.workers = brief, k, max_growth, workers
        m0 = model if m0 is None else m0
        self.m0_ids = {u.unit_id for u in m0}
        self.m0_tokens = m_tokens(m0)
        self.all_files = sorted({e.file for e in sm._retrievable})
        self.state = state or StudyState()
        self.fh = open(trace_path, "a") if trace_path else None
        self.reorg_fh = open(reorg_path, "a") if reorg_path else None
        self.reorg_log: list[dict] = []
        self._evidence_cache: dict[str, SourceElement] = {}   # evidence seen by notes, by element id

    def blocks(self, model: list[SourceModelUnit] | None = None) -> list[dict]:
        return block_view(self.partition, model if model is not None else self.model, self.ws)

    # -- observations (no edit), then one gated rewrite per region -------------
    def take_notes(self, target: dict, question: str, blocks_named: list[str], units: list[SourceModelUnit],
                   evidence: dict, kind: str = "question", tier: str = "", notes: list[dict] | None = None) -> dict:
        if notes is None:
            notes = observe({"kind": kind, "text": question}, units, evidence, self.llm, self.brief)
        for eid, e in evidence.items():
            self._evidence_cache.setdefault(eid, e)
        for n in notes:
            n.update({"step": self.state.step, "question": question, "block": blocks_named[0],
                      "family": target.get("family", "")})
            self.state.notes.setdefault(blocks_named[0], []).append(n)
        tele = {"verdict": "NOTES" if notes else "KNOWN", "applied": False, "new_ids": [], "replaced_ids": []}
        row = {"step": self.state.step, "target": target, "question": question, "target_blocks": blocks_named, "tier": tier,
               "evidence": list(evidence), "shown_units": [u.unit_id for u in units], "notes": notes,
               "proposal": {"why": "", "replace_ids": [], "new_units": []}, **tele, "rejected": [], "kept_old": [], "dup_of": {},
               "n_old": 0, "n_new": 0, "tokens_before": m_tokens(self.model), "tokens_after": m_tokens(self.model)}
        self.state.record(target, question, tele)
        if self.fh:
            self.fh.write(json.dumps(row, default=str) + "\n"); self.fh.flush()
        return row

    def _rewrite_evidence(self, notes: list[dict], units: list[SourceModelUnit]) -> dict[str, SourceElement]:
        """What the notes rest on first, then what the old units rest on,
        bounded so the rewrite and its grounding fit one context. A note's
        evidence seen before a resume (not in this process's cache) falls
        back to the element's compile-time excerpt."""
        seen = [(eid, self._evidence_cache.get(eid) or self.sm.elements.get(eid)) for n in notes for eid in n.get("support", [])]
        return bounded_evidence(
            [(eid, e) for eid, e in seen if e is not None]
            + [(a, self.sm.elements[a]) for u in units for a in u.support_anchors if a in self.sm.elements],
            REWRITE_EVIDENCE_CHARS, REWRITE_EVIDENCE_TOTAL)

    def _prepare(self, region: dict, notes: list[dict], relation_pass: bool = False,
                 evidence_units: list[SourceModelUnit] | None = None) -> dict:
        """LLM half of one region rewrite (thread-safe: the model is only read)."""
        evidence = self._rewrite_evidence(notes, region["units"] + (evidence_units or []))
        proposal = rewrite_region(region, region["units"], notes, evidence, self.llm, self.brief, relation_pass,
                                  context_units=region.get("context_units"))
        n = self.state.reorganized.get(region["name"], 0) + 1
        tele = prepare_rewrite(self.model, region, proposal, evidence, self.llm, f"r{n}", growth=self.growth)
        tele.update({"notes": notes, "summary": proposal["summary"], "region_name": region["name"],
                     "proposed_units": [u["statement"][:200] for u in proposal["units"]], "relation_pass": relation_pass})
        return tele

    def _apply(self, tele: dict) -> dict:
        """Mutation half: gated atomic commit, trace, telemetry."""
        tele = apply_rewrite(self.model, tele)
        region_name, relation_pass = tele.pop("region_name"), tele["relation_pass"]
        tele["step"] = self.state.step
        n = self.state.reorganized.get(region_name, 0) + 1
        if tele["applied"]:
            self.state.reorganized[region_name] = n
        self.reorg_log.append(tele)
        if self.reorg_fh:
            self.reorg_fh.write(json.dumps(tele, default=str) + "\n"); self.reorg_fh.flush()
        print(f"  [reorganize {region_name[:45]}{' (relations)' if relation_pass else ''}] {'APPLIED' if tele['applied'] else 'no-op'} "
              f"{tele['reasons']} old {len(tele['old_ids'])} -> subsumed {len(tele['subsumed'])}, kept {len(tele['kept'])}, new {len(tele['new_ids'])}",
              file=sys.stderr)
        return tele

    def _prepare_safe(self, region: dict, notes: list[dict], relation_pass: bool = False,
                      evidence_units: list[SourceModelUnit] | None = None) -> dict:
        try:
            return self._prepare(region, notes, relation_pass, evidence_units)
        except Exception as e:  # noqa: BLE001 — a failed rewrite of one region is a no-op, not a crash
            print(f"  [rewrite {region['name'][:50]}] ERROR {str(e)[:120]}", file=sys.stderr)
            return {"block": region["name"], "applied": False, "reasons": [f"ERROR {str(e)[:80]}"], "old_ids": [u.unit_id for u in region["units"]],
                    "subsumed": [], "kept": [], "new_ids": [], "ungrounded": 0, "proposed": 0, "notes": notes, "summary": "",
                    "region_name": region["name"], "proposed_units": [], "relation_pass": relation_pass, "_old": [], "_grounded": [],
                    "_lost": set(), "_growth": 0}

    def reorganize(self, block_name: str) -> list[dict]:
        """Consolidation of one block: every entity with notes is rewritten
        from its own units, then one cross-entity pass over the notes that
        span several entities."""
        pending = self.state.notes.get(block_name, [])
        notes = [n for n in pending if not n.get("pair")]      # CONNECT-pair notes wait for consolidate_connect
        if not notes:
            return []
        block = next(b for b in self.blocks() if b["name"] == block_name)
        self.state.notes[block_name] = [n for n in pending if n.get("pair")]
        ent_of_unit = {u.unit_id: entity_key(u) for u in block["units"]}
        by_entity: dict[str, list[dict]] = {}
        cross: list[dict] = []
        for n in notes:
            if n.get("entity"):                                  # a DEEPEN note belongs to its entity
                by_entity.setdefault(n["entity"], []).append(n)
                continue
            ents = {entity_key_of_file(eid.split("#")[0]) for eid in n.get("support", [])}
            ents |= {ent_of_unit[c] for c in n.get("concerns", []) if c in ent_of_unit}
            if n.get("family"):
                ents.add(entity_key_of_file(n["family"]))
            if len(ents) == 1:
                by_entity.setdefault(next(iter(ents)), []).append(n)
            elif ents:
                cross.append(n)
            else:
                by_entity.setdefault("", []).append(n)      # unplaceable: joins the cross pass
        cross += by_entity.pop("", [])
        jobs = []
        regional = [u for u in block["units"] if u.level in ("regional", "global")]
        for ent, ent_notes in sorted(by_entity.items()):
            # representation levels: an entity pass rewrites the entity's OWN units; the block's regional units
            # (shared rules, cross-entity structure) are shown as context and are never absorbed into an entity
            units = [u for u in block["units"] if ent_of_unit[u.unit_id] == ent and u.level not in ("regional", "global")]
            ctx = [u for u in regional if any(entity_key_of_file(a.split("#")[0]) == ent for a in u.support_anchors)][:20]
            jobs.append(({**block, "name": ent, "description": f"entity {ent.split('/')[-1]} of {block['name']}", "units": units,
                          "context_units": ctx}, ent_notes))
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:   # entities are disjoint: LLM work in parallel
            prepared = list(pool.map(lambda job: self._prepare_safe(*job), jobs))
        out = [self._apply(t) for t in prepared]                            # commits in order, one at a time
        if cross:   # relational consolidation: the block's regional units are rewritten here, and only here
            concerned = {c for n in cross for c in n.get("concerns", [])}
            region = {**block, "name": f"{block['name']}:relations", "units": regional}
            out.append(self._rewrite_safe(region, cross, relation_pass=True,
                                          evidence_units=[u for u in block["units"] if u.unit_id in concerned and u.level not in ("regional", "global")]))
        return out

    def _rewrite_safe(self, region: dict, notes: list[dict], relation_pass: bool = False,
                      evidence_units: list[SourceModelUnit] | None = None) -> dict:
        """One rewrite + gated commit of a region (entity, block relations or
        entity pair), traced. A failed proposal (invalid JSON after retries, a
        timeout) is a logged no-op, never the end of the run."""
        try:
            return self._apply(self._prepare_safe(region, notes, relation_pass, evidence_units))
        except Exception as e:  # noqa: BLE001
            print(f"  [rewrite {region['name'][:50]}] ERROR {str(e)[:120]}", file=sys.stderr)
            tele = {"block": region["name"], "applied": False, "reasons": [f"ERROR {str(e)[:80]}"], "old_ids": [u.unit_id for u in region["units"]],
                    "subsumed": [], "kept": [], "new_ids": [], "ungrounded": 0, "proposed": 0, "notes": notes, "summary": "",
                    "proposed_units": [], "relation_pass": relation_pass, "step": self.state.step}
            self.reorg_log.append(tele)
            if self.reorg_fh:
                self.reorg_fh.write(json.dumps(tele, default=str) + "\n"); self.reorg_fh.flush()
            return tele

    def reorganize_all(self) -> None:
        for name in list(self.state.notes):
            self.reorganize(name)

    def family_units(self, block: dict, files: list[str], fam: str) -> list[str]:
        """Unit ids the model holds about this entity: anchored in its files or
        scoped to it, plus the block's regional units (shared rules)."""
        fset = set(files)
        tail = fam.split("/")[-1]
        ids = [u.unit_id for u in block["units"]
               if any(a.split("#")[0] in fset for a in u.support_anchors) or (u.scope or "").split("/")[-1] == tail]
        ids += [u.unit_id for u in block["units"] if u.level == "regional" and u.unit_id not in ids][:20]
        return ids

    def _checkpoint(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint()

    def inspect_entities(self) -> None:
        """Inspection: every entity of every block read IN FULL against the
        model's units about it. A block's entities are observed in parallel
        (notes only read the model) and recorded in order; the notes wait for
        the cycle's one consolidation."""
        done = {(q["target"].get("family"), q["target"].get("chunk", 0)) for q in self.state.question_history
                if q["target"].get("cycle", 1) == self._cycle}            # a later cycle re-inspects everything against the new M
        for b in self.blocks():
            fams: dict[str, list[str]] = {}
            for f in b["files"]:
                fams.setdefault(entity_key_of_file(f), []).append(f)   # versions of one document = one logical entity, read once
            todo = [(fam, files, i, ev) for fam, files in sorted(fams.items())
                    for i, ev in enumerate(full_text_evidence(files, self.ws, self.sm.elements)) if (fam, i) not in done]
            block = next(x for x in self.blocks() if x["name"] == b["name"])

            def observe_one(job):
                fam, files, i, ev = job
                units = relevant_units(self.model, fam.replace("_", " "), must=self.family_units(block, files, fam), k=4)
                try:
                    return units, observe({"kind": "review", "text": f"{fam} (files: {', '.join(files)})"}, units, ev, self.llm, self.brief)
                except Exception as e:  # noqa: BLE001 — one bad model reply must not lose the pass
                    print(f"  [observe {fam.split('/')[-1][:50]}] ERROR {str(e)[:120]}", file=sys.stderr)
                    return units, []
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
                results = list(pool.map(observe_one, todo))
            for (fam, files, i, ev), (units, notes) in zip(todo, results):
                target = {"kind": "REVIEW", "blocks": [b["name"]], "family": fam, "how": "exhaustive", "chunk": i, "cycle": self._cycle}
                row = self.take_notes(target, f"review {fam}", [b["name"]], units, ev, "review", "full", notes=notes)
                print(f"  [{b['name']}] {fam.split('/')[-1][:50]} -> {row['verdict']}", file=sys.stderr)
            self._checkpoint()

    def _cross_entity_units(self) -> list[SourceModelUnit]:
        return [u for u in self.model if u.level == "regional" and u.role in ("relation", "pattern", "mechanism", "distinction", "exception")
                and len({entity_key_of_file(a.split("#")[0]) for a in u.support_anchors}) > 1]

    def _connect_observe(self, a: str, b_: str, question: str, ents: dict[str, dict]) -> tuple:
        """Evidence from both entities for a CONNECT question, observed;
        the notes carry the pair (consolidated by consolidate_connect)."""
        units = ents.get(a, {}).get("units", [])[:8] + ents.get(b_, {}).get("units", [])[:8]
        files = {x.split("#")[0] for u in units for x in u.support_anchors}
        files |= {f for f in self.all_files if entity_key_of_file(f) in (a, b_)}
        evidence, tier = inspect_source(question, self.retriever, self.all_files, max(self.k, 10), scoped=files,
                                        ws=self.ws, elements=self.sm.elements)
        notes = observe({"kind": "connect", "text": question}, units, evidence, self.llm, self.brief)
        for n in notes:
            n["pair"] = [a, b_]
        return units, evidence, tier, notes

    def consolidate_connect(self) -> None:
        """B2: one relation pass per entity pair with notes; units of both
        entities join the evidence; nothing of theirs is rewritten."""
        by_pair: dict[tuple, list[dict]] = {}
        for notes in self.state.notes.values():
            for n in notes:
                if n.get("pair"):
                    by_pair.setdefault(tuple(n["pair"]), []).append(n)
        blocks = self.blocks()
        ents, _ = entity_index(blocks, entity_key)
        for (a, b_), notes in by_pair.items():
            ea, eb = ents.get(a, {"block": notes[0].get("block"), "units": []}), ents.get(b_, {"block": notes[0].get("block"), "units": []})
            region = {**next((x for x in blocks if x["name"] == ea["block"]), blocks[0]),
                      "name": f"{a.split('/')[-1]}<->{b_.split('/')[-1]}", "units": []}
            self._rewrite_safe(region, notes, relation_pass=True, evidence_units=ea["units"] + eb["units"])
            # a pair's notes leave the state only once its pass was attempted, so an interrupted run resumes the rest
            self.state.notes = {b: [n for n in ns if tuple(n.get("pair") or ()) != (a, b_)] for b, ns in self.state.notes.items()}
            self._checkpoint()

    # -- self-study cycle: inspection -> adaptive study -> consolidation ------------
    @staticmethod
    def _note_entities(n: dict) -> list[str]:
        if n.get("pair"):
            return list(n["pair"])
        if n.get("entity"):
            return [n["entity"]]
        return [entity_key_of_file(n["family"])] if n.get("family") else []

    def _study_taken(self) -> list[tuple]:
        out = []
        for q in self.state.question_history:
            t = q["target"]
            if t.get("cycle", 1) != self._cycle:
                continue
            if t.get("how") == "deepen" and t.get("entity"):
                out.append(("DEEPEN", (t["entity"],)))
            elif t.get("how") == "connect" and t.get("pair"):
                out.append(("CONNECT", tuple(t["pair"])))
        return out

    def plan_study(self, n: int) -> list[dict]:
        """One planner call: up to n DEEPEN / CONNECT actions chosen on the
        observation digest (every entity with its notes so far)."""
        blocks = self.blocks()
        ents, _ = entity_index(blocks, entity_key)
        block_of = {entity_key_of_file(f): b["name"] for b in blocks for f in b["files"]}
        by_ent: dict[str, list[dict]] = {}
        for bname, notes in self.state.notes.items():
            for note in notes:
                for e in self._note_entities(note):
                    by_ent.setdefault(e, []).append(note)
                    ents.setdefault(e, {"block": block_of.get(e, note.get("block", bname)), "units": []})
        if self._chars_by_entity is None:
            self._chars_by_entity = {}
            for f in self.all_files:
                e = entity_key_of_file(f)
                self._chars_by_entity[e] = self._chars_by_entity.get(e, 0) + len(self.ws.read_text(f))
        self._ents = ents
        digest = observation_digest(ents, by_ent, self._chars_by_entity)
        try:
            return study_actions(ents, digest, self._cross_entity_units(), self._study_taken(), n, self.brief, self.llm)
        except Exception as e:  # noqa: BLE001 — a bad planner reply costs one round, not the run
            print(f"  [study] planning failed ({str(e)[:100]})", file=sys.stderr)
            return []

    def _study_action(self, a: dict) -> tuple | None:
        """LLM half of one study action (reads the model only): the target,
        its evidence and the observations it added. None on failure."""
        try:
            ents = self._ents
            if a["kind"] == "DEEPEN":
                e = a["entities"][0]
                info = ents.get(e, {"block": "", "units": []})
                units = relevant_units(self.model, a["question"], must=[u.unit_id for u in info["units"][:12]], k=4)
                files = {f for f in self.all_files if entity_key_of_file(f) == e}
                evidence, tier = inspect_source(a["question"], self.retriever, self.all_files, DEEPEN_K, scoped=files,
                                                ws=self.ws, elements=self.sm.elements)
                prior = [n["statement"] for ns in self.state.notes.values() for n in ns
                         if not n.get("pair") and e in self._note_entities(n)][:6]
                text = a["question"] + ("\n\nObservations already recorded about this entity (resolve or sharpen them; "
                                        "do not restate them):\n" + "\n".join(f"- {p}" for p in prior) if prior else "")
                notes = observe({"kind": "deepen", "text": text}, units, evidence, self.llm, self.brief)
                for n in notes:
                    n.update({"entity": e, "kind": "deepen"})
                target = {"kind": "DEEPEN", "blocks": [info["block"]], "entity": e, "why": a.get("why", ""), "how": "deepen",
                          "observation": a.get("observation", ""), "cycle": self._cycle}
                return target, a["question"], units, evidence, tier, notes
            x, y = a["entities"]
            units, evidence, tier, notes = self._connect_observe(x, y, a["question"], ents)
            target = {"kind": "CONNECT", "blocks": sorted({ents[x]["block"], ents[y]["block"]}), "pair": [x, y],
                      "why": a.get("why", ""), "how": "connect", "observation": a.get("observation", ""), "cycle": self._cycle}
            return target, a["question"], units, evidence, tier, notes
        except Exception as e:  # noqa: BLE001 — one failed action must not end the study
            print(f"  [study] {a.get('kind')} {a.get('entities')} failed ({str(e)[:100]})", file=sys.stderr)
            return None

    def run_cycles(self, n: int = 1, k: int = STUDY_K, round_size: int = STUDY_ROUND, min_observations: int = 0,
                   out: Path | None = None) -> None:
        """M^(c+1) = Consolidate(M^(c), Study(Inspect(S, M^(c)))) for c = 1..n. Every cycle re-inspects the
        whole source against the model the previous cycle produced. n = 1 is the main setting (one complete
        learning episode) and runs exactly `run_cycle`. `min_observations` > 0 stops early when a cycle's
        inspection finds fewer new observations than that. A resumed run continues from the cycle it was in;
        m_self_cycle<c>.json snapshots are written under `out`."""
        for c in range(len(self.state.cycles) + 1, n + 1):
            stats = self.run_cycle(k, round_size, cycle=c)
            if out is not None:
                (out / f"m_self_cycle{c}.json").write_text(json.dumps([u.model_dump() for u in self.model], indent=1, default=str))
            print(f"  [cycle {c}] observations {stats['observations']}, study notes {stats['study_notes']}, "
                  f"rewrites {stats['rewrites_applied']}/{stats['rewrites_attempted']}, tokens {stats['tokens']:,}", file=sys.stderr)
            if min_observations and stats["observations"] < min_observations and c < n:
                print(f"  [cycle {c}] fewer than {min_observations} new observations: stopping", file=sys.stderr)
                break

    def run_cycle(self, k: int = STUDY_K, round_size: int = STUDY_ROUND, cycle: int = 1) -> dict:
        """One self-study cycle, read many / write once:
          1. inspection — every entity in full -> observations (no edit);
          2. adaptive study — up to k planner-chosen DEEPEN / CONNECT actions on
             the observations, round_size per planner call, evidence retrieved,
             more observations (no edit);
          3. consolidation — every entity with notes rewritten once, then one
             relation pass per CONNECT pair.
        Returns the cycle's statistics (kept in
        state.cycles); `cycle` tags every target so a later cycle re-inspects every entity."""
        self._cycle = cycle
        h0, r0 = len(self.state.question_history), len(self.reorg_log)
        n_notes = lambda: sum(len(ns) for ns in self.state.notes.values())                           # noqa: E731
        n_before = n_notes()
        self.inspect_entities()
        n_obs = n_notes() - n_before
        done = lambda: sum(1 for q in self.state.question_history                                     # noqa: E731
                           if q["target"].get("how") in ("deepen", "connect") and q["target"].get("cycle", 1) == cycle)
        idle = 0
        while done() < k and idle < 2:
            actions = self.plan_study(min(round_size, k - done()))
            if not actions:
                idle += 1
                continue
            idle = 0
            print("  [study] round: " + ", ".join(f"{a['kind']}({', '.join(e.split('/')[-1] for e in a['entities'])})" for a in actions),
                  file=sys.stderr)
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:   # actions only read the model
                results = list(pool.map(self._study_action, actions))
            for r in results:
                if r is None:
                    continue
                target, question, units, evidence, tier, notes = r
                row = self.take_notes(target, question, target["blocks"], units, evidence, target["how"], tier, notes=notes)
                print(f"  [study] {target['kind']} {target.get('entity') or ' <-> '.join(target.get('pair', []))}: {row['verdict']} ({len(notes)} notes)",
                      file=sys.stderr)
            self._checkpoint()
        n_study = n_notes() - n_before - n_obs
        self.reorganize_all()
        self.consolidate_connect()
        new_reorg = self.reorg_log[r0:]
        stats = {"cycle": cycle, "observations": n_obs, "actions": done(), "study_notes": n_study,
                 "rewrites_applied": sum(1 for t in new_reorg if t.get("applied")), "rewrites_attempted": len(new_reorg),
                 "units": len(self.model), "tokens": m_tokens(self.model), "steps": len(self.state.question_history) - h0}
        prev = next((c for c in self.state.cycles if c["cycle"] == cycle), None)
        if prev is None:
            self.state.cycles.append(stats)
        else:                                                   # resumed cycle: counts accumulate, sizes are current
            for key in ("observations", "study_notes", "rewrites_applied", "rewrites_attempted", "steps"):
                prev[key] += stats[key]
            prev.update({k: stats[k] for k in ("actions", "units", "tokens")})
            stats = prev
        return stats

    def summary(self) -> dict:
        from sourcelearn.reconstruction.reconstruct import structure_stats
        h = self.state.question_history
        ids = {u.unit_id for u in self.model}
        return {"steps": len(h),
                "notes_taken": sum(1 for r in h if r["verdict"] == "NOTES"),
                "reorganizations": dict(self.state.reorganized),
                "rewrite": rewrite_metrics(self.reorg_log),
                "structure_now": structure_stats(self.model),
                "structure_m0_kept": structure_stats([u for u in self.model if u.unit_id in self.m0_ids]),
                "noop_rate": round(sum(1 for r in h if r["verdict"] == "KNOWN") / max(1, len(h)), 3),
                "m0_units": len(self.m0_ids), "units": len(ids), "m0_units_replaced": len(self.m0_ids - ids),
                "rewrite_fraction": round(len(self.m0_ids - ids) / max(1, len(self.m0_ids)), 3),
                "added_units": len(ids - self.m0_ids), "m0_tokens": self.m0_tokens, "tokens": m_tokens(self.model),
                "cycles": list(self.state.cycles),
                "targets": {"inspection": sum(1 for r in h if r["target"].get("how") == "exhaustive"),
                            "deepen": sum(1 for r in h if r["target"].get("how") == "deepen"),
                            "connect": sum(1 for r in h if r["target"].get("how") == "connect")}}


def run_stage(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True)
    p.add_argument("--m0", required=True)
    p.add_argument("--partition", default="", help="partition.json; default: one block per top-level directory")
    p.add_argument("--brief", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, help="one model: planner, observer, writer, grounder")
    p.add_argument("--workers", type=int, default=6, help="parallel LLM calls (entities of a block, actions of a round)")
    args = p.parse_args(argv)
    load_project_env()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    # a kill must still save state.json / m_self.json (the `finally` below), so a rerun resumes
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    trace = Trace(out, "trace")
    llm = LLMClient(model=args.model, trace=trace, seed=0)
    ws_path = Path(args.workspace)
    ws = Workspace(ws_path)
    bundle = CompileBundle.load(args.m0, args.partition or None, args.brief or None)
    m0, brief = bundle.model, bundle.brief
    model = m0
    state_path = out / "state.json"
    state = StudyState.load(state_path) if state_path.exists() else None
    if state is not None and (out / "m_self.json").exists():   # resume: the model as it was at the interruption
        model = CompileBundle.load(out / "m_self.json").model
        print(f"resuming at step {state.step} from {out / 'm_self.json'}", file=sys.stderr)
    sm = SourceModelSession(ws_path, llm)
    retriever = build_retriever("rag_hybrid", sm._retrievable, llm)
    partition = bundle.partition or auto_partition(model)
    if not partition:
        partition = {"blocks": [{"name": "all", "description": "the whole source",
                                 "files": sorted({e.file for e in sm._retrievable})}]}

    def checkpoint() -> None:   # a rerun resumes from here (state.json holds the pending notes)
        (out / "m_self.json").write_text(json.dumps([u.model_dump() for u in model], indent=1, default=str))
        ss.state.save(state_path)

    ss = SelfStudy(model, partition, ws, sm, retriever, llm, brief, CONNECT_K, 0, state,     # no growth cap on a rewrite
                   trace_path=out / "study_trace.jsonl", m0=m0, reorg_path=out / "reorganize_trace.jsonl",
                   workers=args.workers, checkpoint=checkpoint)
    print(f"self-study: {len(partition['blocks'])} blocks, M0 {len(model)} units (~{ss.m0_tokens:,} tok), "
          f"every entity in full, then {STUDY_K} study actions ({STUDY_ROUND} per planner call)", file=sys.stderr)
    try:
        ss.run_cycles()
        (out / "FINISHED").write_text(json.dumps({"steps": ss.state.step}))
    finally:
        (out / "m_self.json").write_text(json.dumps([u.model_dump() for u in model], indent=1, default=str))
        ss.state.save(state_path)
        summ = ss.summary()
        (out / "study_summary.json").write_text(json.dumps(summ, indent=1))
        trace.close()
    print(json.dumps(summ, indent=1))
    return 0
