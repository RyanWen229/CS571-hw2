#!/usr/bin/env python3
"""
re_eval.py -- evaluation of relation extraction on Re-DocRED.

Usage:
    python re_eval.py GOLD.json PRED.json [--json REPORT.json] [--per-relation] [--names rel_info.json]

GOLD.json : a Re-DocRED file (documents with "title", "sents", "vertexSet", "labels").
PRED.json : the same documents, in the same order or matched by title, each with the
            "vertexSet" that the relations refer to and a "labels" list of
            {"h": int, "t": int, "r": str}. "evidence" is ignored.

Two evaluation modes, chosen automatically per document:

  gold entities   -- the predicted vertexSet has exactly the gold mentions (for example,
                     the prediction file was produced from the gold file). Entities are
                     matched by index.
  pipeline        -- the predicted vertexSet came from the student's tagger. A predicted
                     entity is aligned to a gold entity when at least one of its mentions
                     matches a gold mention exactly (same sent_id, start, end). A predicted
                     entity that aligns to no gold entity, or to a gold entity already taken
                     by another predicted entity with more matching mentions, counts as
                     unaligned; every triple that uses it is a false positive.

A predicted triple is correct when its aligned head, aligned tail and relation id equal a
gold triple. Duplicate predicted triples are counted once. Precision, recall and F1 are
micro-averaged over all triples in the file; --per-relation adds a table by relation id,
sorted by gold frequency. Two extra numbers report how the alignment went: the share of
gold entities reached by some predicted entity, and the share of gold triples whose two
arguments were both reachable, which is the recall ceiling for the pipeline.

Exit code is 1 when the files cannot be aligned.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"{path}: expected a JSON list of documents")
    return data


def mention_keys(entity):
    keys = set()
    for m in entity:
        try:
            keys.add((int(m["sent_id"]), int(m["pos"][0]), int(m["pos"][1])))
        except (KeyError, TypeError, ValueError, IndexError):
            sys.exit(f"mention needs sent_id and pos=[start,end]; got {m}")
    return keys


def align_docs(gold, pred):
    if len(gold) != len(pred):
        print(f"warning: {len(gold)} gold documents, {len(pred)} predicted documents", file=sys.stderr)
    by_title = {d.get("title"): d for d in pred if d.get("title") is not None}
    pairs, missing = [], 0
    for i, g in enumerate(gold):
        p = by_title.get(g.get("title"))
        if p is None and i < len(pred) and pred[i].get("title") in (None, g.get("title")):
            p = pred[i]
        if p is None:
            missing += 1
            p = {"title": g.get("title"), "vertexSet": [], "labels": []}
        pairs.append((g, p))
    if missing:
        print(f"warning: {missing} gold documents had no prediction and were scored as empty", file=sys.stderr)
    if missing == len(gold):
        sys.exit("no predicted document could be matched to a gold document (check titles / order)")
    return pairs


def align_entities(gold_vs, pred_vs):
    """Return (mapping pred index -> gold index or None, mode)."""
    gold_keys = [mention_keys(e) for e in gold_vs]
    pred_keys = [mention_keys(e) for e in pred_vs]
    if len(pred_keys) == len(gold_keys) and all(a == b for a, b in zip(pred_keys, gold_keys)):
        return list(range(len(gold_vs))), "gold"
    key_to_gold = {}
    for gi, ks in enumerate(gold_keys):
        for k in ks:
            key_to_gold[k] = gi
    # score every (pred, gold) pair by number of shared mentions, assign greedily, one gold per pred
    cands = []
    for pi, ks in enumerate(pred_keys):
        shared = Counter(key_to_gold[k] for k in ks if k in key_to_gold)
        for gi, n in shared.items():
            cands.append((n, -pi, pi, gi))
    cands.sort(reverse=True)
    mapping = [None] * len(pred_vs)
    used_gold = set()
    for n, _, pi, gi in cands:
        if mapping[pi] is None and gi not in used_gold:
            mapping[pi] = gi
            used_gold.add(gi)
    return mapping, "pipeline"


def triples(labels, mapping=None, doc_idx=0, path="pred"):
    out, dropped = set(), 0
    for l in labels:
        try:
            h, t, r = int(l["h"]), int(l["t"]), str(l["r"])
        except (KeyError, TypeError, ValueError):
            sys.exit(f"{path}: document {doc_idx}: label needs h, t, r; got {l}")
        if mapping is not None:
            if not (0 <= h < len(mapping) and 0 <= t < len(mapping)):
                sys.exit(f"{path}: document {doc_idx}: label {l} refers to an entity index outside vertexSet")
            h, t = mapping[h], mapping[t]
            if h is None or t is None:
                dropped += 1
                out.add(("unaligned", len(out), r))  # unique key, always a false positive
                continue
        out.add((h, t, r))
    return out, dropped


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(pairs, names=None):
    tp, fp, fn = Counter(), Counter(), Counter()
    modes = Counter()
    gold_entities = reached_entities = 0
    gold_triples = reachable_triples = 0
    unaligned_preds = 0
    for i, (g, p) in enumerate(pairs):
        mapping, mode = align_entities(g.get("vertexSet", []), p.get("vertexSet", []))
        modes[mode] += 1
        G, _ = triples(g.get("labels", []), None, i, "gold")
        P, dropped = triples(p.get("labels", []), mapping, i, "pred")
        unaligned_preds += dropped
        reached = {gi for gi in mapping if gi is not None}
        gold_entities += len(g.get("vertexSet", []))
        reached_entities += len(reached)
        gold_triples += len(G)
        reachable_triples += sum(1 for (h, t, r) in G if h in reached and t in reached)
        for x in G & P:
            tp[x[2]] += 1
        for x in P - G:
            fp[x[2]] += 1
        for x in G - P:
            fn[x[2]] += 1
    rels = sorted(set(tp) | set(fp) | set(fn), key=lambda r: -(tp[r] + fn[r]))
    rows = {}
    for r in rels:
        p_, r_, f_ = prf(tp[r], fp[r], fn[r])
        rows[r] = {"name": (names or {}).get(r, ""), "precision": p_, "recall": r_, "f1": f_,
                   "tp": tp[r], "fp": fp[r], "fn": fn[r], "gold": tp[r] + fn[r], "predicted": tp[r] + fp[r]}
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p_, r_, f_ = prf(T, FP, FN)
    summary = {"precision": p_, "recall": r_, "f1": f_, "tp": T, "fp": FP, "fn": FN,
               "gold_triples": T + FN, "predicted_triples": T + FP,
               "documents": len(pairs), "mode": dict(modes),
               "gold_entities": gold_entities, "gold_entities_reached": reached_entities,
               "entity_reach": reached_entities / gold_entities if gold_entities else 0.0,
               "recall_ceiling": reachable_triples / gold_triples if gold_triples else 0.0,
               "predicted_triples_on_unaligned_entities": unaligned_preds}
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description="Relation extraction evaluation for Re-DocRED (micro P/R/F1 over triples).")
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--json", help="write the report as JSON to this path")
    ap.add_argument("--per-relation", action="store_true", help="print a table per relation id")
    ap.add_argument("--names", help="optional JSON file mapping relation ids to names")
    args = ap.parse_args()

    names = load_names(args.names) if args.names else None
    pairs = align_docs(load(args.gold), load(args.pred))
    summary, rows = evaluate(pairs, names)

    mode = "gold entities" if summary["mode"].get("pipeline", 0) == 0 else (
        "pipeline" if summary["mode"].get("gold", 0) == 0 else f"mixed {summary['mode']}")
    print(f"\nRelation extraction over {summary['documents']} documents  (entity mode: {mode})")
    print(f"micro precision {summary['precision']:.4f}   recall {summary['recall']:.4f}   F1 {summary['f1']:.4f}")
    print(f"gold triples {summary['gold_triples']}   predicted {summary['predicted_triples']}   "
          f"tp {summary['tp']}   fp {summary['fp']}   fn {summary['fn']}")
    if mode != "gold entities":
        print(f"gold entities reached by a predicted entity: {summary['gold_entities_reached']}/{summary['gold_entities']} "
              f"({summary['entity_reach']:.3f});  recall ceiling given the entities: {summary['recall_ceiling']:.3f};  "
              f"predicted triples on unaligned entities: {summary['predicted_triples_on_unaligned_entities']}")
    if args.per_relation:
        print(f"\n{'relation':<10}{'name':<34}{'P':>8}{'R':>8}{'F1':>8}{'gold':>7}{'pred':>7}")
        for r, v in rows.items():
            print(f"{r:<10}{v['name'][:32]:<34}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}{v['gold']:>7}{v['predicted']:>7}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_relation": rows}, f, indent=2)
        print(f"\nreport written to {args.json}")


def load_names(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
