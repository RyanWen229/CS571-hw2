#!/usr/bin/env python3
"""
ner_eval.py -- span-level evaluation of entity tagging on Re-DocRED.

Usage:
    python ner_eval.py GOLD.json PRED.json [--json REPORT.json] [--relaxed]

GOLD.json  : a Re-DocRED file (list of documents with "title", "sents", "vertexSet", ...).
PRED.json  : a list of documents in the same order (or matched by title), each with
             "title" and a predicted "vertexSet". vertexSet is a list of entities, each a
             list of mentions {"sent_id": int, "pos": [start, end], "type": str}.
             Grouping mentions into entities does not affect this score; a flat list of
             single-mention entities is fine.

A predicted mention is correct when its sentence, start, end and type all equal a gold
mention. Precision, recall and F1 are reported per type and micro-averaged over all
mentions. With --relaxed, a second table scores type-agnostic overlap: a predicted span
counts as correct if it overlaps any gold span in the same sentence with the same type.

Exit code is 1 when the files cannot be aligned.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

TYPES = ["PER", "ORG", "LOC", "TIME", "NUM", "MISC"]


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"{path}: expected a JSON list of documents")
    return data


def mentions(doc, path, idx, strict_fields=True):
    """Return the set of (sent_id, start, end, type) for a document, with basic validation."""
    out = set()
    n_sents = len(doc.get("sents", []))
    for e_i, entity in enumerate(doc.get("vertexSet", [])):
        for m in entity:
            try:
                s, (a, b), t = int(m["sent_id"]), m["pos"], str(m["type"])
            except (KeyError, TypeError, ValueError):
                sys.exit(f"{path}: document {idx} entity {e_i}: mention needs sent_id, pos=[start,end], type; got {m}")
            if not (0 <= a < b):
                sys.exit(f"{path}: document {idx}: bad span {m['pos']}")
            if strict_fields and n_sents and not (0 <= s < n_sents):
                sys.exit(f"{path}: document {idx}: sent_id {s} out of range (document has {n_sents} sentences)")
            out.add((s, a, b, t))
    return out


def align(gold, pred):
    """Pair gold and predicted documents by title; fall back to position if titles are absent."""
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
            p = {"title": g.get("title"), "vertexSet": []}
        pairs.append((g, p))
    if missing:
        print(f"warning: {missing} gold documents had no prediction and were scored as empty", file=sys.stderr)
    if missing == len(gold):
        sys.exit("no predicted document could be matched to a gold document (check titles / order)")
    return pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pairs, relaxed=False):
    tp, fp, fn = Counter(), Counter(), Counter()
    for i, (g, p) in enumerate(pairs):
        G = mentions(g, "gold", i)
        P = mentions(p, "pred", i, strict_fields=bool(p.get("sents")))
        if not relaxed:
            for m in G & P:
                tp[m[3]] += 1
            for m in P - G:
                fp[m[3]] += 1
            for m in G - P:
                fn[m[3]] += 1
        else:
            g_by = defaultdict(list)
            for s, a, b, t in G:
                g_by[(s, t)].append((a, b))
            matched_gold = set()
            for s, a, b, t in P:
                hit = None
                for ga, gb in g_by.get((s, t), []):
                    if a < gb and ga < b and (s, ga, gb, t) not in matched_gold:
                        hit = (s, ga, gb, t)
                        break
                if hit:
                    tp[t] += 1
                    matched_gold.add(hit)
                else:
                    fp[t] += 1
            for m in G - matched_gold:
                fn[m[3]] += 1
    types = [t for t in TYPES if tp[t] + fp[t] + fn[t]] + sorted(
        {t for t in list(tp) + list(fp) + list(fn) if t not in TYPES})
    rows = {}
    for t in types:
        p, r, f = prf(tp[t], fp[t], fn[t])
        rows[t] = {"precision": p, "recall": r, "f1": f, "tp": tp[t], "fp": fp[t], "fn": fn[t],
                   "gold": tp[t] + fn[t], "predicted": tp[t] + fp[t]}
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, r, f = prf(T, FP, FN)
    rows["MICRO"] = {"precision": p, "recall": r, "f1": f, "tp": T, "fp": FP, "fn": FN,
                     "gold": T + FN, "predicted": T + FP}
    if types:
        rows["MACRO"] = {k: sum(rows[t][k] for t in types) / len(types) for k in ("precision", "recall", "f1")}
    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print(f"{'type':<8}{'P':>8}{'R':>8}{'F1':>8}{'gold':>8}{'pred':>8}{'tp':>7}{'fp':>7}{'fn':>7}")
    for t, v in rows.items():
        if t == "MACRO":
            print(f"{t:<8}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}")
        else:
            print(f"{t:<8}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}"
                  f"{v['gold']:>8}{v['predicted']:>8}{v['tp']:>7}{v['fp']:>7}{v['fn']:>7}")


def main():
    ap = argparse.ArgumentParser(description="Span-level NER evaluation for Re-DocRED (exact span + type).")
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--json", help="write the report as JSON to this path")
    ap.add_argument("--relaxed", action="store_true", help="also report type-matched overlap scores")
    args = ap.parse_args()

    pairs = align(load(args.gold), load(args.pred))
    report = {"documents": len(pairs), "exact": score(pairs)}
    print_table(report["exact"], f"Exact span + type match over {len(pairs)} documents")
    if args.relaxed:
        report["relaxed_overlap"] = score(pairs, relaxed=True)
        print_table(report["relaxed_overlap"], "Relaxed: overlap with a gold span of the same type")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
