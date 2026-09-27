#!/usr/bin/env python3
"""
check_format.py -- validate an HW2 prediction file before submission.

Usage:
    python check_format.py INPUT.json OUTPUT.json [--re] [--names rel_info.json] [--quiet]

INPUT.json  : the unlabeled file the model was run on (title, sents), or the gold file.
OUTPUT.json : the prediction file. NER output must carry a vertexSet; with --re the file
              must also carry labels referring to that vertexSet.

Checks, in order: JSON structure; document count, order and titles; sents unchanged;
mention fields, types, sentence ids and offsets; names (compared ignoring whitespace, since
gold names keep the untokenized surface form); duplicate and overlapping spans (warnings only,
since the gold files themselves contain about 100 repeated mentions);
label fields, index ranges, self-relations, duplicates, unknown relation ids; then
plausibility warnings on mention and relation counts and on argument types of the most
frequent relations. Errors make the exit code 1; warnings do not.
"""
import argparse
import json
import sys
from collections import Counter

TYPES = {"PER", "ORG", "LOC", "TIME", "NUM", "MISC"}
# expected tail types for frequent relations (warnings only)
TAIL_TYPE = {"P17": "LOC", "P131": "LOC", "P150": "LOC", "P19": "LOC", "P20": "LOC", "P27": "LOC",
             "P569": "TIME", "P570": "TIME", "P571": "TIME", "P577": "TIME", "P580": "TIME", "P582": "TIME"}


class Report:
    def __init__(self, quiet=False):
        self.errors, self.warnings, self.quiet = [], [], quiet

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def finish(self):
        for m in (self.errors if self.quiet else self.errors[:200]):
            print("ERROR  " + m)
        if not self.quiet and len(self.errors) > 200:
            print(f"... {len(self.errors) - 200} more errors")
        for m in self.warnings:
            print("WARN   " + m)
        print(f"\n{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return 1 if self.errors else 0


def load(path, R):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        R.err(f"{path}: cannot read JSON ({e})")
        return None
    if not isinstance(data, list):
        R.err(f"{path}: top level must be a JSON list of documents")
        return None
    return data


def check(inp, out, R, want_re, names):
    if len(inp) != len(out):
        R.err(f"document count: input has {len(inp)}, output has {len(out)}")
    n_mentions = Counter()
    n_docs_with_mentions = 0
    n_labels = 0
    tail_violations = Counter()
    tail_checked = Counter()
    for i in range(min(len(inp), len(out))):
        a, b = inp[i], out[i]
        tag = f"doc {i}"
        if not isinstance(b, dict):
            R.err(f"{tag}: not a JSON object")
            continue
        if b.get("title") != a.get("title"):
            R.err(f"{tag}: title differs from input ({b.get('title')!r} vs {a.get('title')!r})")
        if b.get("sents") != a.get("sents"):
            R.err(f"{tag}: sents differ from input (tokens must be copied unchanged)")
        sents = a.get("sents", [])
        vs = b.get("vertexSet")
        if not isinstance(vs, list):
            R.err(f"{tag}: vertexSet missing or not a list")
            continue
        seen = {}
        spans_by_sent = {}
        for ei, ent in enumerate(vs):
            if not isinstance(ent, list) or not ent:
                R.err(f"{tag}: entity {ei} must be a non-empty list of mentions")
                continue
            for mi, m in enumerate(ent):
                where = f"{tag} entity {ei} mention {mi}"
                if not isinstance(m, dict):
                    R.err(f"{where}: not an object"); continue
                t, s, pos = m.get("type"), m.get("sent_id"), m.get("pos")
                if t not in TYPES:
                    R.err(f"{where}: type {t!r} not in {sorted(TYPES)}")
                if not isinstance(s, int) or not (0 <= s < len(sents)):
                    R.err(f"{where}: sent_id {s!r} out of range 0..{len(sents)-1}"); continue
                if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, int) for x in pos)):
                    R.err(f"{where}: pos must be [start, end] of integers, got {pos!r}"); continue
                st, en = pos
                if not (0 <= st < en <= len(sents[s])):
                    R.err(f"{where}: pos {pos} outside sentence {s} (length {len(sents[s])}); end is exclusive"); continue
                text = " ".join(sents[s][st:en])
                if "name" in m and "".join(str(m["name"]).split()) != "".join(text.split()):
                    R.warn(f"{where}: name {m['name']!r} does not match the tokens at pos, {text!r} (offsets may be wrong)")
                key = (s, st, en)
                if key in seen:
                    R.warn(f"{where}: duplicate span {key}, already listed in entity {seen[key]} (evaluators count it once)")
                seen[key] = ei
                for (os_, oe, oei) in spans_by_sent.get(s, []):
                    if st < oe and os_ < en and (os_, oe) != (st, en):
                        R.warn(f"{where}: span {key} overlaps span ({s}, {os_}, {oe}) in entity {oei}")
                spans_by_sent.setdefault(s, []).append((st, en, ei))
                n_mentions[t] += 1
        if vs:
            n_docs_with_mentions += 1
        if want_re:
            labels = b.get("labels")
            if not isinstance(labels, list):
                R.err(f"{tag}: labels missing or not a list (use [] when nothing is predicted)")
                continue
            seen_l = set()
            for li, l in enumerate(labels):
                where = f"{tag} label {li}"
                if not isinstance(l, dict):
                    R.err(f"{where}: not an object"); continue
                h, t_, r = l.get("h"), l.get("t"), l.get("r")
                ok = True
                for nm, v in (("h", h), ("t", t_)):
                    if not isinstance(v, int) or not (0 <= v < len(vs)):
                        R.err(f"{where}: {nm}={v!r} is not an index into vertexSet (0..{len(vs)-1})"); ok = False
                if not isinstance(r, str) or not r:
                    R.err(f"{where}: r must be a relation id string, got {r!r}"); ok = False
                elif names and r not in names:
                    R.err(f"{where}: unknown relation id {r!r}"); ok = False
                if not ok:
                    continue
                if h == t_:
                    R.err(f"{where}: h and t are the same entity ({h})")
                if (h, t_, r) in seen_l:
                    R.err(f"{where}: duplicate triple ({h}, {t_}, {r})")
                seen_l.add((h, t_, r))
                n_labels += 1
                if r in TAIL_TYPE and vs[t_] and isinstance(vs[t_][0], dict):
                    tail_checked[r] += 1
                    if vs[t_][0].get("type") != TAIL_TYPE[r]:
                        tail_violations[r] += 1
    # plausibility
    nd = max(1, min(len(inp), len(out)))
    total = sum(n_mentions.values())
    if total == 0:
        R.warn("no mentions predicted in any document")
    else:
        per_doc = total / nd
        if per_doc < 8 or per_doc > 60:
            R.warn(f"{per_doc:.1f} mentions per document; the gold files average about 26")
        share = {t: n_mentions[t] / total for t in TYPES}
        for t, ref in (("LOC", .32), ("PER", .18), ("TIME", .15), ("MISC", .15), ("ORG", .14), ("NUM", .05)):
            if share[t] > 2.5 * ref or (ref >= .1 and share[t] < ref / 3):
                R.warn(f"type {t} is {share[t]:.0%} of mentions; gold is about {ref:.0%}")
        if n_docs_with_mentions < nd:
            R.warn(f"{nd - n_docs_with_mentions} document(s) have an empty vertexSet")
    if want_re:
        per_doc = n_labels / nd
        if per_doc == 0:
            R.warn("no relations predicted in any document")
        elif per_doc < 5 or per_doc > 120:
            R.warn(f"{per_doc:.1f} relations per document; the gold files average about 35")
        for r, n in tail_checked.items():
            if n >= 20 and tail_violations[r] / n > 0.5:
                R.warn(f"relation {r}: {tail_violations[r]}/{n} tails are not {TAIL_TYPE[r]}; head and tail may be swapped")


def main():
    ap = argparse.ArgumentParser(description="Validate an HW2 prediction file.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--re", action="store_true", help="also check the labels list (relation output)")
    ap.add_argument("--names", help="rel_info.json; when given, relation ids are checked against it")
    ap.add_argument("--quiet", action="store_true", help="print every error (default caps at 200)")
    args = ap.parse_args()
    R = Report(args.quiet)
    inp, out = load(args.input, R), load(args.output, R)
    names = None
    if args.names:
        with open(args.names, encoding="utf-8") as f:
            names = json.load(f)
    if inp is not None and out is not None:
        check(inp, out, R, args.re, names)
    sys.exit(R.finish())


if __name__ == "__main__":
    main()
