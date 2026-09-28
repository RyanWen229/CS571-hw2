#!/usr/bin/env python3
"""Analyze NER type and boundary errors without modifying the gold scorer.

Usage: python ner_error_analysis.py GOLD.json PRED.json [--json REPORT.json]

The script imports document alignment and mention validation from ner_eval.py,
then pairs unmatched gold and predicted spans for diagnostic purposes only.
These pairings do not affect the official precision, recall, or F1 scores.
"""

import argparse
import json
from collections import Counter, defaultdict

from ner_eval import TYPES, align, load, mentions


def overlap(first, second):
    """Return token intersection-over-union for two same-sentence mentions."""
    s1, a1, b1, _ = first
    s2, a2, b2, _ = second
    if s1 != s2:
        return 0.0
    intersection = max(0, min(b1, b2) - max(a1, a2))
    union = max(b1, b2) - min(a1, a2)
    return intersection / union if union else 0.0


def boundary_pattern(gold, pred):
    """Describe how an overlapping predicted boundary differs from gold."""
    _, gold_start, gold_end, _ = gold
    _, pred_start, pred_end, _ = pred
    if pred_start >= gold_start and pred_end <= gold_end:
        return "too_narrow"
    if pred_start <= gold_start and pred_end >= gold_end:
        return "too_wide"
    if pred_end - pred_start == gold_end - gold_start:
        return "shifted"
    return "mixed"


def analyze(pairs):
    """Pair near misses once and summarize type and boundary failure modes."""
    counts = Counter()
    confusion = defaultdict(Counter)
    patterns = Counter()
    missed_by_type = Counter()
    spurious_by_type = Counter()

    for index, (gold_doc, pred_doc) in enumerate(pairs):
        gold = mentions(gold_doc, "gold", index)
        pred = mentions(
            pred_doc, "pred", index, strict_fields=bool(pred_doc.get("sents"))
        )
        exact = gold & pred
        remaining_gold = set(gold - exact)
        remaining_pred = set(pred - exact)
        counts["exact"] += len(exact)

        # Pair identical boundaries with differing types first. These cases
        # populate the conventional gold-type versus predicted-type matrix.
        for gold_mention in sorted(remaining_gold):
            candidates = [
                pred_mention
                for pred_mention in remaining_pred
                if pred_mention[:3] == gold_mention[:3]
            ]
            if not candidates:
                continue
            pred_mention = sorted(candidates, key=lambda item: item[3])[0]
            counts["type_only"] += 1
            confusion[gold_mention[3]][pred_mention[3]] += 1
            remaining_gold.remove(gold_mention)
            remaining_pred.remove(pred_mention)

        # Pair remaining overlaps greedily. Same-type candidates take priority,
        # followed by greatest token IoU and smallest boundary displacement.
        for gold_mention in sorted(remaining_gold):
            candidates = sorted(
                pred_mention
                for pred_mention in remaining_pred
                if overlap(gold_mention, pred_mention) > 0
            )
            if not candidates:
                continue
            pred_mention = max(
                candidates,
                key=lambda item: (
                    item[3] == gold_mention[3],
                    overlap(gold_mention, item),
                    -(abs(item[1] - gold_mention[1]) + abs(item[2] - gold_mention[2])),
                ),
            )
            patterns[boundary_pattern(gold_mention, pred_mention)] += 1
            if pred_mention[3] == gold_mention[3]:
                counts["boundary_only"] += 1
            else:
                counts["boundary_and_type"] += 1
                confusion[gold_mention[3]][pred_mention[3]] += 1
            remaining_gold.remove(gold_mention)
            remaining_pred.remove(pred_mention)

        for mention in remaining_gold:
            missed_by_type[mention[3]] += 1
        for mention in remaining_pred:
            spurious_by_type[mention[3]] += 1

    counts["missed"] = sum(missed_by_type.values())
    counts["spurious"] = sum(spurious_by_type.values())
    paired = (
        counts["type_only"]
        + counts["boundary_only"]
        + counts["boundary_and_type"]
    )
    boundary_involved = counts["boundary_only"] + counts["boundary_and_type"]
    type_involved = counts["type_only"] + counts["boundary_and_type"]

    extra_types = (
        {entity_type for entity_type in confusion}
        | {entity_type for row in confusion.values() for entity_type in row}
    ) - set(TYPES)
    all_types = TYPES + sorted(extra_types)
    matrix = {
        gold_type: {
            pred_type: confusion[gold_type][pred_type] for pred_type in all_types
        }
        for gold_type in all_types
    }

    observations = []
    confused_pairs = [
        (count, gold_type, pred_type)
        for gold_type, row in confusion.items()
        for pred_type, count in row.items()
        if count
    ]
    if confused_pairs:
        count, gold_type, pred_type = max(confused_pairs)
        observations.append(
            f"Most frequent type confusion: {gold_type} -> {pred_type} "
            f"({count} paired spans)."
        )
    if patterns:
        pattern, count = patterns.most_common(1)[0]
        observations.append(f"Most frequent boundary pattern: {pattern} ({count} spans).")
    if missed_by_type:
        entity_type, count = missed_by_type.most_common(1)[0]
        observations.append(f"Most completely missed gold type: {entity_type} ({count} spans).")
    if spurious_by_type:
        entity_type, count = spurious_by_type.most_common(1)[0]
        observations.append(
            f"Most frequent spurious predicted type: {entity_type} ({count} spans)."
        )

    return {
        "counts": dict(counts),
        "paired_near_misses": paired,
        "boundary_involved_share": boundary_involved / paired if paired else 0.0,
        "type_involved_share": type_involved / paired if paired else 0.0,
        "confusion_gold_by_predicted": matrix,
        "boundary_patterns": dict(patterns),
        "missed_by_type": dict(missed_by_type),
        "spurious_by_type": dict(spurious_by_type),
        "observations": observations,
    }


def print_analysis(report):
    """Print a compact human-readable version of an analysis report."""
    counts = report["counts"]
    print("Error analysis")
    print(
        f"paired near misses: {report['paired_near_misses']} "
        f"(type only={counts.get('type_only', 0)}, "
        f"boundary only={counts.get('boundary_only', 0)}, "
        f"both={counts.get('boundary_and_type', 0)})"
    )
    print(
        "among paired near misses: "
        f"boundary involved={report['boundary_involved_share']:.1%}, "
        f"type involved={report['type_involved_share']:.1%}"
    )
    print(
        f"unpaired gold misses={counts.get('missed', 0)}, "
        f"spurious predictions={counts.get('spurious', 0)}"
    )

    matrix = report["confusion_gold_by_predicted"]
    types = list(matrix)
    print("\nType confusion for paired mismatches (rows=gold, columns=predicted)")
    print(f"{'gold':<8}" + "".join(f"{entity_type:>8}" for entity_type in types))
    for gold_type in types:
        print(
            f"{gold_type:<8}"
            + "".join(
                f"{matrix[gold_type][pred_type]:>8}" for pred_type in types
            )
        )

    if report["boundary_patterns"]:
        print(
            "\nBoundary patterns: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(report["boundary_patterns"].items())
            )
        )
    for observation in report["observations"]:
        print(f"- {observation}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic NER boundary/type analysis; does not alter official scoring."
    )
    parser.add_argument("gold")
    parser.add_argument("pred")
    parser.add_argument("--json", help="write the diagnostic report as JSON")
    args = parser.parse_args()

    pairs = align(load(args.gold), load(args.pred))
    report = {"documents": len(pairs), "error_analysis": analyze(pairs)}
    print_analysis(report["error_analysis"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
