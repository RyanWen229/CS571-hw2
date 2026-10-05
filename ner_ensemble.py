#!/usr/bin/env python3
"""Combine a distant-supervised NER model with a supervised checkpoint.

The dev-selected rule keeps all mentions from the distant-supervised BERT for
the listed primary-only types (PER by default). For every other type, it keeps
only exact span-and-type predictions shared by both models. This increases
precision for the harder ORG/LOC/TIME/NUM/MISC categories while retaining the
strong person-name recall of the larger model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return value


def mention_key(mention: dict) -> tuple[int, int, int, str]:
    return (
        int(mention["sent_id"]),
        int(mention["pos"][0]),
        int(mention["pos"][1]),
        str(mention["type"]),
    )


def mentions(document: dict) -> dict[tuple[int, int, int, str], dict]:
    return {
        mention_key(mention): mention
        for entity in document.get("vertexSet", [])
        for mention in entity
    }


def group_mentions(document: dict, selected: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for mention in sorted(selected, key=mention_key):
        sent_id, start, end, entity_type = mention_key(mention)
        sentence = document.get("sents", [])[sent_id]
        name = " ".join(sentence[start:end])
        clean = {
            "name": name,
            "type": entity_type,
            "sent_id": sent_id,
            "pos": [start, end],
        }
        normalized = re.sub(r"\s+", " ", name.casefold()).strip()
        grouped.setdefault((entity_type, normalized), []).append(clean)
    output = dict(document)
    output.pop("labels", None)
    output["vertexSet"] = list(grouped.values())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path, help="distant-supervised NER predictions")
    parser.add_argument("secondary", type=Path, help="supervised NER predictions")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--primary-only-types",
        default="PER",
        help="comma-separated types taken directly from the primary model",
    )
    args = parser.parse_args()

    primary, secondary = load(args.primary), load(args.secondary)
    if len(primary) != len(secondary):
        parser.error("prediction files contain different document counts")
    primary_only = {value.strip() for value in args.primary_only_types.split(",") if value.strip()}
    output = []
    for first, second in zip(primary, secondary):
        if first.get("title") != second.get("title") or first.get("sents") != second.get("sents"):
            parser.error("prediction files are not aligned by title and tokens")
        first_mentions = mentions(first)
        second_keys = set(mentions(second))
        selected = [
            mention
            for key, mention in first_mentions.items()
            if key[3] in primary_only or key in second_keys
        ]
        output.append(group_mentions(first, selected))

    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False)


if __name__ == "__main__":
    main()
