#!/usr/bin/env python3
"""Restore DocRED distant labels for the supplied 10,000-document sample.

The homework's ``unlabeled_10000.json`` was sampled from DocRED's official
``train_distant.json`` after removing ``vertexSet`` and ``labels``. This script
matches records by title and the complete tokenized text, then writes only the
matched weak annotations. Matching the text as well as the title avoids
silently attaching labels to a different article revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def document_key(document: dict) -> str:
    payload = json.dumps(
        [document.get("title"), document.get("sents")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_documents(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as stream:
            documents = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(documents, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match the unlabeled sample to DocRED train_distant.json"
    )
    parser.add_argument("source", type=Path, help="official DocRED train_distant.json")
    parser.add_argument(
        "--unlabeled",
        type=Path,
        default=Path("data/unlabeled_10000.json"),
        help="stripped 10,000-document sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/distant_10000.json"),
        help="destination for matched weak labels",
    )
    args = parser.parse_args()

    unlabeled = load_documents(args.unlabeled)
    distant = load_documents(args.source)
    index: dict[str, dict] = {}
    duplicate_keys: set[str] = set()
    for document in distant:
        key = document_key(document)
        if key in index:
            duplicate_keys.add(key)
        else:
            index[key] = document

    matched: list[dict] = []
    missing: list[str] = []
    for document in unlabeled:
        key = document_key(document)
        if key in duplicate_keys:
            raise SystemExit(
                f"ambiguous match for title {document.get('title', '<untitled>')!r}"
            )
        annotated = index.get(key)
        if annotated is None:
            missing.append(str(document.get("title", "<untitled>")))
        else:
            matched.append(annotated)

    if missing:
        preview = ", ".join(repr(title) for title in missing[:5])
        raise SystemExit(f"{len(missing)} documents were not matched; first: {preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(matched, stream, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {args.output}: {exc}") from exc

    print(f"matched documents: {len(matched)}")
    print(f"weak entity clusters: {sum(len(d.get('vertexSet', [])) for d in matched)}")
    print(f"weak relation labels: {sum(len(d.get('labels', [])) for d in matched)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
