#!/usr/bin/env python3
"""Fast, self-contained BIO tagger for Re-DocRED NER inference.

Usage: python ner_predict.py INPUT.json OUTPUT.json

The model is trained from data/train_revised.json. A cached model is reused when
the training file and feature version have not changed. Mentions with the same
type and case-folded token string are deterministically grouped as one entity.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Iterable


FEATURE_VERSION = 1
HASH_SIZE = 1 << 18
VALID_TYPES = {"PER", "ORG", "LOC", "TIME", "NUM", "MISC"}


def word_shape(word: str) -> str:
    """Map characters to compact case/digit/punctuation classes."""
    mapped = []
    for char in word:
        if char.isupper():
            code = "X"
        elif char.islower():
            code = "x"
        elif char.isdigit():
            code = "d"
        else:
            code = char
        if not mapped or mapped[-1] != code:
            mapped.append(code)
    return "".join(mapped[:12])


def token_features(tokens: list[str], index: int, previous_tag: str) -> list[str]:
    """Return sparse string features for one token."""
    word = tokens[index]
    lower = word.casefold()
    features = [
        "bias",
        f"w={word}",
        f"lower={lower}",
        f"shape={word_shape(word)}",
        f"prevtag={previous_tag}",
        f"len={min(len(word), 15)}",
        f"upper={word.isupper()}",
        f"title={word.istitle()}",
        f"lowercase={word.islower()}",
        f"digit={word.isdigit()}",
        f"hasdigit={any(c.isdigit() for c in word)}",
        f"hyphen={'-' in word}",
        f"alnum={word.isalnum()}",
    ]
    for size in range(1, min(4, len(lower)) + 1):
        features.append(f"pre{size}={lower[:size]}")
        features.append(f"suf{size}={lower[-size:]}")

    for offset in (-2, -1, 1, 2):
        pos = index + offset
        if 0 <= pos < len(tokens):
            nearby = tokens[pos]
            features.extend(
                (f"ctx{offset}={nearby.casefold()}", f"ctxshape{offset}={word_shape(nearby)}")
            )
        else:
            features.append(f"ctx{offset}=<BOUNDARY>")
    before = tokens[index - 1].casefold() if index else "<START>"
    after = tokens[index + 1].casefold() if index + 1 < len(tokens) else "<END>"
    features.extend((f"leftbigram={before}|{lower}", f"rightbigram={lower}|{after}"))
    return features


def sentence_tags(doc: dict) -> list[list[str]]:
    """Create one non-overlapping BIO label sequence per sentence."""
    tags = [["O"] * len(sentence) for sentence in doc.get("sents", [])]
    mentions = []
    for entity in doc.get("vertexSet", []):
        for mention in entity:
            try:
                sent_id = int(mention["sent_id"])
                start, end = map(int, mention["pos"])
                entity_type = str(mention["type"])
            except (KeyError, TypeError, ValueError):
                continue
            if entity_type in VALID_TYPES:
                mentions.append((-(end - start), sent_id, start, end, entity_type))

    for _, sent_id, start, end, entity_type in sorted(mentions):
        if not (0 <= sent_id < len(tags) and 0 <= start < end <= len(tags[sent_id])):
            continue
        if any(tag != "O" for tag in tags[sent_id][start:end]):
            continue
        tags[sent_id][start] = f"B-{entity_type}"
        tags[sent_id][start + 1 : end] = [f"I-{entity_type}"] * (end - start - 1)
    return tags


def training_rows(documents: Iterable[dict]):
    for doc in documents:
        labels = sentence_tags(doc)
        for tokens, tags in zip(doc.get("sents", []), labels):
            previous = "<START>"
            for index, tag in enumerate(tags):
                yield token_features(tokens, index, previous), tag
                previous = tag


def train_model(train_path: Path):
    try:
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:
        raise SystemExit("ner_predict.py requires scikit-learn") from exc

    with train_path.open(encoding="utf-8") as stream:
        documents = json.load(stream)

    vectorizer = FeatureHasher(
        n_features=HASH_SIZE, input_type="string", alternate_sign=False, dtype="float32"
    )
    labels = []

    def feature_stream():
        for features, label in training_rows(documents):
            labels.append(label)
            yield features

    matrix = vectorizer.transform(feature_stream())
    weights = [0.35 if label == "O" else 1.0 for label in labels]
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2e-6,
        max_iter=15,
        tol=1e-3,
        average=True,
        random_state=571,
        n_jobs=-1,
    )
    classifier.fit(matrix, labels, sample_weight=weights)
    return vectorizer, classifier


def load_or_train(train_path: Path, cache_path: Path):
    signature = (FEATURE_VERSION, train_path.stat().st_size, train_path.stat().st_mtime_ns)
    try:
        with cache_path.open("rb") as stream:
            cached = pickle.load(stream)
        if cached.get("signature") == signature:
            return cached["vectorizer"], cached["classifier"]
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        pass

    vectorizer, classifier = train_model(train_path)
    try:
        with cache_path.open("wb") as stream:
            pickle.dump(
                {"signature": signature, "vectorizer": vectorizer, "classifier": classifier},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError:
        pass
    return vectorizer, classifier


def repair_bio(tags: list[str]) -> list[str]:
    repaired = []
    previous = "O"
    for tag in tags:
        if tag.startswith("I-") and previous not in (f"B-{tag[2:]}", tag):
            tag = f"B-{tag[2:]}"
        repaired.append(tag)
        previous = tag
    return repaired


def predict_tags(documents: list[dict], vectorizer, classifier) -> list[list[list[str]]]:
    """Decode all sentences in position-wise batches while using predicted history."""
    sentences = []
    locations = []
    predictions = [[[] for _ in doc.get("sents", [])] for doc in documents]
    for doc_id, doc in enumerate(documents):
        for sent_id, tokens in enumerate(doc.get("sents", [])):
            sentences.append(tokens)
            locations.append((doc_id, sent_id))

    previous = ["<START>"] * len(sentences)
    max_length = max((len(sentence) for sentence in sentences), default=0)
    for index in range(max_length):
        active = [sent_id for sent_id, sentence in enumerate(sentences) if index < len(sentence)]
        rows = [token_features(sentences[sent_id], index, previous[sent_id]) for sent_id in active]
        if not rows:
            continue
        labels = classifier.predict(vectorizer.transform(rows))
        for sentence_id, label in zip(active, labels):
            label = str(label)
            doc_id, sent_id = locations[sentence_id]
            predictions[doc_id][sent_id].append(label)
            previous[sentence_id] = label

    for document in predictions:
        for sent_id, tags in enumerate(document):
            document[sent_id] = repair_bio(tags)
    return predictions


def mentions_from_tags(tokens: list[str], tags: list[str], sent_id: int) -> list[dict]:
    mentions = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if not tag.startswith("B-"):
            index += 1
            continue
        entity_type = tag[2:]
        end = index + 1
        while end < len(tags) and tags[end] == f"I-{entity_type}":
            end += 1
        mentions.append(
            {
                "name": " ".join(tokens[index:end]),
                "type": entity_type,
                "sent_id": sent_id,
                "pos": [index, end],
            }
        )
        index = end
    return mentions


def add_entities(documents: list[dict], predictions: list[list[list[str]]]) -> list[dict]:
    output = []
    for doc, sentence_tags_ in zip(documents, predictions):
        grouped = {}
        for sent_id, (tokens, tags) in enumerate(zip(doc.get("sents", []), sentence_tags_)):
            for mention in mentions_from_tags(tokens, tags, sent_id):
                key = (mention["type"], re.sub(r"\s+", " ", mention["name"].casefold()).strip())
                grouped.setdefault(key, []).append(mention)
        predicted = dict(doc)
        predicted.pop("labels", None)
        predicted["vertexSet"] = list(grouped.values())
        output.append(predicted)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the Re-DocRED NER tagger")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_path = script_dir / "data" / "train_revised.json"
    cache_path = script_dir / "data" / ".ner_model.pkl"
    if not train_path.is_file():
        parser.error(f"training data not found: {train_path}")

    try:
        with args.input.open(encoding="utf-8") as stream:
            documents = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input JSON: {exc}")
    if not isinstance(documents, list):
        parser.error("input JSON must be a list of documents")

    vectorizer, classifier = load_or_train(train_path, cache_path)
    predictions = predict_tags(documents, vectorizer, classifier)
    output = add_entities(documents, predictions)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False)
    except OSError as exc:
        parser.error(f"cannot write output JSON: {exc}")


if __name__ == "__main__":
    main()
