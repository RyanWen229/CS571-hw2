#!/usr/bin/env python3
"""Fast, self-contained BIO tagger for Re-DocRED NER inference.

Usage: python ner_predict.py INPUT.json OUTPUT.json [HYPERPARAMETERS]

The model is trained from data/train_revised.json by default. A cached model is
reused only when both the training file and all hyperparameters match. Mentions
with the same type and case-folded token string are grouped as one entity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# Bump this whenever feature construction changes so stale cached models are ignored.
FEATURE_VERSION = 2
VALID_TYPES = {"PER", "ORG", "LOC", "TIME", "NUM", "MISC"}


@dataclass(frozen=True)
class ModelConfig:
    """Settings that affect features, optimization, or predictions."""

    # Defaults were selected using strict span-level micro-F1 on dev only.
    # Feature hashing keeps the vocabulary bounded without a token-to-column map.
    hash_size: int = 1 << 19
    context_window: int = 3
    affix_length: int = 5
    outside_weight: float = 0.60
    alpha: float = 1e-6
    max_iter: int = 20
    tol: float = 1e-3
    loss: str = "log_loss"
    penalty: str = "l2"
    l1_ratio: float = 0.15
    average: bool = True
    random_state: int = 571


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
        # Collapse repeated classes (for example, "USA2024" becomes "Xd").
        if not mapped or mapped[-1] != code:
            mapped.append(code)
    return "".join(mapped[:12])


def token_features(
    tokens: list[str], index: int, previous_tag: str, config: ModelConfig
) -> list[str]:
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
    # Short affixes help recognize names and common entity-specific endings.
    for size in range(1, min(config.affix_length, len(lower)) + 1):
        features.append(f"pre{size}={lower[:size]}")
        features.append(f"suf{size}={lower[-size:]}")

    # Include a configurable token window with explicit sentence-boundary markers.
    offsets = range(-config.context_window, config.context_window + 1)
    for offset in (offset for offset in offsets if offset):
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

    # Negative lengths make longer mentions win when annotated spans overlap.
    for _, sent_id, start, end, entity_type in sorted(mentions):
        if not (0 <= sent_id < len(tags) and 0 <= start < end <= len(tags[sent_id])):
            continue
        if any(tag != "O" for tag in tags[sent_id][start:end]):
            continue
        tags[sent_id][start] = f"B-{entity_type}"
        tags[sent_id][start + 1 : end] = [f"I-{entity_type}"] * (end - start - 1)
    return tags


def training_rows(documents: Iterable[dict], config: ModelConfig):
    """Yield token features and gold tags in document order."""
    for doc in documents:
        labels = sentence_tags(doc)
        for tokens, tags in zip(doc.get("sents", []), labels):
            # The previous gold tag gives the classifier local sequence context.
            previous = "<START>"
            for index, tag in enumerate(tags):
                yield token_features(tokens, index, previous, config), tag
                previous = tag


def train_model(train_path: Path, config: ModelConfig):
    """Fit the hashed-feature token classifier from the annotated training set."""
    try:
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:
        raise SystemExit("ner_predict.py requires scikit-learn") from exc

    with train_path.open(encoding="utf-8") as stream:
        documents = json.load(stream)

    vectorizer = FeatureHasher(
        n_features=config.hash_size,
        input_type="string",
        alternate_sign=False,
        dtype="float32",
    )
    labels = []

    def feature_stream():
        for features, label in training_rows(documents, config):
            labels.append(label)
            yield features

    matrix = vectorizer.transform(feature_stream())
    # Most tokens are not entities, so reduce their influence during training.
    weights = [config.outside_weight if label == "O" else 1.0 for label in labels]
    classifier = SGDClassifier(
        loss=config.loss,
        penalty=config.penalty,
        alpha=config.alpha,
        l1_ratio=config.l1_ratio,
        max_iter=config.max_iter,
        tol=config.tol,
        average=config.average,
        random_state=config.random_state,
        n_jobs=-1,
    )
    classifier.fit(matrix, labels, sample_weight=weights)
    return vectorizer, classifier


def model_signature(train_path: Path, config: ModelConfig) -> dict:
    """Describe everything that must match before a cached model is reusable."""
    stat = train_path.stat()
    return {
        "feature_version": FEATURE_VERSION,
        "train_size": stat.st_size,
        "train_mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
    }


def default_cache_path(train_path: Path, config: ModelConfig) -> Path:
    """Use a separate cache file for each hyperparameter configuration."""
    settings = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    key = hashlib.sha256(settings).hexdigest()[:12]
    return train_path.parent / f".ner_model_{key}.pkl"


def load_or_train(
    train_path: Path, cache_path: Path, config: ModelConfig, force_retrain: bool = False
):
    """Return a compatible cached model, or train and cache a new one."""
    # File metadata makes this check inexpensive even for a large training file.
    signature = model_signature(train_path, config)
    if not force_retrain:
        try:
            with cache_path.open("rb") as stream:
                cached = pickle.load(stream)
            if cached.get("signature") == signature:
                return cached["vectorizer"], cached["classifier"]
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            # A missing or damaged cache is non-fatal because it can be rebuilt.
            pass

    vectorizer, classifier = train_model(train_path, config)
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
    """Convert orphaned I-tags into valid beginnings of entity spans."""
    repaired = []
    previous = "O"
    for tag in tags:
        if tag.startswith("I-") and previous not in (f"B-{tag[2:]}", tag):
            tag = f"B-{tag[2:]}"
        repaired.append(tag)
        previous = tag
    return repaired


def predict_tags(
    documents: list[dict], vectorizer, classifier, config: ModelConfig
) -> list[list[list[str]]]:
    """Decode all sentences in position-wise batches while using predicted history."""
    sentences = []
    locations = []
    predictions = [[[] for _ in doc.get("sents", [])] for doc in documents]
    for doc_id, doc in enumerate(documents):
        for sent_id, tokens in enumerate(doc.get("sents", [])):
            sentences.append(tokens)
            locations.append((doc_id, sent_id))

    # Decode the same token position across sentences as one batch. This preserves
    # each sentence's predicted-tag history while avoiding one model call per token.
    previous = ["<START>"] * len(sentences)
    max_length = max((len(sentence) for sentence in sentences), default=0)
    for index in range(max_length):
        active = [sent_id for sent_id, sentence in enumerate(sentences) if index < len(sentence)]
        rows = [
            token_features(sentences[sent_id], index, previous[sent_id], config)
            for sent_id in active
        ]
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
    """Convert a valid BIO sequence into Re-DocRED mention dictionaries."""
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
    """Attach predicted mentions to copies of the input documents."""
    output = []
    for doc, sentence_tags_ in zip(documents, predictions):
        grouped = {}
        for sent_id, (tokens, tags) in enumerate(zip(doc.get("sents", []), sentence_tags_)):
            for mention in mentions_from_tags(tokens, tags, sent_id):
                # Repeated surface forms of the same type are treated as one entity.
                key = (mention["type"], re.sub(r"\s+", " ", mention["name"].casefold()).strip())
                grouped.setdefault(key, []).append(mention)
        # Preserve source metadata and sentences, but never carry gold relations
        # into an inference result.
        predicted = dict(doc)
        predicted.pop("labels", None)
        predicted["vertexSet"] = list(grouped.values())
        output.append(predicted)
    return output


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the Re-DocRED NER tagger")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--train",
        type=Path,
        help="annotated training JSON (default: data/train_revised.json)",
    )
    parser.add_argument("--alpha", type=positive_float, default=1e-6,
                        help="SGD regularization strength (default: %(default)g)")
    parser.add_argument("--max-iter", type=positive_int, default=20,
                        help="maximum training epochs (default: %(default)s)")
    parser.add_argument("--tol", type=positive_float, default=1e-3,
                        help="early stopping tolerance (default: %(default)g)")
    parser.add_argument("--outside-weight", type=unit_float, default=0.60,
                        help="training weight for non-entity O tokens (default: %(default)s)")
    parser.add_argument("--context-window", type=positive_int, default=3,
                        help="neighboring tokens on each side (default: %(default)s)")
    parser.add_argument("--affix-length", type=positive_int, default=5,
                        help="longest prefix and suffix feature (default: %(default)s)")
    parser.add_argument("--hash-size", type=positive_int, default=1 << 19,
                        help="number of hashed feature columns (default: %(default)s)")
    parser.add_argument("--loss", choices=("log_loss", "modified_huber", "hinge"),
                        default="log_loss", help="SGD loss function (default: %(default)s)")
    parser.add_argument("--penalty", choices=("l2", "l1", "elasticnet"), default="l2",
                        help="regularization penalty (default: %(default)s)")
    parser.add_argument("--l1-ratio", type=unit_float, default=0.15,
                        help="L1 share for elasticnet penalty (default: %(default)s)")
    parser.add_argument("--average", action=argparse.BooleanOptionalAction, default=True,
                        help="use averaged SGD weights (default: enabled)")
    parser.add_argument("--random-state", type=int, default=571,
                        help="reproducible training seed (default: %(default)s)")
    parser.add_argument("--cache", type=Path,
                        help="model cache path (default: parameter-specific file in data/)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="ignore a compatible cached model")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_path = args.train or script_dir / "data" / "train_revised.json"
    if not train_path.is_file():
        parser.error(f"training data not found: {train_path}")

    config = ModelConfig(
        hash_size=args.hash_size,
        context_window=args.context_window,
        affix_length=args.affix_length,
        outside_weight=args.outside_weight,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        loss=args.loss,
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        average=args.average,
        random_state=args.random_state,
    )
    cache_path = args.cache or default_cache_path(train_path, config)

    try:
        with args.input.open(encoding="utf-8") as stream:
            documents = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input JSON: {exc}")
    if not isinstance(documents, list):
        parser.error("input JSON must be a list of documents")

    vectorizer, classifier = load_or_train(
        train_path, cache_path, config, force_retrain=args.force_retrain
    )
    predictions = predict_tags(documents, vectorizer, classifier, config)
    output = add_entities(documents, predictions)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False)
    except OSError as exc:
        parser.error(f"cannot write output JSON: {exc}")


if __name__ == "__main__":
    main()
