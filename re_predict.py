#!/usr/bin/env python3
"""Fast relation extraction for Re-DocRED.

Usage: python re_predict.py NER_OUTPUT.json RE_OUTPUT.json

The model is an SGD multinomial classifier over ordered entity pairs. It uses
entity types, normalized names, mention counts, mention/sentence distance,
document-title matches, and tokens surrounding or between the closest pair of
mentions. Unrelated training pairs are deterministically subsampled to limit
class imbalance and training time. An optional weakly labeled DocRED corpus can
be added with a tunable source weight. Trained models are cached beside the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


FEATURE_VERSION = 1
NO_RELATION = "no_relation"


@dataclass(frozen=True)
class ModelConfig:
    # Selected using full-pipeline micro-F1 on the development split only.
    hash_size: int = 1 << 19
    negative_ratio: float = 5.0
    alpha: float = 2e-6
    max_iter: int = 25
    tol: float = 1e-3
    class_balance: float = 0.2
    random_state: int = 571
    batch_size: int = 8192
    distant_weight: float = 0.25
    max_relations: int = 2
    relation_no_ratio: float = 0.90


def normalized_name(mention: dict) -> str:
    return re.sub(r"\s+", " ", str(mention.get("name", "")).casefold()).strip()


def entity_type(entity: list[dict]) -> str:
    return str(entity[0].get("type", "UNK")) if entity else "UNK"


def mention_position(mention: dict) -> tuple[int, int, int]:
    try:
        start, end = mention["pos"]
        return int(mention["sent_id"]), int(start), int(end)
    except (KeyError, TypeError, ValueError):
        return 10**9, 10**9, 10**9


def bucket(value: int, limits: tuple[int, ...]) -> str:
    for limit in limits:
        if value <= limit:
            return str(limit)
    return f">{limits[-1]}"


def pair_features(doc: dict, head_index: int, tail_index: int) -> list[str]:
    """Construct sparse categorical features for one ordered entity pair."""
    entities = doc.get("vertexSet", [])
    head, tail = entities[head_index], entities[tail_index]
    head_type, tail_type = entity_type(head), entity_type(tail)
    head_names = {normalized_name(m) for m in head if normalized_name(m)}
    tail_names = {normalized_name(m) for m in tail if normalized_name(m)}
    head_name = min(head_names, key=lambda x: (len(x), x), default="")
    tail_name = min(tail_names, key=lambda x: (len(x), x), default="")
    title = str(doc.get("title", "")).casefold()

    features = [
        "bias",
        f"ht={head_type}",
        f"tt={tail_type}",
        f"types={head_type}>{tail_type}",
        f"hn={head_name}",
        f"tn={tail_name}",
        f"names={head_name}>{tail_name}",
        f"hc={bucket(len(head), (1, 2, 4, 8))}",
        f"tc={bucket(len(tail), (1, 2, 4, 8))}",
        f"h_title={bool(head_name and head_name in title)}",
        f"t_title={bool(tail_name and tail_name in title)}",
    ]
    features.extend(f"hw={token}" for token in re.findall(r"[\w]+", head_name))
    features.extend(f"tw={token}" for token in re.findall(r"[\w]+", tail_name))

    best = None
    for hm in head:
        hs, hstart, hend = mention_position(hm)
        for tm in tail:
            ts, tstart, tend = mention_position(tm)
            sentence_distance = abs(hs - ts)
            token_distance = (
                max(0, max(hstart, tstart) - min(hend, tend))
                if sentence_distance == 0
                else 10**9
            )
            candidate = (sentence_distance, token_distance, hs, hstart, ts, tstart, hend, tend)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return features
    sentence_distance, token_distance, hs, hstart, ts, tstart, hend, tend = best
    features.extend(
        (
            f"sd={bucket(sentence_distance, (0, 1, 2, 4, 8))}",
            f"same_sent={sentence_distance == 0}",
            f"head_first={(hs, hstart) < (ts, tstart)}",
        )
    )
    sentences = doc.get("sents", [])
    if sentence_distance == 0 and 0 <= hs < len(sentences):
        tokens = sentences[hs]
        left_end, right_start = (hend, tstart) if hstart <= tstart else (tend, hstart)
        between = tokens[max(0, left_end) : max(0, right_start)]
        features.append(f"td={bucket(token_distance, (0, 1, 2, 4, 8, 16))}")
        features.extend(f"between={str(token).casefold()}" for token in between[:12])
        if 0 < hstart <= len(tokens):
            features.append(f"hleft={str(tokens[hstart - 1]).casefold()}")
        if 0 <= hend < len(tokens):
            features.append(f"hright={str(tokens[hend]).casefold()}")
        if 0 < tstart <= len(tokens):
            features.append(f"tleft={str(tokens[tstart - 1]).casefold()}")
        if 0 <= tend < len(tokens):
            features.append(f"tright={str(tokens[tend]).casefold()}")
    return features


def positive_labels(doc: dict) -> dict[tuple[int, int], list[str]]:
    labels: dict[tuple[int, int], list[str]] = {}
    entity_count = len(doc.get("vertexSet", []))
    for label in doc.get("labels", []):
        try:
            head, tail = int(label["h"]), int(label["t"])
            relation = str(label["r"])
        except (KeyError, TypeError, ValueError):
            continue
        if head != tail and 0 <= head < entity_count and 0 <= tail < entity_count:
            labels.setdefault((head, tail), []).append(relation)
    return labels


def training_rows(
    documents: Iterable[dict], config: ModelConfig
) -> Iterator[tuple[list[str], str]]:
    rng = random.Random(config.random_state)
    for doc in documents:
        entities = doc.get("vertexSet", [])
        positives = positive_labels(doc)
        possible_negatives = len(entities) * max(0, len(entities) - 1) - len(positives)
        wanted = config.negative_ratio * max(1, sum(map(len, positives.values())))
        keep_probability = min(1.0, wanted / max(1, possible_negatives))
        for head in range(len(entities)):
            for tail in range(len(entities)):
                if head == tail:
                    continue
                relations = positives.get((head, tail))
                if relations:
                    features = pair_features(doc, head, tail)
                    for relation in relations:
                        yield features, relation
                elif rng.random() < keep_probability:
                    yield pair_features(doc, head, tail), NO_RELATION


def train_model(
    train_path: Path, distant_path: Path | None, config: ModelConfig
):
    try:
        import numpy as np
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:
        raise SystemExit("re_predict.py requires scikit-learn and numpy") from exc

    with train_path.open(encoding="utf-8") as stream:
        documents = json.load(stream)
    distant_documents: list[dict] = []
    if distant_path is not None:
        with distant_path.open(encoding="utf-8") as stream:
            distant_documents = json.load(stream)
    vectorizer = FeatureHasher(
        n_features=config.hash_size,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )
    labels: list[str] = []
    source_weights: list[float] = []

    def feature_stream():
        for features, label in training_rows(documents, config):
            labels.append(label)
            source_weights.append(1.0)
            yield features
        for features, label in training_rows(distant_documents, config):
            labels.append(label)
            source_weights.append(config.distant_weight)
            yield features

    matrix = vectorizer.transform(feature_stream())
    counts = Counter(labels)
    largest = max(counts.values(), default=1)
    weights = np.asarray(
        [
            source_weight
            * min(8.0, (largest / counts[label]) ** config.class_balance)
            for label, source_weight in zip(labels, source_weights)
        ],
        dtype=np.float64,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        average=True,
        random_state=config.random_state,
        n_jobs=-1,
    )
    classifier.fit(matrix, labels, sample_weight=weights)
    return vectorizer, classifier


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def model_signature(
    train_path: Path, distant_path: Path | None, config: ModelConfig
) -> dict:
    training_config = asdict(config)
    training_config.pop("max_relations")
    training_config.pop("relation_no_ratio")
    signature = {
        "feature_version": FEATURE_VERSION,
        "train": file_signature(train_path),
        "config": training_config,
    }
    if distant_path is not None:
        signature["distant"] = file_signature(distant_path)
    return signature


def default_cache_path(
    train_path: Path, distant_path: Path | None, config: ModelConfig
) -> Path:
    cache_key = {
        "config": asdict(config),
        "distant": str(distant_path.resolve()) if distant_path is not None else None,
    }
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return train_path.parent / f".re_model_{digest}.pkl"


def load_or_train(
    train_path: Path,
    distant_path: Path | None,
    cache_path: Path,
    config: ModelConfig,
    force_retrain: bool,
):
    signature = model_signature(train_path, distant_path, config)
    if not force_retrain:
        try:
            with cache_path.open("rb") as stream:
                cached = pickle.load(stream)
            if cached.get("signature") == signature:
                return cached["vectorizer"], cached["classifier"]
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            pass
    vectorizer, classifier = train_model(train_path, distant_path, config)
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


def candidate_pairs(documents: list[dict]):
    for doc_index, doc in enumerate(documents):
        count = len(doc.get("vertexSet", []))
        for head in range(count):
            for tail in range(count):
                if head != tail:
                    yield doc_index, head, tail, pair_features(doc, head, tail)


def add_relations(
    documents: list[dict], vectorizer, classifier, config: ModelConfig
) -> list[dict]:
    output = []
    for doc in documents:
        predicted = dict(doc)
        predicted.pop("labels", None)
        predicted["labels"] = []
        output.append(predicted)

    batch: list[tuple[int, int, int, list[str]]] = []

    def predict_batch(rows: list[tuple[int, int, int, list[str]]]) -> None:
        if not rows:
            return
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit("re_predict.py requires numpy") from exc
        probabilities = classifier.predict_proba(
            vectorizer.transform(row[3] for row in rows)
        )
        classes = np.asarray(classifier.classes_)
        no_relation_index = int(np.flatnonzero(classes == NO_RELATION)[0])
        for (doc_index, head, tail, _), row_probabilities in zip(rows, probabilities):
            no_relation_probability = row_probabilities[no_relation_index]
            ranked = np.argsort(row_probabilities)[::-1]
            emitted = 0
            for class_index in ranked:
                relation = str(classes[class_index])
                if relation == NO_RELATION:
                    continue
                if (
                    row_probabilities[class_index]
                    < config.relation_no_ratio * no_relation_probability
                ):
                    break
                output[doc_index]["labels"].append(
                    {"h": head, "t": tail, "r": relation}
                )
                emitted += 1
                if emitted >= config.max_relations:
                    break

    for candidate in candidate_pairs(documents):
        batch.append(candidate)
        if len(batch) >= config.batch_size:
            predict_batch(batch)
            batch.clear()
    predict_batch(batch)
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
    parser = argparse.ArgumentParser(description="Train and run Re-DocRED relation extraction")
    parser.add_argument("input", type=Path, help="NER prediction JSON")
    parser.add_argument("output", type=Path, help="relation prediction JSON")
    parser.add_argument("--train", type=Path, help="default: data/train_revised.json")
    parser.add_argument(
        "--distant-train",
        type=Path,
        help="optional weak labels from prepare_distant_supervision.py",
    )
    parser.add_argument("--cache", type=Path, help="trained model cache path")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--hash-size", type=positive_int, default=1 << 19)
    parser.add_argument("--negative-ratio", type=positive_float, default=5.0)
    parser.add_argument("--alpha", type=positive_float, default=2e-6)
    parser.add_argument("--max-iter", type=positive_int, default=25)
    parser.add_argument("--tol", type=positive_float, default=1e-3)
    parser.add_argument("--class-balance", type=unit_float, default=0.2)
    parser.add_argument("--random-state", type=int, default=571)
    parser.add_argument("--batch-size", type=positive_int, default=8192)
    parser.add_argument(
        "--distant-weight",
        type=positive_float,
        default=0.25,
        help="sample-weight multiplier for weak examples (default: 0.25)",
    )
    parser.add_argument(
        "--max-relations",
        type=positive_int,
        default=2,
        help="maximum predicted relations for an ordered entity pair (default: 2)",
    )
    parser.add_argument(
        "--relation-no-ratio",
        type=positive_float,
        default=0.90,
        help="emit relation when P(relation) >= ratio * P(no_relation)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_path = args.train or script_dir / "data" / "train_revised.json"
    if not train_path.is_file():
        parser.error(f"training data not found: {train_path}")
    if args.distant_train is not None and not args.distant_train.is_file():
        parser.error(f"distant training data not found: {args.distant_train}")
    try:
        with args.input.open(encoding="utf-8") as stream:
            documents = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input JSON: {exc}")
    if not isinstance(documents, list):
        parser.error("input JSON must be a list of documents")

    config = ModelConfig(
        hash_size=args.hash_size,
        negative_ratio=args.negative_ratio,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        class_balance=args.class_balance,
        random_state=args.random_state,
        batch_size=args.batch_size,
        distant_weight=args.distant_weight,
        max_relations=args.max_relations,
        relation_no_ratio=args.relation_no_ratio,
    )
    cache_path = args.cache or default_cache_path(train_path, args.distant_train, config)
    vectorizer, classifier = load_or_train(
        train_path, args.distant_train, cache_path, config, args.force_retrain
    )
    output = add_relations(documents, vectorizer, classifier, config)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False)
    except OSError as exc:
        parser.error(f"cannot write output JSON: {exc}")


if __name__ == "__main__":
    main()
