#!/usr/bin/env python3
"""Contextual BIO tagger for Re-DocRED NER inference.

Usage: python ner_predict.py INPUT.json OUTPUT.json [HYPERPARAMETERS]

The default DistilBERT configuration was selected only on the development
partition. A trained model is cached beside the training data and reused when
the training file and every model-setting hyperparameter match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_TYPES = ("PER", "ORG", "LOC", "TIME", "NUM", "MISC")
LABELS = ("O",) + tuple(
    f"{prefix}-{entity_type}"
    for entity_type in VALID_TYPES
    for prefix in ("B", "I")
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MODEL_VERSION = 3


@dataclass(frozen=True)
class ModelConfig:
    """Settings that affect training, tokenization, or predictions."""

    pretrained_model: str = "distilbert-base-cased"
    epochs: int = 2
    distant_epochs: int = 1
    distant_learning_rate: float = 3e-5
    batch_size: int = 32
    learning_rate: float = 5e-5
    outside_weight: float = 0.60
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_length: int = 512
    random_state: int = 571


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
    for doc, document_tags in zip(documents, predictions):
        grouped = {}
        for sent_id, (tokens, tags) in enumerate(zip(doc.get("sents", []), document_tags)):
            for mention in mentions_from_tags(tokens, tags, sent_id):
                key = (
                    mention["type"],
                    re.sub(r"\s+", " ", mention["name"].casefold()).strip(),
                )
                grouped.setdefault(key, []).append(mention)
        predicted = dict(doc)
        predicted.pop("labels", None)
        predicted["vertexSet"] = list(grouped.values())
        output.append(predicted)
    return output


def flatten_sentences(documents: list[dict], include_labels: bool = False):
    """Flatten document sentences while retaining output locations."""
    tokens, labels, locations = [], [], []
    for doc_id, doc in enumerate(documents):
        doc_labels = sentence_tags(doc) if include_labels else None
        for sent_id, sentence in enumerate(doc.get("sents", [])):
            tokens.append(sentence)
            locations.append((doc_id, sent_id))
            if include_labels:
                labels.append(doc_labels[sent_id])
    return tokens, labels, locations


def encode_sentences(tokenizer, sentences, max_length: int, tags=None):
    """Subword-tokenize sentences and align labels to first word pieces."""
    encoded = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded_labels = None
    if tags is not None:
        encoded_labels = []
        for row_index, sentence_labels in enumerate(tags):
            previous_word = None
            row = []
            for word_id in encoded.word_ids(batch_index=row_index):
                if word_id is None or word_id == previous_word:
                    row.append(-100)
                else:
                    row.append(LABEL_TO_ID[sentence_labels[word_id]])
                previous_word = word_id
            encoded_labels.append(row)
    return encoded, encoded_labels


class EncodedSentences:
    """Small torch-compatible dataset without importing torch at module load."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


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
    """Describe everything that must match before a cache is reusable."""
    signature = {
        "model_version": MODEL_VERSION,
        "train": file_signature(train_path),
        "config": asdict(config),
    }
    if distant_path is not None:
        signature["distant"] = file_signature(distant_path)
    return signature


def default_cache_path(
    train_path: Path, distant_path: Path | None, config: ModelConfig
) -> Path:
    """Use a separate cache directory for each training configuration."""
    cache_key = {
        "config": asdict(config),
        "distant": str(distant_path.resolve()) if distant_path is not None else None,
    }
    settings = json.dumps(cache_key, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(settings).hexdigest()[:12]
    return train_path.parent / f".ner_transformer_{key}"


def cache_matches(cache_path: Path, signature: dict) -> bool:
    try:
        with (cache_path / "ner_signature.json").open(encoding="utf-8") as stream:
            return json.load(stream) == signature
    except (OSError, json.JSONDecodeError):
        return False


def train_or_load(
    train_path,
    distant_path,
    cache_path,
    config,
    device,
    force_retrain=False,
):
    """Load a compatible transformer cache, or fine-tune and save one."""
    try:
        import torch
        from torch.nn import CrossEntropyLoss
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(
            "ner_predict.py requires torch and transformers "
            "(install with: python -m pip install torch transformers)"
        ) from exc

    signature = model_signature(train_path, distant_path, config)
    if not force_retrain and cache_matches(cache_path, signature):
        tokenizer = AutoTokenizer.from_pretrained(cache_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(cache_path).to(device)
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        config.pretrained_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    ).to(device)
    weights = torch.ones(len(LABELS), device=device)
    weights[LABEL_TO_ID["O"]] = config.outside_weight
    loss_function = CrossEntropyLoss(weight=weights, ignore_index=-100)

    def train_stage(path: Path, epochs: int, learning_rate: float, name: str) -> None:
        with path.open(encoding="utf-8") as stream:
            documents = json.load(stream)
        train_tokens, train_tags, _ = flatten_sentences(documents, include_labels=True)
        encodings, labels = encode_sentences(
            tokenizer, train_tokens, config.max_length, train_tags
        )
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
        ordered = sorted(
            range(len(encodings["input_ids"])),
            key=lambda index: len(encodings["input_ids"][index]),
        )
        batches = [
            ordered[index : index + config.batch_size]
            for index in range(0, len(ordered), config.batch_size)
        ]
        random.Random(config.random_state).shuffle(batches)
        loader = DataLoader(
            EncodedSentences(encodings, labels),
            batch_sampler=batches,
            collate_fn=collator,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
        )
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for step, batch in enumerate(loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                gold = batch.pop("labels")
                logits = model(**batch).logits
                loss = loss_function(logits.reshape(-1, len(LABELS)), gold.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += loss.item()
                if step % 100 == 0:
                    print(
                        f"{name} epoch {epoch + 1}/{epochs}, "
                        f"step {step}/{len(loader)}, "
                        f"mean loss {running_loss / step:.4f}",
                        flush=True,
                    )
        del documents, train_tokens, train_tags, encodings, labels, loader

    if distant_path is not None:
        train_stage(
            distant_path,
            config.distant_epochs,
            config.distant_learning_rate,
            "distant",
        )
    train_stage(train_path, config.epochs, config.learning_rate, "human")

    cache_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cache_path)
    tokenizer.save_pretrained(cache_path)
    with (cache_path / "ner_signature.json").open("w", encoding="utf-8") as stream:
        json.dump(signature, stream, indent=2, sort_keys=True)
    return tokenizer, model


def predict_tags(documents, tokenizer, model, config, device):
    """Predict word-level BIO tags from contextual subword representations."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    sentences, _, locations = flatten_sentences(documents)
    encodings, _ = encode_sentences(tokenizer, sentences, config.max_length)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        EncodedSentences(encodings),
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )
    predictions = [[[] for _ in doc.get("sents", [])] for doc in documents]
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_predictions = model(**batch).logits.argmax(dim=-1).cpu().tolist()
            for local_index, token_predictions in enumerate(batch_predictions):
                row_index = offset + local_index
                word_ids = encodings.word_ids(batch_index=row_index)
                tags = ["O"] * len(sentences[row_index])
                previous_word = None
                for token_index, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != previous_word:
                        tags[word_id] = LABELS[token_predictions[token_index]]
                    previous_word = word_id
                doc_id, sent_id = locations[row_index]
                predictions[doc_id][sent_id] = repair_bio(tags)
            offset += len(batch_predictions)
    return predictions


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
    parser.add_argument("--train", type=Path)
    parser.add_argument("--distant-train", type=Path)
    parser.add_argument("--pretrained-model", default="distilbert-base-cased")
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--distant-epochs", type=positive_int, default=1)
    parser.add_argument("--distant-learning-rate", type=positive_float, default=3e-5)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--learning-rate", type=positive_float, default=5e-5)
    parser.add_argument("--outside-weight", type=unit_float, default=0.60)
    parser.add_argument("--weight-decay", type=unit_float, default=0.01)
    parser.add_argument("--warmup-ratio", type=unit_float, default=0.10)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--random-state", type=int, default=571)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("ner_predict.py requires PyTorch") from exc

    script_dir = Path(__file__).resolve().parent
    train_path = args.train or script_dir / "data" / "train_revised.json"
    if not train_path.is_file():
        parser.error(f"training data not found: {train_path}")
    if args.distant_train is not None and not args.distant_train.is_file():
        parser.error(f"distant training data not found: {args.distant_train}")
    config = ModelConfig(
        pretrained_model=args.pretrained_model,
        epochs=args.epochs,
        distant_epochs=args.distant_epochs,
        distant_learning_rate=args.distant_learning_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outside_weight=args.outside_weight,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        random_state=args.random_state,
    )
    random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if args.device == "auto":
        selected = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        selected = args.device
    device = torch.device(selected)

    try:
        with args.input.open(encoding="utf-8") as stream:
            documents = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input JSON: {exc}")
    if not isinstance(documents, list):
        parser.error("input JSON must be a list of documents")

    cache_path = args.cache or default_cache_path(train_path, args.distant_train, config)
    tokenizer, model = train_or_load(
        train_path,
        args.distant_train,
        cache_path,
        config,
        device,
        force_retrain=args.force_retrain,
    )
    predictions = predict_tags(documents, tokenizer, model, config, device)
    output = add_entities(documents, predictions)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False)
    except OSError as exc:
        parser.error(f"cannot write output JSON: {exc}")


if __name__ == "__main__":
    main()
