# CS 571 Homework 2: Entity and Relation Extraction

Re-DocRED is a re-annotated version of DocRED, a dataset for document-level entity and relation extraction on English Wikipedia articles. Each document comes with its entities (grouped mentions, with one of six types) and relation instances between entities, drawn from 96 Wikidata relations. The Re-DocRED authors added relation instances that DocRED annotators had missed, removed logically inconsistent labels, and corrected coreference errors. Unlike the original DocRED test set, the Re-DocRED test set includes its labels.

This repository contains an end-to-end named-entity recognition (NER) and relation-extraction pipeline, exact-match evaluators, error analysis, and the final report. Model and hyperparameter selection use the development split; the test split is reserved for final evaluation.

## One-command interface

Python 3.10 or later is required. Install only the allowed model libraries with `python -m pip install -r requirements.txt`. The `Makefile` provides one command for each required operation:

| Part | Train | Predict one split | Evaluate one split |
|---|---|---|---|
| NER | `make train-ner` | `make predict-ner SPLIT=dev` | `make eval-ner SPLIT=dev` |
| RE | `make train-re` | `make predict-re SPLIT=dev` | `make eval-re SPLIT=dev` |

Replace `dev` with `test` only for the final locked evaluation. Run `make validate SPLIT=dev` or `make validate SPLIT=test` to check both prediction formats. The commands write the four submission files to `predictions/`: `ner_dev.json`, `ner_test.json`, `re_dev.json`, and `re_test.json`. Training caches go under `models/` and intermediate predictions under `artifacts/`.

## Contents

| Path | Content |
|---|---|
| `data/train_revised.json` | Training partition, unchanged from the authors' repository |
| `data/dev_revised.json` | Validation (development) partition, unchanged |
| `data/test_revised.json` | Test partition, unchanged |
| `data/rel_info.json` | Relation ID to relation name map (96 entries), from the original DocRED release |
| `data/unlabeled_10000.json` | 10,000 unlabeled documents (text only) for semi-supervised learning, sampled from DocRED's distantly supervised set |
| `data/distant_10000.json` | The same 10,000 documents with recovered DocRED/Wikidata weak entity and relation labels |
| `prepare_distant_supervision.py` | Exact text matcher that reconstructs `distant_10000.json` from the official DocRED release |
| `ner_predict.py` | From-scratch scikit-learn BIO tagger with hand-written features and beam decoding |
| `ner_eval.py` | Exact span-and-type NER evaluation |
| `ner_error_analysis.py` | NER boundary and type error analysis |
| `re_predict.py` | Hashed-feature relation classifier |
| `re_eval.py` | Gold-entity and pipeline relation evaluation |
| `check_format.py` | Prediction-file format validation |
| `predictions/` | Final dev/test NER and RE submission files |
| `results/` | Saved score reports and error-analysis summaries |
| `Makefile`, `requirements.txt` | One-command training/prediction/evaluation and allowed dependencies |
| `REPORT.tex`, `output/pdf/HW2_Report.pdf` | Report source and compiled report |
| `CITATION.bib` | BibTeX for Re-DocRED and DocRED |
| `LICENSE-Re-DocRED`, `LICENSE-DocRED` | MIT licenses of the two repositories |
| `SHA256SUMS` | Checksums for source data, recovered weak labels, and data licenses |

## Methods

### Named-entity recognition

NER is sentence-level BIO tagging over `PER`, `ORG`, `LOC`, `TIME`, `NUM`, and `MISC`. The model is trained from scratch with scikit-learn; it does not use BERT, embeddings, a pretrained pipeline, or any external NER model.

#### Input and label construction

- Each sentence uses the existing Re-DocRED token list, so predicted positions remain aligned with `[start, end)` offsets.
- If annotated spans overlap, longer spans are assigned first and conflicting shorter spans are skipped. This produces one valid BIO sequence per sentence.

#### Hand-written features

The sparse feature function includes original/lowercase token forms, prefixes and suffixes of lengths 1--5, compressed word shape, length and capitalization/digit/punctuation flags, word and shape features in a three-token window, sentence-boundary markers, lexical bigrams, and the previous BIO tag. Features are mapped to 524,288 nonnegative hashed columns.

#### Training configuration

The model is an averaged multiclass `SGDClassifier` with logistic loss. The dev-selected settings are:

| Setting | Value |
|---|---:|
| Hash columns | `2^19` |
| Loss / penalty | logistic / L2 |
| `alpha` / max epochs | `1e-6` / 20 |
| Context / max affix | 3 / 5 tokens |
| `O`-label loss weight | `0.60` |
| Weak-source weight | `0.50` |
| Random seed | 571 |

Weak training keeps every weak entity token plus `O` tokens within two positions of a weak mention. This uses all 10,000 documents while preventing their roughly two million background tokens from dominating.

#### Decoding and output

Width-4 left-to-right beam decoding uses class log probabilities and rejects illegal `I-X` transitions. A final repair converts any orphaned `I-X` to `B-X`. Mentions with the same type and case-folded surface form are grouped into one Re-DocRED entity.

The first run trains the linear classifier and saves its cache under `models/`. A cache is reused only when training-file metadata and every model-setting hyperparameter match.

#### NER results

Scores use exact mention matching: sentence ID, start, end, and type must all agree with gold.

| Split | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| Development, supervised baseline | 0.7563 | 0.7592 | 0.7578 |
| Development, final distant model | 0.7783 | 0.7854 | **0.7818** |
| Test, supervised baseline | 0.7522 | 0.7506 | 0.7514 |
| Test, final distant model | 0.7766 | 0.7818 | **0.7792** |

The test improvement is +.0278 absolute micro-F1. Final reports are `results/ner_ds_dev_scores.json` and `results/ner_ds_test_scores.json`; error files separate type-only, boundary-only, combined, missed, and spurious cases.

### Relation extraction

Relation extraction operates on the entities in an NER prediction file. Every ordered pair of distinct entities is a candidate. The model is an averaged multiclass linear classifier trained with logistic loss; its classes are the 96 Wikidata relation IDs plus `no_relation`.

#### Pair features

Sparse categorical features are mapped into 524,288 dimensions with nonnegative feature hashing. For each ordered head-tail pair, the representation includes:

- head type, tail type, and their ordered type combination;
- normalized entity names, name tokens, and the ordered name pair;
- mention-count buckets for both entities;
- whether either entity name occurs in the document title;
- closest-mention sentence distance, order, and same-sentence indicator;
- for same-sentence mentions, token distance, up to 12 between-mention tokens, and immediate left/right context for each argument.

All positive ordered pairs are retained. Unrelated pairs are deterministically sampled at a target ratio of five negative examples per positive relation instance. Training examples receive inverse-frequency weights, capped at 8, to reduce domination by frequent relations and `no_relation`.

The selected defaults are L2-regularized averaged SGD, `alpha=2e-6`, at most 25 epochs, tolerance `1e-3`, class-balance exponent `0.2`, batch size 8,192 candidate pairs, and seed 571. The fitted feature hasher and classifier are cached using the training-file metadata and configuration.

#### Distant supervision

The knowledge base is [Wikidata](https://www.wikidata.org/wiki/Wikidata:Introduction), whose statements use item identifiers (`Q...`) and property identifiers (`P...`). This is a natural fit because the 96 DocRED/Re-DocRED relation classes are already Wikidata property IDs. Following the [official DocRED construction method](https://aclanthology.org/P19-1074/), entity mentions are linked and coreferent mentions with the same knowledge-base ID are merged; relations are then labeled from Wikidata properties.

For reproducibility, the experiment restores the released weak annotations instead of making live Wikidata queries. `prepare_distant_supervision.py` matches every record in `unlabeled_10000.json` to official DocRED `train_distant.json` by title and the complete tokenized text. All 10,000 records match exactly, yielding 192,955 weak entity clusters and 147,974 weak relation labels. The output is `data/distant_10000.json` (SHA-256 `fe0eb23cfb502c4a5f1fc764d2c67747a8925d460e05b984767c0b9aaf125045`).

All 10,000 weak documents are added to RE training. Weak rows receive the dev-selected source weight 0.25 while human rows retain weight 1.0. Class-frequency weighting and negative sampling are otherwise unchanged. A calibrated multi-label decoder emits at most two relations for a pair when each relation probability is at least 0.90 times the `no_relation` probability; both the cap and ratio were selected on the final frozen NER dev output.

This two-label decoder fixes the earlier one-relation-per-pair limitation while bounding false positives.

#### Relation results

The relation settings were selected using full-pipeline dev micro-F1 and then locked before test evaluation.

| Split | Training | Entity input | Precision | Recall | Micro-F1 |
|---|---|---|---:|---:|---:|
| Development | Supervised | Predicted | 0.3194 | 0.3569 | 0.3371 |
| Development | + distant, final | Predicted | 0.3255 | 0.3548 | **0.3395** |
| Test | Supervised | Predicted | 0.3219 | 0.3607 | 0.3402 |
| Test | + distant, final | Predicted | 0.3246 | 0.3609 | **0.3418** |
| Test | + distant, final | Gold | 0.4839 | 0.4629 | 0.4732 |

All four final files pass `check_format.py` with zero errors and are accepted by both evaluators. They were also opened in `annotation_viewer.html`; all 500 test documents rendered with populated entity/BIO and relation layers. Final RE reaches 0.3418 full-pipeline test F1.

### Evaluation

`ner_eval.py` micro-averages precision, recall, and F1 over unique mentions. The default score requires an exact span and type match; `--relaxed` additionally reports type-matched overlapping spans.

`re_eval.py` supports two entity modes automatically:

- **Gold entities:** predicted and gold `vertexSet` entries match, so relation arguments are evaluated by entity index.
- **Pipeline entities:** predicted entities are greedily aligned to gold entities through exact shared mention spans. Relations involving an unaligned predicted entity are false positives. The evaluator also reports entity reach and the relation-recall ceiling imposed by NER.

## Reproducing the pipeline

Use Python 3.10 or newer. Install the allowed dependencies:

```bash
python -m pip install -r requirements.txt
```

The six `make` commands in **One-command interface** are the canonical workflow. Run `make train-ner`, `make predict-ner SPLIT=dev`, and `make eval-ner SPLIT=dev`; then run the corresponding three RE commands. Once both stages are fixed, repeat only prediction/evaluation with `SPLIT=test`. If weak annotations must be rebuilt, run `python prepare_distant_supervision.py /path/to/train_distant.json` once before training. For gold-entity RE diagnostics, pass `data/dev_revised.json` directly to `re_predict.py`.

## Citation

Cite both papers when using this data.

Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and Sharifah Mahani Aljunied. 2022. Revisiting DocRED - Addressing the False Negative Problem in Relation Extraction. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 8472–8487, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. https://aclanthology.org/2022.emnlp-main.580/

Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019. DocRED: A Large-Scale Document-Level Relation Extraction Dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 764–777, Florence, Italy. Association for Computational Linguistics. https://aclanthology.org/P19-1074/

Wikimedia Foundation. *Wikidata: Introduction* and *Wikidata: Data model*. https://www.wikidata.org/wiki/Wikidata:Introduction and https://www.wikidata.org/wiki/Help:Data_model (accessed 5 October 2026).

## Source

- Re-DocRED data: https://github.com/tonytan48/Re-DocRED, commit `ccfb54f5ddf5836027c87badda10f6dfc56efaac` (21 August 2023), downloaded 22 September 2026.
- `rel_info.json`: the official DocRED data folder, https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw (linked from https://github.com/thunlp/DocRED).
- `data/distant_10000.json`: exact matches recovered from `train_distant.json` in that official DocRED data folder; Wikidata is the external knowledge base used by DocRED for the weak entity links and relation properties.
- The authors' scoring script is `evaluation.py` in the Re-DocRED repository.

## Partitions

| Partition | Documents | Sentences | Tokens | Entities | Mentions | Relation instances | Relation types | Cross-sentence | No evidence listed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 3,053 | 24,256 | 603,468 | 59,359 | 79,481 | 85,932 | 96 | 53% | 45% |
| validation (dev) | 500 | 4,110 | 101,970 | 9,684 | 13,189 | 17,284 | 95 | 54% | 57% |
| test | 500 | 3,966 | 98,734 | 9,779 | 13,018 | 17,448 | 95 | 53% | 57% |

The document counts match the authors' README (3,053 / 500 / 500), as do the averages of 28.1, 34.6 and 34.9 relation instances per document.

"Cross-sentence" counts relation instances where no mention of the head entity shares a sentence with any mention of the tail entity. "No evidence listed" counts relation instances whose `evidence` list is empty. Most of these are triples added during re-annotation, so the evidence field is incomplete and should not be used as a gold target for evidence extraction.

### Relation to the original DocRED partitions

- `train_revised.json` contains the 3,053 documents of DocRED `train_annotated.json`, with identical sentences.
- `test_revised.json` contains 500 documents from DocRED `dev.json`.
- `dev_revised.json` contains the other 498 documents from DocRED `dev.json`, plus "ABBA Live" from DocRED `test.json` and "Lark Force" from DocRED `train_annotated.json`.
- **"Lark Force" appears in both `train_revised.json` and `dev_revised.json`** with the same text, with 51 relation instances in train and 48 in dev. The files are left as the authors released them. Remove that document from one side if the assignment requires disjoint partitions.

## Unlabeled documents

`data/unlabeled_10000.json` holds 10,000 documents for semi-supervised learning. Each document has only `title` and `sents`, tokenized as in the labeled files. There is no `vertexSet` and no `labels` field.

**Source.** DocRED's distantly supervised training set, `train_distant.json` (101,873 documents; SHA-256 `db6d3cdaab8d36926318bb9339f6fd82d19dbacd186c74d7c20c734355a58b36`), from the official DocRED data folder listed under Source. Its documents come from English Wikipedia like the labeled files, and its entity mentions and relations were produced automatically. Both were removed.

**Exclusions.** Before sampling, 4 documents whose token sequences are identical to a human-annotated DocRED or Re-DocRED document (any partition) were removed, and repeated texts within the distant set were reduced to one copy. This left 101,861 candidate documents. No sampled document shares a title or text with any labeled document.

**Sampling.** The documents were stratified to match Re-DocRED train on document length and entity count. Re-DocRED train was divided into a 5 × 5 grid by quintiles of token count and entity count. The 10,000 documents were allocated to the 25 cells in proportion to Re-DocRED train, and sampled at random within each cell with Python's `random.Random(571)`. Entity counts for the candidates came from the distant set's automatic mentions, which are not included.

| | Re-DocRED train | `unlabeled_10000.json` |
|---|---:|---:|
| Documents | 3,053 | 10,000 |
| Sentences | 24,256 | 81,107 |
| Tokens | 603,468 | 1,989,204 |
| Median / mean tokens per document | 179 / 197.7 | 180 / 198.9 |
| Median / mean sentences per document | 7 / 7.9 | 8 / 8.1 |

Kolmogorov–Smirnov distances to Re-DocRED train are 0.011 for tokens and 0.032 for sentences.

**Use.** Methods that need entity mentions have to predict them first. Code that reads a missing `labels` field as an empty list would treat these documents as having no relations. The included distant-supervision experiment reconstructs the released automatic mentions and Wikidata relations with `prepare_distant_supervision.py`, uses all 10,000 documents, and downweights their training rows to 0.25 based on dev performance.

## Record format

Each file is one JSON array. Each element is one document:

| Field | Type | Content |
|---|---|---|
| `title` | string | Wikipedia article title |
| `sents` | list of lists of strings | Tokens, one inner list per sentence |
| `vertexSet` | list of lists of mentions | Entities. Each inner list is one entity: the mentions that corefer. An entity's position in this list is its index. |
| `labels` | list | Relation instances |

A mention (second mention of entity 0 in the first dev document):

```json
{"name": "Schneider", "pos": [4, 5], "sent_id": 4, "type": "PER",
 "global_pos": [92, 92], "index": "0_1"}
```

- `sent_id` indexes `sents`, starting at 0.
- `pos` is `[start, end)` within `sents[sent_id]`, end exclusive.
- `type` is one of `PER`, `ORG`, `LOC`, `TIME`, `NUM`, `MISC`.
- `name` is the mention string. It does not always equal the tokens at `pos` joined by spaces (2,422 mentions across the three files differ, mostly in spacing around punctuation, such as `"Zest Airways, Inc."` against the tokens `Zest Airways , Inc.`). Use `pos` to locate mentions.
- `global_pos` and `index` are Re-DocRED additions not present in DocRED. In all three files, `global_pos` is the document-level token index of the mention's first token, written twice. `index` is a string of the form `"entity_mention"`. It matches the mention's current position in `vertexSet` for 98.97% of mentions and differs for the rest.

A relation instance (first label of the same document):

```json
{"r": "P580", "h": 11, "t": 6, "evidence": [2]}
```

- `h` and `t` index `vertexSet`. Relations hold between entities, not individual mentions.
- `r` is a Wikidata property ID. `data/rel_info.json` maps it to a name (`P580` is "start time").
- `evidence` lists sentence indices that support the relation. It may be empty (see above).
- An entity pair can have more than one relation.

## Reading the data

```python
import json

rel_names = json.load(open("data/rel_info.json"))
train = json.load(open("data/train_revised.json"))

doc = train[0]
for lab in doc["labels"]:
    head = doc["vertexSet"][lab["h"]][0]["name"]
    tail = doc["vertexSet"][lab["t"]][0]["name"]
    print(head, "|", rel_names[lab["r"]], "|", tail, "| evidence:", lab["evidence"])
```
