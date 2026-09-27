# Re-DocRED (packaged for CS571)

Re-DocRED is a re-annotated version of DocRED, a dataset for document-level entity and relation extraction on English Wikipedia articles. Each document comes with its entities (grouped mentions, with one of six types) and relation instances between entities, drawn from 96 Wikidata relations. The Re-DocRED authors added relation instances that DocRED annotators had missed, removed logically inconsistent labels, and corrected coreference errors. Unlike the original DocRED test set, the Re-DocRED test set includes its labels.

## Contents

| Path | Content |
|---|---|
| `data/train_revised.json` | Training partition, unchanged from the authors' repository |
| `data/dev_revised.json` | Validation (development) partition, unchanged |
| `data/test_revised.json` | Test partition, unchanged |
| `data/rel_info.json` | Relation ID to relation name map (96 entries), from the original DocRED release |
| `data/unlabeled_2000.json` | 2,000 unlabeled documents (text only) for semi-supervised learning, sampled from DocRED's distantly supervised set |
| `CITATION.bib` | BibTeX for Re-DocRED and DocRED |
| `LICENSE-Re-DocRED`, `LICENSE-DocRED` | MIT licenses of the two repositories |
| `SHA256SUMS` | Checksums of every file above |

## Citation

Cite both papers when using this data.

Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and Sharifah Mahani Aljunied. 2022. Revisiting DocRED - Addressing the False Negative Problem in Relation Extraction. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 8472–8487, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. https://aclanthology.org/2022.emnlp-main.580/

Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019. DocRED: A Large-Scale Document-Level Relation Extraction Dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 764–777, Florence, Italy. Association for Computational Linguistics. https://aclanthology.org/P19-1074/

## Source

- Re-DocRED data: https://github.com/tonytan48/Re-DocRED, commit `ccfb54f5ddf5836027c87badda10f6dfc56efaac` (21 August 2023), downloaded 22 September 2026.
- `rel_info.json`: the official DocRED data folder, https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw (linked from https://github.com/thunlp/DocRED).
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

`data/unlabeled_2000.json` holds 2,000 documents for semi-supervised learning. Each document has only `title` and `sents`, tokenized as in the labeled files. There is no `vertexSet` and no `labels` field.

**Source.** DocRED's distantly supervised training set, `train_distant.json` (101,873 documents; SHA-256 `db6d3cdaab8d36926318bb9339f6fd82d19dbacd186c74d7c20c734355a58b36`), from the official DocRED data folder listed under Source. Its documents come from English Wikipedia like the labeled files, and its entity mentions and relations were produced automatically. Both were removed.

**Exclusions.** Before sampling, 4 documents whose token sequences are identical to a human-annotated DocRED or Re-DocRED document (any partition) were removed, and repeated texts within the distant set were reduced to one copy. This left 101,861 candidate documents. No sampled document shares a title or text with any labeled document.

**Sampling.** The documents were stratified to match Re-DocRED train on document length and entity count. Re-DocRED train was divided into a 5 × 5 grid by quintiles of token count and entity count. The 2,000 documents were allocated to the 25 cells in proportion to Re-DocRED train, and sampled at random within each cell with Python's `random.Random(571)`. Entity counts for the candidates came from the distant set's automatic mentions, which are not included.

| | Re-DocRED train | `unlabeled_2000.json` |
|---|---:|---:|
| Documents | 3,053 | 2,000 |
| Sentences | 24,256 | 16,064 |
| Tokens | 603,468 | 394,544 |
| Median / mean tokens per document | 179 / 197.7 | 181 / 197.3 |
| Median / mean sentences per document | 7 / 7.9 | 8 / 8.0 |
| Median / mean entities per document | 19 / 19.4 | 19 / 19.4 (automatic mentions, removed) |

Kolmogorov–Smirnov distances to Re-DocRED train are 0.016 for tokens, 0.010 for entities and 0.017 for sentences.

**Use.** Methods that need entity mentions have to predict them first. Code that reads a missing `labels` field as an empty list would treat these documents as having no relations.

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
